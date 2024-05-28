import networks
import tools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Independent


class WorldModel(nn.Module):
    def __init__(self, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._config = config
        self._float = torch.float32  # Change this as per your requirement

        encoder_cls = {
            'vanilla': networks.ConvEncoder,
            'with_state': networks.ConvEncoderWithState
        }[config.encoder_cls]
        self.encoder = encoder_cls(config.cnn_depth, config.act, config.encoder_kernels)
        self.embed_size = self.encoder.embed_size

        self.dynamics = networks.RSSM(
            config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
            config.dyn_input_layers, config.dyn_output_layers, config.dyn_shared,
            config.dyn_discrete, config.act, config.dyn_mean_act,
            config.dyn_std_act, config.dyn_min_std, config.dyn_cell,
            'stoch' if config.gc_input == 'feat_stoch' else 'full'
        )

        self.heads = {}
        channels = 1 if config.atari_grayscale else 3
        shape = config.size + (channels,)
        self.heads['image'] = networks.ConvDecoder(
            config.cnn_depth, config.act, shape, config.decoder_kernels,
            config.decoder_thin)

        if config.pred_reward:
            self.heads['reward'] = networks.DenseHead(
                [], config.reward_layers, config.units, config.act)

        if config.pred_discount:
            self.heads['discount'] = networks.DenseHead(
                [], config.discount_layers, config.units, config.act, dist='binary')

        if config.pred_embed:
            self.heads['embed'] = networks.DenseHead(
                [self.embed_size], config.value_layers, config.units, config.act)

        self._model_opt = tools.Optimizer(
            'model', config.model_lr, config.opt_eps, config.grad_clip,
            config.weight_decay, opt=config.opt)

        self._scales = dict(
            reward=config.reward_scale, discount=config.discount_scale)

    def train(self, data):
        data = self.preprocess(data)
        optimizer = self._model_opt

        embed = self.encoder(data)
        data['embed'] = embed  # Needed for the embed head
        post, prior = self.dynamics.observe(embed, data['action'])
        kl_balance = tools.schedule(self._config.kl_balance, self._step)
        kl_free = tools.schedule(self._config.kl_free, self._step)
        kl_scale = tools.schedule(self._config.kl_scale, self._step)
        kl_loss, kl_value = self.dynamics.kl_loss(post, prior, kl_balance, kl_free, kl_scale)
        feat = self.dynamics.get_feat(post)

        likes = {}
        for name, head in self.heads.items():
            grad_head = name in self._config.grad_heads
            inp = feat if grad_head else feat.detach()
            pred = head(inp, torch.float32)
            like = pred.log_prob(data[name].float())
            likes[name] = torch.mean(like) * self._scales.get(name, 1.0)

        model_loss = kl_loss - sum(likes.values())
        optimizer.zero_grad()
        model_loss.backward()
        optimizer.step()

        metrics = {f'{name}_loss': -like for name, like in likes.items()}
        metrics['kl_balance'] = kl_balance
        metrics['kl_free'] = kl_free
        metrics['kl_scale'] = kl_scale
        metrics['kl'] = torch.mean(kl_value)
        metrics['prior_ent'] = self.dynamics.get_dist(prior).entropy().mean()
        metrics['post_ent'] = self.dynamics.get_dist(post).entropy().mean()

        return embed, post, feat, kl_value, metrics

    def preprocess(self, obs):
        dtype = self._float
        obs = {k: v.clone() for k, v in obs.items()}
        for k in obs.keys():
            if 'image' in k:
                obs[k] = obs[k].type(dtype) / 255.0 - 0.5
        if self._config.clip_rewards[0] == 'd':
            r_transform = lambda r: r / float(self._config.clip_rewards[1:])
        else:
            r_transform = getattr(torch, self._config.clip_rewards)
        obs['reward'] = r_transform(obs['reward'])
        if 'discount' in obs:
            obs['discount'] *= self._config.discount
        for key, value in obs.items():
            if value is None:
                continue
            obs[key] = value = torch.tensor(value)
            if value.dtype in (torch.float16, torch.float32, torch.float64):
                obs[key] = value.type(dtype)
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        truth = data['image'][:6] + 0.5
        embed = self.encoder(data)
        states, _ = self.dynamics.observe(embed[:6, :5], data['action'][:6, :5])
        recon = self.heads['image'](self.dynamics.get_feat(states)).mean[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine(data['action'][:6, 5:], init)
        openl = self.heads['image'](self.dynamics.get_feat(prior)).mean
        model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        return torch.cat([truth, model, error], 2)

    def get_init_feat(self, obs, state=None, sample=False):
        if state is None:
            batch_size = len(obs['image'])
            latent = self.dynamics.initial(len(obs['image']))
            dtype = self._float
            action = torch.zeros((batch_size, self._config.num_actions), dtype=dtype)
        else:
            latent, action = state
        embed = self.encoder(obs)
        latent, _ = self.dynamics.obs_step(latent, action, embed, sample)
        feat = self.dynamics.get_feat(latent)
        return feat, latent

    def get_init_feat_embed(self, embed, sample=False):
        latent = self.get_init_state_embed(embed, sample)
        feat = self.dynamics.get_feat(latent)
        return feat

    def get_init_state_embed(self, embed, sample=False):
        batch_size = embed.shape[0]
        latent = self.dynamics.initial(batch_size)
        dtype = self._float
        action = torch.zeros((batch_size, self._config.num_actions), dtype=dtype)
        latent, _ = self.dynamics.obs_step(latent, action, embed, sample)
        return latent

# Example usage
# config = ... # Define your configuration
# step = ... # Define your step
# world_model = WorldModel(step, config)
# data = ... # Define your data
# embed, post, feat, kl_value, metrics = world_model.train(data)


class ImagBehavior(nn.Module):

    def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
        super(ImagBehavior, self).__init__()
        self._config = config
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor
        self._reward = reward
        self.actor = networks.ActionHead(
            config.num_actions, config.actor_layers, config.units, config.act,
            config.actor_dist, config.actor_init_std, config.actor_min_std,
            config.actor_dist, config.actor_temp, config.actor_outscale)
        self.value = networks.DenseHead(
            [], config.value_layers, config.units, config.act,
            config.value_head)

        if config.slow_value_target or config.slow_actor_target:
            self._slow_value = networks.DenseHead(
                [], config.value_layers, config.units, config.act)
            self._updates = 0

        self._actor_opt = optim.Adam(self.actor.parameters(), lr=config.actor_lr, eps=config.opt_eps, weight_decay=config.weight_decay)
        self._value_opt = optim.Adam(self.value.parameters(), lr=config.value_lr, eps=config.opt_eps, weight_decay=config.weight_decay)

    def train(self, start, objective=None, imagine=None, tape=None, repeats=None, obs=None):
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}
        actor_tape = torch.autograd.set_detect_anomaly(True)
        with torch.set_grad_enabled(True):
            assert bool(objective) != bool(imagine)
            if objective:
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon, repeats)
                reward = objective(imag_feat, imag_state, imag_action)
            else:
                imag_feat, imag_state, imag_action, reward = imagine(start)
            actor_ent = self.actor(imag_feat).entropy()
            state_ent = self._world_model.dynamics.get_dist(
                imag_state).entropy()
            target, weights = self._compute_target(
                imag_feat, reward, actor_ent, state_ent,
                self._config.slow_actor_target)
            actor_loss, mets = self._compute_actor_loss(
                imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
                weights)
            metrics.update(mets)
        if self._config.slow_value_target != self._config.slow_actor_target:
            target, weights = self._compute_target(
                imag_feat, reward, actor_ent, state_ent,
                self._config.slow_value_target)
        with torch.set_grad_enabled(True):
            value = self.value(imag_feat)[:-1]
            value_loss = -value.log_prob(target.detach())
            if self._config.value_decay:
                value_loss += self._config.value_decay * value.mode()
            value_loss = (weights[:-1] * value_loss).mean()
        metrics['reward_mean'] = reward.mean()
        metrics['reward_std'] = reward.std()
        metrics['actor_ent'] = actor_ent.mean()
        self._actor_opt.zero_grad()
        actor_loss.backward()
        self._actor_opt.step()
        self._value_opt.zero_grad()
        value_loss.backward()
        self._value_opt.step()
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon, repeats=None):
        dynamics = self._world_model.dynamics
        if repeats:
            start = {k: v.repeat(repeats, 1) for k, v in start.items()}
        start = {k: v.view(-1, *v.shape[2:]) for k, v in start.items()}
        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach() if self._stop_grad_actor else feat
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, feat, action
        feat = torch.zeros_like(dynamics.get_feat(start))
        action = policy(feat).mode()
        succ, feats, actions = tools.static_scan(
            step, torch.arange(horizon), (start, feat, action))
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            def unfold(tensor):
                s = tensor.shape
                return tensor.view(s[0], s[1] // repeats, repeats, *s[2:])
            states, feats, actions = torch.nest.map_structure(
                unfold, (states, feats, actions))
        return feats, states, actions

    def _compute_target(self, imag_feat, reward, actor_ent, state_ent, slow):
        reward = reward.float()
        if 'discount' in self._world_model.heads:
            discount = self._world_model.heads['discount'](
                imag_feat).mean()
        else:
            discount = self._config.discount * torch.ones_like(reward)
        if self._config.future_entropy and self._config.actor_entropy() > 0:
            reward += self._config.actor_entropy() * actor_ent
        if self._config.future_entropy and self._config.actor_state_entropy() > 0:
            reward += self._config.actor_state_entropy() * state_ent

        if slow:
            value = self._slow_value(imag_feat).mode()
        else:
            value = self.value(imag_feat).mode()

        target = tools.lambda_return(
            reward[:-1], value[:-1], discount[:-1],
            bootstrap=value[-1], lambda_=self._config.discount_lambda, axis=0)

        weights = torch.cumprod(torch.cat(
            [torch.ones_like(discount[:1]), discount[:-1]], 0), 0)
        return target, weights

    def _compute_actor_loss(self, imag_feat, imag_state, imag_action, target, actor_ent, state_ent, weights):
        metrics = {}
        inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
        policy = self.actor(inp)
        actor_ent = policy.entropy()
        if self._config.imag_gradient == 'dynamics':
            actor_target = target
        elif self._config.imag_gradient == 'reinforce':
            imag_action = imag_action.float()
            actor_target = policy.log_prob(imag_action)[:-1] * (target - self.value(imag_feat[:-1]).mode()).detach()
        elif self._config.imag_gradient == 'both':
            imag_action = imag_action.float()
            actor_target = policy.log_prob(imag_action)[:-1] * (target - self.value(imag_feat[:-1]).mode()).detach()
            mix = self._config.imag_gradient_mix()
            actor_target = mix * target + (1 - mix) * actor_target
            metrics['imag_gradient_mix'] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        if not self._config.future_entropy and self._config.actor_entropy() > 0:
            actor_target += self._config.actor_entropy() * actor_ent[:-1]
        if not self._config.future_entropy and self._config.actor_state_entropy() > 0:
            actor_target += self._config.actor_state_entropy() * state_ent[:-1]
        actor_loss = -(weights[:-1] * actor_target).mean()
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.slow_value_target or self._config.slow_actor_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data.copy_(mix * s.data + (1 - mix) * d.data)
            self._updates += 1

    def act(self, feat, *args):
        return self.actor(feat)
    

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    def __init__(self):
        self.num_actions = 4
        self.actor_layers = 3
        self.units = 256
        self.act = F.elu
        self.actor_dist = 'normal'
        self.actor_init_std = 0.0
        self.actor_min_std = 0.1
        self.actor_temp = 1.0
        self.actor_outscale = 0.0
        self.value_layers = 3
        self.value_head = 'normal'
        self.slow_value_target = True
        self.slow_actor_target = True
        self.weight_decay = 0.0
        self.opt = 'adam'
        self.actor_lr = 0.001
        self.opt_eps = 1e-8
        self.actor_grad_clip = 100.0
        self.value_lr = 0.001
        self.value_grad_clip = 100.0
        self.imag_horizon = 15
        self.imag_sample = True
        self.future_entropy = False
        self.discount_lambda = 0.95
        self.discount = 0.99
        self.actor_entropy = lambda: 0.1
        self.actor_state_entropy = lambda: 0.1
        self.imag_gradient = 'dynamics'
        self.imag_gradient_mix = lambda: 0.5
        self.slow_target_update = 100
        self.slow_target_fraction = 0.99

# Dummy networks classes (these should be replaced with actual implementations)
class ActionHead(nn.Module):
    def __init__(self, num_actions, layers, units, act, dist, init_std, min_std, dist_param, temp, outscale):
        super(ActionHead, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(units, units) for _ in range(layers)])
        self.mean_layer = nn.Linear(units, num_actions)
        self.log_std_layer = nn.Linear(units, num_actions)
        self.act = act

    def forward(self, x):
        for layer in self.layers:
            x = self.act(layer(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        std = torch.exp(log_std)
        return Normal(mean, std)

class DenseHead(nn.Module):
    def __init__(self, shape, layers, units, act, head):
        super(DenseHead, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(units, units) for _ in range(layers)])
        self.output_layer = nn.Linear(units, shape)
        self.act = act

    def forward(self, x):
        for layer in self.layers:
            x = self.act(layer(x))
        return self.output_layer(x)

# Dummy world model class (this should be replaced with an actual implementation)
class WorldModel(nn.Module):
    def __init__(self):
        super(WorldModel, self).__init__()
        self.dynamics = None

# Dummy data preparation
def get_dummy_data(batch_size, seq_len, state_dim, action_dim):
    return {
        'state': torch.randn(batch_size, seq_len, state_dim),
        'action': torch.randint(0, action_dim, (batch_size, seq_len)),
        'reward': torch.randn(batch_size, seq_len),
        'image': torch.randn(batch_size, seq_len, 3, 64, 64)
    }

# Reward function (this should be replaced with an actual implementation)
def reward_function(feat, state, action):
    return torch.randn(feat.shape[0], feat.shape[1])

config = Config()
world_model = WorldModel()
imag_behavior = ImagBehavior(config, world_model, stop_grad_actor=True, reward=reward_function)

# Example data
data = get_dummy_data(batch_size=8, seq_len=16, state_dim=128, action_dim=config.num_actions)

# Training step
start = {
    'state': data['state'][:, 0],
    'action': data['action'][:, 0],
}
imagine = None  # Define this if needed

# Train the ImagBehavior model
imag_feat, imag_state, imag_action, weights, metrics = imag_behavior.train(start, objective=reward_function, imagine=imagine)
print('Training metrics:', metrics)

"""
