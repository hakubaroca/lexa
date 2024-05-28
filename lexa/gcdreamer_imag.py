import networks
import tools
import models
import numpy as np
from tools import get_data_for_off_policy_training, get_future_goal_idxs, get_future_goal_idxs_neg_sampling

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_mlp_model(name, hidden_units, out_dim):
    model = nn.Sequential()
    for units in hidden_units:
        model.add_module(name, nn.Linear(units, activation=F.elu))
    model.add_module(name, nn.Linear(out_dim, activation=torch.tanh))
    return model

def assign_cond(x, cond, y):
    cond = cond.type(x.dtype)
    return x * (1 - cond) + y * cond

class GCDreamerBehavior(models.ImagBehavior):

    def __init__(self, config, world_model, stop_grad):
        super(GCDreamerBehavior, self).__init__(config, world_model, stop_grad)
        if self._config.gc_input == 'skills':
            self._rp_opt = tools.Optimizer(
                self._world_model.rev_pred.parameters(), config.rp_lr, config.opt_eps, config.rp_grad_clip, weight_decay=config.weight_decay)
        if self._config.gc_reward == 'dynamical_distance':
            assert self._config.dd_distance in ['steps_to_go', 'binary']
            if self._config.dd_loss == 'regression':
                dd_out_dim = 1 
                self.dd_loss_fn = nn.MSELoss()
            else:
                if self._config.dd_train_off_policy and self._config.dd_train_imag:
                    raise NotImplementedError
                dd_out_dim = self._config.imag_horizon if self._config.dd_distance == 'steps_to_go' else 2
                if self._config.dd_distance == 'binary':
                    dd_out_dim = 2
                elif self._config.dd_distance == 'steps_to_go':
                    dd_out_dim = self._config.imag_horizon
                    if self._config.dd_neg_sampling_factor > 0:
                        dd_out_dim += 1
                self.dd_loss_fn = nn.CrossEntropyLoss()

            if self._config.dd_train_off_policy and self._config.dd_train_imag:
                self.dd_seq_len = max(self._config.batch_length, self._config.imag_horizon)
            elif self._config.dd_train_imag:
                self.dd_seq_len = self._config.imag_horizon
            else:
                self.dd_seq_len = self._config.batch_length

            self.dd_out_dim = dd_out_dim
            self.dynamical_distance = networks.GC_Distance(out_dim=dd_out_dim, 
                                                           input_type=self._config.dd_inp, normalize_input=self._config.dd_norm_inp)
            self.dd_cur_idxs, self.dd_goal_idxs = get_future_goal_idxs(seq_len=self._config.imag_horizon, 
                                                                       bs=self._config.batch_size*self._config.batch_length)
            self._dd_opt = tools.Optimizer(
                self.dynamical_distance.parameters(), config.value_lr, config.opt_eps, config.value_grad_clip, weight_decay=config.weight_decay)

    def get_actor_inp(self, feat, goal, repeats=None):
        goal = goal.view(1, feat.shape[1], -1)
        goal = goal.repeat(feat.shape[0], 1, 1)
        if repeats:
            goal = goal.unsqueeze(2).repeat(1, 1, repeats, 1).view(feat.shape[0], -1, goal.shape[-1])
        return torch.cat([feat, goal], -1)

    def act(self, feat, obs, latent):
        goal = self._world_model.get_goal(latent)
        _state_rep_dict = {'feat': feat, 'embed': self._world_model.encoder(self._world_model.preprocess(obs))}
        state = _state_rep_dict[self._config.state_rep_for_policy]
        return self.actor(state, goal)

    def train_dd_off_policy(self, off_pol_obs):
        obs = off_pol_obs.permute(1, 0, 2)
        dd_loss = self.get_dynamical_distance_loss(obs, corr_factor=1)
        self._dd_opt.zero_grad()
        dd_loss.backward()
        self._dd_opt.step()
        return dd_loss

    def _gc_reward(self, feat, inp_state=None, action=None, obs=None):
        if self._config.gc_input == 'embed':
            inp_feat, goal_embed = torch.split(feat, [-1, self._world_model.embed_size], dim=-1)
            if self._config.gc_reward == 'l2':
                goal_feat = torch.stack([self._world_model.get_init_feat_embed(e) for e in goal_embed])
                return -torch.mean((goal_feat - inp_feat) ** 2, dim=-1)
            elif self._config.gc_reward == 'cosine':
                goal_feat = torch.stack([self._world_model.get_init_feat_embed(e) for e in goal_embed])
                norm = torch.norm(goal_feat, dim=-1) * torch.norm(inp_feat, dim=-1)
                dot_prod = (goal_feat.unsqueeze(2) @ inp_feat.unsqueeze(3)).squeeze()
                return dot_prod / (norm + 1e-8)
            elif self._config.gc_reward == 'dynamical_distance':
                if self._config.dd_inp == 'feat':
                    inp_feat = inp_state['stoch']
                    goal_feat = torch.stack([self._world_model.get_init_state_embed(e)['stoch'] for e in goal_embed])
                    if len(inp_feat.shape) == 2:
                        inp_feat = inp_feat.unsqueeze(0)
                    dd_out = self.dynamical_distance(torch.cat([inp_feat, goal_feat], dim=-1))
                elif self._config.dd_inp == 'embed': 
                    inp_embed = self._world_model.heads['embed'](inp_feat).mean()
                    dd_out = self.dynamical_distance(torch.cat([inp_embed, goal_embed], dim=-1))
                if self._config.dd_loss == 'regression':
                    reward = -dd_out 
                else:
                    reward = - torch.sum(dd_out * torch.arange(self.dd_out_dim, device=dd_out.device), dim=-1)
                    if self._config.dd_distance == 'steps_to_go':
                        reward /= self.dd_seq_len
                return reward

        elif 'feat' in self._config.gc_input:
            inp_feat, goal_feat = torch.split(feat, 2, dim=-1)
            if self._config.gc_reward == 'l2':
                return -torch.mean((goal_feat - inp_feat) ** 2, dim=-1)
            elif self._config.gc_reward == 'cosine':
                norm = torch.norm(goal_feat, dim=-1) * torch.norm(inp_feat, dim=-1)
                dot_prod = (goal_feat.unsqueeze(2) @ inp_feat.unsqueeze(3)).squeeze()
                return dot_prod / (norm + 1e-8)
            elif self._config.gc_reward == 'dynamical_distance':
                raise AssertionError('should use embed as gc_input')

        elif self._config.gc_input == 'skills':
            if not inp_state:
                raise NotImplementedError
            inp_feat, skill = torch.split(feat, [-1, self._config.skill_dim], dim=-1)
            if self._config.skill_pred_input == 'embed':
                rp_inp = inp_embed = self._world_model.heads['embed'](inp_feat).mean()
            else:
                rp_inp = {**inp_state, 'feat': inp_feat}[self._config.skill_pred_input]
            if self._config.skill_pred_noise > 0:
                noise = torch.randn_like(rp_inp) * self._config.skill_pred_noise
                rp_inp += noise
            pred_skill = self._world_model.rev_pred(rp_inp)
            loss = (pred_skill - skill) ** 2
            return -torch.mean(loss, dim=-1)
        
        if self._config.gc_reward == 'env':
            return self._world_model.heads['reward'](feat).mean()

    def get_dynamical_distance_loss(self, _data, corr_factor=None):
        seq_len, bs = _data.shape[:2]
        def _helper(cur_idxs, goal_idxs, distance):
            loss = 0
            cur_states = _data[cur_idxs[:, 0], cur_idxs[:, 1]]
            goal_states = _data[goal_idxs[:, 0], goal_idxs[:, 1]]
            pred = self.dynamical_distance(torch.cat([cur_states, goal_states], dim=-1))
            if self._config.dd_loss == 'regression':
                _label = distance
                if self._config.dd_norm_reg_label and self._config.dd_distance == 'steps_to_go':
                    _label = _label / self.dd_seq_len
                loss += torch.mean((pred - _label) ** 2)
            else:
                _label = F.one_hot(distance.long(), num_classes=self.dd_out_dim).float()
                loss += self.dd_loss_fn(pred, _label)
            return loss

        idxs = np.random.choice(np.arange(len(self.dd_cur_idxs)), self._config.dd_num_positives)
        loss = _helper(self.dd_cur_idxs[idxs], self.dd_goal_idxs[idxs], self.dd_goal_idxs[idxs][:, 0] - self.dd_cur_idxs[idxs][:, 0])

        if self._config.dd_neg_sampling_factor > 0:
            num_negs = int(self._config.dd_neg_sampling_factor * self._config.dd_num_positives)
            neg_cur_idxs, neg_goal_idxs = get_future_goal_idxs_neg_sampling(num_negs, seq_len, bs, corr_factor)
            loss += _helper(neg_cur_idxs, neg_goal_idxs, torch.ones(num_negs, device=_data.device) * seq_len)
        
        return loss

    def train(self, start, imagine=None, tape=None, repeats=None, obs=None):
        self._update_slow_target()
        metrics = {}

        with torch.set_grad_enabled(True):
            obs = self._world_model.preprocess(obs)
            goal = self._world_model.get_goal(obs, training=True)

            imag_feat, imag_state, imag_action = self._imagine(
                start, self.actor, self._config.imag_horizon, repeats, goal)
            actor_inp = self.get_actor_inp(imag_feat, goal)
            reward = self._gc_reward(actor_inp, imag_state, imag_action, obs)
            if self._config.gc_input == 'skills':
                rp_loss = -torch.mean(reward).float()

            actor_ent = self.actor(actor_inp).entropy()
            state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()

            target, weights = self._compute_target(
                actor_inp, reward, actor_ent, state_ent,
                self._config.slow_actor_target)

            actor_loss, mets = self._compute_actor_loss(
                actor_inp, imag_state, imag_action, target, actor_ent, state_ent,
                weights)
            metrics.update(mets)

        if self._config.slow_value_target != self._config.slow_actor_target:
            target, weights = self._compute_target(
                actor_inp, reward, actor_ent, state_ent,
                self._config.slow_value_target)

        metrics['reward_mean'] = torch.mean(reward)
        metrics['reward_std'] = torch.std(reward)
        metrics['actor_ent'] = torch.mean(actor_ent)

        self._actor_opt.zero_grad()
        actor_loss.backward()
        self._actor_opt.step()

        if self._config.gc_input == 'skills':
            self._rp_opt.zero_grad()
            rp_loss.backward()
            self._rp_opt.step()

        if self._config.gc_reward == 'dynamical_distance' and self._config.dd_train_imag:
            with torch.set_grad_enabled(True):
                _inp = imag_state['stoch'] if 'feat' in self._config.dd_inp \
                    else self._world_model.heads['embed'](imag_feat).mean()
                dd_loss = self.get_dynamical_distance_loss(_inp)
            self._dd_opt.zero_grad()
            dd_loss.backward()
            self._dd_opt.step()

        with torch.set_grad_enabled(True):
            value = self.value(actor_inp)[:-1]
            value_loss = -value.log_prob(target.detach())
            if self._config.value_decay:
                value_loss += self._config.value_decay * value.mean()
            value_loss = (weights[:-1] * value_loss).mean()
        self._value_opt.zero_grad()
        value_loss.backward()
        self._value_opt.step()

        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon, repeats=None, goal=None):
        dynamics = self._world_model.dynamics
        goal = goal.view(-1, goal.shape[-1])
        if repeats:
            start = {k: v.repeat(repeats, 1) for k, v in start.items()}
            goal = goal.repeat(repeats, 1)

        start = {k: v.view(-1, *v.shape[2:]) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach() if self._stop_grad_actor else feat
            action = policy(inp, goal).sample()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, feat, action
        
        feat = torch.zeros_like(dynamics.get_feat(start))
        action = policy(feat, goal).mode()
        succ, feats, actions = tools.static_scan(
            step, torch.arange(horizon), (start, feat, action))

        states = {k: torch.cat([
            start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            def unfold(tensor):
                s = tensor.shape
                return tensor.view(s[0], s[1] // repeats, repeats, *s[2:])
            states, feats, actions = tools.nest.map_structure(
                unfold, (states, feats, actions))

        return feats, states, actions

# Example usage
# config = Config()  # Define your configuration
# world_model = WorldModel()  # Initialize your world model
# behavior = GCDreamerBehavior(config, world_model, stop_grad=True)
# data = get_dummy_data()  # Define your data
# behavior.train(data)

"""
# Define your configuration
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

# Dummy world model class (these should be replaced with actual implementations)
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
        'image': torch.randn(batch_size, seq_len, 3, 64, 64),
        'image_goal': torch.randn(batch_size, 3, 64, 64),
        'goal': torch.randn(batch_size, state_dim),
        'skill': torch.randn(batch_size, 10)
    }

config = Config()
world_model = WorldModel()
gc_dreamer_behavior = GCDreamerBehavior(config, world_model, stop_grad=True)

# Example data
data = get_dummy_data(batch_size=8, seq_len=16, state_dim=128, action_dim=config.num_actions)

# Train the GCDreamerBehavior
start = {
    'state': data['state'][:, 0],
    'action': data['action'][:, 0],
}
obs = data
gc_dreamer_behavior.train(start, obs=obs)
"""
