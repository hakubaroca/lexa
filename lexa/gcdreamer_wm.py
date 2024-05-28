import networks
import tools
import models
import numpy as np
from tools import get_data_for_off_policy_training

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

class GCWorldModel(models.WorldModel):
    def __init__(self, step, config):
        super(GCWorldModel, self).__init__(step, config)

        if config.pred_stoch_state:
            self.heads['stoch_state'] = networks.DenseHead(
                [self._config.dyn_stoch], config.value_layers, config.units, config.act, std='learned')

        if self._config.gc_reward == 'feat_pca':
            feat_size = self.dynamics._stoch + self.dynamics._deter
            self.feat_roll_mean = torch.zeros((feat_size,))
            self.feat_roll_cvar = torch.zeros((feat_size, feat_size))

        if self._config.gc_input == 'skills':
            self.rev_pred = networks.get_mlp_model('feat_rp', [256, 128], self._config.skill_dim)
            if self._config.double_rev_pred:
                self.embed_rev_pred = networks.get_mlp_model('embed_rp', [256, 128], self._config.skill_dim)

    def train(self, data):
        data = self.preprocess(data)
        optimizer = self._model_opt

        embed = self.encoder(data)
        data['embed'] = embed.detach()  # Needed for the embed head
        post, prior = self.dynamics.observe(embed, data['action'])
        data['stoch_state'] = post['stoch'].detach()  # Needed for the embed head
        kl_balance = tools.schedule(self._config.kl_balance, self._step)
        kl_free = tools.schedule(self._config.kl_free, self._step)
        kl_scale = tools.schedule(self._config.kl_scale, self._step)
        kl_loss, kl_value = self.dynamics.kl_loss(post, prior, kl_balance, kl_free, kl_scale)
        feat = self.dynamics.get_feat(post)

        likes = {}
        for name, head in self.heads.items():
            grad_head = (name in self._config.grad_heads)
            inp = feat
            if name == 'reward':
                inp = torch.cat([feat, self.get_goal(data)], dim=-1)
            if name == 'stoch_state':
                inp = embed
            if not grad_head:
                inp = inp.detach()
            pred = head(inp)
            like = pred.log_prob(data[name].float())
            likes[name] = like.mean() * self._scales.get(name, 1.0)
        model_loss = kl_loss - sum(likes.values())
        if self._config.latent_constraint == 'consecutive_state_l2':
            seq_len = feat.shape[1]
            for i in range(seq_len-1):
                model_loss += 0.1 * ((feat[:, i, :] - feat[:, i+1, :]) ** 2).mean().float()

        if self._config.gc_input == 'skills' and self._config.double_rev_pred:
            rp_inp = {**post, 'feat': feat}[self._config.skill_pred_input]
            skill_target = self.rev_pred(rp_inp).detach()
            skill_loss = ((self.embed_rev_pred(embed) - skill_target) ** 2).mean()
            model_loss += skill_loss.float()

        model_parts = [self.encoder, self.dynamics] + list(self.heads.values())
        if self._config.gc_input == 'skills' and self._config.double_rev_pred:
            model_parts += [self.embed_rev_pred]

        optimizer.zero_grad()
        model_loss.backward()
        optimizer.step()

        metrics = {f'{name}_loss': -like for name, like in likes.items()}
        metrics['kl_balance'] = kl_balance
        metrics['kl_free'] = kl_free
        metrics['kl_scale'] = kl_scale
        metrics['kl'] = kl_value.mean()
        metrics['prior_ent'] = self.dynamics.get_dist(prior).entropy().mean()
        metrics['post_ent'] = self.dynamics.get_dist(post).entropy().mean()
        return embed, post, feat, kl_value, metrics

    def get_goal(self, obs, training=False):
        if self._config.gc_input == 'state':
            assert self._config.training_goals == 'env'
            return obs['goal']
        else:
            if not training or self._config.training_goals == 'env':
                _embed = self.encoder({'image': obs['image_goal'], 'state': obs.get('goal', None)})
                if self._config.gc_input == 'embed':
                    return _embed
                elif 'feat' in self._config.gc_input:
                    return self.get_init_feat_embed(_embed) if len(_embed.shape) == 2 else torch.stack([self.get_init_feat_embed(e) for e in _embed])
                elif self._config.gc_input == 'skills':
                    if 'skill' in obs and obs['skill'].numel() > 0:
                        return obs['skill']
                    if self._config.double_rev_pred:
                        return self.embed_rev_pred(_embed)
                    if self._config.skill_pred_input == 'embed':
                        return self.rev_pred(_embed)
                    raise NotImplementedError

            elif self._config.training_goals == 'batch' and self._config.gc_input == 'skills':
                sh = obs['image_goal'].shape
                if len(sh) == 4:
                    return torch.rand((sh[0], self._config.skill_dim), dtype=self._float)
                elif len(sh) == 5:
                    return torch.rand((sh[0], sh[1], self._config.skill_dim), dtype=self._float)

            elif self._config.training_goals == 'batch':
                goal_embed = self.encoder(obs)
                sh = goal_embed.shape
                goal_embed = goal_embed.view(-1, sh[-1])
                ids = torch.randperm(goal_embed.shape[0])
                if self._config.labelled_env_multiplexing:
                    l = goal_embed.shape[0]
                    env_ids = obs['env_idx'].view(-1)
                    oh_ids = torch.nn.functional.one_hot(env_ids, l)
                    cooc = oh_ids[env_ids]
                    ids = torch.multinomial(cooc / cooc.sum(dim=1, keepdim=True), 1).squeeze(1)
                goal_embed = goal_embed[ids]
                goal_embed = goal_embed.view(sh)

                if 'feat' in self._config.gc_input:
                    return torch.stack([self.get_init_feat_embed(e) for e in goal_embed])
                else:
                    return goal_embed

    def get_init_feat(self, obs, state=None, sample=False):
        if state is None:
            batch_size = len(obs['image'])
            latent = self.dynamics.initial(len(obs['image']))
            action = torch.zeros((batch_size, self._config.num_actions), dtype=self._float)
            goal = obs['image_goal']
            goal_state = obs['goal']
            skill = obs['skill']
        else:
            latent, action = state
            goal = latent['image_goal']
            goal_state = latent['goal']
            skill = latent['skill']
        embed = self.encoder(obs)
        latent, _ = self.dynamics.obs_step(latent, action, embed, sample)
        latent['image_goal'] = goal
        latent['goal'] = goal_state
        latent['skill'] = skill
        feat = self.dynamics.get_feat(latent)
        return feat, latent


"""
import torch
import torch.optim as optim

# Config and Network Classes
class Config:
    # Define the required parameters as in the previous example
    pass

class WorldModel:
    def __init__(self):
        self.dynamics = None

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
gc_world_model = GCWorldModel(step=0, config=config)

# Example data
data = get_dummy_data(batch_size=8, seq_len=16, state_dim=128, action_dim=config.num_actions)

# Train the GCWorldModel
embed, post, feat, kl_value, metrics = gc_world_model.train(data)
print('Training metrics:', metrics)

"""