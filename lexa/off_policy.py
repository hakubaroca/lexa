import torch
import torch.nn as nn
import torch.optim as optim

import networks
import tools
import models
import numpy as np
from tools import get_data_for_off_policy_training

class GCOffPolicyOpt:
    def __init__(self, config):
        kw = dict(wd=config.weight_decay, opt=config.opt)

        self._config = config
        self.actor = networks.GC_Actor(config.num_actions, from_images=not self._config.offpolicy_use_embed)
        self._actor_opt = tools.Optimizer('actor', config.actor_lr, config.opt_eps, config.actor_grad_clip, **kw)

    def train_gcbc(self, obs, prev_actions):
        metrics = {}
        s_t, a_t, _, _ = get_data_for_off_policy_training(
            obs[:, :-1], prev_actions[:, 1:], obs[:, 1:], obs[:, 1:], 
            self._config.relabel_mode, relabel_fraction=1.0
        )

        if self._config.gcbc_distance_weighting:
            raise NotImplementedError
        else:
            pred_action = self.actor(s_t)
            loss = torch.mean((pred_action - a_t) ** 2)
        
        self._actor_opt.zero_grad()
        loss.backward()
        self._actor_opt.step()

        metrics.update({'loss': loss.item()})
        metrics = {'replay_' + k: v for k, v in metrics.items()}
        return metrics
