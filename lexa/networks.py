import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Independent, kl_divergence, Bernoulli
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

import tools


class RSSM(nn.Module):
    def __init__(self, stoch=30, deter=200, hidden=200, layers_input=1, layers_output=1,
                 shared=False, discrete=False, act=F.elu, mean_act='none',
                 std_act='softplus', min_std=0.1, cell='gru', feat_mode='full'):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._layers_input = layers_input
        self._layers_output = layers_output
        self._shared = shared
        self._discrete = discrete
        self._act = act
        self._mean_act = mean_act
        self._std_act = std_act
        self._embed = None
        self.feat_mode = feat_mode
        self.feat_size = stoch + deter

        if cell == 'gru':
            self._cell = nn.GRUCell(self._hidden, self._deter)
        else:
            raise NotImplementedError(cell)

        self.initial_state = None

    def initial(self, batch_size, device):
        dtype = torch.float32
        if self._discrete:
            state = {
                'logit': torch.zeros([batch_size, self._stoch, self._discrete], dtype=dtype, device=device),
                'stoch': torch.zeros([batch_size, self._stoch, self._discrete], dtype=dtype, device=device),
                'deter': torch.zeros([batch_size, self._deter], dtype=dtype, device=device)
            }
        else:
            state = {
                'mean': torch.zeros([batch_size, self._stoch], dtype=dtype, device=device),
                'std': torch.zeros([batch_size, self._stoch], dtype=dtype, device=device),
                'stoch': torch.zeros([batch_size, self._stoch], dtype=dtype, device=device),
                'deter': torch.zeros([batch_size, self._deter], dtype=dtype, device=device)
            }
        return state

    def observe(self, embed, action, state=None):
        swap = lambda x: x.permute(1, 0, *range(2, len(x.shape)))
        if state is None:
            state = self.initial(action.size(0), action.device)
        embed, action = swap(embed), swap(action)
        post, prior = self.static_scan(lambda prev, inputs: self.obs_step(prev[0], *inputs),
                                       (action, embed), (state, state))
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        swap = lambda x: x.permute(1, 0, *range(2, len(x.shape)))
        if state is None:
            state = self.initial(action.size(0), action.device)
        action = swap(action)
        prior = self.static_scan(self.img_step, action, state)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state['stoch']
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.view(*shape)
        if self.feat_mode == 'stoch':
            return stoch
        else:
            return torch.cat([stoch, state['deter']], dim=-1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state['logit']
            dist = Independent(tools.OneHotDist(logit), 1)
        else:
            mean, std = state['mean'], state['std']
            dist = Independent(Normal(mean, std), 1)
        return dist

    def obs_step(self, prev_state, prev_action, embed, sample=True):
        if self._embed is None:
            self._embed = embed.shape[-1]
        prior = self.img_step(prev_state, prev_action, None, sample)
        if self._shared:
            post = self.img_step(prev_state, prev_action, embed, sample)
        else:
            x = torch.cat([prior['deter'], embed], dim=-1)
            for i in range(self._layers_output):
                x = self.get(f'obi{i}', nn.Linear, self._hidden, self._act)(x)
            stats = self._suff_stats_layer('obs', x)
            if sample:
                stoch = self.get_dist(stats).sample()
            else:
                stoch = self.get_dist(stats).mean
            post = {'stoch': stoch, 'deter': prior['deter'], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, embed=None, sample=True):
        prev_stoch = prev_state['stoch']
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            prev_stoch = prev_stoch.view(*shape)
        if self._shared:
            if embed is None:
                shape = list(prev_action.shape[:-1]) + [self._embed]
                embed = torch.zeros(shape, dtype=prev_action.dtype, device=prev_action.device)
            x = torch.cat([prev_stoch, prev_action, embed], dim=-1)
        else:
            x = torch.cat([prev_stoch, prev_action], dim=-1)
        for i in range(self._layers_input):
            x = self.get(f'ini{i}', nn.Linear, self._hidden, self._act)(x)
        x = self._cell(x, prev_state['deter'])
        deter = x[0] if isinstance(x, tuple) else x
        for i in range(self._layers_output):
            x = self.get(f'imo{i}', nn.Linear, self._hidden, self._act)(x)
        stats = self._suff_stats_layer('ims', x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mean
        prior = {'stoch': stoch, 'deter': deter, **stats}
        return prior

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            x = self.get(name, nn.Linear, self._stoch * self._discrete, None)(x)
            logit = x.view(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {'logit': logit}
        else:
            x = self.get(name, nn.Linear, 2 * self._stoch, None)(x)
            mean, std = torch.split(x, self._stoch, dim=-1)
            if self._mean_act == 'tanh5':
                mean = 5.0 * torch.tanh(mean / 5.0)
            std = {
                'softplus': lambda: F.softplus(std),
                'abs': lambda: torch.abs(std + 1),
                'sigmoid': lambda: torch.sigmoid(std),
                'sigmoid2': lambda: 2 * torch.sigmoid(std / 2)
            }[self._std_act]()
            std = std + self._min_std
            return {'mean': mean, 'std': std}

    def kl_loss(self, post, prior, balance, free, scale):
        dist = lambda x: self.get_dist(x, torch.float32)
        if balance == 0.5:
            value = kl_divergence(dist(prior), dist(post))
            loss = torch.mean(torch.maximum(value, free))
        else:
            sg = lambda x: {k: v.detach() for k, v in x.items()}
            value = kl_divergence(dist(prior), dist(sg(post)))
            pri = torch.mean(torch.maximum(value, free))
            pos = torch.mean(torch.maximum(kl_divergence(dist(sg(prior)), dist(post)), free))
            loss = balance * pri + (1 - balance) * pos
        loss *= scale
        return loss, value

    def static_scan(self, fn, inputs, start_state):
        outputs = []
        state = start_state
        for input in zip(*inputs):
            state = fn(state, *input)
            outputs.append(state)
        return state, outputs

    def get(self, name, layer, *args, **kwargs):
        if not hasattr(self, name):
            setattr(self, name, layer(*args, **kwargs))
        return getattr(self, name)

class GC_Encoder(nn.Module):
    def __init__(self, depth=8, act=F.leaky_relu, kernels=(4, 4, 4, 4)):
        super().__init__()
        self._act = act
        self._depth = depth
        self._kernels = kernels
        self.convs = nn.ModuleList([nn.Conv2d(1 * self._depth, self._kernels[0], stride=2),
                                     nn.Conv2d(2 * self._depth, self._kernels[1], stride=2),
                                     nn.Conv2d(4 * self._depth, self._kernels[2], stride=2),
                                     nn.Conv2d(8 * self._depth, self._kernels[3], stride=2)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(self._kernels[1]),
                                  nn.BatchNorm2d(self._kernels[3])])

    def forward(self, gc_obs):
        x = gc_obs.view(-1, *gc_obs.shape[-3:])
        for i in range(4):
            x = self._act(self.convs[i](x))
            if i == 1 or i == 3:
                x = self.bns[i//2](x)
        x = x.view(x.size(0), -1)
        return x.view(*gc_obs.shape[:-3], -1)

class GC_Distance(nn.Module):
    def __init__(self, act=F.relu, layers=4, units=128, out_dim=1, input_type='feat', normalize_input=False):
        super().__init__()
        self._layers = layers
        self._units = units
        self._act = act
        self.out_dim = out_dim
        self._input_type = input_type
        self._normalize_input = normalize_input
        self.fcs = nn.ModuleList([nn.Linear(self._units, self._units) for _ in range(self._layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self._units) for _ in range(self._layers)])
        self.out = nn.Linear(self._units, self.out_dim)

    def forward(self, gc_obs, no_softmax=False):
        if self._normalize_input:
            _inp, _goal = gc_obs.chunk(2, dim=-1)
            _inp = _inp / (_inp.norm() + 1e-8)
            _goal = _goal / (_goal.norm() + 1e-8)
            x = torch.cat([_inp, _goal], dim=-1)
        else:
            x = gc_obs

        for index in range(self._layers):
            x = self._act(self.fcs[index](x))
            x = self.bns[index](x)

        out = self.out(x).squeeze()
        if self.out_dim <= 1 or no_softmax:
            return out
        else:
            return F.softmax(out, dim=-1)


class GC_Critic(nn.Module):
    def __init__(self, act=F.relu, layers=4, units=128):
        super().__init__()
        self._layers = layers
        self._encoder = GC_Encoder()
        self._units = units
        self._act = act
        self.fcs = nn.ModuleList([nn.Linear(self._units, self._units) for _ in range(self._layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self._units) for _ in range(self._layers)])
        self.out = nn.Linear(self._units, 1)

    def forward(self, gc_obs, action):
        x = torch.cat([self._encoder(gc_obs), action], dim=-1)
        for index in range(self._layers):
            x = self._act(self.fcs[index](x))
            x = self.bns[index](x)
        return self.out(x).squeeze()

class GC_Actor(nn.Module):
    def __init__(self, size, act=F.relu, layers=10, units=128, from_images=True):
        super().__init__()
        self._size = size
        self._layers = layers
        self._units = units
        self._act = act
        self.from_images = from_images
        if from_images:
            self._encoder = GC_Encoder()
        self.fcs = nn.ModuleList([nn.Linear(self._units, self._units) for _ in range(self._layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self._units) for _ in range(self._layers)])
        self.out = nn.Linear(self._units, self._size)

    def forward(self, gc_obs):
        x = self._encoder(gc_obs) if self.from_images else gc_obs
        for index in range(self._layers):
            x = self._act(self.fcs[index](x))
            x = self.bns[index](x)
        x = self.out(x)
        return torch.tanh(x)

class ConvEncoder(nn.Module):
    def __init__(self, depth=32, act=F.relu, kernels=(4, 4, 4, 4)):
        super(ConvEncoder, self).__init__()
        self._act = act
        self._depth = depth
        self._kernels = kernels
        self.embed_size = depth * 32
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1 * self._depth, kernel_size=self._kernels[0], stride=2)
        self.conv2 = nn.Conv2d(in_channels=1 * self._depth, out_channels=2 * self._depth, kernel_size=self._kernels[1], stride=2)
        self.conv3 = nn.Conv2d(in_channels=2 * self._depth, out_channels=4 * self._depth, kernel_size=self._kernels[2], stride=2)
        self.conv4 = nn.Conv2d(in_channels=4 * self._depth, out_channels=8 * self._depth, kernel_size=self._kernels[3], stride=2)

    def forward(self, obs):
        x = obs['image'].reshape(-1, *obs['image'].shape[-3:])
        x = self._act(self.conv1(x))
        x = self._act(self.conv2(x))
        x = self._act(self.conv3(x))
        x = self._act(self.conv4(x))
        x = x.view(x.size(0), -1)
        shape = list(obs['image'].shape[:-3]) + [x.size(-1)]
        return x.view(*shape)

class ConvEncoderWithState(nn.Module):
    def __init__(self, depth=32, act=F.relu, kernels=(4, 4, 4, 4)):
        super(ConvEncoderWithState, self).__init__()
        self._act = act
        self._depth = depth
        self._kernels = kernels
        self.embed_size = depth * 32 + 9
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1 * self._depth, kernel_size=self._kernels[0], stride=2)
        self.conv2 = nn.Conv2d(in_channels=1 * self._depth, out_channels=2 * self._depth, kernel_size=self._kernels[1], stride=2)
        self.conv3 = nn.Conv2d(in_channels=2 * self._depth, out_channels=4 * self._depth, kernel_size=self._kernels[2], stride=2)
        self.conv4 = nn.Conv2d(in_channels=4 * self._depth, out_channels=8 * self._depth, kernel_size=self._kernels[3], stride=2)

    def forward(self, obs):
        x = obs['image'].reshape(-1, *obs['image'].shape[-3:])
        x = self._act(self.conv1(x))
        x = self._act(self.conv2(x))
        x = self._act(self.conv3(x))
        x = self._act(self.conv4(x))
        x = x.view(x.size(0), -1)
        shape = list(obs['image'].shape[:-3]) + [x.size(-1)]
        x = x.view(*shape)
        return torch.cat([x, obs['state'][..., :9]], dim=-1)


class ConvDecoder(nn.Module):
    def __init__(self, depth=32, act=F.relu, shape=(64, 64, 3), kernels=(5, 5, 6, 6), thin=True):
        super(ConvDecoder, self).__init__()
        self._act = act
        self._depth = depth
        self._shape = shape
        self._kernels = kernels
        self._thin = thin

        if self._thin:
            self.dense = nn.Linear(in_features=32 * self._depth, out_features=32 * self._depth)
        else:
            self.dense = nn.Linear(in_features=128 * self._depth, out_features=2 * 2 * 32 * self._depth)

        self.convT1 = nn.ConvTranspose2d(in_channels=32 * self._depth, out_channels=4 * self._depth,
                                         kernel_size=self._kernels[0], stride=2, padding=0)
        self.convT2 = nn.ConvTranspose2d(in_channels=4 * self._depth, out_channels=2 * self._depth,
                                         kernel_size=self._kernels[1], stride=2, padding=0)
        self.convT3 = nn.ConvTranspose2d(in_channels=2 * self._depth, out_channels=1 * self._depth,
                                         kernel_size=self._kernels[2], stride=2, padding=0)
        self.convT4 = nn.ConvTranspose2d(in_channels=1 * self._depth, out_channels=self._shape[-1],
                                         kernel_size=self._kernels[3], stride=2, padding=0)

    def forward(self, features, dtype=None):
        if self._thin:
            x = self.dense(features)
            x = x.view(-1, 32 * self._depth, 1, 1)
        else:
            x = self.dense(features)
            x = x.view(-1, 32 * self._depth, 2, 2)

        x = self._act(self.convT1(x))
        x = self._act(self.convT2(x))
        x = self._act(self.convT3(x))
        x = self.convT4(x)

        shape = list(features.shape[:-1]) + list(self._shape)
        mean = x.view(*shape)

        if dtype:
            mean = mean.type(dtype)

        return Independent(Normal(mean, torch.ones_like(mean)), len(self._shape))
    
class DenseHead(nn.Module):

    def __init__(self, shape, layers, units, act=nn.ELU(), dist='normal', std=1.0):
        super().__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._act = act
        self._dist = dist
        self._std = std

        self.layers = nn.ModuleList()
        for index in range(self._layers):
            self.layers.append(nn.Linear(self._units, self._units))
        
        self.mean_layer = nn.Linear(self._units, np.prod(self._shape))
        
        if self._std == 'learned':
            self.std_layer = nn.Linear(self._units, np.prod(self._shape))

    def forward(self, features):
        x = features
        for layer in self.layers:
            x = self._act(layer(x))
        mean = self.mean_layer(x)
        mean = mean.view(*features.shape[:-1], *self._shape)

        if self._std == 'learned':
            std = self.std_layer(x)
            std = F.softplus(std) + 0.01
            std = std.view(*features.shape[:-1], *self._shape)
        else:
            std = self._std

        if self._dist == 'normal':
            return Independent(Normal(mean, std), len(self._shape))
        if self._dist == 'huber':
            return Independent(tools.UnnormalizedHuber(mean, std, 1.0), len(self._shape))
        if self._dist == 'binary':
            return Independent(Bernoulli(mean), len(self._shape))
        raise NotImplementedError(self._dist)

# Example usage
# decoder = ConvDecoder()
# features = torch.randn(10, 32 * 32)  # example feature tensor
# output = decoder(features)


class ActionHead(nn.Module):
    def __init__(self, size, layers, units, act=F.elu, dist='trunc_normal',
                 init_std=0.0, min_std=0.1, action_disc=5, temp=0.1, outscale=0):
        super(ActionHead, self).__init__()
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act
        self._min_std = min_std
        self._init_std = init_std
        self._action_disc = action_disc
        self._temp = temp() if callable(temp) else temp
        self._outscale = outscale

        self.layers = nn.ModuleList()
        for index in range(layers):
            kw = {}
            if index == layers - 1 and outscale:
                kw['weight_init'] = nn.init.variance_scaling_(outscale)
            self.layers.append(nn.Linear(units, units, **kw))

        self.out_layer = nn.Linear(units, 2 * size if 'normal' in dist else size)

    def forward(self, *args, dtype=None):
        x = torch.cat(args, -1) if len(args) > 1 else args[0]
        for layer in self.layers:
            x = self._act(layer(x))

        x = self.out_layer(x)
        if dtype:
            x = x.type(dtype)

        if self._dist == 'tanh_normal':
            mean, std = torch.chunk(x, 2, dim=-1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = TransformedDistribution(Normal(mean, std), [TanhTransform()])
            dist = Independent(dist, 1)
        elif self._dist == 'tanh_normal_5':
            mean, std = torch.chunk(x, 2, dim=-1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = TransformedDistribution(Normal(mean, std), [TanhTransform()])
            dist = Independent(dist, 1)
        elif self._dist == 'normal':
            mean, std = torch.chunk(x, 2, dim=-1)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = Independent(Normal(mean, std), 1)
        elif self._dist == 'normal_1':
            mean = x
            dist = Independent(Normal(mean, 1), 1)
        elif self._dist == 'trunc_normal':
            mean, std = torch.chunk(x, 2, dim=-1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = Independent(Normal(mean, std), 1)
        elif self._dist == 'onehot':
            dist = Independent(Bernoulli(logits=x), 1)
        else:
            raise NotImplementedError(self._dist)

        return dist

"""
class GRUCell(tf.keras.layers.AbstractRNNCell):

  def __init__(self, size, norm=False, act=tf.tanh, update_bias=-1, **kwargs):
    super().__init__()
    self._size = size
    self._act = act
    self._norm = norm
    self._update_bias = update_bias
    self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
    if norm:
      self._norm = tfkl.LayerNormalization(dtype=tf.float32)

  @property
  def state_size(self):
    return self._size

  def call(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(tf.concat([inputs, state], -1))
    if self._norm:
      dtype = parts.dtype
      parts = tf.cast(parts, tf.float32)
      parts = self._norm(parts)
      parts = tf.cast(parts, dtype)
    reset, cand, update = tf.split(parts, 3, -1)
    reset = tf.nn.sigmoid(reset)
    cand = self._act(reset * cand)
    update = tf.nn.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]

def get_mlp_model(name, hidden_units, out_dim):
  with tf.name_scope(name) as scope:
    model = tfk.Sequential()
    for units in hidden_units:
        model.add(tfk.layers.Dense(units, activation='elu'))
    model.add(tfk.layers.Dense(out_dim, activation='tanh'))
  return model
"""