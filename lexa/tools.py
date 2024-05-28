import sys
import os
import pipes
import datetime
import io
import json
import pathlib
import pickle
import re
import time
import uuid
import subprocess
import imageio
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import OneHotCategorical, RelaxedOneHotCategorical, Categorical, Normal, MixtureSameFamily, constraints, kl_divergence
from torch.distributions.mixture_same_family import MixtureSameFamily
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def save_cmd(base_dir):
    if not isinstance(base_dir, pathlib.Path):
        base_dir = pathlib.Path(base_dir)
    train_cmd = 'python ' + ' '.join([sys.argv[0]] + [shlex.quote(s) for s in sys.argv[1:]])
    train_cmd += '\n'
    print('\n' + '*' * 80)
    print('Training command:\n' + train_cmd)
    print('*' * 80 + '\n')
    with open(base_dir / "cmd.txt", "w") as f:
        f.write(train_cmd)

def save_git(base_dir):
    # save code revision
    print('Save git commit and diff to {}/git.txt'.format(base_dir))
    cmds = ["echo `git rev-parse HEAD` > {}".format(
        os.path.join(base_dir, 'git.txt')),
        "git diff >> {}".format(
            os.path.join(base_dir, 'git.txt'))]
    print(cmds)
    for cmd in cmds:
        subprocess.run(cmd, shell=True, check=True)

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

class Module(torch.nn.Module):

    def save(self, filename):
        torch.save(self.state_dict(), filename)
        print(f'Save checkpoint with {len(self.state_dict())} tensors.')

    def load(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)
        print(f'Load checkpoint with {len(state_dict)} tensors.')

    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, '_modules'):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)
        return self._modules[name]

def var_nest_names(nest):
    if isinstance(nest, dict):
        items = ' '.join(f'{k}:{var_nest_names(v)}' for k, v in nest.items())
        return '{' + items + '}'
    if isinstance(nest, (list, tuple)):
        items = ' '.join(var_nest_names(v) for v in nest)
        return '[' + items + ']'
    if hasattr(nest, 'name') and hasattr(nest, 'shape'):
        return nest.name + str(tuple(nest.shape)).replace(', ', 'x')
    if hasattr(nest, 'shape'):
        return str(tuple(nest.shape)).replace(', ', 'x')
    return '?'


class Logger:

    def __init__(self, logdir, step):
        self._logdir = Path(logdir)
        self._writer = SummaryWriter(log_dir=str(self._logdir))
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def write(self, fps=False):
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(('fps', self._compute_fps(self.step)))
        print(f'Step {self.step}')
        with (self._logdir / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps({'step': self.step, **dict(scalars)}) + '\n')
        for name, value in scalars:
            self._writer.add_scalar('scalars/' + name, value, self.step)
        for name, value in self._images.items():
            self._writer.add_image(name, value, self.step)
        for name, value in self._videos.items():
            self._add_video(name, value, self.step)
        self._writer.flush()
        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def _add_video(self, name, video, step):
        if len(video.shape) == 4:  # Assuming video shape is (T, H, W, C)
            video = np.transpose(video, (0, 3, 1, 2))  # Convert to (T, C, H, W)
        self._writer.add_video(name, video, step, fps=15)

"""
# Example usage
if __name__ == "__main__":
    logger = Logger(logdir='logs', step=0)
    logger.scalar('accuracy', 0.95)
    logger.image('sample_image', np.random.rand(3, 64, 64))
    logger.video('sample_video', np.random.rand(10, 3, 64, 64))
    logger.write(fps=True)
"""


def graph_summary(writer, step, fn, *args):
    def inner(*args):
        writer.add_scalar('step', step.item(), step.item())
        fn(*args)
    return inner

def video_summary(writer, name, video, step=None, fps=20):
    name = name if isinstance(name, str) else name.decode('utf-8')
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    B, T, H, W, C = video.shape
    try:
        frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
        encoded_gif = encode_gif(frames, fps)
        writer.add_image(name, encoded_gif, step, dataformats='HWC')
    except (IOError, OSError) as e:
        print('GIF summaries require ffmpeg or imageio.', e)
        frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
        writer.add_images(name, frames, step, dataformats='NHWC')

def encode_gif(frames, fps):
    h, w, c = frames[0].shape
    with imageio.get_writer('<bytes>', format='gif', mode='I', fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    return writer.get_result()

def log_eval_metrics(logger, log_prefix, eval_dir, num_eval_eps):
    #keys = pickle.load(open(str(eval_dir)+'/eval_ep_0.pkl', 'rb')).keys()
    multi_task_data = [pickle.load(open(str(eval_dir)+'/eval_ep_'+str(idx)+'.pkl', 'rb')) for idx in range(num_eval_eps)]
    keys = multi_task_data[0].keys()
    for key in keys:
        for idx in range(num_eval_eps):
            logger.scalar(log_prefix+ 'task_'+str(idx)+'/'+key, multi_task_data[idx][key])

        _avg = np.mean([multi_task_data[idx][key] for idx in range(num_eval_eps)])
        logger.scalar(log_prefix + 'avg/'+ key, _avg)
    logger.write()

def simulate(agent, envs, steps=0, episodes=0, state=None):
  # Initialize or unpack simulation state.
  if state is None:
    step, episode = 0, 0
    done = np.ones(len(envs), np.bool)
    length = np.zeros(len(envs), np.int32)
    obs = [None] * len(envs)
    agent_state = None
  else:
    step, episode, done, length, obs, agent_state = state
  all_rewards = []
  #all_gt_rewards = []
  ep_data_lst= []
  while (steps and step < steps) or (episodes and episode < episodes):
    # Reset envs if necessary.
    if done.any():
      indices = [index for index, d in enumerate(done) if d]
      # promises = [envs[i].reset(blocking=False) for i in indices]
      # for index, promise in zip(indices, promises):
      #   obs[index] = promise()
      results = [envs[i].reset() for i in indices]
      for index, result in zip(indices, results):
        obs[index] = result
  
    # Step agents.
    obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
    #action, agent_state = agent(obs, done, agent_state)
    agent_out = agent(obs, done, agent_state)
    if len(agent_out) ==2:
      action, agent_state = agent_out
    else:
      action, agent_state, learned_reward = agent_out
      ep_data = {'learned_reward': learned_reward}
      if 'state' in obs: ep_data['state'] = obs['state']
  
      for key in obs.keys():
        if 'metric_reward' in key:
          ep_data['gt_reward'] = obs[key]
      ep_data_lst.append(ep_data)

    if isinstance(action, dict):
      action = [
          {k: np.array(action[k][i]) for k in action}
          for i in range(len(envs))]
    else:
      action = np.array(action)
    assert len(action) == len(envs)
    # Step envs.
    # promises = [e.step(a, blocking=False) for e, a in zip(envs, action)]
    # obs, _, done = zip(*[p()[:3] for p in promises])
    results = [e.step(a) for e, a in zip(envs, action)]
    obs, _, done = zip(*[p[:3] for p in results])
    obs = list(obs)
    done = np.stack(done)
    episode += int(done.sum())
    length += 1
    step += (done * length).sum()
    length *= (1 - done)
  # Return new state to allow resuming the simulation.
  if len(ep_data_lst) > 0:
    return (step - steps, episode - episodes, done, length, obs, agent_state, ep_data_lst)
  else:
    return (step - steps, episode - episodes, done, length, obs, agent_state)


def save_episodes(directory, episodes):
  directory = pathlib.Path(directory).expanduser()
  directory.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
  filenames = []
  for episode in episodes:
    identifier = str(uuid.uuid4().hex)
    length = len(episode['reward'])
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
      np.savez_compressed(f1, **episode)
      f1.seek(0)
      with filename.open('wb') as f2:
        f2.write(f1.read())
    filenames.append(filename)
  return filenames


def sample_episodes(episodes, length=None, balance=False, seed=0):
  random = np.random.RandomState(seed)
  while True:
    episode = random.choice(list(episodes.values()))
    if length:
      total = len(next(iter(episode.values())))
      available = total - length
      if available < 1:
        print(f'Skipped short episode of length {total}, need length {length}.')
        continue
      if balance:
        index = min(random.randint(0, total), available)
      else:
        index = int(random.randint(0, available + 1))
      episode = {k: v[index: index + length] for k, v in episode.items()}
    yield episode


def load_episodes(directory, limit=None):
  directory = pathlib.Path(directory).expanduser()
  episodes = {}
  total = 0
  for filename in reversed(sorted(directory.glob('*.npz'))):
    try:
      with filename.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    except Exception as e:
      print(f'Could not load episode: {e}')
      continue
    episodes[str(filename)] = episode
    total += len(episode['reward']) - 1
    if limit and total >= limit:
      break
  return episodes


class DtypeDist:

    def __init__(self, dist, dtype=None):
        self._dist = dist
        self._dtype = dtype or torch.float32

    @property
    def name(self):
        return 'DtypeDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        return self._dist.mean().to(self._dtype)

    def mode(self):
        return self._dist.mode().to(self._dtype)

    def entropy(self):
        return self._dist.entropy().to(self._dtype)

    def sample(self, *args, **kwargs):
        return self._dist.sample(*args, **kwargs).to(self._dtype)


class SampleDist:

    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample((self._samples,))
        return torch.mean(samples, dim=0)

    def mode(self):
        samples = self._dist.sample((self._samples,))
        logprob = self._dist.log_prob(samples)
        return samples[torch.argmax(logprob)]

    def entropy(self):
        samples = self._dist.sample((self._samples,))
        logprob = self.log_prob(samples)
        return -torch.mean(logprob, dim=0)
    
  
class OneHotDist(torch.distributions.OneHotCategorical):
    def __init__(self, logits=None, probs=None, dtype=None):
        self._sample_dtype = dtype or torch.float32
        super().__init__(logits=logits, probs=probs)

    def mode(self):
        return super().probs.argmax(dim=-1).type(self._sample_dtype)

    def sample(self, sample_shape=torch.Size()):
        # Straight through biased gradient estimator.
        sample = super().sample(sample_shape).type(self._sample_dtype)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs.unsqueeze(0)
        sample += (probs - probs.detach()).type(self._sample_dtype)
        return sample

class GumbleDist(RelaxedOneHotCategorical):
    def __init__(self, temp, logits=None, probs=None, dtype=None):
        self._sample_dtype = dtype or torch.float32
        self._exact = OneHotCategorical(logits=logits, probs=probs)
        super().__init__(temp, logits=logits, probs=probs)

    def mode(self):
        return self._exact.probs.argmax(dim=-1).type(self._sample_dtype)

    def entropy(self):
        return self._exact.entropy().type(self._sample_dtype)

    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).type(self._sample_dtype)
  
class UnnormalizedHuber(Normal):
    def __init__(self, loc, scale, threshold=1, **kwargs):
        self._threshold = threshold
        super().__init__(loc, scale, **kwargs)

    def log_prob(self, event):
        return -(torch.sqrt((event - self.mean) ** 2 + self._threshold ** 2) - self._threshold)

class SafeTruncatedNormal(Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._clip = clip
        self._mult = mult
        self.low = low
        self.high = high

    def sample(self, sample_shape=torch.Size()):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clamp(event, self.low + self._clip, self.high - self._clip)
            event = event.detach() + (clipped - event).detach()
        if self._mult:
            event *= self._mult
        return event

class TanhBijector:
    def __init__(self):
        self.name = 'tanh'

    def forward(self, x):
        return torch.tanh(x)

    def inverse(self, y):
        dtype = y.dtype
        y = torch.clamp(y, -0.99999997, 0.99999997)
        y = torch.atanh(y)
        return y.to(dtype)

    def forward_log_det_jacobian(self, x):
        log2 = torch.log(torch.tensor(2.0, dtype=x.dtype))
        return 2.0 * (log2 - x - F.softplus(-2.0 * x))


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    assert reward.ndim == value.ndim, (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(reward.ndim))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap.unsqueeze(0)], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns = static_scan(
        lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
        (inputs, pcont), bootstrap, reverse=True)
    if axis != 0:
        returns = returns.permute(dims)
    return returns


def static_scan(fn, inputs, initial_state, reverse=False):
    if reverse:
        inputs = tuple(torch.flip(x, dims=[0]) for x in inputs)
    last = initial_state
    outputs = []
    for i in range(inputs[0].shape[0]):
        inp = tuple(x[i] for x in inputs)
        last = fn(last, inp)
        outputs.append(last)
    if reverse:
        outputs = torch.flip(torch.stack(outputs), dims=[0])
    else:
        outputs = torch.stack(outputs)
    return outputs

class Optimizer(nn.Module):

    def __init__(self, name, lr, eps=1e-4, clip=None, wd=None, wd_pattern=r'.*', opt='adam'):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            'adam': lambda: optim.Adam(self._params, lr=lr, eps=eps),
            'nadam': lambda: optim.NAdam(self._params, lr=lr, eps=eps),
            'adamax': lambda: optim.Adamax(self._params, lr=lr, eps=eps),
            'sgd': lambda: optim.SGD(self._params, lr=lr),
            'momentum': lambda: optim.SGD(self._params, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    @property
    def variables(self):
        return self._params

    def __call__(self, loss, modules):
        modules = modules if hasattr(modules, '__len__') else (modules,)
        varibs = [param for module in modules for param in module.parameters() if param.requires_grad]
        self._params = varibs
        count = sum(p.numel() for p in varibs)
        assert loss.dim() == 0, loss.shape
        
        if self._scaler:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

        grads = [p.grad for p in varibs]
        norm = torch.norm(torch.stack([torch.norm(g.detach()) for g in grads]))

        if self._clip:
            torch.nn.utils.clip_grad_norm_(varibs, self._clip)
        
        if self._wd:
            self._apply_weight_decay(varibs)
        
        if self._scaler:
            self._scaler.step(self._opt)
            self._scaler.update()
        else:
            self._opt.step()
        
        self._opt.zero_grad()

        metrics = {}
        metrics[f'{self._name}_loss'] = loss.item()
        metrics[f'{self._name}_grad_norm'] = norm.item()
        if self._scaler:
            metrics[f'{self._name}_loss_scale'] = self._scaler.get_scale()
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = (self._wd_pattern != r'.*')
        if nontrivial:
            print('Applied weight decay to variables:')
        for var in varibs:
            if re.search(self._wd_pattern, self._name + '/' + var.name):
                if nontrivial:
                    print('- ' + self._name + '/' + var.name)
                var.data.mul_(1 - self._wd)

def args_type(default):
  def parse_string(x):
    if default is None:
      return x
    if isinstance(default, bool):
      return bool(['False', 'True'].index(x))
    if isinstance(default, int):
      return float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, (list, tuple)):
      return tuple(args_type(default[0])(y) for y in x.split(','))
    return type(default)(x)
  def parse_object(x):
    if isinstance(default, (list, tuple)):
      return tuple(x)
    return x
  return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)

def static_scan(fn, inputs, start, reverse=False):
    last = start
    outputs = [[] for _ in torch.flatten(start)]
    indices = range(len(inputs[0]))
    if reverse:
        indices = reversed(indices)
    for index in indices:
        inp = list(map(lambda x: x[index], inputs))
        last = fn(last, inp)
        [o.append(l) for o, l in zip(outputs, torch.flatten(last))]
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    outputs = [torch.stack(x, 0) for x in outputs]
    return outputs

def uniform_mixture(dist, dtype=None):
    if dist.batch_shape[-1] == 1:
        return dist.expand(dist.batch_shape[:-1])
    dtype = dtype or torch.float32
    weights = Categorical(torch.zeros(dist.batch_shape, dtype=dtype))
    return MixtureSameFamily(weights, dist)

def cat_mixture_entropy(dist):
    if isinstance(dist, MixtureSameFamily):
        probs = dist.mixture_distribution.probs
    else:
        probs = dist.probs
    return -torch.mean(torch.mean(probs, dim=2) * torch.log(torch.mean(probs, dim=2) + 1e-8), dim=-1)


def cem_planner(state, num_actions, horizon, proposals, topk, iterations, imagine, objective):
    dtype = torch.float32
    B, P = list(state.values())[0].shape[0], proposals
    H, A = horizon, num_actions
    flat_state = {k: v.repeat(P, 0) for k, v in state.items()}
    mean = torch.zeros((B, H, A), dtype=dtype)
    std = torch.ones((B, H, A), dtype=dtype)
    for _ in range(iterations):
        proposals = torch.randn((B, P, H, A), dtype=dtype)
        proposals = proposals * std[:, None] + mean[:, None]
        proposals = torch.clamp(proposals, -1, 1)
        flat_proposals = proposals.view(B * P, H, A)
        states = imagine(flat_proposals, flat_state)
        scores = objective(states)
        scores = scores.view(B, P).sum(dim=-1)
        _, indices = torch.topk(scores, topk, dim=1, largest=True, sorted=False)
        best = torch.gather(proposals, 1, indices.unsqueeze(-1).expand(-1, -1, H, A))
        mean, var = torch.mean(best, dim=1), torch.var(best, dim=1)
        std = torch.sqrt(var + 1e-6)
    return mean[:, 0, :]

def grad_planner(state, num_actions, horizon, proposals, iterations, imagine, objective, kl_scale, step_size):
    dtype = torch.float32
    B, P = list(state.values())[0].shape[0], proposals
    H, A = horizon, num_actions
    flat_state = {k: v.repeat(P, 0) for k, v in state.items()}
    mean = torch.zeros((B, H, A), dtype=dtype)
    rawstd = 0.54 * torch.ones((B, H, A), dtype=dtype)
    for _ in range(iterations):
        proposals = torch.randn((B, P, H, A), dtype=dtype)
        mean.requires_grad_()
        rawstd.requires_grad_()
        std = F.softplus(rawstd)
        proposals = proposals * std[:, None] + mean[:, None]
        proposals = (proposals.detach().clamp(-1, 1) + proposals - proposals.detach())
        flat_proposals = proposals.view(B * P, H, A)
        states = imagine(flat_proposals, flat_state)
        scores = objective(states)
        scores = scores.view(B, P).sum(dim=-1)
        dist = Normal(mean, std)
        base_dist = Normal(torch.zeros_like(mean), torch.ones_like(std))
        div = kl_divergence(dist, base_dist).sum()
        elbo = scores.sum() - kl_scale * div
        elbo /= torch.tensor(scores.numel(), dtype=dtype)
        grad_mean, grad_rawstd = torch.autograd.grad(elbo, [mean, rawstd])
        grad_mean = grad_mean / (torch.sqrt(torch.var(grad_mean, dim=[1, 2], keepdim=True) + 1e-4))
        grad_rawstd = grad_rawstd / (torch.sqrt(torch.var(grad_rawstd, dim=[1, 2], keepdim=True) + 1e-4))
        mean = torch.clamp(mean + step_size * grad_mean, -1, 1)
        rawstd = rawstd + step_size * grad_rawstd
    return mean[:, 0, :]


class Every:

  def __init__(self, every):
    self._every = every
    self._last = None

  def __call__(self, step):
    if not self._every:
      return False
    if self._last is None:
      self._last = step
      return True
    if step >= self._last + self._every:
      self._last += self._every
      return True
    return False


class EveryNCalls:

  def __init__(self, every):
    self._every = every
    self._last = 0
    self._step = 0
    self.value = False

  def __call__(self, *args):
    if not self._every:
      return False
    self._step += 1
    if self._step >= self._last + self._every or self._last == 0:
      self._last += self._every
      self.value = True
      return True
    self.value = False
    return False


class Once:

  def __init__(self):
    self._once = True

  def __call__(self):
    if self._once:
      self._once = False
      return True
    return False


class Until:

  def __init__(self, until):
    self._until = until

  def __call__(self, step):
    if not self._until:
      return True
    return step < self._until


def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        step = step.float()
        match = re.match(r'linear\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clamp(step / duration, 0, 1)
            return (1 - mix) * initial + mix * final
        match = re.match(r'warmup\((.+),(.+)\)', string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = torch.clamp(step / warmup, 0, 1)
            return scale * value
        match = re.match(r'exp\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        raise NotImplementedError(string)

def am_sampling(obs, actions):

  s_t, a_t, s_tp1, r_t = [], [], [], []
  num_trajs, seq_len = obs.shape[:2]
 
  for traj_idx in range(num_trajs):
    end_idx = np.random.randint(1, seq_len)
   
    #positive 
    s_t.append(torch.concat([obs[traj_idx, end_idx-1], obs[traj_idx, end_idx]], axis =-1))
    s_tp1.append(torch.concat([obs[traj_idx, end_idx],   obs[traj_idx, end_idx]], axis =-1))
    a_t.append(actions[traj_idx, end_idx])
    r_t.append(1)

    #negative
    neg_traj_idx = traj_idx
    while neg_traj_idx == traj_idx:
      neg_traj_idx = np.random.randint(0, num_trajs)
    neg_seq_idx = np.random.randint(1, seq_len)
    s_t.append(torch.concat([obs[traj_idx, end_idx-1], obs[neg_traj_idx, neg_seq_idx]], axis =-1))
    s_tp1.append(torch.concat([obs[traj_idx, end_idx],   obs[neg_traj_idx, neg_seq_idx]], axis =-1))
    a_t.append(actions[traj_idx, end_idx])
    r_t.append(0)
   
  def _expand_and_concat_tensor(_arr):
      return torch.cat([_elem.unsqueeze(0) for _elem in _arr], dim=0)

  s_t =   _expand_and_concat_tensor(s_t)
  s_tp1 = _expand_and_concat_tensor(s_tp1)
  a_t =   _expand_and_concat_tensor(a_t)
  r_t =   torch.tensor(r_t, dtype=torch.float16)
  mask = 1 - r_t

  return s_t, a_t, s_tp1, r_t, mask

def get_future_goal_idxs(seq_len, bs):
   
    cur_idx_list = []
    goal_idx_list = []
    #generate indices grid
    for cur_idx in range(seq_len):
      for goal_idx in range(cur_idx, seq_len):
        cur_idx_list.append(np.concatenate([np.ones((bs,1), dtype=np.int32)*cur_idx, np.arange(bs).reshape(-1,1)], axis = -1))
        goal_idx_list.append(np.concatenate([np.ones((bs,1), dtype=np.int32)*goal_idx, np.arange(bs).reshape(-1,1)], axis = -1))
    
    return np.concatenate(cur_idx_list,0), np.concatenate(goal_idx_list,0)

def get_future_goal_idxs_neg_sampling(num_negs, seq_len, bs, batch_len):
    cur_idxs = np.random.randint((0,0), (seq_len, bs), size=(num_negs,2))
    goal_idxs = np.random.randint((0,0), (seq_len, bs), size=(num_negs,2))
    for i in range(num_negs):
      goal_idxs[i,1] = np.random.choice([j for j in range(bs) if j//batch_len != cur_idxs[i,1]//batch_len])
    return cur_idxs, goal_idxs

def get_data_for_off_policy_training(obs, actions, next_obs, goals, relabel_mode='uniform',
                                     relabel_fraction=0.5, geom_p=0.3, feat_to_embed_func=None):

    num_batches, seq_len = obs.shape[:2]
    num_points = num_batches * seq_len

    def _reshape(_arr):
        return _arr.reshape((num_points,) + _arr.shape[2:])
    
    curr_state = _reshape(obs)
    action = _reshape(actions)
    next_state = _reshape(next_obs)
    _goals = _reshape(goals)
    masks = np.ones(num_points, dtype=np.float16)

    if relabel_fraction > 0:
        # relabelling
        deltas = np.random.geometric(geom_p, size=(num_batches, seq_len))
        next_obs_idxs_for_relabelling = []
        relabel_masks = []
        for i in range(num_batches):
            for j in range(seq_len):
                if relabel_mode == 'uniform':
                    next_obs_idxs_for_relabelling.append(np.random.randint((seq_len * i) + j, seq_len * (i + 1)))
                elif relabel_mode == 'geometric':
                    next_obs_idxs_for_relabelling.append(min((seq_len * i) + j + deltas[i, j] - 1, seq_len * (i + 1) - 1))
                relabel_masks.append(int(next_obs_idxs_for_relabelling[-1] != ((seq_len * i) + j)))
        
        next_obs_idxs_for_relabelling = np.array(next_obs_idxs_for_relabelling).reshape(-1, 1)
        relabel_masks = np.array(relabel_masks, dtype=np.float16).reshape(-1, 1)

        idxs = np.random.permutation(num_points).reshape(-1, 1)
        relabel_idxs, non_relabel_idxs = np.split(idxs, (int(relabel_fraction * num_points),), axis=0)
        curr_state = torch.from_numpy(curr_state[idxs.flatten()])
        action = torch.from_numpy(action[idxs.flatten()])
        next_state = torch.from_numpy(next_state[idxs.flatten()])

        _relabeled_goals = next_state[next_obs_idxs_for_relabelling[relabel_idxs.flatten()].flatten()]
        if feat_to_embed_func is not None:
            _relabeled_goals = feat_to_embed_func(_relabeled_goals)

        _goals = torch.cat([_relabeled_goals, torch.from_numpy(_goals[non_relabel_idxs.flatten()])], axis=0)
        masks = torch.cat([torch.from_numpy(relabel_masks[relabel_idxs.flatten()]).squeeze(), torch.from_numpy(masks[non_relabel_idxs.flatten()])], axis=0)

    s_t = torch.cat([curr_state, _goals], axis=-1)
    s_tp1 = torch.cat([next_state, _goals], axis=-1)

    return s_t, action, s_tp1, masks