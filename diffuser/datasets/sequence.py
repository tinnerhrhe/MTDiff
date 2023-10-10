from collections import namedtuple
import numpy as np
import torch
import pdb
import diffuser.utils as utils
from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset, load_dmc_dataset, load_metaworld_dataset, load_antmaze_dataset, load_maze2d_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
from scipy import stats
import os
Batch = namedtuple('Batch', 'trajectories conditions')
AugBatch = namedtuple('AugBatch', 'trajectories task')
TaskBatch = namedtuple('TaskBatch', 'trajectories conditions task value')
DTBatch = namedtuple('DTBatch', 'actions rtg observations timestep mask')
DT1Batch = namedtuple('DT1Batch', 'actions rtg observations timestep mask task')
PromptDTBatch = namedtuple('PromptDTBatch', 'DTBatch actions rtg observations timestep mask')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')
MTValueBatch = namedtuple('MTValueBatch', 'trajectories conditions task values')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
            #print("episode:",episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions', 'rewards']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        rewards = self.fields.normed_rewards[path_ind, start:end]
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([rewards, actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch
'''load offline DMC dataset'''
class MetaSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, replay_dir_list=[], task_list=[], horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=200000, termination_penalty=0, use_padding=True, seed=None, meta_world=False, maze2d=False, antmaze=False, optimal=True):
        self.replay_dir_list = replay_dir_list
        self.task_list = task_list
        self.reward_scale = 400.0
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.record_values = []
        if meta_world:
            itr = load_metaworld_dataset(self.replay_dir_list, self.task_list, optimal=optimal)
        elif maze2d:
            itr = load_maze2d_dataset(self.task_list)
        elif antmaze:
            itr = load_antmaze_dataset(self.task_list)
        else:
            itr = load_dmc_dataset(self.replay_dir_list, self.task_list)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            if len(episode['rewards']) > self.max_path_length:
                # episode = {k: episode[k][:max_path_length] for k in episode.keys()}
                continue
            self.record_values.append(episode['rewards'].sum())
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
    def normalize(self, keys=['observations', 'actions', 'rewards']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        rewards = self.fields.normed_rewards[path_ind, start:end]
        #observations = self.fields.observations[path_ind, start:end]
        #actions = self.fields.actions[path_ind, start:end]
        #rewards = self.fields.rewards[path_ind, start:end]
        task = np.array(self.task_list.index(self.fields.get_task(path_ind))).reshape(-1,1)#self.get_task_id(self.fields.get_task(path_ind))
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([rewards, actions, observations], axis=-1)
        #if np.any(np.isnan(trajectories)):
        #    print("True->>>")
        batch = TaskBatch(trajectories, conditions, task, 1)
        #print("Batch item load!")
        return batch
class RTGActDataset(MetaSequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=True, seq_length=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.seq_length = seq_length
        self.draw()
    def normalize(self, keys=['observations', 'actions']):#, 'observations', 'rewards'
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            if key == 'rewards':
                self.fields[f'normed_{key}'] = self.fields[key] #/ self.reward_scale
                continue
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
    def __len__(self):
        return len(self.indices)
    def draw(self):
        V = np.array(self.record_values)
        print(V.shape)
        normed_V = (V - V.min()) / (V.max() - V.min())
        #normed_V = normed_V * 2 - 1
        sns.set_palette("hls")
        mpl.rc("figure", figsize=(9, 5))
        fig = sns.distplot(normed_V,bins=20)
        fig.set_xlabel("Normalized Return", fontsize=16)
        fig.set_ylabel("Density", fontsize=16)
        displot_fig = fig.get_figure()
        displot_fig.savefig('./sub-optimal.pdf', dpi = 400)
    def discount_cumsum(self, x, gamma):
        x = x.squeeze(-1)
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[:,-1] = x[:,-1]
        for t in reversed(range(x.shape[-1] - 1)):
            discount_cumsum[:,t] = x[:,t] + gamma * discount_cumsum[:,t + 1]
        return np.expand_dims(discount_cumsum, axis=-1)
    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        path_inds = []
        interval = int(self.fields.n_episodes/len(self.task_list))
        for i in range(len(self.task_list)):
            path_inds.append((path_ind+interval*i)%self.fields.n_episodes)
        observations = np.zeros((len(path_inds), self.seq_length, self.observation_dim))
        count = start
        k = self.seq_length - 1
        while count >= 0 and k >= 0:
            observations[:, k, :] = self.fields.normed_observations[path_inds, count]
            k -= 1
            count -= 1
        actions = self.fields.normed_actions[path_inds, start:end]
        task = np.array([self.task_list.index(self.fields.get_task(path_ind)) for path_ind in path_inds])#self.get_task_id(self.fields.get_task(path_ind))
        rtg = self.discount_cumsum(self.fields['rewards'][path_inds, start:], gamma=self.discount)[:,:end-start] / (self.max_path_length-start)
        batch = TaskBatch(actions, observations, task, rtg[:, 0])
        return batch


class MazeDataset(MetaSequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=True, seq_length=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.seq_length = seq_length
        if normed:
            self.vmin, self.vmax = self._get_bounds()

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.get_value(i)
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        # self.draw()
        return vmin, vmax

    def get_value(self, idx):
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind].sum()
        return rewards

    def normalize(self, keys=['observations', 'actions']):  # , 'observations', 'rewards'
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes * self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def __len__(self):
        return len(self.indices)

    def discount_cumsum(self, x, gamma):
        x = x.squeeze(-1)
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[:, -1] = x[:, -1]
        for t in reversed(range(x.shape[-1] - 1)):
            discount_cumsum[:, t] = x[:, t] + gamma * discount_cumsum[:, t + 1]
        return np.expand_dims(discount_cumsum, axis=-1)

    def normalize_return(self, rewards):
        return (np.sum(rewards, axis=1) - self.vmin) / (self.vmax - self.vmin)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        path_inds = []
        interval = int(self.fields.n_episodes / len(self.task_list))
        for i in range(len(self.task_list)):
            path_inds.append((path_ind + interval * i) % self.fields.n_episodes)
        observations = np.zeros((len(path_inds), self.seq_length, self.observation_dim))
        count = start
        k = self.seq_length - 1
        while count >= 0 and k >= 0:
            observations[:, k, :] = self.fields.normed_observations[path_inds, count]
            k -= 1
            count -= 1
        actions = self.fields.normed_actions[path_inds, start:end]
        task = np.array([self.task_list.index(self.fields.get_task(path_ind)) for path_ind in
                         path_inds])  # self.get_task_id(self.fields.get_task(path_ind))
        # rtg = self.discount_cumsum(self.fields['rewards'][path_inds, start:], gamma=self.discount)[:,:end-start] / (self.max_path_length-start)
        rtg = self.normalize_return(self.fields['rewards'][path_inds])
        # rtg = self.discount_cumsum(self.fields['rewards'][path_inds, :], gamma=self.discount)[:, :end - start] / 400.
        batch = TaskBatch(actions, observations, task, rtg)
        return batch
