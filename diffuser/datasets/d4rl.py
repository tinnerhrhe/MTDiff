import os
import collections
import numpy as np
import gym
import pdb
from pathlib import Path
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)
import d4rl
import h5py
from tqdm import tqdm
@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def get_dataset(env):
    dataset = env.get_dataset()

    if 'antmaze' in str(env).lower():
        ## the antmaze-v0 environments have a variety of bugs
        ## involving trajectory segmentation, so manually reset
        ## the terminal and timeout fields
        dataset = antmaze_fix_timeouts(dataset)
        dataset = antmaze_scale_rewards(dataset)
        get_max_delta(dataset)

    return dataset

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1

def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        #keys = [observation, action, reward]
        terminals = np.zeros_like(episode['reward'])
        #terminals[terminals.shape[0]-1] = 1
        episode = {k+'s': episode[k][:-1] for k in episode.keys()}
        episode['terminals'] = terminals[:-1]
        return episode
def load_metaworld_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode
def load_dmc_dataset(replay_dir_list,task_list):
    for i in range(len(replay_dir_list)):       # loop
        _replay_dir = replay_dir_list[i]
        _task = task_list[i]
        '''
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        print(f'Loading data from {_replay_dir} and Relabel...', "worker_id:", worker_id)      # each worker will run this function
        '''
        #print(_replay_dir.glob('*.npz'))
        eps_fns = sorted(_replay_dir.glob('*.npz'))
        #print(eps_fns)
        #_,__, eps_fns = os.waalk(_replay_dir)
        for eps_fn in eps_fns:
            # load a npz file to represent an episodic sample. The keys include 'observation', 'action', 'reward', 'discount', 'physics'
            episode = load_episode(eps_fn)
            episode['task']=_task
            yield episode
def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys
def get_maze_dataset(h5path=None):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    # Run a few quick sanity checks
    for key in ['observations', 'actions', 'rewards', 'terminals']:
        assert key in data_dict, 'Dataset is missing key %s' % key
    N_samples = data_dict['observations'].shape[0]
    if data_dict['rewards'].shape == (N_samples, 1):
        data_dict['rewards'] = data_dict['rewards'][:, 0]
    assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
        str(data_dict['rewards'].shape))
    if data_dict['terminals'].shape == (N_samples, 1):
        data_dict['terminals'] = data_dict['terminals'][:, 0]
    assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
        str(data_dict['rewards'].shape))
    return data_dict
def load_metaworld_dataset(replay_dir,task_list,optimal=True):
    data_ = collections.defaultdict(list)
    #save = [
    print(task_list)
    #replay_dir = '/NAS2020/Workspaces/DRLGroup/hrhe/metaworld.pytorch/logs'
    leng = 2000 if optimal else 1000
    for task_id in range(len(task_list)):       # loop
        _replay_dir = os.path.join(replay_dir, task_list[task_id])
        _replay_dir = Path(_replay_dir) / Path(os.listdir(_replay_dir)[0]) / 'dataset'
        eps_fns = sorted(_replay_dir.glob('*.npz'), key=lambda x:int(str(x)))
        for eps_fn in [eps_fns[i] for i in range(leng)]:#eps_fns[-20:]:#x:#eps_fns[-20:]:
            episode = load_metaworld_episode(eps_fn)
            for i in range(episode["terminals"].shape[0]):
                done_bool = bool(episode['terminals'][i])
                for k in episode:
                    data_[k].append(episode[k][i])
                if done_bool:
                    episode_data = {}
                    for k in data_:
                        episode_data[k] = np.array(data_[k])
                    episode_data['task']= task_list[task_id]
                    yield episode_data
                    data_ = collections.defaultdict(list)

def maze2d_set_terminals(env, dataset):
    env = load_environment(env) if type(env) == str else env
    goal = np.array(env._target)
    threshold = 0.5
    xy = dataset['observations'][:,:2]
    distances = np.linalg.norm(xy - goal, axis=-1)
    at_goal = distances < threshold
    timeouts = np.zeros_like(dataset['terminals'])

    timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]

    timeout_steps = np.where(timeouts)[0]
    path_lengths = timeout_steps[1:] - timeout_steps[:-1]
    #print((path_lengths<10000).sum())
    print(
        f'[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | '
        f'min length: {path_lengths.min()} | max length: {path_lengths.max()}'
    )

    dataset['timeouts'] = timeouts
    return dataset
def load_maze2d_dataset(task_list):
    #save = [
    print(task_list)
    dir = f'your path'
    for task_id in task_list:       # loop
        dataset = get_maze_dataset(h5path=f'{dir}/{task_id}-sparse.hdf5')
        dataset = maze2d_set_terminals(task_id, dataset)
        N = dataset['rewards'].shape[0]
        data_ = collections.defaultdict(list)
        use_timeouts = 'timeouts' in dataset
        episode_step = 0
        for i in range(N):
            done_bool = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == 600 - 1)
            for k in dataset:
                if 'metadata' in k: continue
                data_[k].append(dataset[k][i])
            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                #if 'maze2d' in env.name:
                #    episode_data = process_maze2d_episode(episode_data)
                episode_data['task'] = task_id
                yield episode_data
                data_ = collections.defaultdict(list)
            episode_step += 1
def load_antmaze_dataset(task_list):
    #save = [
    print(task_list)
    for task_id in task_list:       # loop
        env = gym.make(task_id)
        dataset = env.get_dataset()
        #dataset = maze2d_set_terminals(task_id, dataset)
        N = dataset['rewards'].shape[0]
        data_ = collections.defaultdict(list)
        use_timeouts = 'timeouts' in dataset
        episode_step = 0
        for i in range(N):
            done_bool = bool(dataset['timeouts'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == env._max_episode_steps - 1)
            for k in dataset:
                if 'metadata' in k: continue
                data_[k].append(dataset[k][i])
            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                episode_data['task'] = task_id
                yield episode_data
                data_ = collections.defaultdict(list)
            episode_step += 1


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
