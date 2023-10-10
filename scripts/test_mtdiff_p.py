import os
import inspect
import random
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
#import diffuser.datasets.dmc as dmc
import matplotlib.pyplot as plt
import pdb
from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict
import diffuser.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
from scipy import stats
import metaworld
import time
import torch
import statistics
from gym import wrappers
from imageio import imwrite
#from cv2 import imwrite
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def get_task_id(str):
        task_dict = {
            'quadruped_walk': np.array([0]),
            'quadruped_jump': np.array([1]),
            'quadruped_run': np.array([2]),
            'quadruped_roll_fast': np.array([3])
            }
        return task_dict[str]

# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#
DTYPE = torch.float
DEVICE = 'cuda'
def cycle(dl):
    while True:
        for data in dl:
            yield data

def to_torch(x, dtype=None, device=None):
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)
        # import pdb; pdb.set_trace()
    return torch.tensor(x, dtype=dtype, device=device)
class Parser(utils.Parser):
    dataset: str = 'test-'
    config: str = 'config.locomotion'


args = Parser().parse_args('plan')

# -----------------------------------------------------------------------------#
# ---------------------------------- loading ----------------------------------#
# -----------------------------------------------------------------------------#

# load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, device=args.device, seed=args.seed,
)
#diffusion_experiment = utils.load_diffusion(
#    "/mnt/petrelfs/hehaoran","MTdiffuser_remote/logs_",
#    epoch=args.diffusion_epoch, device=args.device, seed=args.seed,
#)
## ensure that the diffusion model and value function are compatible with each other
# utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema.to(args.device)
diffusion.clip_denoised = True
dataset = diffusion_experiment.dataset
## initialize value guide
# value_function = value_experiment.ema
#"""
task_metaworld = ['basketball-v2', 'bin-picking-v2',  'button-press-topdown-v2',
 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2',
'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2',
'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2',
 'faucet-close-v2',  'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2',
 'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2',
 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2',
 'plate-slide-back-side-v2',  'soccer-v2', 'push-wall-v2',  'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2',
'window-close-v2','assembly-v2','button-press-topdown-wall-v2','hammer-v2','peg-unplug-side-v2',
                               'reach-wall-v2', 'stick-push-v2', 'stick-pull-v2', 'box-close-v2']

task_list = ['basketball-v2', 'bin-picking-v2',  'button-press-topdown-v2',
 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2',
'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2',
'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2',
 'faucet-close-v2',  'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2',
 'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2',
 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2',
 'plate-slide-back-side-v2',  'soccer-v2', 'push-wall-v2',  'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2',
'window-close-v2','assembly-v2','button-press-topdown-wall-v2','hammer-v2','peg-unplug-side-v2',
                               'reach-wall-v2', 'stick-push-v2', 'stick-pull-v2', 'box-close-v2']
meta_task=np.array(args.task_list_metaworld.index(args.env_id))#get_task_id(args.task)
value = np.array([args.value])


# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#
def evaluate_metaworld(rtg, guidance):
    num_eval = 50
    task = [metaworld.MT1(env).train_tasks[i] for i in range(num_eval) for env in task_metaworld]
    mt1 = [metaworld.MT1(env) for i in range(num_eval) for env in task_metaworld]
    env_list = [mt1[i].train_classes[task_metaworld[i]]() for j in range(num_eval) for i in range(len(task_metaworld))]
    seed = 0
    for i in range(len(env_list)):
        env_list[i].set_task(task[i])
        env_list[i].seed(seed)
    score = 0
    total_success = 0
    env_success_rate = [0 for i in env_list]
    episode_rewards = [0 for i in env_list]
    max_episode_length = 500
    obs_list = [env.reset()[None] for env in env_list]
    obs = np.concatenate(obs_list, axis=0)
    cond_task = torch.tensor([i for j in range(num_eval) for i in range(len(task_metaworld))],
                                 device=args.device).reshape(-1, )
    conditions = torch.zeros([obs.shape[0], 2, obs.shape[-1]], device=args.device)
    rtg = torch.ones((len(env_list),), device=args.device) * rtg
    for j in range(max_episode_length):
        obs =dataset.normalizer.normalize(obs, 'observations')
        conditions = torch.cat([conditions[:, 1:, :], to_torch(obs, device=args.device).unsqueeze(1)], dim=1)
        samples = diffusion.conditional_sample(conditions, task=cond_task,
                                                       value=rtg,
                                                       verbose=False, horizon=14, guidance=1.2)
        action = samples.trajectories[:, 0, :]
        action = action.reshape(-1, dataset.action_dim)
        action = to_np(action)
        action = dataset.normalizer.unnormalize(action, 'actions')
        obs_list = []
        for i in range(len(env_list)):
            next_observation, reward, terminal, info = env_list[i].step(action[i])
            if i == 0:
                print("step", j, "Reward:", reward)
            obs_list.append(next_observation[None])
            episode_rewards[i] += reward
            if info['success'] > 1e-8:
                env_success_rate[i] = 1
        obs = np.concatenate(obs_list, axis=0)
    for i in range(len(task_metaworld)):
        tmp = []
        tmp_suc = 0
        for j in range(num_eval):
            tmp.append(episode_rewards[i + j * len(task_metaworld)])
            tmp_suc += env_success_rate[i + j * len(task_metaworld)]
        this_score = statistics.mean(tmp)
        success = tmp_suc / num_eval
        total_success += success
        score += this_score
        print(f"task:{task_metaworld[i]},success rate:{success}, mean episodic return:{this_score}, "
              #f"std:{statistics.stdev(tmp)}")
              )
    print('Total success rate:', total_success / len(task_metaworld))
if __name__ == '__main__':
    evaluate_metaworld(9., 1.2)

