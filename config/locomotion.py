import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]

logbase = 'logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.TasksAttTemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'env_id': 'dial-turn-v2',
        'horizon': 32,#100
        'n_diffusion_steps': 200,
        'action_weight': 10,
        'num_tasks':4,
        'loss_weights': None,
        'loss_discount': 1.0,
        'predict_epsilon': True,
        'dim_mults': (1,2),
        'is_unet': False,
        'attention': True,
        'renderer': 'utils.MuJoCoRenderer',
        'replay_dir_metaworld': './collect/metaworld',
        #'task_list_metaworld':['basketball-v2'],
        'task_list_metaworld':['basketball-v2', 'bin-picking-v2',  'button-press-topdown-v2',
 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2',
'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2',
'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2',
 'faucet-close-v2',  'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2',
 'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2',
 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2',
 'plate-slide-back-side-v2',  'soccer-v2',
 'push-wall-v2',  'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2', 'window-close-v2'],
        'loader': 'datasets.MTValueDataset',
        #'normalizer': 'GaussianNormalizer',#SafeLimitsNormalizer
        'normalizer': 'SafeLimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        "is_walker": False,
        'optimal': True,
        'max_path_length': 1000,
        'discount': 0.99,
        'termination_penalty': -0,
        'normed': True,
        #multi-task dataset
        'dataset_dir':'./collect/walker/dataset',
        'task_list':['walker_run','walker_walk','walker_flip','walker_stand'],
            #['quadruped_walk', 'quadruped_jump','quadruped_run','quadruped_roll_fast'],
        'data_type_list':['replay','replay','replay','replay'],
        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 200000,
        'loss_type': 'huber',
        'n_train_steps': 2e6,
        'batch_size': 128,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 100000,
        'sample_freq': 20000,
        'n_saves': 20,
        'save_parallel': False,
        'n_reference': 8,
        'is_mt45': False,
        'bucket': None,
        'device': 'cuda:0',
        'seed': None,
        'inv_task': 'pick-place-v2'
    },


}


#------------------------ overrides ------------------------#

