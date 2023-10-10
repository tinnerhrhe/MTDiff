import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import diffuser.utils as utils


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d'
    config: str = 'config.locomotion'

args = Parser().parse_args('diffusion')

task_list = ['maze2d-1', 'maze2d-2', 'maze2d-3', 'maze2d-4', 'maze2d-5', 'maze2d-6', 'maze2d-7', 'maze2d-8']
#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#
dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    replay_dir_list=[],
    task_list=task_list,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=600,
    ## value-specific kwargs
    discount=args.discount,
    termination_penalty=args.termination_penalty,
    normed=True,
    meta_world=False,
    maze2d=True,
    seq_length=5,
)

#render_config = utils.Config(
#    args.renderer,
#    savepath=(args.savepath, 'render_config.pkl'),
#    env=args.dataset,
#)

dataset = dataset_config()
dic = {
        'maze2d-1': utils.MAZE_1,
        'maze2d-2': utils.MAZE_2,
        'maze2d-3': utils.MAZE_3,
        'maze2d-4': utils.MAZE_4,
        'maze2d-5': utils.MAZE_5,
        'maze2d-6': utils.MAZE_6,
        'maze2d-7': utils.MAZE_7,
        'maze2d-8': utils.MAZE_8,
    }
prompt_trajectories = [utils.parse_maze(dic[task_id]) for task_id in task_list]
#renderer = render_config()
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim
reward_dim = 1

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim+1,# + action_dim,# + reward_dim,
    cond_dim=observation_dim,
    num_tasks=3,
    dim_mults=args.dim_mults,
    attention=args.attention,
    device=args.device,
    train_device=args.device,
    prompt_trajectories=prompt_trajectories,
    verbose=False,
    task_list=args.task_list,
    action_dim=action_dim,
)
diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.MazeTrainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    envs=task_list,
    task_list=task_list,
    is_unet=args.is_unet,
    trainer_device=args.device,
    horizon=args.horizon,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)
renderer=None
trainer = trainer_config(diffusion, dataset, renderer)


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)
print('Testing forward...', end=' ', flush=True)
n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)