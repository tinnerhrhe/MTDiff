import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import diffuser.utils as utils
import numpy as np
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'test-'
    config: str = 'config.locomotion'

args = Parser().parse_args('diffusion')
if args.is_mt45:
    task_list = ['basketball-v2', 'button-press-topdown-v2',
     'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2',
     'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2',
     'door-open-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2',
     'faucet-close-v2', 'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2',
     'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2',
     'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2',
     'plate-slide-back-side-v2', 'soccer-v2', 'push-wall-v2', 'shelf-place-v2', 'sweep-into-v2', 'sweep-v2',
     'window-open-v2',
     'window-close-v2', 'assembly-v2', 'button-press-topdown-wall-v2', 'hammer-v2', 'peg-unplug-side-v2',
     'reach-wall-v2', 'stick-push-v2', 'stick-pull-v2']
else:
    task_list = ['basketball-v2', 'bin-picking-v2', 'button-press-topdown-v2',
 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2',
'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2',
'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2',
 'faucet-close-v2',  'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2',
 'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2',
 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2',
 'plate-slide-back-side-v2',  'soccer-v2', 'push-wall-v2',  'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2',
'window-close-v2','assembly-v2','button-press-topdown-wall-v2','hammer-v2','peg-unplug-side-v2',
                               'reach-wall-v2', 'stick-push-v2', 'stick-pull-v2', 'box-close-v2']

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#
prompt_trajectories = [np.load(f"./metaworld_prompts/{task_list[ind]}_prompt.npy", allow_pickle=True) for ind in range(len(task_list))]
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
    max_path_length=500,
    ## value-specific kwargs
    optimal=args.optimal,
    discount=args.discount,
    termination_penalty=args.termination_penalty,
    normed=args.normed,
    meta_world=True,
    seq_length=2,
)

dataset = dataset_config()
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
    num_tasks=50,
    dim_mults=args.dim_mults,
    attention=args.attention,
    device=args.device,
    train_device=args.device,
    prompt_trajectories=prompt_trajectories,
    verbose=False,
    task_list=task_list,
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
    utils.MetaworldTrainer,
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
batch = utils.batchify(dataset[0])
#loss, _ = diffusion.loss(*batch, device=args.device)
#loss.backward()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
#args.n_train_steps = 5e4
#args.n_steps_per_epoch = 1000
n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)

