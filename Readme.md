# Readme

This is the code for the paper "Diffusion Model is an Effective Planner and Data Synthesizer for Multi-Task Reinforcement Learning".  We propose a diffusion-based effective planner and data synthesizer for multi-task RL

![](E:\2023-2\nips23\image\models.jpg)

## Instructions 

Train a MTDiff-p with:

```
python scripts/train_diffusion_meta.py --model models.Tasksmeta --diffusion models.GaussianActDiffusion --loss_type statehuber --loader datasets.RTGActDataset --device cuda:0
```

Train a MTDiff-s with:

```
python scripts/train_augmeta.py --model models.TasksAug --diffusion models.AugDiffusion --loss_type statehuber --loader datasets.AugDataset --device cuda:0
```

You can tune any hyperparameters in the `config` for experiments.

Plan using MTDiff-p on MT50-rand with:

```
python scripts/diff_test.py --diffusion_loadpath model_saved_path --diffusion_epoch selected_epoch --device cuda:0
```

Synthesize data using MTDiff-s for `task_x` with:

```
python scripts/augdata.py --diffusion_loadpath model_saved_path --diffusion_epoch selected_epoch --device cuda:0 --env_id task_x
```

You can tune any hyperparameters in the `config` and files to guide sampling.

## Acknowledgment 

Our code for MTDiff is partly based on the Diffuser from  https://github.com/jannerm/diffuser and Decision Diffuser from https://github.com/anuragajay/decision-diffuser.

