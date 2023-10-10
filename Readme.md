# Diffusion Model is an Effective Planner and Data Synthesizer for Multi-Task Reinforcement Learning

This is the official code for the paper ["Diffusion Model is an Effective Planner and Data Synthesizer for Multi-Task Reinforcement Learning"](https://arxiv.org/abs/2305.18459). 
We propose a diffusion-based effective planner and data synthesizer for multi-task RL.

![](E:\2023-2\nips23\image\models.jpg)
## Dataset
You can download our dataset via this [Google Drive link](https://drive.google.com/drive/folders/1Ce11F4C6ZtmEoVUzpzoZLox4noWcxCEb?usp=sharing).
## Instructions 

Train a MTDiff-p with:

```
python scripts/mtdiff_p_meta.py --model models.Tasksmeta --diffusion models.GaussianActDiffusion --loss_type statehuber --loader datasets.RTGActDataset --device cuda:0
```

Train a MTDiff-s with:

```
python scripts/mtdiff_s.py --model models.TasksAug --diffusion models.AugDiffusion --loss_type statehuber --loader datasets.AugDataset --device cuda:0
```

You can tune any hyperparameters in the `config` for experiments.

Conduct generative planning using MTDiff-p on MT50-rand with:

```
python scripts/test_mtdiff_p.py --diffusion_loadpath model_saved_path --diffusion_epoch selected_epoch --device cuda:0
```


You can tune any hyperparameters in the `config/locomotion.py` and `diffusion.py` to guide sampling.

## Acknowledgment 

Our code for MTDiff is partly based on the Diffuser from  https://github.com/jannerm/diffuser and Decision Diffuser from https://github.com/anuragajay/decision-diffuser.

## References
```bib
@inproceedings{he2023mtdiff,
  title={Diffusion Model is an Effective Planner and Data Synthesizer for Multi-Task Reinforcement Learning},
  author={Haoran He and Chenjia Bai and Kang Xu and Zhuoran Yang and Weinan Zhang and Dong Wang and Bin Zhao and Xuelong Li},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2023}
}
```