import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pickle
import pdb
import transformers
from transformers import TransfoXLModel, TransfoXLConfig
from .GPT2 import GPT2Model
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
    AttentionBlock
)
from torch.distributions import Bernoulli
def get_prompt(trajectories, num_episodes, num_steps, is_meta, is_maze=False):
    trj_ids = np.random.choice(
        np.arange(len(trajectories)),
        size=num_episodes,
        replace=False,
    )
    if is_meta:
        T_start, T_end = 0, 170
    elif is_maze:
        T_start, T_end = 0, 1000
    else:
        T_start, T_end = 500, 1000
    start_steps = np.random.choice(
        np.arange(T_start, T_end-num_steps),
        size=num_episodes,
        replace=True,
    )
    p_trj_obs = np.array([trajectories[trj_ids[i]]['observations'][start_steps[i]:start_steps[i]+num_steps]
                          for i in range(num_episodes)])
    p_trj_actions = np.array([trajectories[trj_ids[i]]['actions'][start_steps[i]:start_steps[i]+num_steps]
                              for i in range(num_episodes)])
    p_trj = np.concatenate([p_trj_obs, p_trj_actions], axis=-1).reshape(num_episodes*num_steps, -1) #TODO
    return p_trj
def get_prompt_batch(prompt_trajectories, cond, num_episodes=2, num_steps=20):
    cond = cond.long()
    trajectories = [get_prompt(prompt_trajectories[ind], num_episodes, num_steps) for ind in cond]
    prompt_timestep = 0 #TODO
    return torch.tensor(np.array(trajectories)), prompt_timestep
def get_prompt_batchs(prompt_trajectories, cond, num_episodes=2, num_steps=20, is_meta=True):
    cond = cond.long()
    trajectories = [get_prompt(prompt_trajectories[ind], num_episodes, num_steps, is_meta) for ind in cond]
    return torch.tensor(np.array(trajectories))
def get_prompt_maze2d(prompt_trajectories, cond, num_episodes=1, num_steps=50, is_meta=False, is_maze=True):
    cond = cond.long()
    trajectories = [get_prompt(prompt_trajectories[ind], num_episodes, num_steps, is_meta, is_maze=is_maze) for ind in cond]
    return torch.tensor(np.array(trajectories))
class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)
class ConResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )
        self.context_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(32, out_channels),
            Rearrange('batch t -> batch t 1'),
        )
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t, context):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) * self.time_mlp(t) +self.context_mlp(context)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)
class TasksResidualTemporalBlock(ResidualTemporalBlock):

    def __init__(self, inp_channels, out_channels, time_embed_dim, task_embed_dim, horizon, kernel_size=5):
        super().__init__(inp_channels+32, out_channels, time_embed_dim, horizon, kernel_size=kernel_size)
        self.linear_map = nn.Sequential(
            nn.Linear(32, 128),
            nn.Mish(),
            nn.Linear(128, 32)

        )
    def forward(self, x, t, context):#cond, value
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        '''
        emb_c = self.linear_map_tasks(cond)
        emb_c = emb_c.view(*emb_c.shape, 1)
        emb_c = emb_c.expand(-1, -1, x.shape[-1])
        emb_v = self.linear_map_values(value)
        emb_v = emb_v.view(*emb_v.shape, 1)
        emb_v = emb_v.expand(-1, -1, x.shape[-1])
        '''
        context = self.linear_map(context)
        context = context.view(*context.shape, 1)
        context = context.expand(-1, -1, x.shape[-1])
        x = torch.cat([x, context], dim=1)
        return super().forward(x, t)



class Tasksmeta(nn.Module):
    "MT-Diffuser with a Transformer backbone"

    def __init__(
            self,
            horizon,
            transition_dim,
            cond_dim,
            num_tasks,
            dim=128,
            dim_mults=(1, 2, 4, 8),
            attention=False,
            depth=56,
            mlp_ratio=4.0,
            hidden_size=256,
            num_heads=8,
            train_device=None,
            prompt_trajectories=None,
            task_list=None,
            action_dim=None,
            max_ep_len=1000,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.dim = dim
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.state_dim = transition_dim - 1
        self.action_dim = action_dim
        self.prompt_trajectories = prompt_trajectories
        self.task_list = task_list
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer=4,
            n_head=2,
            n_inner=4 * 256,
            activation_function='mish',
            n_positions=1024,
            n_ctx=1023,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(2*dim),
            nn.Linear(2*dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, 2*dim),
        )
        self.return_mlp = nn.Sequential(
            nn.Linear(1, dim * 2),
            nn.Mish(),
            nn.Linear(dim * 2, 4*dim),
            nn.Mish(),
            nn.Linear(dim * 4, 2 * dim),
        )
        self.prompt_embed = nn.Sequential(
            nn.LayerNorm(self.state_dim+self.action_dim),
            nn.Linear(self.state_dim + self.action_dim, hidden_size*2),  # TODO
            nn.Mish(),
            nn.Linear(hidden_size * 2, 4 * hidden_size),
            nn.Mish(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.embed_obs = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, 2 * hidden_size),  # TODO
            nn.Mish(),
            nn.Linear(hidden_size * 2, 4 * hidden_size),
            nn.Mish(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

        self.mask_dist = Bernoulli(probs=0.8)
        self.transformer = GPT2Model(config)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.embed_act = nn.Linear(self.action_dim, hidden_size)
        self.position_emb = nn.Parameter(torch.zeros(1, 24+horizon, hidden_size))
        self.predict_act = torch.nn.Linear(hidden_size, self.action_dim)

    def forward(self, x, time, cond, value, context_mask, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None):
        '''
            x : [ batch x horizon x transition ]
        '''
        cond = cond.long()
        prompt_data = get_prompt_batchs(self.prompt_trajectories, cond, num_episodes=1, num_steps=20, is_meta=True)
        prompt_data = prompt_data.to(device=x.device, dtype=torch.float32).reshape(cond.shape[0], -1, self.state_dim+self.action_dim)#TODO
        prompt_embeddings = self.prompt_embed(prompt_data)
        obs_embeddings = self.embed_obs(x_condition)
        t = self.time_mlp(time).unsqueeze(1)
        if not force:
            mask = self.mask_dist.sample(sample_shape=(cond.shape[0], 1)).to(x.device)
        else:
            mask = 1 if return_cond else 0
        batch_size, seq_length = x.shape[0], x.shape[1]
        value = value.view(-1, 1)
        value = self.return_mlp(value)
        cond_return = (value * mask).unsqueeze(1) + 1e-8
        act_embeddings = self.embed_act(x)
        addition_length = 20 + 1 + 1 + 2  #TODO
        addition_attention_mask = torch.ones((batch_size, addition_length), dtype=torch.long, device=x.device)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=x.device)
            '''set attention mask'''
        stacked_inputs = torch.cat((t, cond_return, prompt_embeddings, obs_embeddings, act_embeddings), dim=1)
        stacked_attention_mask = torch.cat(
            (addition_attention_mask, attention_mask), dim=1)

        stacked_inputs = t * stacked_inputs + cond_return + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = self.embed_ln(stacked_inputs)
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        act_preds = self.predict_act(x[:, -seq_length:, :])  # predict next state given state and action
        return act_preds
class Tasksmaze(nn.Module):
    "MT-Diffuser with a Transformer backbone"

    def __init__(
            self,
            horizon,
            transition_dim,
            cond_dim,
            num_tasks,
            dim=64,
            dim_mults=(1, 2, 4, 8),
            attention=False,
            depth=56,
            mlp_ratio=4.0,
            hidden_size=128,
            num_heads=8,
            train_device=None,
            prompt_trajectories=None,
            task_list=None,
            action_dim=None,
            max_ep_len=1000,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.dim = dim
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.state_dim = transition_dim - 1
        self.action_dim = action_dim
        self.prompt_trajectories = prompt_trajectories
        self.task_list = task_list
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer=4,
            n_head=2,
            n_inner=4 * 256,
            activation_function='mish',
            n_positions=1024,
            n_ctx=1023,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(2*dim),
            nn.Linear(2*dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, 2*dim),
        )
        self.return_mlp = nn.Sequential(
            nn.Linear(1, dim * 2),
            nn.Mish(),
            nn.Linear(dim * 2, 4*dim),
            nn.Mish(),
            nn.Linear(dim * 4, 2 * dim),
        )
        self.prompt_embed = nn.Sequential(
            #nn.LayerNorm(1),
            nn.Linear(1, hidden_size*2),  # TODO
            nn.Mish(),
            nn.Linear(hidden_size * 2, 4 * hidden_size),
            nn.Mish(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.embed_obs = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, 2 * hidden_size),  # TODO
            nn.Mish(),
            nn.Linear(hidden_size * 2, 4 * hidden_size),
            nn.Mish(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

        self.mask_dist = Bernoulli(probs=0.8)
        self.transformer = GPT2Model(config)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.embed_act = nn.Linear(self.action_dim, hidden_size)
        self.position_emb = nn.Parameter(torch.zeros(1, 42+horizon, hidden_size))
        self.predict_act = torch.nn.Linear(hidden_size, self.action_dim)

    def forward(self, x, time, cond, value, context_mask, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None):
        '''
            x : [ batch x horizon x transition ]
        '''
        cond = cond.long()
        prompt_data = torch.tensor([self.prompt_trajectories[ind] for ind in cond]).to(device=x.device, dtype=torch.float32).reshape(cond.shape[0], -1, 1)
        prompt_embeddings = self.prompt_embed(prompt_data)
        obs_embeddings = self.embed_obs(x_condition)
        t = self.time_mlp(time).unsqueeze(1)
        if not force:
            mask = self.mask_dist.sample(sample_shape=(cond.shape[0], 1)).to(x.device)
        else:
            mask = 1 if return_cond else 0
        batch_size, seq_length = x.shape[0], x.shape[1]
        value = value.view(-1, 1)
        value = self.return_mlp(value)
        cond_return = (value * mask).unsqueeze(1) + 1e-8
        act_embeddings = self.embed_act(x)
        addition_length = 35 + 1 + 1 + 5  #TODO
        addition_attention_mask = torch.ones((batch_size, addition_length), dtype=torch.long, device=x.device)
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=x.device)
            '''set attention mask'''
        stacked_inputs = torch.cat((t, cond_return, prompt_embeddings, obs_embeddings, act_embeddings), dim=1)
        stacked_attention_mask = torch.cat(
            (addition_attention_mask, attention_mask), dim=1)
        stacked_inputs = t * stacked_inputs + cond_return + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = self.embed_ln(stacked_inputs)
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        act_preds = self.predict_act(x[:, -seq_length:, :])
        return act_preds

class TasksmetaAug(nn.Module):
    "MT-Diffuser with a Transformer backbone"

    def __init__(
            self,
            horizon,
            transition_dim,
            cond_dim,
            num_tasks,
            dim=128,
            dim_mults=(1, 2, 4, 8),
            attention=False,
            depth=56,
            mlp_ratio=4.0,
            hidden_size=256,
            num_heads=8,
            train_device=None,
            prompt_trajectories=None,
            task_list=None,
            action_dim=None,
            max_ep_len=1000,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.dim = dim
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.state_dim = transition_dim - 1
        self.action_dim = action_dim
        self.prompt_trajectories = prompt_trajectories
        self.task_list = task_list
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer=6,
            n_head=4,
            n_inner=4 * 256,
            activation_function='mish',
            n_positions=1024,
            n_ctx=1023,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(2*dim),
            nn.Linear(2*dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, 2*dim),
        )
        self.reward_mlp = nn.Sequential(
            nn.Linear(1, dim * 2),
            nn.Mish(),
            nn.Linear(dim * 2, 2*dim),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(self.state_dim, dim * 2),
            nn.Mish(),
            nn.Linear(dim * 2, 2 * dim),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(self.action_dim, dim * 2),
            nn.Mish(),
            nn.Linear(dim * 2, 2 * dim),
        )
        self.prompt_embed = nn.Sequential(
            nn.LayerNorm(self.state_dim + self.action_dim),
            nn.Linear(self.state_dim + self.action_dim, hidden_size * 2),  # TODO
            nn.Mish(),
            nn.Linear(hidden_size * 2, 4 * hidden_size),
            nn.Mish(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.mask_dist = Bernoulli(probs=0.8)
        self.transformer = GPT2Model(config)
        self.embed_ln = nn.LayerNorm(hidden_size)
        ''' 1(T_i)+50(prompt)+16(modeling sequence)'''
        self.position_emb = nn.Parameter(torch.zeros(1, 101, hidden_size))

        # note: we don't predict states or returns for the paper
        self.predict_act = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.Mish(),
            nn.Linear(dim * 2, self.action_dim),
        )
        self.predict_state = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.Mish(),
            nn.Linear(dim * 2, self.state_dim),
        )
        self.predict_reward = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.Mish(),
            nn.Linear(dim * 2, 1),
        )

    def forward(self, x, time, cond, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None):
        '''
            x : [ batch x horizon x transition ]
        '''
        cond = cond.long()
        prompt_data = get_prompt_batchs(self.prompt_trajectories, cond, num_episodes=1, num_steps=20, is_meta=True)
        prompt_data = prompt_data.to(device=x.device, dtype=torch.float32).reshape(cond.shape[0], -1, self.state_dim+self.action_dim)#TODO
        prompt_embeddings = self.prompt_embed(prompt_data)
        states, actions, rewards, next_states = x[:, :, :self.state_dim], x[:, :, self.state_dim:self.state_dim+self.action_dim], \
                                               x[:, :, self.state_dim+self.action_dim:self.state_dim+self.action_dim+1], \
                                               x[:, :, self.state_dim+self.action_dim+1:]
        state_embeddings = self.state_mlp(states)
        next_state_embeddings = self.state_mlp(next_states)
        action_embeddings = self.action_mlp(actions)
        reward_embeddings = self.reward_mlp(rewards)
        t = self.time_mlp(time).unsqueeze(1)
        batch_size, seq_length = x.shape[0], x.shape[1]
        addition_length = 20 + 1 #TODO
        addition_attention_mask = torch.ones((batch_size, addition_length), dtype=torch.long, device=x.device)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, 4 * seq_length), dtype=torch.long, device=x.device)
            '''set attention mask'''
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings, reward_embeddings, next_state_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 4 * seq_length, self.hidden_size)
        all_inputs = torch.cat((t, prompt_embeddings, stacked_inputs), dim=1)
        stacked_attention_mask = torch.cat(
            (addition_attention_mask, attention_mask), dim=1)
        final_inputs = t * all_inputs + self.position_emb[:, :all_inputs.shape[1], :]
        final_inputs = self.embed_ln(final_inputs)
        transformer_outputs = self.transformer(
            inputs_embeds=final_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state'][:, -4 * seq_length:, :]
        x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)
        act_preds = self.predict_act(x[:, 1])
        state_preds = self.predict_state(x[:, 0])
        reward_preds = self.predict_reward(x[:, 2])
        next_state_preds = self.predict_state(x[:,3])
        output = torch.cat([state_preds, act_preds, reward_preds, next_state_preds], dim=-1)
        return output
