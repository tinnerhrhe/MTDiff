from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb
import math
import einops
import time
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import diffuser.utils as utils
from .helpers import (
    vp_beta_schedule,
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    apply_conditioning_v1,
    one_hot_dict,
    Losses,
    SinusoidalPosEmb,
)
ACTIVATION_MAP = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "linear": nn.Identity,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
}

Sample = namedtuple('Sample', 'trajectories values chains')


@torch.no_grad()
def default_sample_fn(model, x, cond, task, value,  context_mask, t, **sample_kwargs):
    b, *_, device = *x.shape, x.device
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, task=task, value=value, context_mask=context_mask, t=t)
    #model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = 0.5*torch.randn_like(x)
    nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))) #TODO:add nonzero_mask
    #noise = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    #noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True, beta_schedule='vp',
        action_weight=1.0, loss_discount=1.0, loss_weights=None, drop_prob=0.25,
    ):
        super().__init__()
        self.drop_prob = torch.tensor(drop_prob)
        print(self.drop_prob)
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim# + 1#TODO
        self.model = model
        self.guide_s = 0.6
        self.act_rew_dim = self.action_dim# + 1#TODO
        """add beta schedule"""
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)
        #betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        #print(loss_weights)
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.act_rew_dim)
        ##loss_weights = self.get_loss_weights(loss_discount)
        ##self.loss_fn = Losses[loss_type](loss_weights)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.act_rew_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        ##loss_weights[0, :self.act_rew_dim] = action_weight
        loss_weights[0, :] = action_weight
        return loss_weights
   # """
    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:  #via the equation between x_0 and x_t
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, task, value, context_mask, t):
        #double batch
        #x = x.repeat(2,1,1)
        #t = t.repeat(2,1,1)
        batch_size = x.shape[0]
        noise_return_cond = self.model(x, task[batch_size:], value[:batch_size], context_mask[batch_size:], t,
                                       force=True, return_cond=True)
        noise_task_cond = self.model(x, task[:batch_size], value[:batch_size], context_mask[:batch_size], t, force=True)
        noise_uncond = self.model(x, task[batch_size:], value[:batch_size], context_mask[batch_size:], t, force=True)
        #noise = (1+self.guide_s)*noise_cond-self.guide_s*noise_uncond
        noise = noise_uncond + self.guide_s * (noise_return_cond-noise_uncond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, task, value, guidance, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device
        context_mask = torch.zeros_like(task).to(device)
        # double the batch
        batch_size = shape[0]
        task = task.repeat(2)
        value = value.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[batch_size:] = 1. # makes second half of batch context free
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None
        self.guide_s = guidance
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, task, value, context_mask, t, **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, task, value, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        task = task.repeat(batch_size)
        value = value.repeat(batch_size)
        #task = [one_hot_dict[t] for t in task]
        return self.p_sample_loop(shape, cond, task, value, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, task, value, t):
        #x_start = torch.cat(x_start)
        noise = torch.randn_like(x_start)
        context_mask = torch.bernoulli(torch.zeros_like(task)+self.drop_prob).to(x_start.device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, 0)##self.act_rew_dim) #only conditon on current observation and get a_t and r_t

        x_recon = self.model(x_noisy, task, value, context_mask, t)
        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        #task = torch.zeros((batch_size),device=x.device)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x[:,:,1:1+self.action_dim], *args, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)
class GaussianInvDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=True, predict_epsilon=True, beta_schedule='vp',
        action_weight=1.0, loss_discount=1.0, loss_weights=None, drop_prob=0.25,
    ):
        super().__init__()
        self.drop_prob = torch.tensor(drop_prob)
        print(self.drop_prob)
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + 1# + action_dim# + 1#TODO
        self.model = model
        self.guide_s = 1.6
        self.act_rew_dim = self.action_dim# + 1#TODO
        """add beta schedule"""
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)
        #betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        #print(loss_weights)
        ##loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        ##self.loss_fn = Losses[loss_type](loss_weights, self.act_rew_dim)
        #loss_weights = self.get_loss_weights(loss_discount)
        #self.loss_fn = Losses[loss_type](loss_weights)
        #self.action_weight = action_weight
        #self.inv_model = ARInvModel(hidden_dim=256, observation_dim=self.observation_dim, action_dim=self.action_dim)

    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = 1
        dim_weights = torch.ones(self.observation_dim+1, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[:10, :] = 0
        loss_weights[10, :] = self.action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:  #via the equation between x_0 and x_t
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, task, value, context_mask, t):
        batch_size = x.shape[0]
        noise_task_cond = self.model(x, t, task[:batch_size], value[:batch_size], context_mask[:batch_size], x_condition=cond,
                                     force=True, return_cond=True)
        noise_uncond = self.model(x, t, task[batch_size:], value[:batch_size], context_mask[batch_size:], x_condition=cond, force=True)
        #noise = (1+self.guide_s)*noise_cond-self.guide_s*noise_uncond
        noise = noise_uncond + self.guide_s * (noise_task_cond-noise_uncond) #+ 1.2 * (noise_return_cond-noise_uncond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, task, value, guidance, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device
        context_mask = torch.zeros_like(task).to(device)
        batch_size = shape[0]
        task = task.repeat(2)
        value = value.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[batch_size:] = 1. # makes second half of batch context free
        x = torch.randn(shape, device=device)
        chain = [x] if return_chain else None
        self.guide_s = guidance
        values = None
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, task, value, context_mask, t, **sample_kwargs)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        #x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, task, value, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)
        return self.p_sample_loop(shape, cond, task, value, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, task, value, t):
        cond = einops.rearrange(cond, 'i j h k -> (i j) h k')
        task = einops.rearrange(task, 'i j -> (i j)') #task.reshape(-1, self.num_tasks)
        value = einops.rearrange(value, 'i j 1 -> (i j) 1')
        noise = torch.randn_like(x_start)
        context_mask = None
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t, task, value, context_mask, x_condition=cond)
        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, *args):
        x = x.reshape(-1, self.horizon, self.observation_dim+self.action_dim+1)
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffusion_loss, info = self.p_losses(x, *args, t)
        x_t = x[:, :-1, self.action_dim + 1:]
        a_t = x[:, :-1, :self.action_dim]
        x_t_1 = x[:, 1:, self.action_dim + 1:]
        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
        task = args[1]
        task = einops.rearrange(task, 'i j -> (i j)')
        task = task.repeat(self.horizon - 1)
        a_t = a_t.reshape(-1, self.action_dim)
        pred_a_t = self.inv_model.forward0(x_comb_t, task)
        inv_loss = F.mse_loss(pred_a_t, a_t)
        loss = (1 / 2) * (diffusion_loss + inv_loss)
        info.update({'inv_loss': inv_loss})
        return loss, info

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)
class GaussianActDiffusion(GaussianInvDiffusion):
    def __init__(self, *args, loss_discount=1.0, loss_type='l1', action_weight=10, **kwargs):
        super().__init__(*args, loss_discount=loss_discount, loss_type=loss_type, action_weight=action_weight, **kwargs)
        self.transition_dim = self.action_dim
        self.action_weight = action_weight
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses[loss_type](loss_weights)
    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        #self.action_weight = 1
        dim_weights = torch.ones(self.action_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        loss_weights[0, :] = self.action_weight

        return loss_weights
    def loss(self, x, *args):
        x = einops.rearrange(x, 'i j h k -> (i j) h k')
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffusion_loss, info = self.p_losses(x, *args, t) #TODO
        loss = diffusion_loss
        return loss, info
    def conditional_sample(self, cond, task, value, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = cond.shape[0]
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        return self.p_sample_loop(shape, cond, task, value, **sample_kwargs)
class AugDiffusion(GaussianInvDiffusion):
    def __init__(self, *args, loss_discount=1.0, loss_type='l1', action_weight=10, **kwargs):
        super().__init__(*args, loss_discount=loss_discount, loss_type=loss_type, action_weight=action_weight, **kwargs)
        self.transition_dim = self.action_dim + 2*self.observation_dim + 1
        self.action_weight = action_weight
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses[loss_type](loss_weights)
    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        dim_weights = torch.ones(self.action_dim+self.observation_dim*2+1, dtype=torch.float32)

        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        return loss_weights
    def loss(self, x, *args):
        x = einops.rearrange(x, 'i j h k -> (i j) h k')
        batch_size = len(x)
        task = args[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffusion_loss, info = self._losses(x, task, t)
        loss = diffusion_loss
        return loss, info
    def _losses(self, x_start, task, t):
        task = einops.rearrange(task, 'i j -> (i j)')
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t, task)
        assert noise.shape == x_recon.shape
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info
    def conditional_sample(self, task, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = task.shape[0]
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        return self._sample_loop(shape, task, **sample_kwargs)

    @torch.no_grad()
    def _sample_loop(self, shape, task, verbose=True, return_chain=False,
                      sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        chain = [x] if return_chain else None
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x = self.default_sample_fn(x, task, t, **sample_kwargs)

            #progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        #x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return x

    @torch.no_grad()
    def default_sample_fn(self, x, task, t, **sample_kwargs):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self._mean_variance(x=x, task=task, t=t)
        noise = 1.0 * torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))  # TODO:add nonzero_mask
        return model_mean + nonzero_mask * (1.0 * model_log_variance).exp() * noise
    def _mean_variance(self, x, task, t):
        batch_size = x.shape[0]
        noise = self.model(x, t, task)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
