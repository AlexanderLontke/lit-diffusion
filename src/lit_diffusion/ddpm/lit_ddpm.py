from typing import Dict, Optional, Union

import numpy as np

from tqdm import tqdm

import torch
from torch import nn
import pytorch_lightning as pl

# Util
from lit_diffusion.util import instantiate_python_class_from_string_config
from lit_diffusion.constants import TRAINING_LOSS_METRIC_KEY
from lit_diffusion.ddpm.util import extract_into_tensor, default
from lit_diffusion.constants import (
    DiffusionTarget,
)

# Beta Schedule
from lit_diffusion.ddpm.beta_schedule import make_beta_schedule


class LitDDPM(pl.LightningModule):
    def __init__(
        self,
        p_theta_model: nn.Module,
        diffusion_target: Union[str, DiffusionTarget],
        schedule_type: str,
        beta_schedule_steps: int,
        beta_schedule_linear_start: float,
        beta_schedule_linear_end: float,
        learning_rate: float,
        data_key: Optional[str] = None,
        learning_rate_scheduler_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["p_theta_model"])
        self.p_theta_model = p_theta_model
        self.diffusion_target = DiffusionTarget(diffusion_target)

        # Fix beta schedule
        self.beta_schedule_steps = beta_schedule_steps
        self.betas = torch.tensor(
            make_beta_schedule(
                schedule=schedule_type,
                n_timestep=beta_schedule_steps,
                linear_start=beta_schedule_linear_start,
                linear_end=beta_schedule_linear_end,
            ),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        # Cache values often used for approximating p_{\theta}(x_{t-1}|x_{t})
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - self.alphas_cumprod)
        )
        # Cache values often used to sample from p_{\theta}(x_{t-1}|x_{t})
        self.register_buffer("sqrt_alphas", np.sqrt(self.alphas))
        self.register_buffer("sqrt_betas", np.sqrt(self.betas))

        # Setup loss
        self.loss = nn.MSELoss(reduction="sum")

        # Setup learning and scheduler
        self.learning_rate = learning_rate
        self.learning_rate_scheduler_config = learning_rate_scheduler_config

        # Data access
        self.data_key = data_key

    # Methods relating to approximating p_{\theta}(x_{t-1}|x_{t})
    def training_step(self, x_0):
        if self.data_key:
            x_0 = x_0[self.data_key]
        t = torch.randint(
            0, self.beta_schedule_steps, (x_0.shape[0],), device=self.device
        ).long()
        loss = self.p_loss(x_0=x_0, t=t)
        self.log(
            TRAINING_LOSS_METRIC_KEY,
            loss,
            prog_bar=True,
            on_epoch=True,
        )
        return loss

    def p_loss(self, x_0, t):
        """
        Calculates the variational lower bound loss of p_{\theta}(x_{t-1}|x_{t}) based on the simplified
        difference between the distribution q(x_{t}|x_{t+1}) and p_{\theta}(x_{t}|x_{t+1})
        :param x_0: sample without any noise
        :param t: timestep for which the difference of distributions should be calculated
        :return: Simplified loss for the KL_divergence of q and p_{\theta}
        """
        noised_x, noise = self.q_sample(
            x_0=x_0,
            t=t,
        )
        model_x = self.p_theta_model(noised_x, t)

        # Determine target
        if self.diffusion_target == DiffusionTarget.X_0:
            target = x_0
        elif self.diffusion_target == DiffusionTarget.EPS:
            target = noise
        else:
            raise NotImplementedError(
                f"Diffusion target {self.diffusion_target} not supported"
            )

        loss_t_simple = self.loss(model_x, target)
        return loss_t_simple

    def q_sample(self, x_0, t):
        """
        Samples x_t from the distribution q(x_t|x_{t-1}) given x_0 and t, made possible through the
        re-parametrization trick
        :param x_0: sample without any noise
        :param t: timestep for which a noised sample x_t should be created
        :return: x_t and noise used to generate it
        """
        epsilon_noise = torch.randn_like(x_0)
        noised_x = (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
            * epsilon_noise
        )
        return noised_x, epsilon_noise

    # Methods related to sampling from the distribution p_{\theta}(x_{t-1}|x_{t})
    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        Method implementing one DDPM sampling step
        :param x_t: sample at current timestep
        :param t: denotes current timestep
        :return: x at timestep t-1
        """
        z = torch.rand((1,)) if t > 1 else 0
        model_estimation = self.p_theta_model(x_t, t)
        # Note: 1 - alpha_t = beta_t
        x_t_minus_one = (
            1.0
            / self.sqrt_alphas[t]
            * (
                x_t
                - (self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t])
                * model_estimation
            )
            + self.sqrt_betas[t] * z
        )
        return x_t_minus_one

    @torch.no_grad()
    def p_sample_loop(self, shape, starting_noise: Optional[torch.Tensor] = None):
        x_t = default(starting_noise, lambda: torch.randn(shape, device=self.device))
        for t in tqdm(
            reversed(range(self.beta_schedule_steps)),
            desc="DDPM sampling:",
            total=self.beta_schedule_steps,
        ):
            x_t = self.p_sample(
                x_t=x_t,
                t=t,
            )
        return x_t

    def configure_optimizers(self):
        """
        Lightning method used to configure an optimizer for training
        :return: torch Optimizer class
        """
        return_dict = {}
        lr = self.learning_rate
        params = list(self.p_theta_model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        # If a learning rate scheduler config was given, initialize a lr-scheduler
        if self.learning_rate_scheduler_config:
            return_dict["lr_scheduler"] = instantiate_python_class_from_string_config(
                class_config=self.learning_rate_scheduler_config, optimizer=opt
            )
        return_dict["optimizer"] = opt
        return return_dict
