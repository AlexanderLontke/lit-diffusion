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
        alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod)
        )
        # Cache values often used to sample from p_{\theta}(x_{t-1}|x_{t})
        alphas_cumprod_prev = alphas_cumprod.roll(shifts=1, dims=0)
        alphas_cumprod_prev[0] = 1.0
        self.register_buffer(
            "posterior_mean_coef1",
            torch.sqrt(alphas_cumprod_prev) * self.betas / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.sqrt(self.alphas)
            * (1.0 - alphas_cumprod_prev)
            / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_variance",
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) * self.betas,
        )
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(np.maximum(self.posterior_variance, 1e-20)),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_m1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0)
        )

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
    def q_posterior(self, x_0, x_t, t):
        """
        Returns the posterior distribution q(x_{t-1}|x_t,x_0) which is defined by its mean and variance
        :param x_0: x at timestep 0
        :param x_t: x at timestep t
        :param t: current timestep t
        :return: mean, variance and log of the variance based on x_0, x_t, and t
        """
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_0.shape) * x_0
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_mean_variance(self, x_t, t):
        """
        Calculates an approximation of x_0, given eps_{\theta} and based off that returns the posterior
        distribution p_{\theta}(x_{t-1}|x_t,x_0)
        :param x_t: x noised at timestep t
        :param t: current timestep t
        :return: approximated mean and variance of the posterior distribution
        """
        model_output = self.model(x_t, t)
        if self.diffusion_target == DiffusionTarget.EPS:
            x_0_predicted = (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
                * model_output
            )
        elif self.diffusion_target == DiffusionTarget.X_0:
            x_0_predicted = model_output
        else:
            raise NotImplementedError(
                f"Diffusion Target {self.diffusion_target} is not implemented"
            )

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_0=x_0_predicted, x_t=x_t, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        Method implementing one DDPM sampling step
        :param x_t: sample at current timestep
        :param t: denotes current timestep
        :return: x at timestep t-1
        """
        b, *_ = x_t.shape
        model_mean, _, model_log_variance = self.p_mean_variance(x=x_t, t=t)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self, shape, starting_noise: Optional[torch.Tensor] = None, batch_size: int = 1
    ):
        x_t = default(
            starting_noise,
            lambda: torch.randn((batch_size, *shape), device=self.device),
        )
        for t in tqdm(
            reversed(range(self.beta_schedule_steps)),
            desc="DDPM sampling:",
            total=self.beta_schedule_steps,
        ):
            x_t = self.p_sample(
                x_t=x_t,
                t=torch.full(
                    size=(batch_size,),
                    fill_value=t,
                    device=self.device,
                    dtype=torch.long,
                ),
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


if __name__ == "__main__":
    import numpy as np

    betas = make_beta_schedule(
        schedule="linear",
        n_timestep=10,
        linear_start=0.0001,
        linear_end=0.02,
    )
    print(betas)
    betas_prev = torch.tensor(
        betas,
        dtype=torch.float32,
        device=torch.device("cpu"),
        requires_grad=False,
    ).roll(shifts=1, dims=0)
    betas_prev[0] = 1.0
    print(betas_prev)
    print(np.append(1.0, betas[:-1]))
