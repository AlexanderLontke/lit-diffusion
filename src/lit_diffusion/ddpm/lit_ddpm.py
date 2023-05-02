from typing import Any, Callable, Dict, Optional, Union


import torch
from torch import nn

# Lit Diffusion DDPM
from lit_diffusion.ddpm.util import extract_into_tensor
from lit_diffusion.ddpm.constants import (
    DDPMDiffusionTarget,
)

# Lit Diffusion
from lit_diffusion.diffusion_base.lit_diffusion_base import LitDiffusionBase
from lit_diffusion.diffusion_base.constants import (
    LOSS_DICT_TARGET_KEY,
    LOSS_DICT_MODEL_OUTPUT_KEY,
    LOSS_DICT_LOSSES_KEY,
    P_MEAN_VAR_DICT_LOG_VARIANCE_KEY,
    P_MEAN_VAR_DICT_MEAN_KEY,
    P_MEAN_VAR_DICT_VARIANCE_KEY,
    P_MEAN_VAR_DICT_PRED_X_0_KEY,
)


class LitDDPM(LitDiffusionBase):
    def __init__(
        self,
        diffusion_target: Union[str, DDPMDiffusionTarget],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Set diffusion training target
        self.diffusion_target = DDPMDiffusionTarget(diffusion_target)

        # Setup loss
        # MSE loss is equivalent to L2 loss,
        # averaging is delayed to enable t-schedule sampling
        self.loss = nn.MSELoss(reduction="none")

    # Methods relating to approximating p_{\theta}(x_{t-1}|x_{t})
    def p_loss(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates the variational lower bound loss of p_{\theta}(x_{t-1}|x_{t}) based on the simplified
        difference between the distribution q(x_{t}|x_{t+1}) and p_{\theta}(x_{t}|x_{t+1})
        :param x_0: sample without any noise
        :param t: timestep for which the difference of distributions should be calculated
        :param model_kwargs: Any additional model key word arguments that might be needed
        :return: Simplified loss for the KL_divergence of q and p_{\theta}
        """
        noised_x, noise = self.q_sample(
            x_0=x_0,
            t=t,
        )
        model_x = self.p_theta_model(noised_x, t, **model_kwargs)

        # Determine target
        if self.diffusion_target == DDPMDiffusionTarget.X_0:
            target = x_0
        elif self.diffusion_target == DDPMDiffusionTarget.EPS:
            target = noise
        else:
            raise NotImplementedError(
                f"Diffusion target {self.diffusion_target} not supported"
            )
        return {
            LOSS_DICT_LOSSES_KEY: self.loss(model_x, target).mean(
                dim=list(range(1, len(model_x.shape)))
            ),
            LOSS_DICT_MODEL_OUTPUT_KEY: model_x,
            LOSS_DICT_TARGET_KEY: target,
        }

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
    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool,
        denoised_fn: Optional[Callable] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates an approximation of x_0, given eps_{\theta} and based off that returns the posterior
        distribution p_{\theta}(x_{t-1}|x_t,x_0)
        :param x_t: the [N x C x ...] tensor at timestep t
        :param t: a 1-D Tensor of timesteps
        :param clip_denoised: if True, clip the denoised signal into [-1, 1]
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'p_mean': the model mean output.
                 - 'p_variance': the model variance output.
                 - 'p_log_variance': the log of 'variance'.
                 - 'pred_x_0': the prediction for x_0.
        """
        model_output = self.p_theta_model(x_t, t, **model_kwargs)
        if self.diffusion_target == DDPMDiffusionTarget.EPS:
            x_0_predicted = (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract_into_tensor(self.sqrt_recip_m1_alphas_cumprod, t, x_t.shape)
                * model_output
            )
        elif self.diffusion_target == DDPMDiffusionTarget.X_0:
            x_0_predicted = model_output
        else:
            raise NotImplementedError(
                f"Diffusion Target {self.diffusion_target} is not implemented"
            )

        # Apply additional processing steps
        if denoised_fn is not None:
            x_0_predicted = denoised_fn(x_0_predicted)
        if clip_denoised:
            x_0_predicted = x_0_predicted.clamp(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_0=x_0_predicted, x_t=x_t, t=t
        )
        return {
            P_MEAN_VAR_DICT_MEAN_KEY: model_mean,
            P_MEAN_VAR_DICT_VARIANCE_KEY: posterior_variance,
            P_MEAN_VAR_DICT_LOG_VARIANCE_KEY: posterior_log_variance,
            P_MEAN_VAR_DICT_PRED_X_0_KEY: x_0_predicted,
        }
