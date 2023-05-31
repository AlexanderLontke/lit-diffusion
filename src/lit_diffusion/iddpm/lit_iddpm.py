"""
Lightning Module implementing Improved Denoising Diffusion Probabilistic Models
https://arxiv.org/abs/2102.09672

Code adapted from:
https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
"""
from typing import Any, Callable, Dict, Optional, Union

# Numpy
import numpy as np

# PyTorch
import torch

# Lit Dffusion IDDPM
from lit_diffusion.iddpm.losses import normal_kl, discretized_gaussian_log_likelihood
from lit_diffusion.iddpm.util import extract_into_tensor, mean_flat
from lit_diffusion.iddpm.constants import (
    IDDPMTargetType,
    IDDPMVarianceType,
    IDDPMLossType,
)

# LitDiffusion
from lit_diffusion.diffusion_base.lit_diffusion_base import LitDiffusionBase
from lit_diffusion.diffusion_base.constants import (
    LOSS_DICT_TARGET_KEY,
    LOSS_DICT_NOISED_INPUT_KEY,
    LOSS_DICT_NOISE_KEY,
    LOSS_DICT_MODEL_OUTPUT_KEY,
    LOSS_DICT_LOSSES_KEY,
    P_MEAN_VAR_DICT_MEAN_KEY,
    P_MEAN_VAR_DICT_VARIANCE_KEY,
    P_MEAN_VAR_DICT_LOG_VARIANCE_KEY,
    P_MEAN_VAR_DICT_PRED_X_0_KEY,
)


class LitIDDPM(LitDiffusionBase):
    def __init__(
        self,
        diffusion_target: Union[str, IDDPMTargetType],
        model_variance_type: Union[str, IDDPMVarianceType],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_variance_type = IDDPMVarianceType(model_variance_type)
        self.diffusion_target = IDDPMTargetType(diffusion_target)
        self.num_timesteps = int(self.betas.shape[0])

    # PyTorch Lightning functions
    def p_loss(
        self, x_0: torch.Tensor, t: torch.Tensor, model_kwargs: Optional[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses for a single timestep.

        :param x_0: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        x_t, noise = self.q_sample(x_0, t)

        terms = {}

        if (
            self.loss_type == IDDPMLossType.KL
            or self.loss_type == IDDPMLossType.RESCALED_KL
        ):
            vb_out = self._vb_terms_bpd(
                x_start=x_0,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
            terms["loss"] = vb_out["output"]
            if self.loss_type == IDDPMLossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
            # TODO determine the correct values for these variables
            model_output = None
            target = None
        elif (
            self.loss_type == IDDPMLossType.MSE
            or self.loss_type == IDDPMLossType.RESCALED_MSE
        ):
            model_output = self.p_theta_model(
                x_t, self._scale_timesteps(t), **model_kwargs
            )

            if self.model_var_type in [
                IDDPMVarianceType.LEARNED,
                IDDPMVarianceType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    frozen_out=frozen_out,
                    x_start=x_0,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == IDDPMLossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                IDDPMTargetType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_0, x_t=x_t, t=t
                )[0],
                IDDPMTargetType.START_X: x_0,
                IDDPMTargetType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_0.shape
            mse_term = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = mse_term + terms["vb"]
            else:
                terms["loss"] = mse_term
        else:
            raise NotImplementedError(self.loss_type)

        return {
            LOSS_DICT_LOSSES_KEY: terms["loss"],
            LOSS_DICT_NOISED_INPUT_KEY: x_t,
            LOSS_DICT_NOISE_KEY: noise,
            LOSS_DICT_TARGET_KEY: target,
            LOSS_DICT_MODEL_OUTPUT_KEY: model_output,
        }

    def _vb_terms_bpd(
        self, x_start, x_t, t, clip_denoised=True, model_kwargs=None, frozen_out=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            x_t,
            t,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            frozen_out=frozen_out,
        )
        kl = normal_kl(
            true_mean,
            true_log_variance_clipped,
            out[P_MEAN_VAR_DICT_MEAN_KEY],
            out[P_MEAN_VAR_DICT_LOG_VARIANCE_KEY],
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start,
            means=out[P_MEAN_VAR_DICT_MEAN_KEY],
            log_scales=0.5 * out[P_MEAN_VAR_DICT_LOG_VARIANCE_KEY],
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out[P_MEAN_VAR_DICT_PRED_X_0_KEY]}

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool,
        denoised_fn: Optional[Callable] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        frozen_out=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param x_t: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        model_output = (
            frozen_out
            if frozen_out
            else self.p_theta_model(x_t, self._scale_timesteps(t), **model_kwargs)
        )

        if self.model_variance_type in [
            IDDPMVarianceType.LEARNED,
            IDDPMVarianceType.LEARNED_RANGE,
        ]:
            assert model_output.shape == (B, C * 2, *x_t.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if self.model_var_type == IDDPMVarianceType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x_t.shape
                )
                max_log = extract_into_tensor(np.log(self.betas), t, x_t.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixed_large, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                IDDPMVarianceType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                IDDPMVarianceType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = extract_into_tensor(model_variance, t, x_t.shape)
            model_log_variance = extract_into_tensor(model_log_variance, t, x_t.shape)

        # Find x_0 from model output
        if self.diffusion_target == IDDPMTargetType.START_X:
            pred_xstart = model_output
        elif self.diffusion_target == IDDPMTargetType.PREVIOUS_X:
            pred_xstart = self._predict_xstart_from_xprev(
                x_t=x_t, t=t, xprev=model_output
            )
        elif self.diffusion_target:
            pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
        else:
            raise NotImplementedError(self.diffusion_target)

        # Apply additional processing steps
        if denoised_fn is not None:
            pred_xstart = denoised_fn(pred_xstart)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)

        if self.diffusion_target in [
            IDDPMTargetType.START_X,
            IDDPMTargetType.EPSILON,
        ]:
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x_t, t=t
            )
        elif self.diffusion_target == IDDPMTargetType.PREVIOUS_X:
            model_mean = model_output
        else:
            raise NotImplementedError(self.diffusion_target)

        assert (
            model_mean.shape
            == model_log_variance.shape
            == pred_xstart.shape
            == x_t.shape
        )
        return {
            P_MEAN_VAR_DICT_MEAN_KEY: model_mean,
            P_MEAN_VAR_DICT_VARIANCE_KEY: model_variance,
            P_MEAN_VAR_DICT_LOG_VARIANCE_KEY: model_log_variance,
            P_MEAN_VAR_DICT_PRED_X_0_KEY: pred_xstart,
        }

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
