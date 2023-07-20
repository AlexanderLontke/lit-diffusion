from abc import abstractmethod
from contextlib import contextmanager, nullcontext
from typing import Any, List, Optional, Dict, Callable

from tqdm import tqdm

# Numpy
import numpy as np

# PyTorch
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

# Lightning
import pytorch_lightning as pl

# LitDiffusion
from lit_diffusion.beta_schedule.beta_schedule import make_beta_schedule
from lit_diffusion.iddpm.sampler import (
    ScheduleSampler,
    UniformSampler,
    LossAwareSampler,
)
from lit_diffusion.util import instantiate_python_class_from_string_config
from lit_diffusion.ddpm.util import default, extract_into_tensor
from lit_diffusion.utils.lit_ema import LitEma
from lit_diffusion.utils.vlb_utils import (
    mean_flat,
    normal_kl,
    discretized_gaussian_log_likelihood,
)
from lit_diffusion.diffusion_base.constants import (
    LOGGING_TRAIN_PREFIX,
    LOGGING_VAL_PREFIX,
    TRAINING_LOSS_METRIC_KEY,
    LOSS_DICT_TARGET_KEY,
    LOSS_DICT_LOSSES_KEY,
    LOSS_DICT_MODEL_OUTPUT_KEY,
    LOSS_DICT_RECONSTRUCTED_INPUT_KEY,
    P_MEAN_VAR_DICT_MEAN_KEY,
    P_MEAN_VAR_DICT_LOG_VARIANCE_KEY,
    P_MEAN_VAR_DICT_PRED_X_0_KEY,
)

SAMPLE_KEY = "sample"


class LitDiffusionBase(pl.LightningModule):
    def __init__(
        self,
        p_theta_model: nn.Module,
        schedule_type: str,
        beta_schedule_steps: int,
        beta_schedule_linear_start: float,
        beta_schedule_linear_end: float,
        learning_rate: float,
        data_key: str,
        use_ema: bool = False,
        clip_denoised: bool = False,
        schedule_sampler: Optional[ScheduleSampler] = None,
        stack_inputs_keys: Optional[List[str]] = None,
        auxiliary_p_theta_model_input: Optional[Dict] = None,
        learning_rate_scheduler_config: Optional[Dict] = None,
        training_metrics: Optional[Dict[str, Callable]] = None,
        validation_metrics: Optional[Dict[str, Callable]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["p_theta_model"])
        # P_theta model
        self.p_theta_model = p_theta_model
        self.stack_inputs_keys = stack_inputs_keys
        self.auxiliary_p_theta_model_input = auxiliary_p_theta_model_input

        # Make beta schedule
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
        # Maintain an exponentially moving average of the p_theta model
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_model = LitEma(self.p_theta_model)
            print(f"Keeping EMAs of {len(list(self.ema_model.buffers()))}.")

        # Clip during sampling
        self.clip_denoised = clip_denoised

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

        # Setup timestep scheduler
        self.schedule_sampler = schedule_sampler or UniformSampler(beta_schedule_steps)

        # Setup metrics
        self.training_metrics = training_metrics
        self.validation_metrics = validation_metrics

        # Setup learning and scheduler
        self.learning_rate = learning_rate
        self.learning_rate_scheduler_config = learning_rate_scheduler_config

        # Data access
        self.data_key = data_key

    # Methods relating to calling the de-noising model
    def format_p_theta_model_input(self, x_0, batch):
        model_kwargs = {}
        if self.auxiliary_p_theta_model_input:
            model_kwargs = {
                model_kwarg: batch[data_key]
                for model_kwarg, data_key in self.auxiliary_p_theta_model_input.items()
            }

        # stack data from kwargs onto x if it is desired
        if self.stack_inputs_keys:
            for k in self.stack_inputs_keys:
                x_0 = torch.cat([x_0, model_kwargs.pop(k)], dim=1)
        return x_0, model_kwargs

    @abstractmethod
    def p_loss(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """
        Called during training and validation to determine model loss based on a single de-noising timestep
        :param x_0: Starting input
        :param t: sampled timestep
        :param model_kwargs: additional model key word arguments
        :return: Tuple containing loss, reconstructed input, noise, model output and the model's target
        """
        raise NotImplementedError("Abstract method call")

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        ), noise

    def q_sample_inverse(self, x_t: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        """
        Computes the reconstructed x based on a given noised x and the noise as well as the corresponding timestep
        :param x_t: noised input
        :param noise: noise which was added onto the input
        :param t: timestep corresponding to the level of the noise
        :return: De-noised x
        """
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recip_m1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def _train_val_step(
        self, batch, metrics_dict: Optional[Dict[str, Callable]], logging_prefix: str
    ):
        # Get data sample
        x_0 = batch[self.data_key]

        # Determine any further required inputs from the data set
        x_0, model_kwargs = self.format_p_theta_model_input(x_0=x_0, batch=batch)

        # Randomly sample current time step
        batch_size, *_ = x_0.shape
        t, weights = self.schedule_sampler.sample(
            batch_size=batch_size, device=self.device
        )

        # Get model outputs and loss
        loss_dict = self.p_loss(x_0=x_0, t=t, model_kwargs=model_kwargs)
        losses = loss_dict[LOSS_DICT_LOSSES_KEY]
        reconstructed_x_0 = loss_dict[LOSS_DICT_RECONSTRUCTED_INPUT_KEY]
        model_output = loss_dict[LOSS_DICT_MODEL_OUTPUT_KEY]
        target = loss_dict[LOSS_DICT_TARGET_KEY]
        del loss_dict

        # Update sampler if necessary
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, losses.detach())

        # Calculate final loss based on timestep sampler weights
        loss = (losses * weights).mean()

        # Log trainings loss
        self.log(
            logging_prefix + TRAINING_LOSS_METRIC_KEY,
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        # Log any additional metrics
        if metrics_dict:
            for metric_name, metric_function in metrics_dict.items():
                self._log_metric(
                    logging_prefix=logging_prefix,
                    metric_name=metric_name,
                    metric_function=metric_function,
                    batch_size=batch_size,
                    model_output=model_output,
                    target=target,
                )
                self._log_metric(
                    logging_prefix=logging_prefix + "reconstructed_x0_",
                    metric_name=metric_name,
                    metric_function=metric_function,
                    batch_size=batch_size,
                    model_output=reconstructed_x_0,
                    target=x_0,
                )
        return loss

    def _log_metric(
        self,
        logging_prefix: str,
        metric_name: str,
        metric_function: Callable,
        batch_size: int,
        model_output: torch.Tensor,
        target: torch.Tensor,
    ):
        if hasattr(metric_function, "device") and hasattr(metric_function, "to"):
            metric_function = metric_function.to(target.device)
        logging_kwargs = {
            "on_step": True,
            "on_epoch": True,
            "batch_size": batch_size,
        }
        value = metric_function(model_output, target)
        if isinstance(value, Dict):
            self.log_dict(
                dictionary={logging_prefix + k: v for k, v in value.items()},
                **logging_kwargs,
            )
        else:
            self.log(name=logging_prefix + metric_name, value=value, **logging_kwargs)

    # Pytorch Lightning Methods for training
    def training_step(self, batch) -> STEP_OUTPUT:
        # Apply Model
        return self._train_val_step(
            batch=batch,
            metrics_dict=self.training_metrics,
            logging_prefix=LOGGING_TRAIN_PREFIX,
        )

    @torch.no_grad()
    def validation_step(
        self, batch, *args: Any, **kwargs: Any
    ) -> Optional[STEP_OUTPUT]:
        # Apply Model
        self._train_val_step(
            batch=batch,
            metrics_dict=self.validation_metrics,
            logging_prefix=LOGGING_VAL_PREFIX,
        )
        return None

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

    # Helper methods for managing ema model
    # Copied from https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/ddpm.py
    def on_train_batch_end(self, *args, **kwargs):
        # Update EMA model after every batch
        if self.use_ema:
            self.ema_model(self.p_theta_model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.ema_model.store(self.p_theta_model.parameters())
            self.ema_model.copy_to(self.p_theta_model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.ema_model.restore(self.p_theta_model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    # VLB loss
    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape, f"Expected shape {x_start.shape} but got {x_t.shape}"
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

    # Sampling methods
    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool,
        denoised_fn: Optional[Callable] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        frozen_out: Optional[torch.Tensor] = None,
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
        :param frozen_out: Half-detached version of model output, still learn the variance using the variational bound, but don't let
            it affect our mean prediction.
        :return: a dict with the following keys:
                 - 'p_mean': the model mean output.
                 - 'p_variance': the model variance output.
                 - 'p_log_variance': the log of 'variance'.
                 - 'pred_x_0': the prediction for x_0.
        """
        raise NotImplementedError("Called abstract function")

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t,
        clip_denoised: bool = True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = (
            out[P_MEAN_VAR_DICT_MEAN_KEY]
            + nonzero_mask
            * torch.exp(0.5 * out[P_MEAN_VAR_DICT_LOG_VARIANCE_KEY])
            * noise
        )
        return {SAMPLE_KEY: sample, "pred_xstart": out[P_MEAN_VAR_DICT_PRED_X_0_KEY]}

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        batch_size: int = 1,
        use_ema_weights: bool = False,
        clip_denoised: bool = False,
        starting_noise: Optional[torch.Tensor] = None,
        safe_intermediaries_every_n_steps: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        denoise_fn: Optional[Callable] = None,
    ):
        result = []
        x_t = default(
            starting_noise,
            lambda: torch.randn((batch_size, *shape), device=self.device),
        )
        if not model_kwargs:
            model_kwargs = {}
        # Use EMA model weights if desired
        with self.ema_scope("P sample loop") if use_ema_weights else nullcontext():
            for t in tqdm(
                reversed(range(self.beta_schedule_steps)),
                desc=f"DDPM sampling on device {self.device}:",
                total=self.beta_schedule_steps,
            ):
                # If specified safe the intermediaries as well
                if safe_intermediaries_every_n_steps:
                    if t % safe_intermediaries_every_n_steps == 0:
                        result.append(x_t)
                # Sample x_t based off x_{t-1}
                x_t = self.p_sample(
                    x=x_t,
                    t=torch.full(
                        size=(batch_size,),
                        fill_value=t,
                        device=self.device,
                        dtype=torch.long,
                    ),
                    model_kwargs=model_kwargs,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoise_fn,
                )[SAMPLE_KEY]
            result.append(x_t)
            return result
