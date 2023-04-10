from typing import Optional

import torch
from torch import nn


class AdapterPThetaModel(nn.Module):
    def __init__(
        self,
        original_p_theta_model: nn.Module,
        p_theta_model_output_index: int = 0,
        p_theta_model_call_timestep_key: Optional[str] = None,
        output_mask_key: Optional[str] = None,
    ):
        super().__init__()
        # Store p_theta_model
        self._p_theta_model = original_p_theta_model
        self.p_theta_model_call_timestep_key = p_theta_model_call_timestep_key
        self.p_theta_model_output_index = p_theta_model_output_index
        self.output_mask_key = output_mask_key

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, *args, **kwargs):
        # If an argument for the timestep exists in the model forward call include it in the
        # key word arguments
        if self.p_theta_model_call_timestep_key:
            kwargs[self.p_theta_model_call_timestep_key] = t

        # Call the p_theta model's forward method with all necessary arguments and return the result
        model_output = list(self._p_theta_model(x_t, *args, **kwargs))[
            self.p_theta_model_output_index
        ]

        # Mask output if mask is available
        if self.output_mask_key:
            output_mask = kwargs.pop(self.output_mask_key)
            model_output = output_mask * model_output
        return model_output
