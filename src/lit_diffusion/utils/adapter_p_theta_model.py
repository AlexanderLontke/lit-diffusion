from typing import List, Optional

import torch
from torch import nn


class AdapterPThetaModel(nn.Module):
    def __init__(
        self,
        original_p_theta_model: nn.Module,
        stack_inputs_keys: Optional[List[str]] = None,
        p_theta_model_output_index: int = 0,
        p_theta_model_call_timestep_key: Optional[str] = None,
        output_mask_key: Optional[str] = None,
    ):
        super().__init__()
        # Store p_theta_model
        self._p_theta_model = original_p_theta_model
        self.stack_inputs_keys = stack_inputs_keys
        self.p_theta_model_call_timestep_key = p_theta_model_call_timestep_key
        self.p_theta_model_output_index = p_theta_model_output_index
        self.output_mask_key = output_mask_key

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, *args, **kwargs):
        # If an argument for the timestep exists in the model forward call include it in the
        # key word arguments
        if self.p_theta_model_call_timestep_key:
            kwargs[self.p_theta_model_call_timestep_key] = t

        # stack data from kwargs onto x if it is desired
        if self.stack_inputs_keys:
            for k in self.stack_inputs_keys:
                x_t = torch.cat([x_t, kwargs.pop(k)], dim=1)

        # If an output mask key was given remove it from model input
        if self.output_mask_key:
            output_mask = kwargs.pop(self.output_mask_key)

        # Call the p_theta model's forward method with all necessary arguments and return the result
        model_output = self._p_theta_model(x_t, *args, **kwargs)

        # Make sure only x_{t-1} is returned
        model_output = model_output[
            self.p_theta_model_output_index
        ] if isinstance(model_output, tuple) else model_output

        # Mask output if mask is available
        if self.output_mask_key:
            model_output = output_mask * model_output

        return model_output
