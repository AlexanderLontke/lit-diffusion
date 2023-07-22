from typing import List, Optional, Union

import torch
from torch import nn


class ContextAdapterPThetaModel(nn.Module):
    def __init__(
        self,
        original_p_theta_model: nn.Module,
        p_theta_model_call_context_key: str,
        context_keys: Optional[List[Union[str, int]]] = None,
        context_embedder: Optional[nn.Module] = None,
    ):
        super().__init__()
        # Store p_theta_model
        self._p_theta_model = original_p_theta_model

        # Context
        self.context_embedder = context_embedder
        self.p_theta_model_call_context_key = p_theta_model_call_context_key
        self.context_keys = context_keys

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, *args, **kwargs):
        # print("x_t", x_t.shape, "t", t.shape, "args", args, "kwargs", kwargs.keys())
        context = []
        for context_key in self.context_keys:
            context.append(kwargs.pop(context_key))
        context = torch.concat(context, dim=1)

        # Run it through the embedder if available
        if self.context_embedder is not None:
            context = self.context_embedder(context)

        # Add context to p_theta model call
        assert (
            self.p_theta_model_call_context_key not in kwargs.keys()
        ), f"{self.p_theta_model_call_context_key} already exists in kwargs of model call"
        kwargs[self.p_theta_model_call_context_key] = context

        # Call the p_theta model's forward method with all necessary arguments and return the result
        model_output = self._p_theta_model(x_t, t, *args, **kwargs)

        return model_output
