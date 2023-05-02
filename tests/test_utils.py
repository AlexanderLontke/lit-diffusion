from torch import nn

from lit_diffusion.utils.adapter_p_theta_model import AdapterPThetaModel
from conftest import OTHER_IMG_KWARG_KEY


def test_adapter_model_stacking(x_t_batch, t_batch, batch_model_kwargs):
    p_theta_adapter = AdapterPThetaModel(
        original_p_theta_model=nn.Identity(),
        stack_inputs_keys=[OTHER_IMG_KWARG_KEY]
    )
    other_img = batch_model_kwargs[OTHER_IMG_KWARG_KEY]
    other_img_channels = other_img.shape[1]
    model_output = p_theta_adapter.forward(x_t=x_t_batch, t=t_batch, **batch_model_kwargs)
    x_b, x_c, x_h, x_w = x_t_batch.shape
    m_b, m_c, m_h, m_w = model_output.shape
    assert (m_b, m_h, m_w) == (x_b, x_h, x_w)  # All dimensions should match except for the channel dim
    # Make sure the stacked channels are the last in the index order
    assert (model_output[:, -other_img_channels:, ::] == other_img).all()

