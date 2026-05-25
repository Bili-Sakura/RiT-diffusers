import pytest

torch = pytest.importorskip("torch")

from diffusers.models.transformers.transformer_rit import RiTTransformer2DModel
from diffusers.schedulers.scheduling_flow_match_rit import RiTFlowMatchScheduler


def test_rit_transformer_forward():
    model = RiTTransformer2DModel(
        input_size=4,
        patch_size=1,
        in_channels=8,
        hidden_size=32,
        depth=2,
        num_heads=4,
        in_context_len=0,
        use_cls=True,
    )
    latents = torch.randn(2, 8, 4, 4)
    timesteps = torch.tensor([0.5, 0.25])
    class_labels = torch.tensor([1, 2])
    cls_token = torch.randn(2, 8)

    output = model(latents, timesteps, class_labels, cls_token=cls_token)

    assert output.sample.shape == latents.shape
    assert output.cls_sample.shape == (2, 8)


def test_scheduler_x_prediction_step():
    scheduler = RiTFlowMatchScheduler(pred_type="x", sample_eps=1e-5)
    sample = torch.ones(1, 4, 2, 2)
    model_output = torch.full_like(sample, 2.0)
    output = scheduler.step(
        model_output,
        torch.tensor([0.25]),
        sample,
        next_timestep=torch.tensor([0.5]),
        return_dict=True,
    )
    expected_velocity = (model_output - sample) / (1.0 - 0.25)
    expected = sample + (0.5 - 0.25) * expected_velocity
    assert output.prev_sample.shape == sample.shape
    assert torch.allclose(output.prev_sample, expected)


def test_scheduler_coupled_noise_config():
    scheduler = RiTFlowMatchScheduler(coupled_noise=True, latent_channels=384)
    assert scheduler._cfg("coupled_noise") is True
    assert scheduler.time_dist_shift > 1.0
