import torch

from screening.flow import image_pred_to_velocity_pred
from screening.models import (
    MultiScreenForClassFlowMatching,
    MultiScreenForContextFlowMatching,
)


def test_image_pred_to_velocity_pred_matches_clean_to_noise_velocity():
    clean_image = torch.tensor([[[[2.0]]]])
    noise_image = torch.tensor([[[[-1.0]]]])
    timestep = torch.tensor([0.25])
    noisy_image = (
        timestep[:, None, None, None] * clean_image
        + (1.0 - timestep[:, None, None, None]) * noise_image
    )

    velocity = image_pred_to_velocity_pred(
        pred_image=clean_image,
        noisy_image=noisy_image,
        timestep=timestep,
    )

    torch.testing.assert_close(velocity, clean_image - noise_image)


def test_init_class_flow_matching_model():
    image_size = 32
    patch_size = 16
    h_patch = image_size // patch_size
    w_patch = image_size // patch_size

    num_repeats = 8

    model = MultiScreenForClassFlowMatching(
        hidden_dim=64,
        num_heads=4,
        num_blocks=4,
        num_classes=10,
        num_repeats=num_repeats,
        bottleneck_dim=16,
        in_channels=3,
        patch_size=patch_size,
    )

    assert isinstance(model, MultiScreenForClassFlowMatching)

    images = torch.randn(2, 3, image_size, image_size)
    label_ids = torch.randint(0, 10, (2,))
    timestep = torch.rand(2)
    attention_mask = torch.ones(
        1, num_repeats + num_repeats + h_patch * w_patch
    )  # [1, num_patches]

    output = model(
        images=images,
        label_ids=label_ids,
        timestep=timestep,
        attention_mask=attention_mask,
    )

    assert output.shape == (2, 3, 32, 32)


def test_generate_class_flow_matching_model():
    image_size = 32
    patch_size = 16
    num_repeats = 8

    model = MultiScreenForClassFlowMatching(
        hidden_dim=64,
        num_heads=4,
        num_blocks=4,
        num_classes=10,
        num_repeats=num_repeats,
        bottleneck_dim=16,
        in_channels=3,
        patch_size=patch_size,
    )

    assert isinstance(model, MultiScreenForClassFlowMatching)

    label_ids = torch.randint(0, 10, (2,))

    images = model.generate(
        label_ids=label_ids,
        width=image_size,
        height=image_size,
        num_steps=20,
        cfg_scale=3.0,
    )

    assert isinstance(images, list)
    assert len(images) == 2


def test_generate_context_flow_matching_model():
    image_size = 32
    patch_size = 16
    num_repeats = 8

    model = MultiScreenForContextFlowMatching(
        hidden_dim=64,
        num_heads=4,
        num_blocks=4,
        label2id={
            "cat": 0,
            "dog": 1,
            "car": 2,
            "tree": 3,
            "house": 4,
            "person": 5,
            "bicycle": 6,
            "flower": 7,
            "sky": 8,
            "water": 9,
        },
        num_repeats=num_repeats,
        bottleneck_dim=16,
        in_channels=3,
        patch_size=patch_size,
    )

    assert isinstance(model, MultiScreenForContextFlowMatching)

    images = model.generate(
        prompts=["cat dog", "car tree"],
        width=image_size,
        height=image_size,
        num_steps=20,
        cfg_scale=3.0,
    )

    assert isinstance(images, list)
    assert len(images) == 2


def test_generate_context_flow_matching_model_with_negative_prompts():
    image_size = 32
    patch_size = 16
    num_repeats = 8

    model = MultiScreenForContextFlowMatching(
        hidden_dim=64,
        num_heads=4,
        num_blocks=4,
        label2id={
            "cat": 0,
            "dog": 1,
            "car": 2,
            "tree": 3,
            "house": 4,
            "person": 5,
            "bicycle": 6,
            "flower": 7,
            "sky": 8,
            "water": 9,
        },
        num_repeats=num_repeats,
        bottleneck_dim=16,
        in_channels=3,
        patch_size=patch_size,
    )

    assert isinstance(model, MultiScreenForContextFlowMatching)

    images = model.generate(
        prompts=["cat dog", "car tree"],
        negative_prompts=["flower", "person sky"],
        width=image_size,
        height=image_size,
        num_steps=20,
        cfg_scale=3.0,
    )

    assert isinstance(images, list)
    assert len(images) == 2
