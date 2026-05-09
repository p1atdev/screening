from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from train.abcd_digits import TrainConfig as ABCDigitsTrainConfig
from train.abcd_digits import load_config as load_abcd_digits_config
from train.mnist import TrainConfig as MNISTTrainConfig
from train.mnist import load_config as load_mnist_config
from train.flow_matching import TrainConfig as FlowMatchingTrainConfig
from train.flow_matching import apply_cfg_dropout
from train.flow_matching import flow_matching_loss
from train.flow_matching import load_config as load_flow_matching_config


def test_load_abcd_digits_config_uses_yaml_and_overrides(tmp_path: Path):
    config_path = tmp_path / "abcd_digits.yaml"
    config_path.write_text(
        "\n".join(
            [
                "train_samples: 8",
                "val_samples: 4",
                "n_lines: 50",
                "hidden_dim: 16",
                "num_heads: 4",
                "betas: [0.8, 0.9]",
                "gradient_checkpointing: true",
                "precision: bf16",
                "wandb_mode: disabled",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_abcd_digits_config(config_path, overrides={"train_samples": 16})

    assert cfg.train_samples == 16
    assert cfg.val_samples == 4
    assert cfg.betas == (0.8, 0.9)
    assert cfg.gradient_checkpointing is True
    assert cfg.precision == "bf16"
    assert cfg.wandb_mode == "disabled"


def test_abcd_digits_train_config_rejects_incompatible_head_shape():
    with pytest.raises(ValidationError):
        ABCDigitsTrainConfig(hidden_dim=10, num_heads=4)


def test_abcd_digits_train_config_rejects_unknown_precision():
    with pytest.raises(ValidationError):
        ABCDigitsTrainConfig(precision="tf32")


def test_load_mnist_config_uses_yaml_and_overrides(tmp_path: Path):
    config_path = tmp_path / "mnist.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data_dir: ./mnist-data",
                "download: false",
                "hidden_dim: 32",
                "num_heads: 4",
                "num_blocks: 2",
                "patch_size: 7",
                "betas: [0.8, 0.9]",
                "gradient_checkpointing: true",
                "precision: bf16",
                "wandb_project: screening-mnist-test",
                "wandb_run_name: smoke",
                "wandb_mode: offline",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_mnist_config(config_path, overrides={"batch_size": 16})

    assert cfg.data_dir == Path("mnist-data")
    assert cfg.download is False
    assert cfg.batch_size == 16
    assert cfg.hidden_dim == 32
    assert cfg.betas == (0.8, 0.9)
    assert cfg.gradient_checkpointing is True
    assert cfg.precision == "bf16"
    assert cfg.wandb_project == "screening-mnist-test"
    assert cfg.wandb_run_name == "smoke"
    assert cfg.wandb_mode == "offline"


def test_mnist_train_config_rejects_incompatible_head_shape():
    with pytest.raises(ValidationError):
        MNISTTrainConfig(hidden_dim=10, num_heads=4)


def test_mnist_train_config_rejects_unknown_precision():
    with pytest.raises(ValidationError):
        MNISTTrainConfig(precision="tf32")


def test_mnist_train_config_rejects_unknown_wandb_mode():
    with pytest.raises(ValidationError):
        MNISTTrainConfig(wandb_mode="dry-run")


def test_load_flow_matching_config_uses_yaml_and_overrides(tmp_path: Path):
    config_path = tmp_path / "flow_matching.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data_dir: ./one-image",
                "image_path: ./one-image/single.webp",
                "image_mode: RGB",
                "image_size: 32",
                "label_id: 0",
                "num_classes: 1",
                "hidden_dim: 32",
                "num_heads: 4",
                "num_blocks: 2",
                "patch_size: 16",
                "betas: [0.8, 0.9]",
                "cfg_dropout_prob: 0.25",
                "loss_type: v-loss",
                "gradient_checkpointing: true",
                "precision: bf16",
                "wandb_project: screening-flow-test",
                "wandb_run_name: smoke",
                "wandb_mode: offline",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_flow_matching_config(config_path, overrides={"batch_size": 4})

    assert cfg.data_dir == Path("one-image")
    assert cfg.image_path == Path("one-image/single.webp")
    assert cfg.batch_size == 4
    assert cfg.hidden_dim == 32
    assert cfg.betas == (0.8, 0.9)
    assert cfg.cfg_dropout_prob == 0.25
    assert cfg.loss_type == "v-loss"
    assert cfg.gradient_checkpointing is True
    assert cfg.precision == "bf16"
    assert cfg.wandb_project == "screening-flow-test"
    assert cfg.wandb_run_name == "smoke"
    assert cfg.wandb_mode == "offline"
    assert cfg.in_channels == 3


def test_flow_matching_train_config_rejects_incompatible_head_shape():
    with pytest.raises(ValidationError):
        FlowMatchingTrainConfig(hidden_dim=10, num_heads=4)


def test_flow_matching_train_config_rejects_odd_hidden_dim():
    with pytest.raises(ValidationError):
        FlowMatchingTrainConfig(hidden_dim=15, num_heads=5)


def test_flow_matching_train_config_rejects_bad_image_shape():
    with pytest.raises(ValidationError):
        FlowMatchingTrainConfig(image_size=30, patch_size=16)


def test_flow_matching_train_config_rejects_bad_cfg_dropout_prob():
    with pytest.raises(ValidationError):
        FlowMatchingTrainConfig(cfg_dropout_prob=1.1)


def test_flow_matching_train_config_rejects_unknown_loss_type():
    with pytest.raises(ValidationError):
        FlowMatchingTrainConfig(loss_type="eps-loss")


def test_apply_cfg_dropout_can_force_unconditional_labels():
    labels = torch.tensor([0, 1, 2])

    kept, keep_fraction = apply_cfg_dropout(
        label_ids=labels,
        uncond_id=9,
        dropout_prob=0.0,
    )
    dropped, drop_fraction = apply_cfg_dropout(
        label_ids=labels,
        uncond_id=9,
        dropout_prob=1.0,
    )

    torch.testing.assert_close(kept, labels)
    torch.testing.assert_close(dropped, torch.full_like(labels, 9))
    assert keep_fraction == 0.0
    assert drop_fraction == 1.0


def test_flow_matching_v_loss_uses_noisy_image_and_target_image_velocity():
    target_images = torch.tensor([[[[2.0]]]])
    noise_images = torch.tensor([[[[-1.0]]]])
    pred_images = torch.tensor([[[[1.25]]]])
    timestep = torch.tensor([0.25])
    noisy_images = (
        timestep[:, None, None, None] * target_images
        + (1.0 - timestep[:, None, None, None]) * noise_images
    )

    loss = flow_matching_loss(
        pred_images=pred_images,
        target_images=target_images,
        noisy_images=noisy_images,
        timestep=timestep,
        loss_type="v-loss",
    )

    expected_pred_velocity = (pred_images - noisy_images) / (1.0 - timestep)[
        :, None, None, None
    ]
    expected_target_velocity = (target_images - noisy_images) / (1.0 - timestep)[
        :, None, None, None
    ]
    expected_loss = torch.nn.functional.mse_loss(
        expected_pred_velocity,
        expected_target_velocity,
    )
    torch.testing.assert_close(loss, expected_loss)
