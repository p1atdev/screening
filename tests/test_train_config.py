from pathlib import Path

import pytest
from pydantic import ValidationError

from train.abcd_digits import TrainConfig as ABCDigitsTrainConfig
from train.abcd_digits import load_config as load_abcd_digits_config
from train.mnist import TrainConfig as MNISTTrainConfig
from train.mnist import load_config as load_mnist_config


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
