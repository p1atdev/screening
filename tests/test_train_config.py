from pathlib import Path

import pytest
from pydantic import ValidationError

from train.abcd_digits import TrainConfig, load_config


def test_load_config_uses_yaml_and_overrides(tmp_path: Path):
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
                "wandb_mode: disabled",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path, overrides={"train_samples": 16})

    assert cfg.train_samples == 16
    assert cfg.val_samples == 4
    assert cfg.betas == (0.8, 0.9)
    assert cfg.wandb_mode == "disabled"


def test_train_config_rejects_incompatible_head_shape():
    with pytest.raises(ValidationError):
        TrainConfig(hidden_dim=10, num_heads=4)
