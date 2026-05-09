from pathlib import Path

from PIL import Image
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
from train.context_flow_matching import TrainConfig as ContextFlowMatchingTrainConfig
from train.context_flow_matching import apply_prompt_dropout
from train.context_flow_matching import build_optimizer
from train.context_flow_matching import load_config as load_context_flow_matching_config
from train.context_flow_matching import metadata_to_prompt
from train.context_flow_matching import optimizer_eval
from train.context_flow_matching import optimizer_train
from train.context_flow_matching import pil_grid
from train.context_flow_matching import sample_prompt_groups
from train.context_flow_matching import sample_wandb_key


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


def test_load_context_flow_matching_config_uses_nested_samples(tmp_path: Path):
    config_path = tmp_path / "context_flow_matching.yaml"
    config_path.write_text(
        "\n".join(
            [
                "image_dir: ./images",
                "tags_dir: ./tags",
                "label2id_path: ./label2id.json",
                "image_size: 32",
                "hidden_dim: 32",
                "num_heads: 4",
                "num_blocks: 2",
                "patch_size: 16",
                "max_context_len: 12",
                "optimizer: radam_schedule_free",
                "loss_type: v-loss",
                "samples:",
                "  every: 7",
                "  num_steps: 3",
                "  cfg_scale: 2.5",
                "  prompts:",
                "    - general 1girl",
                "  dataset_prompts: 2",
                "  num_images_per_prompt: 2",
                "  columns: 2",
                "gradient_checkpointing: true",
                "precision: bf16",
                "wandb_mode: offline",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_context_flow_matching_config(config_path, overrides={"batch_size": 4})

    assert cfg.image_dir == Path("images")
    assert cfg.tags_dir == Path("tags")
    assert cfg.label2id_path == Path("label2id.json")
    assert cfg.batch_size == 4
    assert cfg.optimizer == "radam_schedule_free"
    assert cfg.loss_type == "v-loss"
    assert cfg.samples.every == 7
    assert cfg.samples.num_steps == 3
    assert cfg.samples.cfg_scale == 2.5
    assert cfg.samples.prompts == ["general 1girl"]
    assert cfg.samples.dataset_prompts == 2
    assert cfg.samples.num_images_per_prompt == 2
    assert cfg.samples.columns == 2
    assert cfg.gradient_checkpointing is True
    assert cfg.precision == "bf16"
    assert cfg.wandb_mode == "offline"


def test_context_flow_matching_train_config_rejects_bad_sample_shape():
    with pytest.raises(ValidationError):
        ContextFlowMatchingTrainConfig(
            image_size=32,
            patch_size=16,
            samples={"width": 30},
        )


def test_context_flow_matching_train_config_rejects_unknown_optimizer():
    with pytest.raises(ValidationError):
        ContextFlowMatchingTrainConfig(optimizer="lion")


def test_context_build_optimizer_can_use_radam_schedule_free():
    cfg = ContextFlowMatchingTrainConfig(
        image_size=32,
        patch_size=16,
        optimizer="radam_schedule_free",
    )
    model = torch.nn.Linear(2, 1)

    optimizer = build_optimizer(model.parameters(), cfg)
    optimizer_train(optimizer)
    optimizer_eval(optimizer)

    assert optimizer.__class__.__name__ == "RAdamScheduleFree"


def test_metadata_to_prompt_filters_unknown_tags_in_stable_order():
    metadata = {
        "rating": "general",
        "character_tags": {
            "known_character": 0.9,
            "unknown_character": 0.8,
        },
        "general_tags": {
            "1girl": 0.99,
            "unknown_general": 0.7,
        },
    }
    label2id = {
        "general": 0,
        "known_character": 1,
        "1girl": 2,
    }

    prompt = metadata_to_prompt(
        metadata=metadata,
        label2id=label2id,
        shuffle_tags=False,
        filter_unknown_tags=True,
        include_rating=True,
        include_character_tags=True,
        include_general_tags=True,
        tag_separator=" ",
    )

    assert prompt == "general known_character 1girl"


def test_apply_prompt_dropout_can_force_unconditional_prompts():
    prompts = ["general 1girl", "sensitive solo"]

    kept, keep_fraction = apply_prompt_dropout(prompts=prompts, dropout_prob=0.0)
    dropped, drop_fraction = apply_prompt_dropout(prompts=prompts, dropout_prob=1.0)

    assert kept == prompts
    assert dropped == ["", ""]
    assert keep_fraction == 0.0
    assert drop_fraction == 1.0


def test_context_sample_prompt_groups_keep_prompt_batches():
    class DummyDataset:
        def __len__(self) -> int:
            return 2

        def prompt_for_index(self, index: int, shuffle_tags: bool = False) -> str:
            return f"dataset_{index}_{shuffle_tags}"

    cfg = ContextFlowMatchingTrainConfig(
        image_size=32,
        patch_size=16,
        samples={
            "prompts": ["general 1girl"],
            "dataset_prompts": 1,
            "num_images_per_prompt": 3,
        },
    )

    groups = sample_prompt_groups(cfg, DummyDataset())

    assert groups == [
        ("general 1girl", ["general 1girl"] * 3),
        ("dataset_0_False", ["dataset_0_False"] * 3),
    ]


def test_context_sample_prompt_groups_default_to_config_prompts_only():
    class DummyDataset:
        def __len__(self) -> int:
            return 2

        def prompt_for_index(self, index: int, shuffle_tags: bool = False) -> str:
            return f"dataset_{index}_{shuffle_tags}"

    cfg = ContextFlowMatchingTrainConfig(
        image_size=32,
        patch_size=16,
        samples={
            "prompts": ["first", "second"],
            "num_images_per_prompt": 2,
        },
    )

    groups = sample_prompt_groups(cfg, DummyDataset())

    assert cfg.samples.dataset_prompts == 0
    assert groups == [
        ("first", ["first", "first"]),
        ("second", ["second", "second"]),
    ]


def test_pil_grid_uses_auto_columns():
    images = [Image.new("RGB", (8, 8), (index * 20, 0, 0)) for index in range(5)]

    grid = pil_grid(images)

    assert grid.mode == "RGB"
    assert grid.size == (40, 28)


def test_sample_wandb_key_uses_prompt_index():
    assert sample_wandb_key(0) == "samples/grid_00"
    assert sample_wandb_key(12) == "samples/grid_12"
