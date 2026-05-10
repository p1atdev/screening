import argparse
import csv
import json
import math
import random
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, Self

from PIL import Image
import schedulefree
import torch
import torch.nn.functional as F
import wandb
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from screening.flow import image_pred_to_velocity_pred
from screening.models import MultiScreenForContextFlowMatching

Precision = Literal["fp32", "fp16", "bf16"]
WandbMode = Literal["online", "offline", "disabled"]
ImageMode = Literal["L", "RGB", "RGBA"]
LossType = Literal["x-loss", "v-loss"]
OptimizerName = Literal["adamw", "radam_schedule_free"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "context_flow_matching_animeface.yaml"
DEFAULT_IMAGE_DIR = Path(
    "~/WDC20/Documents/python/anime-face-dev/data/animeface-20s-1.8M"
)
DEFAULT_TAGS_DIR = Path("~/WDC20/Documents/python/anime-face-dev/data/tags")


class SampleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    at_start: bool = True
    every: int | None = Field(default=1000, gt=0)
    num_steps: int = Field(default=64, gt=0)
    cfg_scale: float = Field(default=3.0, ge=0.0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    max_context_len: int | None = Field(default=None, gt=0)
    prompts: list[str] = Field(default_factory=list)
    dataset_prompts: int = Field(default=0, ge=0)
    num_images_per_prompt: int = Field(default=1, gt=0)
    seed: int | None = 123
    columns: int | None = Field(default=None, gt=0)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    image_dir: Path = DEFAULT_IMAGE_DIR
    tags_dir: Path = DEFAULT_TAGS_DIR
    label2id_path: Path = PROJECT_ROOT / "configs" / "animeface-label2id.json"
    image_glob: str = "*.webp"
    image_mode: ImageMode = "RGB"
    image_size: int = Field(default=256, gt=0)
    max_samples: int | None = Field(default=None, gt=0)
    shuffle_tags: bool = True
    filter_unknown_tags: bool = True
    include_rating: bool = True
    include_character_tags: bool = True
    include_general_tags: bool = True
    tag_separator: str = Field(default=" ", min_length=1)
    max_context_len: int = Field(default=64, gt=0)

    hidden_dim: int = Field(default=64, gt=0)
    num_heads: int = Field(default=8, gt=0)
    num_blocks: int = Field(default=8, gt=0)
    num_repeats: int = Field(default=8, gt=0)
    bottleneck_dim: int = Field(default=16, gt=0)
    patch_size: int = Field(default=16, gt=0)
    window_threshold: float = Field(default=256.0, gt=0.0)

    batch_size: int = Field(default=16, gt=0)
    epochs: int = Field(default=1, gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    lr: float = Field(default=1e-3, gt=0.0)
    optimizer: OptimizerName = "adamw"
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    grad_clip_norm: float | None = Field(default=1.0, gt=0.0)
    min_timestep: float = Field(default=0.0, ge=0.0, lt=1.0)
    max_timestep: float = Field(default=1.0, gt=0.0, le=1.0)
    cfg_dropout_prob: float = Field(default=0.1, ge=0.0, le=1.0)
    loss_type: LossType = "x-loss"

    log_every: int = Field(default=20, gt=0)
    checkpoint_every: int | None = Field(default=1000, gt=0)
    samples: SampleConfig = Field(default_factory=SampleConfig)
    output_dir: Path = PROJECT_ROOT / "output" / "context_flow_matching_animeface"
    checkpoint_path: Path | None = None

    seed: int = 123
    device: str = Field(default="auto", min_length=1)
    num_workers: int = Field(default=4, ge=0)
    gradient_checkpointing: bool = True
    precision: Precision = "fp32"
    wandb_project: str = Field(default="screening-context-flow-matching", min_length=1)
    wandb_run_name: str | None = None
    wandb_mode: WandbMode = "online"

    @property
    def in_channels(self) -> int:
        return {"L": 1, "RGB": 3, "RGBA": 4}[self.image_mode]

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for timestep Fourier features")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        sample_width = (
            self.samples.width if self.samples.width is not None else self.image_size
        )
        sample_height = (
            self.samples.height if self.samples.height is not None else self.image_size
        )
        if sample_width % self.patch_size != 0 or sample_height % self.patch_size != 0:
            raise ValueError("sample width/height must be divisible by patch_size")
        if self.min_timestep >= self.max_timestep:
            raise ValueError("min_timestep must be smaller than max_timestep")
        for beta in self.betas:
            if not 0.0 <= beta < 1.0:
                raise ValueError("betas must be in [0.0, 1.0)")
        return self

    def to_log_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["in_channels"] = self.in_channels
        return data


class AnimeFaceRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_id: str
    image_path: Path
    tags_path: Path


class AnimeFaceContextDataset(Dataset[tuple[torch.Tensor, str, str]]):
    def __init__(
        self,
        rows: list[AnimeFaceRow],
        label2id: dict[str, int],
        image_mode: ImageMode,
        image_size: int,
        shuffle_tags: bool,
        filter_unknown_tags: bool,
        include_rating: bool,
        include_character_tags: bool,
        include_general_tags: bool,
        tag_separator: str,
    ):
        if not rows:
            raise ValueError("rows must not be empty")
        self.rows = rows
        self.label2id = label2id
        self.shuffle_tags = shuffle_tags
        self.filter_unknown_tags = filter_unknown_tags
        self.include_rating = include_rating
        self.include_character_tags = include_character_tags
        self.include_general_tags = include_general_tags
        self.tag_separator = tag_separator
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Lambda(lambda tensor: tensor * 2.0 - 1.0),
            ]
        )
        self.image_mode = image_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        row = self.rows[index]
        image = load_image_tensor(
            row.image_path,
            image_mode=self.image_mode,
            transform=self.transform,
        )
        metadata = load_json(row.tags_path)
        prompt = metadata_to_prompt(
            metadata=metadata,
            label2id=self.label2id,
            shuffle_tags=self.shuffle_tags,
            filter_unknown_tags=self.filter_unknown_tags,
            include_rating=self.include_rating,
            include_character_tags=self.include_character_tags,
            include_general_tags=self.include_general_tags,
            tag_separator=self.tag_separator,
        )
        return image, prompt, row.image_id

    def prompt_for_index(self, index: int, shuffle_tags: bool = False) -> str:
        row = self.rows[index]
        metadata = load_json(row.tags_path)
        return metadata_to_prompt(
            metadata=metadata,
            label2id=self.label2id,
            shuffle_tags=shuffle_tags,
            filter_unknown_tags=self.filter_unknown_tags,
            include_rating=self.include_rating,
            include_character_tags=self.include_character_tags,
            include_general_tags=self.include_general_tags,
            tag_separator=self.tag_separator,
        )


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return PROJECT_ROOT / expanded


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_label2id(path: Path) -> dict[str, int]:
    raw_label2id = load_json(path)
    if not isinstance(raw_label2id, dict):
        raise ValueError(f"{path} must contain a JSON object")
    label2id: dict[str, int] = {}
    for label, index in raw_label2id.items():
        if not isinstance(label, str) or not isinstance(index, int):
            raise ValueError(f"{path} must map string labels to integer IDs")
        label2id[label] = index
    return label2id


def load_config_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        return {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    if any(not isinstance(key, str) for key in raw_config):
        raise ValueError(f"{path} must use string keys")
    return dict(raw_config)


def load_config(path: Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    data = load_config_file(path)
    if overrides is not None:
        data.update(overrides)
    return TrainConfig.model_validate(data)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train MultiScreenForContextFlowMatching on AnimeFace tags."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.print_config:
        print(yaml.safe_dump(cfg.to_log_dict(), sort_keys=False))
        raise SystemExit(0)
    return cfg


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_precision_config(precision: Precision, device: torch.device) -> None:
    if precision == "fp32":
        return
    if not torch.amp.autocast_mode.is_autocast_available(device.type):
        raise RuntimeError(
            f"mixed precision is not supported for device type {device.type!r}"
        )
    if precision == "bf16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "bf16 mixed precision requires a CUDA device with bf16 support"
            )


def precision_dtype(precision: Precision) -> torch.dtype | None:
    if precision == "fp32":
        return None
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def precision_autocast(device: torch.device, precision: Precision) -> Any:
    dtype = precision_dtype(precision)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_grad_scaler(
    device: torch.device,
    precision: Precision,
) -> torch.amp.GradScaler | None:
    if precision == "fp16":
        return torch.amp.GradScaler(device.type)
    return None


def convert_image(image: Image.Image, image_mode: ImageMode) -> Image.Image:
    if image_mode == "RGB" and image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert(image_mode)


def load_image_tensor(
    path: Path,
    image_mode: ImageMode,
    transform: transforms.Compose,
) -> torch.Tensor:
    image = Image.open(path)
    image = convert_image(image, image_mode=image_mode)
    return transform(image)


def metadata_to_tags(
    metadata: dict[str, Any],
    label2id: dict[str, int],
    shuffle_tags: bool,
    filter_unknown_tags: bool,
    include_rating: bool,
    include_character_tags: bool,
    include_general_tags: bool,
) -> list[str]:
    tags: list[str] = []
    if include_rating:
        rating = metadata.get("rating", "general")
        if isinstance(rating, str):
            tags.append(rating)
    if include_character_tags:
        character_tags = metadata.get("character_tags", {})
        if isinstance(character_tags, dict):
            tags.extend(str(tag) for tag in character_tags.keys())
    if include_general_tags:
        general_tags = metadata.get("general_tags", {})
        if isinstance(general_tags, dict):
            tags.extend(str(tag) for tag in general_tags.keys())

    if filter_unknown_tags:
        tags = [tag for tag in tags if tag in label2id]
    if shuffle_tags:
        random.shuffle(tags)
    return tags


def metadata_to_prompt(
    metadata: dict[str, Any],
    label2id: dict[str, int],
    shuffle_tags: bool,
    filter_unknown_tags: bool,
    include_rating: bool,
    include_character_tags: bool,
    include_general_tags: bool,
    tag_separator: str,
) -> str:
    tags = metadata_to_tags(
        metadata=metadata,
        label2id=label2id,
        shuffle_tags=shuffle_tags,
        filter_unknown_tags=filter_unknown_tags,
        include_rating=include_rating,
        include_character_tags=include_character_tags,
        include_general_tags=include_general_tags,
    )
    return tag_separator.join(tags)


def build_rows(cfg: TrainConfig) -> list[AnimeFaceRow]:
    image_dir = resolve_path(cfg.image_dir)
    tags_dir = resolve_path(cfg.tags_dir)
    rows: list[AnimeFaceRow] = []
    for image_path in sorted(image_dir.glob(cfg.image_glob)):
        image_id = image_path.stem
        tags_path = tags_dir / f"{image_id}.tags.json"
        if not tags_path.exists():
            continue
        rows.append(
            AnimeFaceRow(
                image_id=image_id,
                image_path=image_path,
                tags_path=tags_path,
            )
        )
        if cfg.max_samples is not None and len(rows) >= cfg.max_samples:
            break

    if not rows:
        raise FileNotFoundError(
            f"No image/tag pairs found in image_dir={image_dir} tags_dir={tags_dir}"
        )
    return rows


def build_dataset(
    cfg: TrainConfig, label2id: dict[str, int]
) -> AnimeFaceContextDataset:
    return AnimeFaceContextDataset(
        rows=build_rows(cfg),
        label2id=label2id,
        image_mode=cfg.image_mode,
        image_size=cfg.image_size,
        shuffle_tags=cfg.shuffle_tags,
        filter_unknown_tags=cfg.filter_unknown_tags,
        include_rating=cfg.include_rating,
        include_character_tags=cfg.include_character_tags,
        include_general_tags=cfg.include_general_tags,
        tag_separator=cfg.tag_separator,
    )


def collate_batch(rows: list[tuple[torch.Tensor, str, str]]) -> dict[str, Any]:
    images, prompts, image_ids = zip(*rows, strict=True)
    return {
        "images": torch.stack(list(images), dim=0),
        "prompts": list(prompts),
        "image_ids": list(image_ids),
    }


def move_images(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    return batch["images"].to(device, non_blocking=True)


def sample_timesteps(
    batch_size: int,
    mean: float = -0.8,
    std: float = 0.8,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    z = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(z)


def make_noisy_images(
    images: torch.Tensor,
    min_timestep: float,
    max_timestep: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = images.size(0)
    timestep = (
        sample_timesteps(batch_size, device=images.device)
        * (max_timestep - min_timestep)
        + min_timestep
    )
    noise = torch.randn_like(images)
    noisy_images = (
        timestep[:, None, None, None] * images
        + (1.0 - timestep[:, None, None, None]) * noise
    )
    return noisy_images, noise, timestep


def apply_prompt_dropout(
    prompts: list[str],
    dropout_prob: float,
) -> tuple[list[str], float]:
    if dropout_prob <= 0.0:
        return prompts, 0.0

    dropped_prompts = []
    dropped_count = 0
    for prompt in prompts:
        if random.random() < dropout_prob:
            dropped_prompts.append("")
            dropped_count += 1
        else:
            dropped_prompts.append(prompt)
    return dropped_prompts, dropped_count / max(len(prompts), 1)


def flow_matching_loss(
    pred_images: torch.Tensor,
    target_images: torch.Tensor,
    noisy_images: torch.Tensor,
    timestep: torch.Tensor,
    loss_type: LossType,
) -> torch.Tensor:
    if loss_type == "x-loss":
        return F.mse_loss(pred_images, target_images)

    pred_velocity = image_pred_to_velocity_pred(
        pred_image=pred_images,
        noisy_image=noisy_images,
        timestep=timestep,
    )
    target_velocity = image_pred_to_velocity_pred(
        pred_image=target_images,
        noisy_image=noisy_images,
        timestep=timestep,
    )
    return F.mse_loss(pred_velocity, target_velocity)


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    cfg: TrainConfig,
) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer == "radam_schedule_free":
        return schedulefree.RAdamScheduleFree(
            parameters,
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"unsupported optimizer: {cfg.optimizer}")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def format_parameter_count(count: int) -> str:
    return f"{count:,}"


def optimizer_train(optimizer: torch.optim.Optimizer) -> None:
    train_fn = getattr(optimizer, "train", None)
    if callable(train_fn):
        train_fn()


def optimizer_eval(optimizer: torch.optim.Optimizer) -> None:
    eval_fn = getattr(optimizer, "eval", None)
    if callable(eval_fn):
        eval_fn()


def set_train_mode(
    model: MultiScreenForContextFlowMatching,
    optimizer: torch.optim.Optimizer,
) -> None:
    model.train()
    optimizer_train(optimizer)


def set_eval_mode(
    model: MultiScreenForContextFlowMatching,
    optimizer: torch.optim.Optimizer,
) -> None:
    model.eval()
    optimizer_eval(optimizer)


def encode_context(
    model: MultiScreenForContextFlowMatching,
    prompts: list[str],
    images: torch.Tensor,
    max_context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    context, context_mask = model.context_encoder.encode_prompts(
        prompts,
        max_token_length=max_context_len,
    )
    batch_size, _channels, height, width = images.size()
    num_patches = height * width // model.patch_size**2
    device = images.device
    attention_mask = torch.cat(
        [
            torch.ones((batch_size, num_patches), device=device, dtype=torch.long),
            context_mask.to(device=device, dtype=torch.long),
            torch.ones(
                (batch_size, model.num_repeats), device=device, dtype=torch.long
            ),
        ],
        dim=1,
    )
    return context.to(device=device), attention_mask


def forward_loss(
    model: MultiScreenForContextFlowMatching,
    images: torch.Tensor,
    prompts: list[str],
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    noisy_images, noise, timestep = make_noisy_images(
        images=images,
        min_timestep=cfg.min_timestep,
        max_timestep=cfg.max_timestep,
    )
    prompts, cfg_dropout_fraction = apply_prompt_dropout(
        prompts=prompts,
        dropout_prob=cfg.cfg_dropout_prob,
    )
    context, attention_mask = encode_context(
        model=model,
        prompts=prompts,
        images=noisy_images,
        max_context_len=cfg.max_context_len,
    )

    with precision_autocast(device, cfg.precision):
        pred_images = model(
            images=noisy_images,
            context=context,
            timestep=timestep,
            attention_mask=attention_mask,
        )
        loss = flow_matching_loss(
            pred_images=pred_images,
            target_images=images,
            noisy_images=noisy_images,
            timestep=timestep,
            loss_type=cfg.loss_type,
        )

    with torch.no_grad():
        noise_mse = F.mse_loss(noisy_images, images)
        pred_mse = F.mse_loss(pred_images.float(), images.float())
        velocity_mse = flow_matching_loss(
            pred_images=pred_images.float(),
            target_images=images.float(),
            noisy_images=noisy_images.float(),
            timestep=timestep.float(),
            loss_type="v-loss",
        )
        noise_norm = noise.float().pow(2).mean().sqrt()
        prompt_lengths = [
            len(prompt.split(cfg.tag_separator)) if prompt else 0 for prompt in prompts
        ]
        avg_prompt_len = sum(prompt_lengths) / max(len(prompt_lengths), 1)
    metrics = {
        "train/noisy_mse": float(noise_mse.item()),
        "train/pred_mse": float(pred_mse.item()),
        "train/velocity_mse": float(velocity_mse.item()),
        "train/timestep_mean": float(timestep.float().mean().item()),
        "train/noise_rms": float(noise_norm.item()),
        "train/cfg_dropout_fraction": cfg_dropout_fraction,
        "train/prompt_length": avg_prompt_len,
    }
    return loss, metrics


def save_checkpoint(
    path: Path,
    model: MultiScreenForContextFlowMatching,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    step: int,
    epoch: int,
) -> None:
    set_eval_mode(model, optimizer)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.to_log_dict(),
            "step": step,
            "epoch": epoch,
        },
        path,
    )


def pil_grid(
    images: list[Image.Image],
    columns: int | None = None,
    padding: int = 4,
) -> Image.Image:
    if not images:
        raise ValueError("images must not be empty")
    nrow = columns if columns is not None else math.ceil(math.sqrt(len(images)))
    tensors = torch.stack(
        [transforms.ToTensor()(image.convert("RGB")) for image in images]
    )
    grid = make_grid(
        tensors,
        nrow=max(nrow, 1),
        padding=padding,
        pad_value=1.0,
    )
    return transforms.ToPILImage()(grid)


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def append_sample_manifest(
    manifest_path: Path,
    rows: list[dict[str, str | int]],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not manifest_path.exists()
    fieldnames = ["step", "kind", "index", "path", "prompt"]
    with manifest_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def sample_dimensions(cfg: TrainConfig) -> tuple[int, int]:
    width = cfg.samples.width if cfg.samples.width is not None else cfg.image_size
    height = cfg.samples.height if cfg.samples.height is not None else cfg.image_size
    return width, height


def sample_max_context_len(cfg: TrainConfig) -> int:
    if cfg.samples.max_context_len is not None:
        return cfg.samples.max_context_len
    return cfg.max_context_len


def sample_prompt_groups(
    cfg: TrainConfig,
    dataset: AnimeFaceContextDataset,
) -> list[tuple[str, list[str]]]:
    prompts = list(cfg.samples.prompts)
    if cfg.samples.dataset_prompts > 0:
        dataset_count = min(cfg.samples.dataset_prompts, len(dataset))
        prompts.extend(
            dataset.prompt_for_index(index, shuffle_tags=False)
            for index in range(dataset_count)
        )
    if not prompts:
        prompts = [""]

    return [
        (prompt, [prompt] * cfg.samples.num_images_per_prompt) for prompt in prompts
    ]


def format_paths(paths: list[Path]) -> str:
    return ", ".join(str(path) for path in paths)


def sample_wandb_key(prompt_index: int) -> str:
    return f"samples/grid_{prompt_index:02d}"


@torch.no_grad()
def save_training_samples(
    model: MultiScreenForContextFlowMatching,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    dataset: AnimeFaceContextDataset,
    step: int,
    device: torch.device,
) -> list[Path]:
    sample_dir = resolve_path(cfg.output_dir) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sample_dir / "manifest.csv"

    prompt_groups = sample_prompt_groups(cfg, dataset)
    prompts = [prompt for _, group_prompts in prompt_groups for prompt in group_prompts]
    width, height = sample_dimensions(cfg)
    if cfg.samples.seed is not None:
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(cfg.samples.seed + step)
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()
            torch.cuda.manual_seed_all(cfg.samples.seed + step)
        else:
            cuda_state = None
    else:
        generator_state = None
        cuda_state = None

    set_eval_mode(model, optimizer)
    generated_images = model.generate(
        prompts=prompts,
        width=width,
        height=height,
        num_steps=cfg.samples.num_steps,
        cfg_scale=cfg.samples.cfg_scale,
        device=device,
        max_context_len=sample_max_context_len(cfg),
        seed=cfg.samples.seed,
    )

    if generator_state is not None:
        torch.random.set_rng_state(generator_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    rows: list[dict[str, str | int]] = []
    sample_paths: list[Path] = []
    grid_paths: list[Path] = []
    wandb_logs: dict[str, wandb.Image] = {}
    offset = 0
    for prompt_index, (prompt, group_prompts) in enumerate(prompt_groups):
        group_images = generated_images[offset : offset + len(group_prompts)]
        for sample_index, image in enumerate(group_images):
            flat_index = offset + sample_index
            sample_path = (
                sample_dir / f"step_{step:08d}_prompt_{prompt_index:02d}"
                f"_sample_{sample_index:02d}.png"
            )
            image.save(sample_path)
            sample_paths.append(sample_path)
            rows.append(
                {
                    "step": step,
                    "kind": "sample",
                    "index": flat_index,
                    "path": relative_path(sample_path),
                    "prompt": prompt,
                }
            )

        grid = pil_grid(group_images, columns=cfg.samples.columns)
        grid_path = sample_dir / f"step_{step:08d}_prompt_{prompt_index:02d}_grid.png"
        grid.save(grid_path)
        grid_paths.append(grid_path)
        rows.append(
            {
                "step": step,
                "kind": "grid",
                "index": prompt_index,
                "path": relative_path(grid_path),
                "prompt": prompt,
            }
        )
        wandb_logs[sample_wandb_key(prompt_index)] = wandb.Image(
            str(grid_path),
            caption=prompt,
        )
        offset += len(group_prompts)

    append_sample_manifest(manifest_path, rows)

    wandb.log(wandb_logs, step=step)
    if wandb.run is not None:
        for grid_path in grid_paths:
            wandb.save(str(grid_path), policy="now")
        for sample_path in sample_paths:
            wandb.save(str(sample_path), policy="now")

    return grid_paths


def write_resolved_config(cfg: TrainConfig) -> None:
    output_dir = resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_log_dict(), f, sort_keys=False)


def train(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    validate_precision_config(cfg.precision, device)
    label2id = load_label2id(resolve_path(cfg.label2id_path))
    dataset = build_dataset(cfg, label2id=label2id)
    write_resolved_config(cfg)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )

    model = MultiScreenForContextFlowMatching(
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_blocks=cfg.num_blocks,
        label2id=label2id,
        num_repeats=cfg.num_repeats,
        bottleneck_dim=cfg.bottleneck_dim,
        in_channels=cfg.in_channels,
        patch_size=cfg.patch_size,
        window_threshold=cfg.window_threshold,
        label_splitter=cfg.tag_separator,
    ).to(device)
    model.set_gradient_checkpointing(cfg.gradient_checkpointing)
    trainable_params, total_params = count_parameters(model)

    optimizer = build_optimizer(model.parameters(), cfg)
    grad_scaler = make_grad_scaler(device=device, precision=cfg.precision)

    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        mode=cfg.wandb_mode,
        config=cfg.to_log_dict() | {"num_labels": len(label2id)},
    )
    wandb.watch(model, log="gradients", log_freq=max(cfg.log_every, 1))

    step = 0
    last_sample_step: int | None = None
    running_loss = 0.0
    running_metrics: dict[str, float] = {}
    running_count = 0

    print(
        " ".join(
            [
                f"device={device}",
                f"images={len(dataset)}",
                f"batches_per_epoch={len(train_loader)}",
                f"image_size={cfg.image_size}",
                f"in_channels={cfg.in_channels}",
                f"labels={len(label2id)}",
                f"trainable_params={format_parameter_count(trainable_params)}",
                f"total_params={format_parameter_count(total_params)}",
            ]
        )
    )

    if cfg.samples.at_start:
        grid_paths = save_training_samples(
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            dataset=dataset,
            step=step,
            device=device,
        )
        print(f"saved initial sample grids: {format_paths(grid_paths)}")
        last_sample_step = step

    for epoch in range(cfg.epochs):
        set_train_mode(model, optimizer)
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        for raw_batch in progress:
            images = move_images(raw_batch, device=device)
            prompts = raw_batch["prompts"]
            optimizer.zero_grad(set_to_none=True)
            loss, batch_metrics = forward_loss(
                model=model,
                images=images,
                prompts=prompts,
                cfg=cfg,
                device=device,
            )

            if grad_scaler is None:
                loss.backward()
                if cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip_norm
                    )
                optimizer.step()
            else:
                grad_scaler.scale(loss).backward()
                if cfg.grad_clip_norm is not None:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip_norm
                    )
                grad_scaler.step(optimizer)
                grad_scaler.update()

            step += 1
            loss_value = float(loss.item())
            running_loss += loss_value
            for key, value in batch_metrics.items():
                running_metrics[key] = running_metrics.get(key, 0.0) + value
            running_count += 1
            progress.set_postfix(loss=f"{loss_value:.4f}")

            if step % cfg.log_every == 0:
                denom = max(running_count, 1)
                logs = {
                    "train/loss": running_loss / denom,
                    "optimizer/lr": optimizer.param_groups[0].get(
                        "scheduled_lr", optimizer.param_groups[0]["lr"]
                    ),
                    "epoch": epoch + 1,
                }
                logs.update(
                    {key: value / denom for key, value in running_metrics.items()}
                )
                wandb.log(logs, step=step)
                print(
                    " ".join(
                        [
                            f"step={step}",
                            f"epoch={epoch + 1}",
                            f"loss={logs['train/loss']:.5f}",
                            f"lr={logs['optimizer/lr']:.2e}",
                        ]
                    )
                )
                running_loss = 0.0
                running_metrics = {}
                running_count = 0

            if cfg.samples.every is not None and step % cfg.samples.every == 0:
                grid_paths = save_training_samples(
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    dataset=dataset,
                    step=step,
                    device=device,
                )
                print(f"saved sample grids: {format_paths(grid_paths)}")
                last_sample_step = step
                set_train_mode(model, optimizer)

            if cfg.checkpoint_every is not None and step % cfg.checkpoint_every == 0:
                checkpoint_path = (
                    resolve_path(cfg.output_dir)
                    / "checkpoints"
                    / f"step_{step:08d}.ckpt"
                )
                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    step=step,
                    epoch=epoch,
                )
                set_train_mode(model, optimizer)

            if cfg.max_steps is not None and step >= cfg.max_steps:
                break

        if cfg.max_steps is not None and step >= cfg.max_steps:
            break

    if last_sample_step != step:
        final_grid_paths = save_training_samples(
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            dataset=dataset,
            step=step,
            device=device,
        )
        print(f"saved final sample grids: {format_paths(final_grid_paths)}")

    checkpoint_path = (
        resolve_path(cfg.checkpoint_path)
        if cfg.checkpoint_path is not None
        else resolve_path(cfg.output_dir) / "checkpoints" / "last.ckpt"
    )
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        step=step,
        epoch=max(epoch, 0),
    )
    print(f"saved checkpoint: {checkpoint_path}")
    run.finish()


if __name__ == "__main__":
    train(parse_args())
