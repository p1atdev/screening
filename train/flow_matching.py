import argparse
import csv
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, Self

from PIL import Image
import torch
import torch.nn.functional as F
import wandb
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from screening.image import tensor_to_pil
from screening.models import MultiScreenForClassFlowMatching

Precision = Literal["fp32", "fp16", "bf16"]
WandbMode = Literal["online", "offline", "disabled"]
ImageMode = Literal["L", "RGB", "RGBA"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "flow_matching_one_image.yaml"


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_dir: Path = PROJECT_ROOT / "data" / "one-image"
    image_path: Path | None = None
    image_glob: str = "*.webp"
    image_mode: ImageMode = "RGB"
    image_size: int = Field(default=128, gt=0)
    label_id: int = Field(default=0, ge=0)
    num_classes: int = Field(default=1, gt=0)
    repeats_per_epoch: int = Field(default=1024, gt=0)

    hidden_dim: int = Field(default=64, gt=0)
    num_heads: int = Field(default=4, gt=0)
    num_blocks: int = Field(default=4, gt=0)
    num_repeats: int = Field(default=8, gt=0)
    bottleneck_dim: int = Field(default=16, gt=0)
    patch_size: int = Field(default=16, gt=0)
    window_threshold: float = Field(default=256.0, gt=0.0)

    batch_size: int = Field(default=16, gt=0)
    epochs: int = Field(default=20, gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    lr: float = Field(default=1e-3, gt=0.0)
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    grad_clip_norm: float | None = Field(default=None, gt=0.0)
    min_timestep: float = Field(default=0.0, ge=0.0, lt=1.0)
    max_timestep: float = Field(default=1.0, gt=0.0, le=1.0)
    cfg_dropout_prob: float = Field(default=0.1, ge=0.0, le=1.0)

    log_every: int = Field(default=20, gt=0)
    sample_every: int = Field(default=100, gt=0)
    checkpoint_every: int | None = Field(default=None, gt=0)
    sample_at_start: bool = True
    sample_num_images: int = Field(default=4, gt=0)
    sample_num_steps: int = Field(default=32, gt=0)
    sample_cfg_scale: float = Field(default=1.0, ge=0.0)
    output_dir: Path = PROJECT_ROOT / "output" / "flow_matching_one_image"
    checkpoint_path: Path | None = None

    seed: int = 123
    device: str = Field(default="auto", min_length=1)
    num_workers: int = Field(default=0, ge=0)
    gradient_checkpointing: bool = False
    precision: Precision = "fp32"
    wandb_project: str = Field(default="screening-flow-matching", min_length=1)
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
        if self.label_id >= self.num_classes:
            raise ValueError("label_id must be smaller than num_classes")
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


class ImageClassDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        rows: list[tuple[Path, int]],
        image_mode: ImageMode,
        image_size: int,
        repeats_per_epoch: int,
    ):
        if not rows:
            raise ValueError("rows must not be empty")
        self.rows = rows
        self.repeats_per_epoch = repeats_per_epoch
        self.image_tensors = [
            load_image_tensor(path, image_mode=image_mode, image_size=image_size)
            for path, _label_id in rows
        ]
        self.labels = [
            torch.tensor(label_id, dtype=torch.long) for _path, label_id in rows
        ]

    def __len__(self) -> int:
        return self.repeats_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row_index = index % len(self.rows)
        return self.image_tensors[row_index], self.labels[row_index]


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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
        description="Train MultiScreenForClassFlowMatching on image/class pairs."
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
    path: Path, image_mode: ImageMode, image_size: int
) -> torch.Tensor:
    image = Image.open(path)
    image = convert_image(image, image_mode=image_mode)
    transform = transforms.Compose(
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
    return transform(image)


def build_dataset(cfg: TrainConfig) -> ImageClassDataset:
    if cfg.image_path is not None:
        image_paths = [resolve_project_path(cfg.image_path)]
    else:
        data_dir = resolve_project_path(cfg.data_dir)
        image_paths = sorted(data_dir.glob(cfg.image_glob))

    if not image_paths:
        raise FileNotFoundError(
            f"No images found for image_path={cfg.image_path} "
            f"data_dir={cfg.data_dir} image_glob={cfg.image_glob!r}"
        )

    rows = [(path, cfg.label_id) for path in image_paths]
    return ImageClassDataset(
        rows=rows,
        image_mode=cfg.image_mode,
        image_size=cfg.image_size,
        repeats_per_epoch=cfg.repeats_per_epoch,
    )


def move_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def sample_timesteps(
    batch_size: int,
    mean: float = -0.8,
    std: float = 0.8,
    device: torch.device = torch.device("cpu"),
):
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


def apply_cfg_dropout(
    label_ids: torch.Tensor,
    uncond_id: int,
    dropout_prob: float,
) -> tuple[torch.Tensor, float]:
    if dropout_prob <= 0.0:
        return label_ids, 0.0

    drop_mask = torch.rand(label_ids.shape, device=label_ids.device) < dropout_prob
    dropped_label_ids = torch.where(
        drop_mask,
        torch.full_like(label_ids, fill_value=uncond_id),
        label_ids,
    )
    return dropped_label_ids, float(drop_mask.float().mean().item())


def forward_loss(
    model: MultiScreenForClassFlowMatching,
    images: torch.Tensor,
    label_ids: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    noisy_images, noise, timestep = make_noisy_images(
        images=images,
        min_timestep=cfg.min_timestep,
        max_timestep=cfg.max_timestep,
    )
    label_ids, cfg_dropout_fraction = apply_cfg_dropout(
        label_ids=label_ids,
        uncond_id=model.label_embedding.uncond_id,
        dropout_prob=cfg.cfg_dropout_prob,
    )
    with precision_autocast(device, cfg.precision):
        pred_images = model(
            images=noisy_images,
            label_ids=label_ids,
            timestep=timestep,
        )
        loss = F.mse_loss(pred_images, images)

    with torch.no_grad():
        noise_mse = F.mse_loss(noisy_images, images)
        pred_mse = F.mse_loss(pred_images.float(), images.float())
        noise_norm = noise.float().pow(2).mean().sqrt()
    metrics = {
        "train/noisy_mse": float(noise_mse.item()),
        "train/pred_mse": float(pred_mse.item()),
        "train/timestep_mean": float(timestep.float().mean().item()),
        "train/noise_rms": float(noise_norm.item()),
        "train/cfg_dropout_fraction": cfg_dropout_fraction,
    }
    return loss, metrics


def save_checkpoint(
    path: Path,
    model: MultiScreenForClassFlowMatching,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    step: int,
    epoch: int,
) -> None:
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


def pil_grid(images: list[Image.Image], columns: int, padding: int = 4) -> Image.Image:
    if not images:
        raise ValueError("images must not be empty")
    images = [image.convert("RGB") for image in images]
    width, height = images[0].size
    rows = math.ceil(len(images) / columns)
    grid = Image.new(
        "RGB",
        (
            columns * width + (columns + 1) * padding,
            rows * height + (rows + 1) * padding,
        ),
        color=(255, 255, 255),
    )
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        x = padding + col * (width + padding)
        y = padding + row * (height + padding)
        grid.paste(image, (x, y))
    return grid


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
    fieldnames = ["step", "kind", "index", "path"]
    with manifest_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def save_training_samples(
    model: MultiScreenForClassFlowMatching,
    cfg: TrainConfig,
    dataset: ImageClassDataset,
    step: int,
    device: torch.device,
) -> Path:
    sample_dir = resolve_project_path(cfg.output_dir) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sample_dir / "manifest.csv"

    target_tensor = dataset.image_tensors[0]
    target_image = tensor_to_pil(target_tensor)[0]
    target_path = sample_dir / "target.png"
    if not target_path.exists():
        target_image.save(target_path)

    label_ids = torch.full(
        (cfg.sample_num_images,),
        fill_value=cfg.label_id,
        dtype=torch.long,
        device=device,
    )
    model.eval()
    generated_images = model.generate(
        label_ids=label_ids,
        width=cfg.image_size,
        height=cfg.image_size,
        num_steps=cfg.sample_num_steps,
        cfg_scale=cfg.sample_cfg_scale,
        device=device,
    )

    rows: list[dict[str, str | int]] = []
    sample_paths: list[Path] = []
    for index, image in enumerate(generated_images):
        sample_path = sample_dir / f"step_{step:08d}_sample_{index:02d}.png"
        image.save(sample_path)
        sample_paths.append(sample_path)
        rows.append(
            {
                "step": step,
                "kind": "sample",
                "index": index,
                "path": relative_path(sample_path),
            }
        )

    grid = pil_grid(
        [target_image, *generated_images], columns=cfg.sample_num_images + 1
    )
    grid_path = sample_dir / f"step_{step:08d}_grid.png"
    grid.save(grid_path)
    rows.append(
        {
            "step": step,
            "kind": "grid",
            "index": -1,
            "path": relative_path(grid_path),
        }
    )
    append_sample_manifest(manifest_path, rows)

    wandb.log(
        {
            "samples/grid": wandb.Image(str(grid_path)),
        },
        step=step,
    )
    if wandb.run is not None:
        wandb.save(str(grid_path), policy="now")
        for sample_path in sample_paths:
            wandb.save(str(sample_path), policy="now")

    return grid_path


def write_resolved_config(cfg: TrainConfig) -> None:
    output_dir = resolve_project_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_log_dict(), f, sort_keys=False)


def train(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    validate_precision_config(cfg.precision, device)
    dataset = build_dataset(cfg)
    write_resolved_config(cfg)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    model = MultiScreenForClassFlowMatching(
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_blocks=cfg.num_blocks,
        num_classes=cfg.num_classes,
        num_repeats=cfg.num_repeats,
        bottleneck_dim=cfg.bottleneck_dim,
        in_channels=cfg.in_channels,
        patch_size=cfg.patch_size,
        window_threshold=cfg.window_threshold,
    ).to(device)
    model.set_gradient_checkpointing(cfg.gradient_checkpointing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    grad_scaler = make_grad_scaler(device=device, precision=cfg.precision)

    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        mode=cfg.wandb_mode,
        config=cfg.to_log_dict(),
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
                f"images={len(dataset.rows)}",
                f"batches_per_epoch={len(train_loader)}",
                f"image_size={cfg.image_size}",
                f"in_channels={cfg.in_channels}",
            ]
        )
    )

    if cfg.sample_at_start:
        grid_path = save_training_samples(
            model=model,
            cfg=cfg,
            dataset=dataset,
            step=step,
            device=device,
        )
        print(f"saved initial samples: {grid_path}")
        last_sample_step = step

    for epoch in range(cfg.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        for raw_batch in progress:
            images, labels = move_batch(raw_batch, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, batch_metrics = forward_loss(
                model=model,
                images=images,
                label_ids=labels,
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
                    "optimizer/lr": optimizer.param_groups[0]["lr"],
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

            if step % cfg.sample_every == 0:
                grid_path = save_training_samples(
                    model=model,
                    cfg=cfg,
                    dataset=dataset,
                    step=step,
                    device=device,
                )
                print(f"saved samples: {grid_path}")
                last_sample_step = step
                model.train()

            if cfg.checkpoint_every is not None and step % cfg.checkpoint_every == 0:
                checkpoint_path = (
                    resolve_project_path(cfg.output_dir)
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

            if cfg.max_steps is not None and step >= cfg.max_steps:
                break

        if cfg.max_steps is not None and step >= cfg.max_steps:
            break

    if last_sample_step != step:
        final_grid_path = save_training_samples(
            model=model,
            cfg=cfg,
            dataset=dataset,
            step=step,
            device=device,
        )
        print(f"saved final samples: {final_grid_path}")

    checkpoint_path = (
        resolve_project_path(cfg.checkpoint_path)
        if cfg.checkpoint_path is not None
        else resolve_project_path(cfg.output_dir) / "checkpoints" / "last.ckpt"
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
