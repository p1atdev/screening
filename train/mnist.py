import argparse
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, Self

import torch
import torch.nn.functional as F
import wandb
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from screening import MultiScreenForImageClassification

Precision = Literal["fp32", "fp16", "bf16"]
WandbMode = Literal["online", "offline", "disabled"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "mnist.yaml"


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_dir: Path = DEFAULT_DATA_DIR
    download: bool = True
    hidden_dim: int = Field(default=64, gt=0)
    num_heads: int = Field(default=4, gt=0)
    num_blocks: int = Field(default=4, gt=0)
    patch_size: int = Field(default=7, gt=0)
    window_threshold: float = Field(default=256.0, gt=0.0)
    batch_size: int = Field(default=128, gt=0)
    eval_batch_size: int = Field(default=256, gt=0)
    epochs: int = Field(default=5, gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    max_train_batches: int | None = Field(default=None, gt=0)
    max_val_batches: int | None = Field(default=None, gt=0)
    train_subset: int | None = Field(default=None, gt=0)
    val_subset: int | None = Field(default=None, gt=0)
    lr: float = Field(default=1e-3, gt=0.0)
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    log_every: int = Field(default=50, gt=0)
    seed: int = 123
    device: str = Field(default="auto", min_length=1)
    num_workers: int = Field(default=2, ge=0)
    gradient_checkpointing: bool = False
    precision: Precision = "fp32"
    wandb_project: str = Field(default="screening-mnist", min_length=1)
    wandb_run_name: str | None = None
    wandb_mode: WandbMode = "online"
    checkpoint_path: Path | None = None

    @model_validator(mode="after")
    def validate_model_shape(self) -> Self:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        for beta in self.betas:
            if not 0.0 <= beta < 1.0:
                raise ValueError("betas must be in [0.0, 1.0)")
        return self

    def to_log_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


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
        description="Train MultiScreenForImageClassification on MNIST."
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


def maybe_subset(
    dataset: Dataset[Any],
    size: int | None,
    seed: int,
) -> Dataset[Any]:
    if size is None or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader[Any], DataLoader[Any]]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root=cfg.data_dir,
        train=True,
        transform=transform,
        download=cfg.download,
    )
    val_dataset = datasets.MNIST(
        root=cfg.data_dir,
        train=False,
        transform=transform,
        download=cfg.download,
    )
    train_dataset = maybe_subset(train_dataset, cfg.train_subset, cfg.seed)
    val_dataset = maybe_subset(val_dataset, cfg.val_subset, cfg.seed + 1)

    pin_memory = get_device(cfg.device).type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def make_patch_position_ids(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch_size, _, height, width = images.size()
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"patch_size={patch_size} is larger than image size {height}x{width}"
        )

    grid_height = (height - patch_size) // patch_size + 1
    grid_width = (width - patch_size) // patch_size + 1
    rows = torch.arange(grid_height, device=images.device)
    cols = torch.arange(grid_width, device=images.device)
    row_ids, col_ids = torch.meshgrid(rows, cols, indexing="ij")
    position_ids = torch.stack([row_ids, col_ids], dim=-1).view(-1, 2)
    return position_ids.unsqueeze(0).expand(batch_size, -1, -1)


def image_logits(
    model: MultiScreenForImageClassification,
    images: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    patch_logits = model(
        image_feature=images,
        position_ids=make_patch_position_ids(images, patch_size),
    )
    return patch_logits.mean(dim=1)


def move_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float(preds.eq(labels).float().mean().item())


@torch.no_grad()
def evaluate(
    model: MultiScreenForImageClassification,
    loader: DataLoader[Any],
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_index, raw_batch in enumerate(loader):
        if cfg.max_val_batches is not None and batch_index >= cfg.max_val_batches:
            break

        images, labels = move_batch(raw_batch, device)
        with precision_autocast(device, cfg.precision):
            logits = image_logits(model, images, cfg.patch_size)
            loss = F.cross_entropy(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int(logits.argmax(dim=-1).eq(labels).sum().item())
        total_examples += batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def save_checkpoint(
    path: Path,
    model: MultiScreenForImageClassification,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    step: int,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.to_log_dict(),
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def train(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    validate_precision_config(cfg.precision, device)
    train_loader, val_loader = build_loaders(cfg)

    model = MultiScreenForImageClassification(
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_blocks=cfg.num_blocks,
        num_classes=10,
        in_channels=1,
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
    running_loss = 0.0
    running_acc = 0.0
    running_count = 0
    best_accuracy = -1.0

    print(f"device={device} train_batches={len(train_loader)} val_batches={len(val_loader)}")
    for epoch in range(cfg.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        for batch_index, raw_batch in enumerate(progress):
            if (
                cfg.max_train_batches is not None
                and batch_index >= cfg.max_train_batches
            ):
                break

            images, labels = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with precision_autocast(device, cfg.precision):
                logits = image_logits(model, images, cfg.patch_size)
                loss = F.cross_entropy(logits, labels)

            if grad_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

            step += 1
            batch_acc = accuracy(logits.detach(), labels)
            running_loss += float(loss.item())
            running_acc += batch_acc
            running_count += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.3f}")

            if step % cfg.log_every == 0:
                print(
                    " ".join(
                        [
                            f"step={step}",
                            f"epoch={epoch + 1}",
                            f"loss={running_loss / max(running_count, 1):.4f}",
                            f"acc={running_acc / max(running_count, 1):.4f}",
                            f"lr={optimizer.param_groups[0]['lr']:.2e}",
                        ]
                    )
                )
                wandb.log(
                    {
                        "train/loss": running_loss / max(running_count, 1),
                        "train/accuracy": running_acc / max(running_count, 1),
                        "optimizer/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch + 1,
                    },
                    step=step,
                )
                running_loss = 0.0
                running_acc = 0.0
                running_count = 0

            if cfg.max_steps is not None and step >= cfg.max_steps:
                break

        metrics = evaluate(model=model, loader=val_loader, cfg=cfg, device=device)
        print(
            f"epoch={epoch + 1} val_loss={metrics['loss']:.4f} "
            f"val_acc={metrics['accuracy']:.4f}"
        )
        wandb.log(
            {
                "val/loss": metrics["loss"],
                "val/accuracy": metrics["accuracy"],
                "epoch": epoch + 1,
            },
            step=step,
        )
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            wandb.log({"val/best_accuracy": best_accuracy}, step=step)
            if cfg.checkpoint_path is not None:
                save_checkpoint(
                    path=cfg.checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    step=step,
                    epoch=epoch,
                    metrics=metrics,
                )

        if cfg.max_steps is not None and step >= cfg.max_steps:
            break

    run.finish()


if __name__ == "__main__":
    train(parse_args())
