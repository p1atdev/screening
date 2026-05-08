import argparse
import math
import random
from pathlib import Path
from typing import Any, Literal, Self, cast

import torch
import torch.nn.functional as F
import wandb
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from screening import ABCDTokenizer, MultiScreen, generate_by_line_count

WandbMode = Literal["online", "offline", "disabled"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "abcd_digits.yaml"


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    train_samples: int = Field(default=2000, gt=0)
    val_samples: int = Field(default=200, gt=0)
    n_lines: int = Field(default=64, ge=50)
    depth: float = Field(default=0.5, ge=0.0, le=1.0)
    n_digits: int = Field(default=6, ge=2)
    hidden_dim: int = Field(default=64, gt=0)
    num_heads: int = Field(default=4, gt=0)
    num_blocks: int = Field(default=4, gt=0)
    window_threshold: float = Field(default=256.0, gt=0.0)
    batch_size: int = Field(default=16, gt=0)
    eval_batch_size: int = Field(default=32, gt=0)
    epochs: int = Field(default=5, gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    lr: float = Field(default=1e-3, gt=0.0)
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(default=0.0, ge=0.0)
    log_every: int = Field(default=20, gt=0)
    eval_every: int = Field(default=200, gt=0)
    seed: int = 123
    device: str = Field(default="auto", min_length=1)
    num_workers: int = Field(default=0, ge=0)
    wandb_project: str = Field(default="screening-abcdigits", min_length=1)
    wandb_run_name: str | None = None
    wandb_mode: WandbMode = "online"
    checkpoint_path: str | None = None

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


class ABCDigitsCompletionDataset(Dataset[dict[str, str]]):
    def __init__(self, samples: list[str], n_digits: int):
        self.rows = []
        for text in samples:
            prompt = text[:-n_digits]
            answer = text[-n_digits:]
            self.rows.append(
                {
                    "text": text,
                    "prompt": prompt,
                    "answer": answer,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


class ABCDigitsCollator:
    def __init__(self, tokenizer: ABCDTokenizer, n_digits: int):
        self.tokenizer = tokenizer
        self.n_digits = n_digits

    def __call__(self, rows: list[dict[str, str]]) -> dict[str, Any]:
        full_texts = [row["text"] for row in rows]
        prompt_texts = [row["prompt"] for row in rows]
        answer_texts = [row["answer"] for row in rows]

        encoded = self.tokenizer.encode_batch(full_texts, add_special_tokens=False)
        input_ids = encoded.input_ids[:, :-1]
        labels = encoded.input_ids[:, 1:]
        attention_mask = encoded.attention_mask[:, :-1]

        loss_mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, row in enumerate(rows):
            prompt_len = len(row["prompt"])
            full_len = len(row["text"])
            loss_mask[i, prompt_len - 1 : full_len - 1] = True

        prompt_ids = [
            self.tokenizer.encode(text, add_special_tokens=False).input_ids
            for text in prompt_texts
        ]
        answer_ids = torch.stack(
            [
                self.tokenizer.encode(text, add_special_tokens=False).input_ids
                for text in answer_texts
            ]
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids,
        }


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
        description="Train MultiScreen on the ABCDigits synthetic retrieval task."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--val-samples", type=int)
    parser.add_argument("--n-lines", type=int)
    parser.add_argument("--depth", type=float)
    parser.add_argument("--n-digits", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--num-blocks", type=int)
    parser.add_argument("--window-threshold", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-run-name", type=str)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--checkpoint-path", type=str)

    args = parser.parse_args()
    cfg = load_config(args.config)
    override_fields = [
        "train_samples",
        "val_samples",
        "n_lines",
        "depth",
        "n_digits",
        "hidden_dim",
        "num_heads",
        "num_blocks",
        "window_threshold",
        "batch_size",
        "eval_batch_size",
        "epochs",
        "max_steps",
        "lr",
        "weight_decay",
        "log_every",
        "eval_every",
        "seed",
        "device",
        "num_workers",
        "wandb_project",
        "wandb_run_name",
        "checkpoint_path",
    ]
    overrides = {
        field: getattr(args, field)
        for field in override_fields
        if getattr(args, field) is not None
    }
    if args.beta1 is not None or args.beta2 is not None:
        overrides["betas"] = (
            cfg.betas[0] if args.beta1 is None else args.beta1,
            cfg.betas[1] if args.beta2 is None else args.beta2,
        )
    if args.wandb_mode is not None:
        overrides["wandb_mode"] = cast(WandbMode, args.wandb_mode)

    cfg = load_config(args.config, overrides=overrides)
    if args.print_config:
        print(yaml.safe_dump(cfg.to_log_dict(), sort_keys=False))
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


def build_datasets(cfg: TrainConfig) -> tuple[ABCDigitsCompletionDataset, ...]:
    samples = generate_by_line_count(
        n_lines=cfg.n_lines,
        depth=cfg.depth,
        n_digits=cfg.n_digits,
        n_trials=cfg.train_samples + cfg.val_samples,
    )
    train_samples = samples[: cfg.train_samples]
    val_samples = samples[cfg.train_samples :]
    return (
        ABCDigitsCompletionDataset(train_samples, n_digits=cfg.n_digits),
        ABCDigitsCompletionDataset(val_samples, n_digits=cfg.n_digits),
    )


def move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    moved = dict(batch)
    for key in ["input_ids", "labels", "attention_mask", "loss_mask", "answer_ids"]:
        moved[key] = batch[key].to(device)
    return moved


def make_position_ids(input_ids: torch.Tensor) -> torch.Tensor:
    seq_len = input_ids.size(1)
    return (
        torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )


def answer_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    denom = loss_mask.sum().clamp_min(1)
    return (losses * loss_mask).sum() / denom


def forward_loss(
    model: MultiScreen,
    batch: dict[str, Any],
) -> torch.Tensor:
    input_ids = batch["input_ids"]
    logits = model(
        input_ids=input_ids,
        position_ids=make_position_ids(input_ids),
        attention_mask=batch["attention_mask"],
    )
    return answer_loss(
        logits=logits,
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
    )


def pad_token_lists(
    token_lists: list[torch.Tensor],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(tokens.numel() for tokens in token_lists)
    input_ids = torch.full(
        (len(token_lists), max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(token_lists), max_len),
        dtype=torch.long,
        device=device,
    )
    for i, tokens in enumerate(token_lists):
        tokens = tokens.to(device)
        input_ids[i, : tokens.numel()] = tokens
        attention_mask[i, : tokens.numel()] = 1
    return input_ids, attention_mask


@torch.no_grad()
def greedy_answer_ids(
    model: MultiScreen,
    prompt_ids: list[torch.Tensor],
    n_digits: int,
    pad_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    generated = [tokens.clone() for tokens in prompt_ids]
    answer_chunks: list[torch.Tensor] = []

    for _ in range(n_digits):
        input_ids, attention_mask = pad_token_lists(
            generated,
            pad_token_id=pad_token_id,
            device=device,
        )
        logits = model(
            input_ids=input_ids,
            position_ids=make_position_ids(input_ids),
            attention_mask=attention_mask,
        )
        last_positions = attention_mask.sum(dim=1) - 1
        next_logits = logits[
            torch.arange(logits.size(0), device=device), last_positions
        ]
        next_ids = next_logits.argmax(dim=-1)
        answer_chunks.append(next_ids.cpu())
        for i, token_id in enumerate(next_ids.cpu()):
            generated[i] = torch.cat([generated[i], token_id.reshape(1)])

    return torch.stack(answer_chunks, dim=1)


@torch.no_grad()
def evaluate(
    model: MultiScreen,
    loader: DataLoader[dict[str, Any]],
    tokenizer: ABCDTokenizer,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_loss_tokens = 0
    exact = 0
    token_correct = 0
    token_total = 0
    total_examples = 0

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        input_ids = batch["input_ids"]
        logits = model(
            input_ids=input_ids,
            position_ids=make_position_ids(input_ids),
            attention_mask=batch["attention_mask"],
        )
        loss = answer_loss(logits, batch["labels"], batch["loss_mask"])
        loss_tokens = int(batch["loss_mask"].sum().item())
        total_loss += float(loss.item()) * loss_tokens
        total_loss_tokens += loss_tokens

        pred_ids = greedy_answer_ids(
            model=model,
            prompt_ids=raw_batch["prompt_ids"],
            n_digits=cfg.n_digits,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        )
        target_ids = raw_batch["answer_ids"]
        matches = pred_ids.eq(target_ids)
        exact += int(matches.all(dim=1).sum().item())
        token_correct += int(matches.sum().item())
        token_total += int(matches.numel())
        total_examples += int(target_ids.size(0))

    return {
        "val/loss": total_loss / max(total_loss_tokens, 1),
        "val/exact_match_accuracy": exact / max(total_examples, 1),
        "val/token_accuracy": token_correct / max(token_total, 1),
    }


def save_checkpoint(
    path: str,
    model: MultiScreen,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    step: int,
    epoch: int,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.to_log_dict(),
            "step": step,
            "epoch": epoch,
        },
        checkpoint_path,
    )


def train(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    tokenizer = ABCDTokenizer()
    train_dataset, val_dataset = build_datasets(cfg)
    collator = ABCDigitsCollator(tokenizer=tokenizer, n_digits=cfg.n_digits)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collator,
    )

    model = MultiScreen(
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_blocks=cfg.num_blocks,
        vocab_size=tokenizer.vocab_size,
        window_threshold=cfg.window_threshold,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        mode=cfg.wandb_mode,
        config=cfg.to_log_dict() | {"vocab_size": tokenizer.vocab_size},
    )
    wandb.watch(model, log="gradients", log_freq=max(cfg.log_every, 1))

    step = 0
    running_loss = 0.0
    running_count = 0
    best_exact = -math.inf

    for epoch in range(cfg.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        for raw_batch in progress:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = forward_loss(model, batch)
            loss.backward()
            optimizer.step()

            step += 1
            running_loss += float(loss.item())
            running_count += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if step % cfg.log_every == 0:
                avg_loss = running_loss / max(running_count, 1)
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "optimizer/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch + 1,
                    },
                    step=step,
                )
                running_loss = 0.0
                running_count = 0

            should_eval = step % cfg.eval_every == 0
            is_last_step = cfg.max_steps is not None and step >= cfg.max_steps
            if should_eval or is_last_step:
                metrics = evaluate(
                    model=model,
                    loader=val_loader,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    device=device,
                )
                wandb.log(metrics | {"epoch": epoch + 1}, step=step)
                model.train()

                if metrics["val/exact_match_accuracy"] > best_exact:
                    best_exact = metrics["val/exact_match_accuracy"]
                    if cfg.checkpoint_path is not None:
                        save_checkpoint(
                            path=cfg.checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            cfg=cfg,
                            step=step,
                            epoch=epoch,
                        )

            if is_last_step:
                run.finish()
                return

    metrics = evaluate(
        model=model,
        loader=val_loader,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
    )
    wandb.log(metrics | {"epoch": cfg.epochs}, step=step)
    if cfg.checkpoint_path is not None:
        save_checkpoint(
            path=cfg.checkpoint_path,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            step=step,
            epoch=cfg.epochs - 1,
        )
    run.finish()


if __name__ == "__main__":
    train(parse_args())
