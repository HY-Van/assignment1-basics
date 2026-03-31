from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb

from cs336_basics.model import TransformerLM
from cs336_basics.optim import AdamW
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import (
    clip_gradients,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    load_checkpoint,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Transformer LM.")
    parser.add_argument("--train-data-path", type=Path, required=True)
    parser.add_argument("--valid-data-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)
    parser.add_argument("--special-tokens", nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--total-iters", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--min-lr", type=float, required=True)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoint.pt"))
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    return parser.parse_args()


def _load_token_ids(path: Path, tokenizer: Tokenizer | None = None) -> torch.Tensor:
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path)).to(torch.long)
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu").to(torch.long)
    if tokenizer is None:
        raise ValueError("A tokenizer is required to encode text datasets.")
    text = path.read_text(encoding="utf-8")
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _estimate_loss(
    model: TransformerLM,
    dataset: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
    eval_batches: int,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(eval_batches):
            x, y = get_batch(dataset, batch_size, context_length, device)
            logits = model(x)
            loss = cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
            losses.append(float(loss.item()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def _maybe_init_wandb(args: argparse.Namespace) -> bool:
    if args.wandb_project is None:
        return False
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    return True


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, args.special_tokens)
    train_data = _load_token_ids(args.train_data_path, tokenizer)
    valid_data = _load_token_ids(args.valid_data_path, tokenizer)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=dtype,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        _optimizer_to_device(optimizer, device)

    use_wandb = _maybe_init_wandb(args)
    model.train()

    for iteration in range(start_iter, args.total_iters):
        lr = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.total_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        loss.backward()
        clip_gradients(model.parameters(), args.grad_clip)

        grad_sq_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_sq_norm += float(torch.sum(param.grad.detach().float() ** 2).item())
        grad_norm = grad_sq_norm**0.5

        optimizer.step()

        if iteration % args.log_every == 0:
            print(
                f"iter={iteration} train_loss={loss.item():.6f} "
                f"lr={lr:.8f} grad_norm={grad_norm:.6f}"
            )
            if use_wandb:
                wandb.log(
                    {
                        "iter": iteration,
                        "train/loss": float(loss.item()),
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                    },
                    step=iteration,
                )

        if iteration % args.eval_every == 0:
            valid_loss = _estimate_loss(
                model=model,
                dataset=valid_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
                eval_batches=args.eval_batches,
            )
            print(f"iter={iteration} valid_loss={valid_loss:.6f}")
            if use_wandb:
                wandb.log({"iter": iteration, "valid/loss": valid_loss}, step=iteration)

        if (iteration + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, iteration + 1, args.checkpoint_path)

    save_checkpoint(model, optimizer, args.total_iters, args.checkpoint_path)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
