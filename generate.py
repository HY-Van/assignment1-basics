from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.generation import generate
from cs336_basics.model import TransformerLM
from cs336_basics.optim import AdamW
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer LM.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)
    parser.add_argument("--special-tokens", nargs="*", default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--eos-token", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, args.special_tokens)
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
    optimizer = AdamW(model.parameters(), lr=1e-3)
    load_checkpoint(args.checkpoint_path, model, optimizer)
    model.eval()

    prompt_ids = torch.tensor(tokenizer.encode(args.prompt), device=device, dtype=torch.long)
    eos_token_id = None
    if args.eos_token is not None:
        eos_ids = tokenizer.encode(args.eos_token)
        if len(eos_ids) != 1:
            raise ValueError("eos_token must encode to exactly one token.")
        eos_token_id = eos_ids[0]

    output_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
        context_length=args.context_length,
    )
    print(tokenizer.decode(output_ids.squeeze(0).tolist()))


if __name__ == "__main__":
    main()
