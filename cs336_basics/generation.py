from __future__ import annotations

import torch
from torch import Tensor, nn

from cs336_basics.model import softmax


def sample_next_token(logits: Tensor, temperature: float = 1.0, top_p: float = 1.0) -> Tensor:
    next_logits = logits[..., -1, :]
    if temperature == 0:
        return torch.argmax(next_logits, dim=-1)
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1].")

    probs = softmax(next_logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative_probs <= top_p
        keep[..., 0] = True
        first_over = cumulative_probs >= top_p
        keep = keep | (first_over & (torch.cumsum(first_over.to(torch.int64), dim=-1) == 1))
        sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(dim=-1, index=sampled_sorted).squeeze(-1)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int | None = None,
    context_length: int | None = None,
) -> Tensor:
    generated = prompt_ids.clone()
    if generated.ndim == 1:
        generated = generated.unsqueeze(0)

    for _ in range(max_new_tokens):
        model_input = generated
        if context_length is not None and model_input.shape[-1] > context_length:
            model_input = model_input[..., -context_length:]

        logits = model(model_input)
        next_token = sample_next_token(logits, temperature=temperature, top_p=top_p).unsqueeze(-1)
        generated = torch.cat((generated, next_token), dim=-1)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

    return generated
