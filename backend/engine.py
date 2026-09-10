"""
DiffusionEngine — wraps LLaDA's iterative unmasking generation loop so that
it *yields* the full current token sequence after every refinement step,
instead of only returning the final tensor.

Adapted from reference implementation:
    https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch
import torch.nn.functional as F

LLADA_MASK_ID = 126336


@dataclass
class TokenState:
    id: int
    text: str
    masked: bool


@dataclass
class DiffusionState:
    step: int
    total_steps: int
    tokens: List[TokenState]


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


def _decode_single_token(tokenizer, token_id: int, mask_id: int) -> str:
    """Decode a single token id for display purposes."""
    if token_id == mask_id:
        return "[MASK]"
    return tokenizer.decode([token_id], skip_special_tokens=False)


class DiffusionEngine:
    """Thin wrapper around a loaded LLaDA model/tokenizer that exposes every
    intermediate diffusion state instead of only the final output tensor.
    """

    def __init__(
        self,
        model,
        tokenizer,
        mask_id: int = LLADA_MASK_ID,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_id = mask_id
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        steps: int = 16,
        gen_length: int = 32,
        block_length: int = 32,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
    ) -> Iterator[DiffusionState]:
        if gen_length % block_length != 0:
            raise ValueError("gen_length must be a multiple of block_length")

        num_blocks = gen_length // block_length
        if steps % num_blocks != 0:
            raise ValueError("steps must be a multiple of (gen_length // block_length)")
        steps_per_block = steps // num_blocks

        messages = [{"role": "user", "content": prompt}]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_ids = self.tokenizer(
            chat_prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.device)

        prompt_len = input_ids.shape[1]

        x = torch.full(
            (1, prompt_len + gen_length), self.mask_id, dtype=torch.long, device=self.device
        )
        x[:, :prompt_len] = input_ids

        global_step = 0

        for num_block in range(num_blocks):
            block_start = prompt_len + num_block * block_length
            block_end = prompt_len + (num_block + 1) * block_length

            block_mask_index = x[:, block_start:block_end] == self.mask_id
            num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == self.mask_id

                logits = self.model(x).logits

                logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # (1, L)

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(f"Unknown remasking strategy: {remasking}")

                x0_p[:, block_end:] = -float("inf")

                x0 = torch.where(mask_index, x0, x)
                neg_inf = torch.tensor(-float("inf"), device=x.device, dtype=x0_p.dtype)
                confidence = torch.where(mask_index, x0_p, neg_inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    k = int(num_transfer_tokens[j, i])
                    if k > 0:
                        _, select_index = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

                global_step += 1

                gen_ids = x[0, prompt_len:].tolist()
                tokens = [
                    TokenState(
                        id=token_id,
                        text=_decode_single_token(self.tokenizer, token_id, self.mask_id),
                        masked=(token_id == self.mask_id),
                    )
                    for token_id in gen_ids
                ]

                yield DiffusionState(step=global_step, total_steps=steps, tokens=tokens)