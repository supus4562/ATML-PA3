"""
Example-Weighted Causal Cross-Entropy Loss for Task 2.
Implements the exact mathematical specification from ATML PA3 Section 2:
L = - (1 / |B|) * sum_{i in B} (1 / |R_i|) * sum_{t in R_i} log p(y_{i,t} | I_i, q_i, y_{i,<t})
Masks visual tokens, prompts, and padding, weighting each caption or short answer equally.
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F


def compute_example_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Compute example-weighted causal cross-entropy loss.
    Input:
      logits: [B, seq_len, vocab_size]
      labels: [B, seq_len] with non-supervised tokens set to ignore_index (-100)
    Output:
      mean_loss: Scalar tensor representing the batch loss
      example_losses: [B] tensor of per-example losses
      total_valid_tokens: int count of supervised response tokens in the batch
    """
    # Shift logits and labels for next-token causal prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    B, S, V = shift_logits.shape
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    )
    loss_matrix = loss_flat.view(B, S)

    tokens_per_example = (shift_labels != ignore_index).sum(dim=-1).float()  # [B]
    denom = tokens_per_example.clamp(min=1.0)
    example_losses = loss_matrix.sum(dim=-1) / denom  # [B]

    has_tokens = (tokens_per_example > 0.0).float()
    batch_denom = has_tokens.sum().clamp(min=1.0)
    mean_loss = (example_losses * has_tokens).sum() / batch_denom

    total_valid_tokens = int(tokens_per_example.sum().item())
    return mean_loss, example_losses, total_valid_tokens


def compute_reference_answer_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    im_end_id: Optional[int] = None,
    ignore_index: int = -100,
) -> Tuple[float, int]:
    """
    Calculates reference-answer negative log-likelihood (NLL) for development evaluation:
    'For answer NLL, exclude termination and average token NLL within each example before averaging examples.'
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Mask out termination token if provided
    if im_end_id is not None:
        term_mask = (shift_labels == im_end_id)
        if term_mask.any():
            shift_labels = shift_labels.clone()
            shift_labels[term_mask] = ignore_index

    B, S, V = shift_logits.shape
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    )
    loss_matrix = loss_flat.view(B, S)

    tokens_per_example = (shift_labels != ignore_index).sum(dim=-1).float()
    denom = tokens_per_example.clamp(min=1.0)
    example_nll = loss_matrix.sum(dim=-1) / denom

    has_tokens = (tokens_per_example > 0.0).float()
    batch_denom = has_tokens.sum().clamp(min=1.0)

    mean_nll = float(((example_nll * has_tokens).sum() / batch_denom).item())
    total_tokens = int(tokens_per_example.sum().item())
    return mean_nll, total_tokens
