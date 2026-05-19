"""
A number of functions that help with evaluating a base model.
"""
import math
import numpy as np
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record the losses
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        loss2d = loss2d.view(-1) # flatten
        y = y.view(-1) # flatten
        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb


def evaluate_bpr(model, batches, steps, token_residues):
    """Bits per residue — same as bpb but with residue counts instead of byte counts."""
    return evaluate_bpb(model, batches, steps, token_residues)


@torch.no_grad()
def evaluate_ngram_losses(model, batches, steps):
    """
    Returns the mean cross-entropy loss (in nats) at each sequence position n=0..T-1,
    averaged over all batches. In the paper's notation (arXiv 2602.07488), position n
    here corresponds to L_{n+1}: predicting the (n+1)-th token given n+1 tokens of context.

    Returns a float64 numpy array of shape (T,).

    Unlike evaluate_bpb, this returns raw nats with no byte normalization, and keeps
    the T dimension instead of collapsing it, so the caller can study how loss varies
    with context length — the key measurement for fitting γ (entropy decay exponent).
    """
    device = model.get_device()
    sum_nats = None  # lazily initialized from first batch shape
    count = None
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none')  # (B, T)
        if sum_nats is None:
            T = loss2d.shape[1]
            sum_nats = torch.zeros(T, dtype=torch.float64, device=device)
            count = torch.zeros(T, dtype=torch.int64, device=device)
        valid = (y >= 0)  # mask out ignore_index=-1 tokens (e.g. from SFT masking)
        sum_nats += (loss2d * valid).to(torch.float64).sum(dim=0)
        count += valid.sum(dim=0)
    # sum reduce across all DDP ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(sum_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    mean_nats = (sum_nats / count.clamp(min=1)).cpu().numpy()
    return mean_nats  # shape (T,), float64
