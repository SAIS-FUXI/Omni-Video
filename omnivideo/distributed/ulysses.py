import torch
import torch.distributed as dist

from ..modules.attention import flash_attention
from .util import all_to_all


def distributed_attention(
    q,
    k,
    v,
    seq_lens,
    window_size=(-1, -1),
    sp_group=None,
):
    """
    Performs distributed attention based on DeepSpeed Ulysses attention mechanism.
    please refer to https://arxiv.org/pdf/2309.14509

    Args:
        q:           [B, Lq // p, Nq, C1].
        k:           [B, Lk // p, Nk, C1].
        v:           [B, Lk // p, Nk, C2]. Nq must be divisible by Nk.
        seq_lens:    [B], length of each sequence in batch
        window_size: (left right). If not (-1, -1), apply sliding window local attention.
        sp_group:    sequence parallel process group.
    """
    if not dist.is_initialized():
        raise ValueError("distributed group should be initialized.")

    # Fail fast on incompatible shapes (otherwise dist.all_to_all may hang/crash).
    sp_world_size = dist.get_world_size(group=sp_group) if sp_group is not None else dist.get_world_size()
    if q.size(2) % sp_world_size != 0:
        raise ValueError(f"q.num_heads={q.size(2)} must be divisible by sp_size={sp_world_size}")
    if k.size(2) % sp_world_size != 0:
        raise ValueError(f"k.num_heads={k.size(2)} must be divisible by sp_size={sp_world_size}")
    if v.size(2) % sp_world_size != 0:
        raise ValueError(f"v.num_heads={v.size(2)} must be divisible by sp_size={sp_world_size}")

    # gather q/k/v sequence
    q = all_to_all(q, scatter_dim=2, gather_dim=1, group=sp_group)
    k = all_to_all(k, scatter_dim=2, gather_dim=1, group=sp_group)
    v = all_to_all(v, scatter_dim=2, gather_dim=1, group=sp_group)

    # apply attention
    x = flash_attention(
        q,
        k,
        v,
        k_lens=seq_lens,
        window_size=window_size,
    )

    # scatter q/k/v sequence
    x = all_to_all(x, scatter_dim=1, gather_dim=2, group=sp_group)
    return x
