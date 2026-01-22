import torch
import torch.distributed as dist


_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_SIZE = 1


def init_distributed_group():
    """r initialize distributed group.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')


def init_sequence_parallel_group(sp_size):
    """
    Initialize sequence parallel groups.
    """
    if not dist.is_initialized():
        return None

    world_size = dist.get_world_size()
    assert world_size % sp_size == 0

    rank = dist.get_rank()
    num_sp_groups = world_size // sp_size

    sp_group = None
    for i in range(num_sp_groups):
        ranks = list(range(i * sp_size, (i + 1) * sp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            sp_group = group

    global _SEQUENCE_PARALLEL_GROUP, _SEQUENCE_PARALLEL_SIZE
    _SEQUENCE_PARALLEL_GROUP = sp_group
    _SEQUENCE_PARALLEL_SIZE = sp_size

    return sp_group


def init_distributed_groups(sp_size):
    """
    Initialize orthogonal groups for FSDP and Sequence Parallelism.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # SP Group
    sp_group = init_sequence_parallel_group(sp_size)

    # DP Group (for FSDP)
    # Ranks in DP group are [rank % sp_size, rank % sp_size + sp_size, ...]
    num_sp_groups = world_size // sp_size
    dp_group = None
    for i in range(sp_size):
        ranks = [i + j * sp_size for j in range(num_sp_groups)]
        group = dist.new_group(ranks)
        if rank in ranks:
            dp_group = group

    return dp_group, sp_group


def get_sequence_parallel_group():
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_size():
    return _SEQUENCE_PARALLEL_SIZE


def get_sequence_parallel_rank():
    if _SEQUENCE_PARALLEL_GROUP is None:
        return 0
    return dist.get_rank(_SEQUENCE_PARALLEL_GROUP)


def get_rank(group=None):
    if group is None:
        group = _SEQUENCE_PARALLEL_GROUP
    return dist.get_rank(group=group)


def get_world_size(group=None):
    if group is None:
        group = _SEQUENCE_PARALLEL_GROUP
    return dist.get_world_size(group=group)


class AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scatter_dim, gather_dim, group=None):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.group = group
        world_size = get_world_size(group)
        if world_size <= 1:
            return x

        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group)
        return torch.cat(outputs, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        return AllToAll.apply(grad_output, ctx.gather_dim, ctx.scatter_dim,
                             ctx.group), None, None, None


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    return AllToAll.apply(x, scatter_dim, gather_dim, group)


def all_gather(tensor, group=None):
    world_size = get_world_size(group)
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor.contiguous(), group=group)
    return tensor_list


class GatherForward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dim, group=None):
        ctx.dim = dim
        ctx.group = group
        world_size = get_world_size(group)
        if world_size <= 1:
            return x

        output = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(output, x.contiguous(), group=group)
        return torch.cat(output, dim=dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        world_size = get_world_size(ctx.group)
        if world_size <= 1:
            return grad_output, None, None

        return grad_output.chunk(world_size,
                                 dim=ctx.dim)[get_rank(ctx.group)], None, None


def gather_forward(input, dim, group=None):
    """
    Gather sequence.
    """
    return GatherForward.apply(input, dim, group)
