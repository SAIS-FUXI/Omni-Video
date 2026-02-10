import logging
import os
import pickle as pkl
import types
from functools import partial

import torch
import torch.distributed as dist

from .distributed.fsdp import shard_model
from .modules.t5 import T5EncoderModel
from .modules.unified_model import UnifiedWanWithMixedConditionModel
from .modules.vae2_1 import Wan2_1_VAE as WanVAE
from .x2x_gen_unified import OmniVideoX2XUnified


class OmniVideoX2XUnified1_3B(OmniVideoX2XUnified):
    """
    Single-model 1.3B variant for V2V.

    This class keeps the same public interface as OmniVideoX2XUnified, but uses
    only one diffusion model. Compatibility aliases are kept so existing helper
    code that references high_noise_model / low_noise_model continues to work.
    """

    def __init__(
        self,
        config,
        checkpoint_dir,
        vlm_in_dim=4096,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        sp_size=None,
        t5_cpu=False,
        use_visual_context_adapter=False,
        visual_context_adapter_patch_size=(1, 4, 4),
        max_context_len=None,
        init_on_cpu=True,
        wan_config=None,
        fsdp_shard_size=None,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.dit_fsdp = dit_fsdp
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.get("boundary", 0.875)
        self.param_dtype = config.param_dtype

        # Handle FSDP group initialization
        if dit_fsdp and fsdp_shard_size is not None and fsdp_shard_size > 1:
            world_size = dist.get_world_size()
            assert world_size % fsdp_shard_size == 0
            num_groups = world_size // fsdp_shard_size
            self.fsdp_group = None
            for i in range(num_groups):
                ranks = list(range(i * fsdp_shard_size, (i + 1) * fsdp_shard_size))
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self.fsdp_group = group
            self.fsdp_shard_size = fsdp_shard_size
            self.fsdp_rank = self.rank % fsdp_shard_size
            logging.info(
                f"Rank {rank}: Initialized FSDP group with size {self.fsdp_shard_size}, fsdp_rank {self.fsdp_rank}"
            )
        else:
            self.fsdp_group = None
            self.fsdp_shard_size = 1
            self.fsdp_rank = 0

        shard_fn = partial(shard_model, device_id=device_id, process_group=self.fsdp_group)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint), device="cpu")

        # Handle USP initialization
        self.use_usp = use_usp
        if use_usp:
            from .distributed.util import get_rank, get_sequence_parallel_size, init_sequence_parallel_group

            sp_group = init_sequence_parallel_group(config.get("sp_size", 1) if sp_size is None else sp_size)
            self.sp_group = sp_group
            self.sp_size = get_sequence_parallel_size()
            self.sp_rank = get_rank(sp_group)
            logging.info(f"Rank {rank}: Initialized USP with size {self.sp_size}, sp_rank {self.sp_rank}")
        else:
            self.sp_group = None
            self.sp_size = 1
            self.sp_rank = 0

        # Load a single UnifiedWanWithMixedConditionModel for 1.3B
        if wan_config is not None:
            logging.info("Initializing single UnifiedWanModel from provided config")
            self.model = UnifiedWanWithMixedConditionModel(
                wan_config=wan_config,
                vlm_in_dim=vlm_in_dim,
                precision_dtype=self.param_dtype,
                device_id=device_id if not self.init_on_cpu else "cpu",
                rank=rank,
                dit_fsdp=dit_fsdp,
                use_usp=use_usp,
                use_visual_context_adapter=use_visual_context_adapter,
                visual_context_adapter_patch_size=visual_context_adapter_patch_size,
                max_context_len=max_context_len,
                skip_init=True,
            ).to(self.param_dtype)
        else:
            logging.info(f"Creating single UnifiedWanModel from {checkpoint_dir}")
            self.model = UnifiedWanWithMixedConditionModel.from_pretrained(
                wan_ckpt_dir=checkpoint_dir,
                wan_subfolder=config.get("checkpoint", None),
                vlm_in_dim=vlm_in_dim,
                precision_dtype=self.param_dtype,
                device_id=device_id if not self.init_on_cpu else "cpu",
                rank=rank,
                dit_fsdp=dit_fsdp,
                use_usp=use_usp,
                use_visual_context_adapter=use_visual_context_adapter,
                visual_context_adapter_patch_size=visual_context_adapter_patch_size,
                max_context_len=max_context_len,
            ).to(self.param_dtype)

        # Compatibility aliases for existing helper code paths
        self.high_noise_model = self.model
        self.low_noise_model = self.model

        if use_usp:
            from .distributed.full_sequence_parallel import sp_attn_forward, sp_cross_attn_forward, sp_dit_forward

            m = self.model
            m.wan_model.sp_group = self.sp_group
            for block in m.wan_model.blocks:
                block.self_attn.sp_group = self.sp_group
                block.cross_attn.sp_group = self.sp_group
                block.sp_group = self.sp_group
                block.self_attn.forward = types.MethodType(sp_attn_forward, block.self_attn)
                block.cross_attn.forward = types.MethodType(sp_cross_attn_forward, block.cross_attn)
            m.wan_model.forward = types.MethodType(sp_dit_forward, m.wan_model)

        if dit_fsdp:
            self.model.wan_model = shard_fn(self.model.wan_model)

        self.model.enable_eval()

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            self.model.vlm_norm.to(self.device)
            self.model.vlm_proj.to(self.device)
            if self.model.visual_context_adapter:
                self.model.visual_context_adapter.to(self.device)
        elif not self.init_on_cpu:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt_en

        # Load special tokens from checkpoint directory
        special_tokens_path = os.path.join(checkpoint_dir, "special_tokens.pkl")
        if os.path.exists(special_tokens_path):
            with open(special_tokens_path, "rb") as f:
                self.special_tokens = pkl.load(f)
            assert isinstance(self.special_tokens, dict), "special_tokens should be a dictionary"
            for key, value in self.special_tokens.items():
                self.special_tokens[key] = value.to(self.param_dtype).to(self.device)
            logging.info(f"Loaded special token embeddings from {special_tokens_path}: {list(self.special_tokens.keys())}")
        else:
            self.special_tokens = None
            logging.info(f"No special_tokens.pkl found at {special_tokens_path}, special_tokens disabled")

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        """
        1.3B single-model behavior: always use the same model, ignore boundary split.
        """
        if self.dit_fsdp:
            return self.model
        if offload_model or self.init_on_cpu:
            if next(self.model.parameters()).device.type == "cpu":
                self.model.to(self.device)
        return self.model

