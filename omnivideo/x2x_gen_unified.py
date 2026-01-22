import gc
import logging
import math
import os
import pickle as pkl
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE as WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .modules.unified_model import UnifiedWanWithMixedConditionModel
        
class OmniVideoX2XUnified:

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
        visual_context_adapter_patch_size=(1,4,4),
        max_context_len=None,
        init_on_cpu=True,
        wan_config=None,
        fsdp_shard_size=None,
    ):
        r"""
        Initializes the Wan text-to-video generation model with adapter components.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.dit_fsdp = dit_fsdp
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.get('boundary', 0.875)
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
            logging.info(f"Rank {rank}: Initialized FSDP group with size {self.fsdp_shard_size}, fsdp_rank {self.fsdp_rank}")
        else:
            self.fsdp_group = None
            self.fsdp_shard_size = 1
            self.fsdp_rank = 0

        shard_fn = partial(shard_model, device_id=device_id, process_group=self.fsdp_group)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        # Initialize VAE on CPU first to avoid competing for GPU memory during model loading
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device='cpu')

        # Handle USP initialization
        self.use_usp = use_usp
        if use_usp:
            from .distributed.util import init_sequence_parallel_group, get_sequence_parallel_size, get_rank
            
            # Initialize sequence parallel group
            # Note: This expects torch.distributed to be initialized already
            sp_group = init_sequence_parallel_group(config.get('sp_size', 1) if sp_size is None else sp_size)
            self.sp_group = sp_group
            self.sp_size = get_sequence_parallel_size()
            self.sp_rank = get_rank(sp_group)
            logging.info(f"Rank {rank}: Initialized USP with size {self.sp_size}, sp_rank {self.sp_rank}")
        else:
            self.sp_group = None
            self.sp_size = 1
            self.sp_rank = 0

        # Load UnifiedWanWithMixedConditionModels
        if wan_config is not None:
            logging.info(f"Initializing UnifiedWanModels from provided config (skipping from_pretrained weight init)")
            self.high_noise_model = UnifiedWanWithMixedConditionModel(
                            wan_config=wan_config,
                            vlm_in_dim=vlm_in_dim,
                            precision_dtype=self.param_dtype,
                            device_id=device_id if not self.init_on_cpu else 'cpu',
                            rank=rank,
                            dit_fsdp=dit_fsdp,
                            use_usp=use_usp,
                            use_visual_context_adapter=use_visual_context_adapter,
                            visual_context_adapter_patch_size=visual_context_adapter_patch_size,
                            max_context_len=max_context_len,
                            skip_init=True,
                        ).to(self.param_dtype)
            self.low_noise_model = UnifiedWanWithMixedConditionModel(
                            wan_config=wan_config,
                            vlm_in_dim=vlm_in_dim,
                            precision_dtype=self.param_dtype,
                            device_id=device_id if not self.init_on_cpu else 'cpu',
                            rank=rank,
                            dit_fsdp=dit_fsdp,
                            use_usp=use_usp,
                            use_visual_context_adapter=use_visual_context_adapter,
                            visual_context_adapter_patch_size=visual_context_adapter_patch_size,
                            max_context_len=max_context_len,
                            skip_init=True,
                        ).to(self.param_dtype)
        else:
            logging.info(f"Creating High Noise UnifiedWanModel from {checkpoint_dir}")
            self.high_noise_model = UnifiedWanWithMixedConditionModel.from_pretrained(
                            wan_ckpt_dir=checkpoint_dir,
                            wan_subfolder=config.get('high_noise_checkpoint', 'high_noise_model'),
                            vlm_in_dim=vlm_in_dim,
                            precision_dtype=self.param_dtype,
                            device_id=device_id if not self.init_on_cpu else 'cpu',
                            rank=rank,
                            dit_fsdp=dit_fsdp,
                            use_usp=use_usp,
                            use_visual_context_adapter=use_visual_context_adapter,
                            visual_context_adapter_patch_size=visual_context_adapter_patch_size,
                            max_context_len=max_context_len,
                        ).to(self.param_dtype)
            
            logging.info(f"Creating Low Noise UnifiedWanModel from {checkpoint_dir}")
            self.low_noise_model = UnifiedWanWithMixedConditionModel.from_pretrained(
                            wan_ckpt_dir=checkpoint_dir,
                            wan_subfolder=config.get('low_noise_checkpoint', 'low_noise_model'),
                            vlm_in_dim=vlm_in_dim,
                            precision_dtype=self.param_dtype,
                            device_id=device_id if not self.init_on_cpu else 'cpu',
                            rank=rank,
                            dit_fsdp=dit_fsdp,
                            use_usp=use_usp,
                            use_visual_context_adapter=use_visual_context_adapter,
                            visual_context_adapter_patch_size=visual_context_adapter_patch_size,
                            max_context_len=max_context_len,
                        ).to(self.param_dtype)

        # Apply USP patches if enabled
        if use_usp:
            from .distributed.full_sequence_parallel import (sp_dit_forward, sp_attn_forward, sp_cross_attn_forward)
            for m in [self.high_noise_model, self.low_noise_model]:
                m.wan_model.sp_group = sp_group
                for block in m.wan_model.blocks:
                    block.self_attn.sp_group = sp_group
                    block.cross_attn.sp_group = sp_group
                    block.sp_group = sp_group
                    block.self_attn.forward = types.MethodType(sp_attn_forward, block.self_attn)
                    block.cross_attn.forward = types.MethodType(sp_cross_attn_forward, block.cross_attn)
                m.wan_model.forward = types.MethodType(sp_dit_forward, m.wan_model)

        # Apply FSDP if enabled
        if dit_fsdp:
            for m in [self.high_noise_model, self.low_noise_model]:
                m.wan_model = shard_fn(m.wan_model)
        
        self.high_noise_model.enable_eval()
        self.low_noise_model.enable_eval()

        if dist.is_initialized():
            dist.barrier()
        
        # Device placement depends on mode:
        # - FSDP: shard_model() already moves wan_model to GPU during wrapping
        #         Only need to move companion modules (vlm_norm, vlm_proj, visual_context_adapter)
        # - Non-FSDP with init_on_cpu: Models stay on CPU, loaded on-demand
        # - Non-FSDP without init_on_cpu: Load high_noise_model to GPU initially
        if dit_fsdp:
            # FSDP: wan_model is already on GPU from shard_fn()
            # Move companion modules to GPU (they are NOT wrapped by FSDP)
            for m in [self.high_noise_model, self.low_noise_model]:
                m.vlm_norm.to(self.device)
                m.vlm_proj.to(self.device)
                if m.visual_context_adapter:
                    m.visual_context_adapter.to(self.device)
        elif not self.init_on_cpu:
            # Non-FSDP, non-CPU init: load high_noise_model to GPU (first model needed)
            self.high_noise_model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt_en

        # Load special tokens from checkpoint directory
        special_tokens_path = os.path.join(checkpoint_dir, "special_tokens.pkl")
        if os.path.exists(special_tokens_path):
            with open(special_tokens_path, 'rb') as f:
                self.special_tokens = pkl.load(f)
            assert isinstance(self.special_tokens, dict), "special_tokens should be a dictionary"
            # Convert to proper dtype and device
            for key, value in self.special_tokens.items():
                self.special_tokens[key] = value.to(self.param_dtype).to(self.device)
            logging.info(f"Loaded special token embeddings from {special_tokens_path}: {list(self.special_tokens.keys())}")
        else:
            self.special_tokens = None
            logging.info(f"No special_tokens.pkl found at {special_tokens_path}, special_tokens disabled")

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.
        
        For FSDP: Both models stay on GPU (sharded), just select the right one.
        For non-FSDP: Offload/load models as needed.
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'

        required_model = getattr(self, required_model_name)
        offload_model_obj = getattr(self, offload_model_name)

        # FSDP: Don't do manual offloading - both models stay on GPU (sharded)
        if self.dit_fsdp:
            return required_model

        # Non-FSDP: Handle model offloading
        if offload_model or self.init_on_cpu:
            # Offload the model we don't need
            if next(offload_model_obj.parameters()).device.type == 'cuda':
                offload_model_obj.to('cpu')
            # Load the model we need
            if next(required_model.parameters()).device.type == 'cpu':
                required_model.to(self.device)
                
        return required_model


    def generate(self,
                 input_prompt,
                 precomputed_context=None,
                 ar_vision_input=None,
                 visual_emb=None,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 precision_dtype=torch.bfloat16,
                 offload_model=True,
                 classifier_free_ratio=0.0,
                 unconditioned_context=None,
                 condition_mode="auto"):
        r"""
        Generates video frames from text prompt using diffusion process with adapter.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            visual_emb (`torch.Tensor`, *optional*, defaults to None):
                Visual embedding for the visual context adapter
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            classifier_free_ratio (`float`, *optional*, defaults to 0.0):
                Ratio for classifier-free guidance during training
            unconditioned_context (`torch.Tensor`, *optional*, defaults to None):
                Unconditioned context for classifier-free guidance
            condition_mode (`str`, *optional*, defaults to "auto"):
                Mode for conditioning, options:
                - "auto": Automatically determine based on inputs (default)
                - "full": Use context + visual_emb + aligned_emb
                - "aligned_emb_with_text": Use aligned_emb + context
                - "aligned_emb_only": Use aligned_emb only
                - "visual_with_aligned_emb": Use visual_emb + aligned_emb
                - "text_only": Use context only

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # Text conditioning:
        # - If `precomputed_context` is provided (e.g., from offline dataset pickle), use it directly.
        # - Otherwise, run T5 encoder on `input_prompt` (legacy behavior).
        def _to_2d_tensor(x):
            if x is None: return None
            if isinstance(x, list) and len(x) == 1: x = x[0]
            if not isinstance(x, torch.Tensor): x = torch.tensor(x)
            if x.dim() == 3 and x.size(0) == 1: x = x.squeeze(0)
            return x

        # Positive context
        if precomputed_context is not None:
            context = [_to_2d_tensor(precomputed_context).to(self.device, dtype=precision_dtype)]
        else:
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([input_prompt], self.device)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = [t.to(self.device) for t in self.text_encoder([input_prompt], torch.device('cpu'))]

        # Negative context (null)
        # Always encode negative prompt to allow for flexibility
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context_null = [t.to(self.device) for t in self.text_encoder([n_prompt], torch.device('cpu'))]

        # Sync contexts within SP group to ensure bit-exact parity
        if self.use_usp and dist.is_initialized() and self.sp_size > 1:
            # Source rank for broadcast is the global rank of the first rank in the SP group
            src_global_rank = self.rank - self.sp_rank
            for ctx in [context, context_null]:
                for i in range(len(ctx)):
                    dist.broadcast(ctx[i], src=src_global_rank, group=self.sp_group)

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=precision_dtype,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync', noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
            torch.amp.autocast('cuda', dtype=self.param_dtype),
            torch.no_grad(),
            no_sync_low_noise(),
            no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            # Build null AR vision for CFG (only needed when ar_vision_input is provided)
            ar_vision_input_null = None
            if condition_mode != 'text_only' and ar_vision_input is not None:
                # Determine VLM dimension from ar_vision_input
                vlm_dim = ar_vision_input[0].size(-1) if isinstance(ar_vision_input, list) else ar_vision_input.size(-1)
                
                # Use provided uncond_ar_vision or create zeros online
                if isinstance(unconditioned_context, dict) and ('uncond_ar_vision' in unconditioned_context):
                    uncond_ar_vision = unconditioned_context['uncond_ar_vision']
                else:
                    # Create zero tensor with 2 tokens as null embedding (same as training)
                    uncond_ar_vision = torch.zeros(2, vlm_dim, device=self.device, dtype=precision_dtype) + 1e-6
                
                if isinstance(ar_vision_input, list):
                    ar_vision_input_null = [uncond_ar_vision for _ in range(len(ar_vision_input))]
                else:
                    ar_vision_input_null = uncond_ar_vision
            
            # Prepare arguments for the model
            arg_c = {
                'context': context, 
                'seq_len': seq_len,
                'ar_vision_input': ar_vision_input,
                'visual_emb': visual_emb,
                'special_token_dict': self.special_tokens,
                'classifier_free_ratio': 0.0,  # During inference, we don't use random dropout
                'unconditioned_context': unconditioned_context,
                'condition_mode': condition_mode
            }
               
            arg_null = {
                'context': context_null, 
                'seq_len': seq_len,
                'ar_vision_input': ar_vision_input_null,
                'visual_emb': visual_emb, 
                'special_token_dict': self.special_tokens,
                'classifier_free_ratio': 0.0,
                'unconditioned_context': unconditioned_context,
                'condition_mode': condition_mode
            }

            for t in tqdm(timesteps):
                timestep = t.unsqueeze(0)

                # Select model based on timestep boundary
                model = self._prepare_model_for_timestep(t, boundary, offload_model)

                noise_pred_cond = model(latents, t=timestep, **arg_c)[0]
                noise_pred_uncond = model(latents, t=timestep, **arg_null)[0]

                # Standard CFG formula: uncond + scale * (cond - uncond)
                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]
            
            # Offload both models after sampling (only for non-FSDP)
            if offload_model and not self.dit_fsdp:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()
            
            # Synchronize before VAE decoding
            if dist.is_initialized():
                dist.barrier()
            
            # VAE decode: only on leader rank of the group to save time and memory
            # Must move both model AND scale tensors (mean, std) to GPU
            def move_vae_to_device(device):
                self.vae.model.to(device)
                self.vae.mean = self.vae.mean.to(device)
                self.vae.std = self.vae.std.to(device)
                self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]
            
            if self.use_usp:
                is_decode_rank = (self.sp_rank == 0)
            elif self.dit_fsdp and self.fsdp_shard_size > 1:
                is_decode_rank = (self.fsdp_rank == 0)
            else:
                is_decode_rank = (self.rank == 0)

            if is_decode_rank:
                move_vae_to_device(self.device)
                videos = self.vae.decode(latents)
                move_vae_to_device('cpu')  # Free GPU memory
            else:
                videos = [None]

        # Cleanup after context manager exit
        del noise
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if self.use_usp:
            return videos[0] if self.sp_rank == 0 else None
        elif self.dit_fsdp and self.fsdp_shard_size > 1:
            return videos[0] if self.fsdp_rank == 0 else None
        else:
            return videos[0] if self.rank == 0 else None
