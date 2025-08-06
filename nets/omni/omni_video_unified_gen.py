import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from nets.third_party.wan.distributed.fsdp import shard_model
from nets.third_party.wan.modules.model import WanModel
from nets.third_party.wan.modules.t5 import T5EncoderModel
from nets.third_party.wan.modules.vae import WanVAE
from nets.third_party.wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from nets.third_party.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from nets.omni.modules.omni_video_model import OmniVideoMixedConditionModel
        
class OmniVideoX2XUnified:

    def __init__(
        self,
        config,
        checkpoint_dir,
        adapter_checkpoint_dir,
        vision_head_ckpt_dir,
        adapter_in_channels=1152,
        adapter_out_channels=4096,
        adapter_query_length=64,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        use_visual_context_adapter=False,
        visual_context_adapter_patch_size=(1,4,4),
        max_context_len=None,
    ):
        r"""
        Initializes the Wan text-to-video generation model with adapter components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            adapter_checkpoint_dir (`str`):
                Path to directory containing adapter checkpoints
            adapter_in_channels (`int`, *optional*, defaults to 1152):
                Input channels for the adapter
            adapter_out_channels (`int`, *optional*, defaults to 4096):
                Output channels for the adapter
            adapter_query_length (`int`, *optional*, defaults to 64):
                Query length for the adapter
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.dit_fsdp = dit_fsdp

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating OmniVideoMixedConditionModel from {checkpoint_dir} and {adapter_checkpoint_dir}")
        # Initialize the OmniVideoMixedConditionModel
        self.model = OmniVideoMixedConditionModel.from_pretrained(
                        wan_ckpt_dir=checkpoint_dir,
                        adapter_ckpt_dir=adapter_checkpoint_dir,
                        vision_head_ckpt_dir=vision_head_ckpt_dir, 
                        adapter_in_channels=adapter_in_channels,
                        adapter_out_channels=adapter_out_channels,
                        adapter_query_length=adapter_query_length,
                        precision_dtype=self.param_dtype,
                        device_id=device_id,
                        rank=rank,
                        dit_fsdp=dit_fsdp,
                        use_usp=use_usp,
                        use_visual_context_adapter=use_visual_context_adapter,
                        visual_context_adapter_patch_size=visual_context_adapter_patch_size,
                        max_context_len=max_context_len,
                    )
        
        self.model.enable_eval()

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from nets.third_party.wan.distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.wan_model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.wan_model.forward = types.MethodType(usp_dit_forward, self.model.wan_model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 aligned_emb=None,
                 ar_vision_input=None,
                 visual_emb=None,
                 ref_images=None,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 special_tokens=None,
                 classifier_free_ratio=0.0,
                 unconditioned_context=None,
                 condition_mode="auto",
                 use_visual_as_input=False):
        r"""
        Generates video frames from text prompt using diffusion process with adapter.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            aligned_emb (`torch.Tensor`, *optional*, defaults to None):
                Aligned embedding for the adapter
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
            special_tokens (`dict`, *optional*, defaults to None):
                Dictionary of special token embeddings
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
            use_visual_as_input (`bool`, *optional*, defaults to False):
                Whether to use visual embedding as part of the input

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

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        if ar_vision_input is not None: 
            aligned_emb = None # make sure aligned_emb is not used when ar_vision_input is provided
        # Process aligned embedding if provided
        if aligned_emb is not None and not isinstance(aligned_emb, torch.Tensor):
            aligned_emb = torch.tensor(aligned_emb, dtype=torch.float32, device=self.device)
        
        if aligned_emb is not None and aligned_emb.dim() == 1:
            # Add batch dimension if needed
            aligned_emb = aligned_emb.unsqueeze(0)

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

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

            ar_vision_input_null = None
            aligned_emb_null = None
            visual_emb_null = None
            ref_images_null = None
            # import pdb; pdb.set_trace();
            if not condition_mode == 'text_only':
                assert type(unconditioned_context) is dict, "unconditioned_context must be provided as a dict if condition_mode is not 'text_only'"

                if ar_vision_input is not None and 'uncond_ar_vision' in unconditioned_context:
                    if isinstance(ar_vision_input, list):
                        ar_vision_input_null = [unconditioned_context['uncond_ar_vision'] for _ in range(len(ar_vision_input))]
                    else:
                        ar_vision_input_null = unconditioned_context['uncond_ar_vision']

                if context is not None and isinstance(context, list):
                    context_null = [unconditioned_context['uncond_context'] for c in context]
                elif context is not None and isinstance(context, torch.Tensor):
                    context_null = unconditioned_context['uncond_context']
                    if context_null is not None and context_null.dim() < context.dim():
                        context_null = context_null.unsqueeze(0)
                else:
                    context_null = context_null # use t5 null context

                if visual_emb is not None:
                    if isinstance(visual_emb, list):
                        visual_emb_null = [torch.zeros_like(ve) for ve in visual_emb]
                    else:
                        visual_emb_null = torch.zeros_like(visual_emb) if isinstance(visual_emb, torch.Tensor) else None
            
             # Prepare arguments for the model
            arg_c = {
                'context': context, 
                'seq_len': seq_len,
                'aligned_emb': aligned_emb,
                'ar_vision_input': ar_vision_input,
                'visual_emb': visual_emb,
                'ref_images': ref_images,
                'special_token_dict': special_tokens,
                'classifier_free_ratio': 0.0,  # During inference, we don't use random dropout
                'unconditioned_context': unconditioned_context,
                'condition_mode': condition_mode
            }
               
            arg_null = {
                'context': context_null, 
                'seq_len': seq_len,
                'aligned_emb': aligned_emb_null, #TODO : not finished
                'ar_vision_input': ar_vision_input_null,
                'visual_emb': visual_emb, # TODO: not finished
                'ref_images': ref_images, # TODO: not finished
                'special_token_dict': special_tokens,
                'classifier_free_ratio': 0.0,
                'unconditioned_context': unconditioned_context,
                'condition_mode': condition_mode
            }

            for _, t in enumerate(tqdm(timesteps)):
                if use_visual_as_input and visual_emb is not None:
                    assert len(latents) == 1, "currently only support one latent at a time"
                    latent_model_input = [latents[0] + visual_emb]
                else:
                    latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                
                noise_pred = noise_pred_cond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.dit_fsdp:
                if self.rank == 0:
                    videos = self.vae.decode(x0)
            else:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if self.dit_fsdp:
            return videos[0] if self.rank == 0 else None
        else:
            return videos[0]
