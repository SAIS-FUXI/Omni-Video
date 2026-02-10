import argparse
import copy
import logging
import os
import sys
import warnings
import json
import decord
import gc
warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image
from torchvision import transforms
import numpy as np

from omnivideo.configs import WAN_CONFIGS, SIZE_CONFIGS
from omnivideo.utils.utils import cache_video, str2bool
from omnivideo.x2x_gen_unified_1_3B import OmniVideoX2XUnified1_3B

# VLM (Vision-Language Model) helpers
from omnivideo.vllm_model import (
    load_qwen3vl_model_and_processor,
    generate_caption_and_extract_features,
    offload_qwen3vl_to_cpu,
    load_qwen3vl_to_gpu,
)

# ============================================================================
# OmniVideo Offload Helpers
# ============================================================================

def offload_model_to_cpu(model):
    """Offload OmniVideo model components to CPU to free GPU memory for Qwen3-VL"""
    logging.info("[Offload] Moving OmniVideo components to CPU...")
    if hasattr(model, 'model') and model.model is not None:
        model.model.to('cpu')
    if hasattr(model, 'text_encoder') and model.text_encoder is not None:
        if hasattr(model.text_encoder, 'model'):
            model.text_encoder.model.to('cpu')
    if hasattr(model, 'vae') and model.vae is not None:
        if hasattr(model.vae, 'model'):
            model.vae.model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()

def load_model_to_gpu(model, target_device):
    """Load OmniVideo model components back to GPU after Qwen3-VL is done"""
    logging.info(f"[Offload] Moving OmniVideo components to {target_device}...")
    if hasattr(model, 'model') and model.model is not None:
        model.model.to(target_device)
    gc.collect()
    torch.cuda.empty_cache()


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert "v2v" in args.task, f"Only v2v tasks are supported. Got: {args.task}"

    # Default OmniVideo fine-tuned checkpoints live under ckpt_dir:
    #   - {ckpt_dir}/transformer/pytorch_model.pt
    if (not getattr(args, "new_checkpoint", None)) or (not str(args.new_checkpoint).strip()):
        default_ckpt = os.path.join(args.ckpt_dir, "transformer", "pytorch_model.pt")
        if os.path.exists(default_ckpt):
            args.new_checkpoint = default_ckpt

    # Default sampling steps
    if args.sample_steps is None:
        args.sample_steps = 40

    if args.sample_shift is None:
        args.sample_shift = 5.0

    # Default number of frames for video tasks
    if args.frame_num is None:
        args.frame_num = 81

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate videos using Wan 1.3B with mixed conditions (Standalone v2v)"
    )
    
    # Basic arguments
    parser.add_argument(
        "--task",
        type=str,
        default="v2v-1.3B",
        help="The task to run."
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory."
    )
    parser.add_argument(
        "--new_checkpoint",
        type=str,
        default=None,
        help="Optional path to the single model checkpoint. If omitted, will use {ckpt_dir}/model.pt when present."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from."
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="The prompt list to generate the video from."
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video."
    )
    
    # Sampling parameters
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample."
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="The sampling steps."
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers."
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale."
    )
    
    parser.add_argument(
        "--use_usp",
        type=str2bool,
        default=False,
        help="Whether to use sequence parallelism (USP/Ulysses)"
    )
    parser.add_argument(
        "--sp_size",
        type=int,
        default=None,
        help="Sequence parallel size"
    )
    
    parser.add_argument(
        "--t5_fsdp",
        type=str2bool,
        default=False,
        help="Whether to use FSDP for T5"
    )
    
    parser.add_argument(
        "--dit_fsdp",
        type=str2bool,
        default=False,
        help="Whether to use FSDP for DiT"
    )
    
    # Context length parameter
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=6144,
        help="Max context length for concatenated conditioning tokens"
    )
    
    # Classifier-free guidance parameters
    parser.add_argument(
        "--classifier_free_ratio",
        type=float,
        default=0.0,
        help="Ratio for classifier-free guidance during inference."
    )
    parser.add_argument(
        "--sample_fps",
        type=int,
        default=8,
        help="FPS of the generated video."
    )

    ## sampling rate and skip num
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=3,
        help="Sampling rate for the video."
    )
    parser.add_argument(
        "--skip_num",
        type=int,
        default=0,
        help="Skip number for the video."
    )
    # ============================================================================
    # V2V E2E Arguments (Online feature extraction)
    # ============================================================================
    parser.add_argument(
        "--qwen3vl_model_path",
        type=str,
        default=None,
        help="Path to Qwen3-VL model for online feature extraction"
    )
    parser.add_argument(
        "--qwen3vl_dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Data type for Qwen3-VL model"
    )
    parser.add_argument(
        "--video_max_duration",
        type=float,
        default=5.0,
        help="Maximum video duration for Qwen3-VL processing (seconds)"
    )
    parser.add_argument(
        "--qwen3vl_device_map",
        type=str,
        default="auto",
        help="Device map for Qwen3-VL model parallelism. Use 'auto' to distribute across all visible GPUs, or 'cuda:0' for single GPU"
    )
    parser.add_argument(
        "--qwen3vl_temperature",
        type=float,
        default=0.0,
        help="Temperature for Qwen3-VL generation. 0.0 for greedy decoding (no sampling)"
    )

    args = parser.parse_args()
    _validate_args(args)
    return args

def _init_logging(rank):
    # save log to file 
    log_file = f'log_{rank}.log'
    
    # Ensure the root logger is clean so our configuration takes effect
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[
                logging.StreamHandler(stream=sys.stdout), 
                logging.FileHandler(log_file)
            ],
            force=True)
        # Verify logging is working immediately
        logging.info(f"Logging initialized for rank {rank}. Output will be sent to terminal and {log_file}")
    else:
        logging.basicConfig(level=logging.ERROR, force=True)


def transform_frames_to_tensor(frames, target_size=(480, 832)):
    """
    Transform a list of frames (PIL Images or numpy arrays) to tensors with resize and center crop
    """
    # Define the transformation pipeline
    h, w = frames[0].shape[:2]
    ratio = float(target_size[1]) / float(target_size[0]) # w/h
    if w < h * ratio:
        crop_size = (int(float(w) / ratio), w)
    else:
        crop_size = (h, int(float(h) * ratio))

    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Process each frame
    tensor_frames = []
    for frame in frames:
        # Convert to PIL Image if it's a numpy array
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        # Apply transformations
        tensor_frame = transform(frame)
        tensor_frames.append(tensor_frame)
    
    # Stack all frames into a single tensor [T, C, H, W]
    return torch.stack(tensor_frames)

def read_video_frames(video_path, frame_num, sampling_rate=1, skip_num=0, target_size=(480, 832)):
    """
    读取视频并提取指定数量的帧 (Using decord)
    """
    try:
        vr = decord.VideoReader(video_path)
    except Exception as e:
        logging.error(f"无法打开视频: {video_path}, error: {e}")
        return None
    
    total_frames = len(vr)
    
    # Calculate sampling rate
    while total_frames < frame_num * sampling_rate + skip_num:
        sampling_rate -= 1
        if sampling_rate <= 0:
            logging.warning(f"Video frame count not enough: {video_path}, total: {total_frames}, need: {frame_num} frames")
            return None
    
    logging.info(f"Using sampling rate: {sampling_rate}")

    # Check aspect ratio compatibility
    try:
        first_frame = vr[0].asnumpy()
    except Exception as e:
        logging.error(f"Failed to read first frame: {video_path}, error: {e}")
        return None

    height, width = first_frame.shape[:2]
    logging.info(f"Video info: total_frames={total_frames}, resolution=w{width}xh{height}")

    if (target_size[0] > target_size[1] and height < width) or (target_size[0] < target_size[1] and height > width):
        logging.info(f'target_size {target_size} cur video: {height} {width}, so skip')
        return None
    
    indices = [skip_num + i * sampling_rate for i in range(frame_num)]
    
    try:
        frames = vr.get_batch(indices).asnumpy()
    except Exception as e:
        logging.error(f"Failed to extract frames: {video_path}, error: {e}")
        return None
    
    # frames is (N, H, W, C) numpy array (RGB), transform expects this or PIL
    frames = transform_frames_to_tensor(frames, target_size) # [T, C, H, W]
    return frames

def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    print(f'world_size: {world_size}, rank: {rank}, local_rank: {local_rank}', flush=True)

    # Always initialize distributed for torchrun compatibility
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size)

    if '1.3B' in args.task:
        cfg = copy.deepcopy(WAN_CONFIGS['t2v-1.3B'])
    else:
        cfg = copy.deepcopy(WAN_CONFIGS['t2v-1.3B'])
    
    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        seed_tensor = torch.tensor([args.base_seed], device=torch.device(f"cuda:{local_rank}"), dtype=torch.int64)
        dist.broadcast(seed_tensor, src=0)
        base_seed = int(seed_tensor.item())
        
        if args.dit_fsdp:
            args.base_seed = base_seed
        else:
            sp_size = args.sp_size if args.sp_size is not None else 1
            dp_rank_for_seed = rank // sp_size
            args.base_seed = base_seed + dp_rank_for_seed * 1000
    
    sp_size = args.sp_size if (args.sp_size is not None and args.sp_size > 0) else 1
    sp_rank = rank % sp_size
    
    if args.dit_fsdp:
        dp_world_size = 1
        dp_rank = 0
    else:
        dp_world_size = world_size // sp_size
        dp_rank = rank // sp_size
    
    if dp_world_size == 0: dp_world_size = 1 # Safety
    
    logging.info(f"Rank {rank}: dp_world_size={dp_world_size}, dp_rank={dp_rank}, sp_rank={sp_rank}, dit_fsdp={args.dit_fsdp}")

    # Load prompts from JSONL file
    if hasattr(args, 'prompt_file') and args.prompt_file:
        if not args.prompt_file.endswith(".jsonl"):
            raise ValueError(f"Only JSONL format is supported for prompt_file. Got: {args.prompt_file}")
        with open(args.prompt_file, 'r') as f:
            prompt_list = [json.loads(line.strip(" \t\n")) for line in f.readlines() if line.strip()]
    elif hasattr(args, 'prompt') and args.prompt:
        prompt_list = [{'prompt': args.prompt}]
    else:
        raise ValueError("Either --prompt_file (JSONL) or --prompt must be provided")
    
    precision_dtype = cfg.param_dtype
    unconditioned_context = None

    # Initialize model (1.3B specific class)
    logging.info("Creating OmniVideoX2XUnified1_3B pipeline.")
    omni_video = OmniVideoX2XUnified1_3B(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        vlm_in_dim=cfg.vlm_in_dim,
        device_id=device,
        rank=rank,
        use_usp=args.use_usp,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        sp_size=args.sp_size,
        use_visual_context_adapter=cfg.use_visual_context_adapter,
        visual_context_adapter_patch_size=cfg.visual_context_adapter_patch_size,
        max_context_len=args.max_context_len,
        init_on_cpu=True,
        wan_config=cfg if (args.new_checkpoint is not None) else None,
    )

    # Load single checkpoint for 1.3B
    if hasattr(args, 'new_checkpoint') and args.new_checkpoint is not None and str(args.new_checkpoint).strip():
        if dist.is_initialized():
            dist.barrier()

        state_dict = torch.load(args.new_checkpoint, map_location="cpu")
        if 'module' in state_dict:
            state_dict = state_dict['module']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        logging.info(f"Rank {rank}: Casting state dict to {precision_dtype}")
        for k in list(state_dict.keys()):
            if isinstance(state_dict[k], torch.Tensor):
                state_dict[k] = state_dict[k].to(precision_dtype)

        missing, unexpected = omni_video.model.load_state_dict(state_dict, strict=False)
        if rank == 0:
            logging.info(
                f"[model] load_state_dict: missing_keys={len(missing):,}, unexpected_keys={len(unexpected):,}"
            )
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(f"Loaded trained checkpoint into model from {args.new_checkpoint}")

        if dist.is_initialized():
            dist.barrier()
    
    # Output directory
    output_dir = os.path.join(os.getcwd(), "outputs")
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    logging.info("Generating video ...")
    print("Generating video ...", flush=True)
    target_size = SIZE_CONFIGS[args.size]

    # Initialize Qwen3-VL state
    qwen3vl_model = None
    qwen3vl_processor = None

    for idx, prompt_dict in enumerate(prompt_list):
        if idx % dp_world_size != dp_rank:
            continue

        visual_emb = None
        ar_vision_input = None
        precomputed_context = None

        prompt = prompt_dict.get('edit_prompt', prompt_dict.get('prompt', prompt_dict.get('Prompt', '')))
        if not prompt:
            logging.warning(f"Skipping sample {idx}: no prompt found in {prompt_dict.keys()}")
            continue
        
        if sp_rank == 0:
            logging.info(f"Processing prompt: {prompt[:100]}...")

        source_video_path = prompt_dict.get('source_clip_path', 
                                            prompt_dict.get('video_path_src', 
                                                            prompt_dict.get('video', None)))
        
        if source_video_path is not None:
            if sp_rank == 0:
                logging.info(f"Reading video: {source_video_path}")
            frames_tensor = read_video_frames(source_video_path, args.frame_num, args.sampling_rate, args.skip_num, (target_size[1], target_size[0]))
            if frames_tensor is None:
                if sp_rank == 0:
                    logging.warning(f"Failed to read video, skipping: {source_video_path}")
                continue

            # Extract VAE latent features
            if sp_rank == 0:
                logging.info("Extracting VAE latent features...")
            with torch.no_grad():
                frames_tensor = frames_tensor.to(device)
                omni_video.vae.model.to(device)
                omni_video.vae.mean = omni_video.vae.mean.to(device)
                omni_video.vae.std = omni_video.vae.std.to(device)
                omni_video.vae.scale = [omni_video.vae.mean, 1.0 / omni_video.vae.std]
                
                latent_feature = omni_video.vae.encode(frames_tensor.transpose(0,1).unsqueeze(0))[0]
                
                omni_video.vae.model.to('cpu')
                omni_video.vae.mean = omni_video.vae.mean.to('cpu')
                omni_video.vae.std = omni_video.vae.std.to('cpu')
                omni_video.vae.scale = [omni_video.vae.mean, 1.0 / omni_video.vae.std]
                
                target_latent_frames = (args.frame_num - 1) // 4 + 1
                if latent_feature.dim() == 4:
                    latent_feature = latent_feature.unsqueeze(0)
                
                if latent_feature.shape[2] > target_latent_frames:
                    latent_feature = latent_feature[:, :, :target_latent_frames]
                
            visual_emb = latent_feature
            
            # Online Qwen3-VL feature extraction
            if args.qwen3vl_model_path:
                if dist.is_initialized():
                    dist.barrier()
                
                offload_model_to_cpu(omni_video)
                
                if rank == 0:
                    if qwen3vl_model is None:
                        qwen3vl_model, qwen3vl_processor = load_qwen3vl_model_and_processor(
                            model_path=args.qwen3vl_model_path,
                            device='cuda',
                            dtype=args.qwen3vl_dtype,
                            device_map=args.qwen3vl_device_map,
                        )
                    else:
                        load_qwen3vl_to_gpu(qwen3vl_model, args.qwen3vl_device_map)
                    
                    target_video_caption, qwen_features = generate_caption_and_extract_features(
                        model=qwen3vl_model,
                        processor=qwen3vl_processor,
                        source_video_path=source_video_path,
                        edit_prompt=prompt,
                        source_caption_system_prompt=cfg.source_caption_system_prompt,
                        target_caption_system_prompt=cfg.target_caption_system_prompt,
                        feature_extraction_system_prompt=cfg.feature_extraction_system_prompt,
                        video_max_duration=args.video_max_duration,
                        temperature=args.qwen3vl_temperature,
                    )
                    ar_vision_input = qwen_features['vlm_last_hidden_states'].to(device)
                    if ar_vision_input.dim() == 2:
                        ar_vision_input = ar_vision_input.unsqueeze(0)
                    
                    offload_qwen3vl_to_cpu(qwen3vl_model)
                
                # Broadcast results
                if dist.is_initialized() and world_size > 1:
                    if rank == 0:
                        caption_bytes = target_video_caption.encode('utf-8')
                        caption_len = torch.tensor([len(caption_bytes)], device=device)
                    else:
                        caption_len = torch.tensor([0], device=device)
                    dist.broadcast(caption_len, src=0)
                    
                    if rank == 0:
                        caption_tensor = torch.tensor(list(caption_bytes), dtype=torch.uint8, device=device)
                    else:
                        caption_tensor = torch.zeros(caption_len.item(), dtype=torch.uint8, device=device)
                    dist.broadcast(caption_tensor, src=0)
                    
                    if rank != 0:
                        target_video_caption = bytes(caption_tensor.cpu().tolist()).decode('utf-8')
                    
                    if rank == 0:
                        ar_shape = torch.tensor(list(ar_vision_input.shape), device=device)
                    else:
                        ar_shape = torch.zeros(3, dtype=torch.long, device=device)
                    dist.broadcast(ar_shape, src=0)
                    
                    if rank != 0:
                        ar_vision_input = torch.zeros(tuple(ar_shape.tolist()), device=device, dtype=torch.float32)
                    dist.broadcast(ar_vision_input, src=0)
                
                load_model_to_gpu(omni_video, device)
                if dist.is_initialized():
                    dist.barrier()
                
                # Extract T5 features
                t5_device = torch.device('cpu') if getattr(omni_video, 't5_cpu', False) else device
                if not getattr(omni_video, 't5_cpu', False):
                    omni_video.text_encoder.model.to(device)
                
                target_t5_emb = omni_video.text_encoder([target_video_caption], t5_device)
                if isinstance(target_t5_emb, list): target_t5_emb = target_t5_emb[0]
                target_t5_emb = target_t5_emb.to(device)
                
                edit_t5_emb = omni_video.text_encoder([prompt], t5_device)
                if isinstance(edit_t5_emb, list): edit_t5_emb = edit_t5_emb[0]
                edit_t5_emb = edit_t5_emb.to(device)
                
                if not getattr(omni_video, 't5_cpu', False):
                    omni_video.text_encoder.model.cpu()
                
                precomputed_context = torch.cat([target_t5_emb, edit_t5_emb], dim=0)
            
        # Ensure temporal padding
        if visual_emb is not None:
            vc_patch_size = cfg.visual_context_adapter_patch_size
            if isinstance(vc_patch_size, (list, tuple)) and vc_patch_size[0] > 1:
                t_patch = vc_patch_size[0]
                t = visual_emb.shape[2]
                if t % t_patch != 0:
                    num_to_pad = t_patch - (t % t_patch)
                    pad_tensor = visual_emb[:, :, 0:1].repeat(1, 1, num_to_pad, 1, 1)
                    visual_emb = torch.cat([pad_tensor, visual_emb], dim=2)

        # Generate video
        video = omni_video.generate(
            prompt,
            precomputed_context=precomputed_context,
            visual_emb=visual_emb,
            ar_vision_input=ar_vision_input,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            classifier_free_ratio=args.classifier_free_ratio,
            unconditioned_context=unconditioned_context,
            condition_mode=cfg.condition_mode,
            precision_dtype=precision_dtype,
        )

        is_save_rank = (rank == 0) if args.dit_fsdp else (sp_rank == 0)
        
        if video is not None and is_save_rank:
            sample_id = prompt_dict.get('id', prompt_dict.get('sample_id', idx))
            if source_video_path:
                source_basename = os.path.splitext(os.path.basename(source_video_path))[0]
                save_file = os.path.join(output_dir, f"{source_basename}_id{sample_id}_edited.mp4")
            else:
                save_file = os.path.join(output_dir, f"output_id{sample_id}_edited.mp4")

            logging.info(f"Saving generated video to {save_file}")
            cache_video(
                tensor=video[None],
                save_file=save_file,
                fps=args.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))

    # Final cleanup
    if dist.is_initialized():
        try:
            torch.cuda.synchronize()
            dist.barrier()
            dist.destroy_process_group()
        except Exception as e:
            logging.error(f"Rank {rank}: Error during cleanup: {e}")
    
    if qwen3vl_model is not None:
        del qwen3vl_model
        del qwen3vl_processor
        gc.collect()

    torch.cuda.empty_cache()

if __name__ == "__main__":
    rank = int(os.getenv("RANK", 0))
    _init_logging(rank)
    args = _parse_args()
    generate(args)
