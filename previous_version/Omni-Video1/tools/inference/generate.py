#!/usr/bin/env python3
"""
This script provides a unified interface for video and image generation tasks including:
- Text-to-Video (T2V)
- Text-to-Image (T2I) 
- Image-to-Image (I2I)
- Video-to-Video (V2V)

Usage:
    python generate.py --prompt "A beautiful sunset" --task t2v --size 832*480
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import pickle as pkl
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from torchvision import transforms

# Optional CV2 import for video processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV (cv2) not available. Video processing will be disabled.")
    CV2_AVAILABLE = False
    cv2 = None

# Local imports - import the generator from its new location
try:
    from nets.omni.omni_video_generator import OmniVideoGenerator
    from nets.third_party.wan.configs import SIZE_CONFIGS
    from nets.third_party.wan.utils.utils import cache_video, cache_image, str2bool
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all dependencies are installed and paths are configured correctly.")
    sys.exit(1)


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        ValueError: If validation fails
    """
    # Basic validation - removed ckpt_dir requirement since paths are now fixed
    if not args.prompt:
        raise ValueError("Please provide a prompt using --prompt argument")
        
    # Set defaults
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None: 
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832", "352*640", "640*352"]:
            args.sample_shift = 3.0

    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I validation
    if "t2i" in args.task and args.frame_num != 1:
        raise ValueError(f"Frame number must be 1 for t2i task, got {args.frame_num}")

    # Set random seed if not provided
    if args.base_seed < 0:
        args.base_seed = torch.randint(0, 2**32, (1,)).item()


def str2tuple(v: str) -> tuple:
    """
    Convert string to tuple.
    
    Examples:
        '1,2,2' -> (1, 2, 2)
        '(1,2,2)' -> (1, 2, 2)
        
    Args:
        v: String representation of a tuple
        
    Returns:
        Parsed tuple with integer values
    """
    v = v.strip()
    if v.startswith('(') and v.endswith(')'):
        v = v[1:-1]
    
    return tuple(int(x.strip()) for x in v.split(','))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OmniVideo: Unified Video Generation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic arguments
    parser.add_argument(
        "--task", type=str, default="t2v",
        choices=["t2v", "t2i", "i2i", "v2v"],
        help="Generation task type"
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--size", type=str, default="832*480",
        help="Output size in format 'width*height'"
    )
    parser.add_argument(
        "--frame_num", type=int, default=None,
        help="Number of frames to generate (should be 4n+1 for videos)"
    )
    parser.add_argument(
        "--sample_fps", type=int, default=8,
        help="FPS of the generated video"
    )
    
    # Model paths (now fixed)
    parser.add_argument(
        "--ckpt_dir", type=str,
        help="Path to the main model checkpoint directory (now fixed)"
    )
    parser.add_argument(
        "--adapter_ckpt_dir", type=str,
        help="Path to the adapter checkpoint (now fixed)"
    )
    parser.add_argument(
        "--vision_head_ckpt_dir", type=str,
        help="Path to the vision head checkpoint (now fixed)"
    )
    parser.add_argument(
        "--new_checkpoint", type=str,
        help="Path to additional checkpoint to load (now fixed)"
    )
    parser.add_argument(
        "--ar_model_path", type=str,
        help="Path to the AR model (now fixed)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--sample_solver", type=str, default='unipc',
        choices=['unipc', 'dpm++'],
        help="Sampling solver"
    )
    parser.add_argument(
        "--sample_steps", type=int, default=None,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--sample_shift", type=float, default=None,
        help="Sampling shift factor"
    )
    parser.add_argument(
        "--sample_guide_scale", type=float, default=5.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--base_seed", type=int, default=-1,
        help="Random seed (-1 for random)"
    )
    
    # Input files
    parser.add_argument(
        "--src_file_path", type=str, default=None,
        help="Source image/video path for editing tasks"
    )
    parser.add_argument(
        "--save_file", type=str,
        help="Output file path (auto-generated if not specified)"
    )
    
    # Advanced parameters
    parser.add_argument(
        "--adapter_in_channels", type=int, default=1152,
        help="Adapter input channels"
    )
    parser.add_argument(
        "--adapter_out_channels", type=int, default=4096,
        help="Adapter output channels"
    )
    parser.add_argument(
        "--adapter_query_length", type=int, default=256,
        help="Adapter query length"
    )
    parser.add_argument(
        "--use_visual_context_adapter", type=str2bool, default=False,
        help="Whether to use visual context adapter"
    )
    parser.add_argument(
        "--visual_context_adapter_patch_size", type=str2tuple, default=(1, 4, 4),
        help="Visual context adapter patch size (e.g., '1,4,4')"
    )
    parser.add_argument(
        "--use_visual_as_input", type=str2bool, default=False,
        help="Whether to use visual as input"
    )
    parser.add_argument(
        "--condition_mode", type=str, default="full",
        help="Conditioning mode"
    )
    parser.add_argument(
        "--max_context_len", type=int, default=1024,
        help="Maximum context length"
    )
    
    # Classifier-free guidance
    parser.add_argument(
        "--classifier_free_ratio", type=float, default=0.0,
        help="Classifier-free guidance ratio"
    )
    parser.add_argument(
        "--unconditioned_context_path", type=str,
        help="Path to unconditioned context embeddings (now fixed)"
    )
    parser.add_argument(
        "--unconditioned_context_length", type=int, default=256,
        help="Unconditioned context length"
    )
    parser.add_argument(
        "--special_tokens_path", type=str,
        help="Path to special tokens file (now fixed)"
    )
    
    # AR model parameters
    parser.add_argument(
        "--ar_model_num_video_frames", type=int, default=8,
        help="Number of video frames for AR model"
    )
    parser.add_argument(
        "--ar_query", type=str,
        help="Query for AR model"
    )
    parser.add_argument(
        "--ar_conv_mode", type=str, default="llama_3",
        help="AR model conversation mode"
    )
    
    # Video processing
    parser.add_argument(
        "--sampling_rate", type=int, default=3,
        help="Video sampling rate"
    )
    parser.add_argument(
        "--skip_num", type=int, default=0,
        help="Number of frames to skip"
    )
    
    args = parser.parse_args()
    validate_args(args)
    return args


def init_logging(rank: int) -> None:
    """Initialize logging configuration."""
    log_file = f'omnivideo_generate_rank{rank}.log'
    
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file)
            ]
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def transform_image_to_tensor(image: Union[Image.Image, np.ndarray], 
                            target_size: Tuple[int, int] = (480, 832)) -> torch.Tensor:
    """
    Transform PIL Image or numpy array to tensor with resize and center crop.
    
    Args:
        image: Input image
        target_size: Target size (height, width)
        
    Returns:
        Transformed tensor
    """
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        w, h = image.size
        
    ratio = float(target_size[1]) / float(target_size[0])  # w/h
    
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
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    return transform(image)


def extract_vae_features(image_path: str, vae, device: torch.device, 
                        target_size: Tuple[int, int]) -> Optional[torch.Tensor]:
    """
    Extract VAE features from image.
    
    Args:
        image_path: Path to image file
        vae: VAE model instance
        device: Computation device
        target_size: Target image size
        
    Returns:
        VAE encoded features or None if failed
    """
    if not os.path.exists(image_path):
        logging.warning(f"Image file not found: {image_path}")
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform_image_to_tensor(image, target_size)
        image_tensor = image_tensor.unsqueeze(1).to(device)  # [C, 1, H, W]

        with torch.no_grad():
            latent_feature = vae.encode([image_tensor])
            latent_feature = latent_feature[0]
        
        return latent_feature
        
    except Exception as e:
        logging.error(f"Failed to extract VAE features from {image_path}: {e}")
        return None


def read_video_frames(video_path: str, frame_num: int, sampling_rate: int = 3, 
                     skip_num: int = 0, target_size: Tuple[int, int] = (480, 832)) -> Optional[torch.Tensor]:
    """
    Read video frames and convert to tensor.
    
    Args:
        video_path: Path to video file
        frame_num: Number of frames to extract
        sampling_rate: Frame sampling rate
        skip_num: Number of frames to skip at beginning
        target_size: Target frame size (height, width)
        
    Returns:
        Frame tensor [T, C, H, W] or None if failed
    """
    if not os.path.exists(video_path):
        logging.warning(f"Video file not found: {video_path}")
        return None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return None
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logging.info(f"Video info: frames={total_frames}, fps={fps}, size={width}x{height}")

        # Adjust sampling rate if needed
        while total_frames < frame_num * sampling_rate + skip_num:
            sampling_rate -= 1
            if sampling_rate == 0:
                logging.warning(f"Cannot extract {frame_num} frames from video")
                return None
                
        logging.info(f"Using sampling rate: {sampling_rate}")

        # Check aspect ratio compatibility
        target_aspect = target_size[1] / target_size[0]  # w/h
        video_aspect = width / height
        
        if abs(target_aspect - video_aspect) > 0.5:  # Significant aspect ratio difference
            logging.warning(f"Aspect ratio mismatch: target={target_aspect:.2f}, video={video_aspect:.2f}")
        
        # Extract frames
        frames = []
        current_frame = 0
        
        while current_frame < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame < skip_num:
                current_frame += 1
                continue
            
            if (current_frame - skip_num) % sampling_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
            current_frame += 1
            
            if len(frames) >= frame_num:
                break
        
        if len(frames) != frame_num:
            logging.warning(f"Extracted {len(frames)} frames, expected {frame_num}")
            return None
        
        # Convert to tensor
        frame_tensors = []
        for frame in frames:
            frame_tensor = transform_image_to_tensor(Image.fromarray(frame), target_size)
            frame_tensors.append(frame_tensor)
        
        return torch.stack(frame_tensors)  # [T, C, H, W]
        
    except Exception as e:
        logging.error(f"Failed to read video frames: {e}")
        return None
        
    finally:
        cap.release()


def main():
    """Main function for video generation."""
    args = parse_args()
    
    # Initialize distributed environment
    rank = int(os.getenv("RANK", 0))
    init_logging(rank)
    
    try:
        # Create generator instance
        generator = OmniVideoGenerator(args)
        generator.setup_distributed()
        
        # Synchronize random seed across processes
        if dist.is_initialized():
            base_seed = [args.base_seed] if rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            args.base_seed = base_seed[0] + rank * 1000
        
        # Load models and data
        generator.load_special_tokens()
        generator.load_unconditioned_context()
        generator.initialize_models()
        
        # Set up output directory
        output_dir = "./output"
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            
        # Generate content
        logging.info(f"Starting generation with prompt: '{args.prompt}'")
        
        # Handle size configuration
        if args.size in SIZE_CONFIGS:
            target_size = SIZE_CONFIGS[args.size]
        else:
            # Parse size string like "832*480"
            try:
                w, h = args.size.split('*')
                target_size = (int(h), int(w))  # (height, width)
            except:
                target_size = (480, 832)  # default
                logging.warning(f"Invalid size format: {args.size}, using default {target_size}")
        logging.info(f"Target size: {target_size}")
        
        prompt = args.prompt.strip()
        src_file_path = args.src_file_path
        if src_file_path and not os.path.exists(src_file_path):
            logging.error(f"Source file not found: {src_file_path}")
            return
        
        # Get AR model predictions
        ar_query = args.ar_query or "<video>\n Describe this video and its style in a very detailed manner."
        ar_caption_ids, ar_caption = generator.ar_model.generate(src_file_path, prompt)
        
        logging.info(f"task id: {ar_caption_ids}")
        # Determine task based on AR model output, this is a simple version of task detection
        gen_mode_1 = torch.any(ar_caption_ids == 128003)
        gen_mode_2 = torch.any(ar_caption_ids == 128002)
        
        if not gen_mode_1 and not gen_mode_2:
            logging.info("OmniVideo model did not suggest any generation task, it is a understanding task")
            print(f"OmniVideo model output: {ar_caption}")
            return
        
        # Set task type based on AR model output (overrides command line task)
        if (gen_mode_1 or gen_mode_2) and src_file_path:
            if src_file_path.endswith('.png') or src_file_path.endswith('.jpg'):
                task = 'i2i'
            elif src_file_path.endswith('.mp4'):
                task = 'v2v'
            else:
                logging.error("Source file type not supported, currently only support png, jpg and mp4")
                return
        elif gen_mode_1:
            task = 't2v'
        elif gen_mode_2:
            task = 't2i'
        else:
            logging.error("Could not determine appropriate task")
            return
            
        logging.info(f"Determined task: {task} (overriding command line task: {args.task})")
        
        # Update frame_num based on detected task
        if task in ['t2i', 'i2i'] and args.frame_num > 1:
            args.frame_num = 1
            logging.info("Updated frame_num to 1 for image generation task")
        elif task in ['t2v', 'v2v'] and args.frame_num == 1:
            args.frame_num = 81
            logging.info("Updated frame_num to 81 for video generation task")
        
        # Process visual input if needed
        visual_emb = None
        if task == 'i2i' and src_file_path:
            visual_emb = extract_vae_features(
                src_file_path, generator.omnivideo_x2x.vae, 
                generator.device, (target_size[1], target_size[0])
            )
            if visual_emb is not None:
                visual_emb = visual_emb[:, 0:1]
                
        elif task == 'v2v' and src_file_path:
            frames_tensor = read_video_frames(
                src_file_path, args.frame_num, args.sampling_rate, 
                args.skip_num, (target_size[1], target_size[0])
            )
            if frames_tensor is not None:
                frames_tensor = frames_tensor.to(generator.device)
                with torch.no_grad():
                    visual_emb = generator.omnivideo_x2x.vae.encode(
                        frames_tensor.transpose(0,1).unsqueeze(0)
                    )[0]
        
        # Generate embeddings
        vlm_last_hidden_states = generator.ar_model.general_emb(
            prompt, src_file_path, task_type=task
        )
        
        # logging for input shape
        logging.info(f"Input shape: {vlm_last_hidden_states.shape}")
        logging.info(f"Visual emb shape: {visual_emb.shape if visual_emb is not None else None}")
        logging.info(f"Task: {task}")
        logging.info(f"Prompt: {prompt}")
        logging.info(f"Src file path: {src_file_path if src_file_path is not None else None}")
        logging.info(f"Size: {target_size}")
        logging.info(f"Frame num: {args.frame_num}")

        # Generate content
        result = generator.omnivideo_x2x.generate(
            prompt,
            visual_emb=visual_emb,
            ar_vision_input=vlm_last_hidden_states,
            size=(target_size[0], target_size[1]),  # (width, height)
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            special_tokens=generator.special_tokens,
            classifier_free_ratio=args.classifier_free_ratio,
            unconditioned_context=generator.unconditioned_context,
            condition_mode=args.condition_mode,
            use_visual_as_input=args.use_visual_as_input,
        )
        
        if result is None:
            logging.warning("Generation failed - no output produced")
            return
            
        # Save result
        if not args.save_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')[:50]
            
            suffix = '.png' if '2i' in task else '.mp4'
            rank_suffix = f"_rank{rank}" if generator.world_size > 1 else ""
            
            filename = (f"{task}_{args.size}_{args.sample_solver}_seed{args.base_seed}_"
                       f"cfg{args.sample_guide_scale}_steps{args.sample_steps}_"
                       f"frames{args.frame_num}{rank_suffix}_{safe_prompt}_{timestamp}{suffix}")
            
            args.save_file = os.path.join(output_dir, filename)
        
        # Save based on task type
        if args.frame_num == 1:
            logging.info(f"Saving image to {args.save_file}")
            cache_image(
                tensor=result.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
        else:
            logging.info(f"Saving video to {args.save_file}")
            cache_video(
                tensor=result[None],
                save_file=args.save_file,
                fps=args.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
        
        logging.info(f"Generation completed successfully: {args.save_file}")
        
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        raise
        
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main() 