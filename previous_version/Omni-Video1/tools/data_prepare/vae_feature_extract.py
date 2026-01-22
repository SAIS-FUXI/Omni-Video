import argparse
import logging
import os
import sys
import pickle
import json
import warnings
from tqdm import tqdm

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.cuda.amp as amp
import numpy as np
from PIL import Image
import cv2

from wan.configs import WAN_CONFIGS
from wan.modules.vae import WanVAE
from wan.modules.t5 import T5EncoderModel
from wan.utils.utils import str2bool

warnings.filterwarnings('ignore')

def _init_logging():
    """Initialize logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)]
    )

def _parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract video latent features and save to pickle files"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(WAN_CONFIGS.keys()),
        help="Task name, choose from WAN_CONFIGS"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Model checkpoint directory path"
    )
    parser.add_argument(
        "--video_list_path",
        type=str,
        required=True,
        help="Input video file list path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory path"
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="Number of frames to extract, should be 4n+1"
    )
    parser.add_argument(
        "--t5_cpu",
        type=str2bool,
        default=False,
        help="Whether to place T5 model on CPU"
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=3,
        help="Video frame sampling rate"
    )
    parser.add_argument(
        "--skip_num",
        type=int,
        default=0,
        help="Skip first N frames"
    )
    parser.add_argument(
        "--target_size",
        type=str,
        default="480,832",
        help="Target size (height, width)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    assert os.path.exists(args.ckpt_dir), f"Checkpoint directory does not exist: {args.ckpt_dir}"
    assert os.path.exists(args.video_list_path), f"Video file does not exist: {args.video_list_path}"
    assert args.task in WAN_CONFIGS, f"Unsupported task: {args.task}"
    assert args.frame_num % 4 == 1, f"Frame number should be 4n+1, current: {args.frame_num}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args

def get_video_frames_cnt(video_path):
    """
    Get the frame count of a video

    Args:
        video_path: Video file path

    Returns:
        frame_count: Number of frames in the video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = 0 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1   
    cap.release()

    return frame_count

def transform_frames_to_tensor(frames, target_size=(480, 832)):
    """
    Transform a list of frames (PIL Images or numpy arrays) to tensors with resize and center crop
    
    Args:
        frames: List of frames (PIL Images or numpy arrays)
        target_size: Target size (height, width) for the output tensors
        
    Returns:
        torch.Tensor: Tensor of shape [T, C, H, W] normalized to [-1, 1]
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

def read_video_frames(video_path, frame_num, sampling_rate=3, skip_num=0, target_size=(480, 832)):
    """
    Read video and extract specified number of frames
    
    Args:
        video_path: Video file path
        frame_num: Number of frames to extract
    
    Returns:
        frames: Extracted frames, shape [frame_num, H, W, 3]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video information
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = get_video_frames_cnt(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logging.info(f"Video info: Total frames={total_frames}, FPS={fps}, Resolution={width}x{height}")
    
    # Calculate sampling interval
    # if total_frames < frame_num * sampling_rate + skip_num:
    #     return None

    # reduce sampling_rate if total_frames < frame_num * sampling_rate + skip_num
    while total_frames < frame_num * sampling_rate + skip_num:
        sampling_rate -= 1
        if sampling_rate <= 0:
            logging.warning(f"Insufficient video frames, skipping: {video_path}")
            return None
        logging.info(f"Reducing sampling rate to: {sampling_rate}")
    
    # Skip videos with aspect ratio differences
    if (target_size[0] > target_size[1] and height < width) or (target_size[0] < target_size[1] and height > width):
        logging.info(f'target_size {target_size} cur video: {height} {width}, so skip')
        return None
    
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
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        current_frame += 1
        
        # Exit loop if enough frames collected
        if len(frames) >= frame_num:
            break
    
    cap.release()
    
    # Ensure we got enough frames
    if len(frames) != frame_num:
        # Return None instead of throwing exception if insufficient frames
        return None
    
    frames = transform_frames_to_tensor(frames, target_size) # [T, C, H, W]

    return frames

def extract_features(args):
    """Extract video features and save"""    
    # Get configuration
    cfg = WAN_CONFIGS[args.task]

    # Initialize distributed environment
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
        
    if world_size > 1:  # Initialize distributed environment
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size
        )
    
    # Initialize VAE model
    logging.info(f"Initializing VAE model...")
    vae = WanVAE(
        vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint),
        device=device
    )
    
    # Initialize T5 text encoder
    logging.info(f"Initializing T5 text encoder...")
    text_device = torch.device('cpu') if args.t5_cpu else device
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=text_device,
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer)
    )
    
    # Read video frames
    with open(args.video_list_path, 'r') as f:
        video_list = json.load(f)
        target_size = [int(it) for it in args.target_size.split(',')]
        sampling_rate = int(args.sampling_rate)
        extracted_num = 0
    
        for idx, item in enumerate(video_list):
            try:
                if idx % world_size != rank:
                    continue
                
                if idx % 500 == 0:
                    logging.info(f"Processing video {idx}")
                
                video_path = item['video']
                prompt = item['conversations'][1]['value']
                prompt = prompt.replace('\n', ' ')

                if not os.path.exists(video_path):
                    logging.warning(f"Video file does not exist, skipping: {video_path}")
                    continue
                        
                # Create output filename
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                if len(video_name) < 4:
                    video_name = f"{video_path.split('/')[-2]}_{video_name}"
                output_path = os.path.join(args.output_dir, f"{video_name}_features.pkl")
                if os.path.exists(output_path):
                    logging.info(f"{output_path} already exists, skipping") # skip if already exist
                    continue
                
                logging.info(f"Reading video: {video_path}")
                frames_tensor = read_video_frames(video_path, args.frame_num, sampling_rate, args.skip_num, target_size)
                if frames_tensor is None:
                    logging.warning(f"Failed to read video, skipping: {video_path}")
                    continue

                # Process text prompt
                if prompt and len(prompt) > 10:
                    logging.info(f"Encoding text prompt: {prompt}")
                    with torch.no_grad():
                        text_emb = text_encoder([prompt], text_device)
                        if not args.t5_cpu:
                            text_emb = [t.cpu() for t in text_emb]
                else:
                    logging.info(f"{video_path} no text prompt provided, skipping")
                    # text_emb = None
                    continue

                # Convert frames to tensor and preprocess
                logging.info("Preprocessing video frames...")

                # Extract features using VAE
                logging.info("Extracting latent features using VAE...")
                with torch.no_grad():
                    # Move frames to device
                    frames_tensor = frames_tensor.to(device)
                    # Extract features [1, C, T, H, W]
                    latent_feature = vae.encode(frames_tensor.transpose(0,1).unsqueeze(0))
                    # Move back to CPU
                    latent_feature = latent_feature[0].cpu()


                # Save features to pickle file
                logging.info(f"Saving features to: {output_path}")
                data = {
                    'latent_feature': latent_feature,
                    'prompt': prompt,
                    'text_emb': text_emb,
                    'video_path': video_path,
                    'frame_num': args.frame_num,
                }

                with open(output_path, 'wb') as f:
                    pickle.dump(data, f)
                extracted_num += 1
            except Exception as e:
                logging.error(f"Error processing video: {e}")
                continue

    logging.info("Feature extraction completed!")

if __name__ == "__main__":
    _init_logging()
    args = _parse_args()
    extract_features(args)
