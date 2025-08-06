#!/usr/bin/env python3
"""
Video Sample Reformatting Tool

This script processes log files to extract source, target, and generated video paths,
then creates merged comparison videos showing all three videos side by side.

Usage:
    python reformat_video_sample.py --input_log /path/to/logfile.log --output_dir /path/to/output
"""

import os
import re
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import imageio
from tqdm import tqdm

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )

def parse_log_file(log_file_path):
    """
    Parse log file to extract video paths.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        list: List of dictionaries containing video paths for each sample
    """
    video_samples = []
    current_sample = {}
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Extract src_image path
        src_match = re.search(r'src_image path: (.+)', line)
        if src_match:
            current_sample['src_path'] = src_match.group(1)
        
        # Extract tgt_image path
        tgt_match = re.search(r'tgt_image path: (.+)', line)
        if tgt_match:
            current_sample['tgt_path'] = tgt_match.group(1)
        
        # Extract generated video path
        gen_match = re.search(r'Saving generated video to (.+)', line)
        if gen_match:
            current_sample['gen_path'] = gen_match.group(1)
            
            # When we have all three paths, add to samples and reset
            if 'src_path' in current_sample and 'tgt_path' in current_sample:
                video_samples.append(current_sample.copy())
                current_sample = {}
    
    logging.info(f"Found {len(video_samples)} video samples in log file")
    return video_samples

def read_video(video_path):
    """
    Read video file and return frames.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        tuple: (frames, fps, original_size)
    """
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return None, None, None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return None, None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps, (width, height)

def resize_video_frames(frames, target_size):
    """
    Resize video frames to target size.
    
    Args:
        frames (list): List of video frames
        target_size (tuple): Target (width, height)
        
    Returns:
        list: Resized frames
    """
    if not frames:
        return frames
    
    target_width, target_height = target_size
    resized_frames = []
    
    for frame in frames:
        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        resized_frames.append(resized_frame)
    
    return resized_frames

def merge_videos_horizontally(src_frames, tgt_frames, gen_frames):
    """
    Merge three video frame sequences horizontally.
    
    Args:
        src_frames (list): Source video frames
        tgt_frames (list): Target video frames  
        gen_frames (list): Generated video frames
        
    Returns:
        list: Merged video frames
    """
    # Find the minimum number of frames across all videos
    min_frames = min(len(src_frames), len(tgt_frames), len(gen_frames))
    
    if min_frames == 0:
        logging.error("One or more videos have no frames")
        return []
    
    merged_frames = []
    
    for i in range(min_frames):
        # Concatenate frames horizontally
        merged_frame = np.hstack([src_frames[i], tgt_frames[i], gen_frames[i]])
        merged_frames.append(merged_frame)
    
    return merged_frames

def merge_videos_horizontally_couple(src_frames, gen_frames):
    """
    Merge three video frame sequences horizontally.
    
    Args:
        src_frames (list): Source video frames
        tgt_frames (list): Target video frames  
        gen_frames (list): Generated video frames
        
    Returns:
        list: Merged video frames
    """
    # Find the minimum number of frames across all videos
    min_frames = min(len(src_frames), len(gen_frames))
    
    if min_frames == 0:
        logging.error("One or more videos have no frames")
        return []
    
    merged_frames = []
    
    for i in range(min_frames):
        # Concatenate frames horizontally
        merged_frame = np.hstack([src_frames[i], gen_frames[i]])
        merged_frames.append(merged_frame)
    
    return merged_frames

def save_video(frames, output_path, fps=30.0):
    """
    Save video frames to file using imageio.
    
    Args:
        frames (list): Video frames to save (as numpy arrays in BGR format)
        output_path (str): Output video file path
        fps (float): Frames per second
    """
    if not frames:
        logging.error("No frames to save")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Convert BGR frames to RGB for imageio (opencv uses BGR, imageio expects RGB)
        rgb_frames = []
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(rgb_frame)
        
        # Write video using imageio
        with imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8) as writer:
            for frame in rgb_frames:
                writer.append_data(frame)
        
        logging.info(f"Saved merged video to: {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save video to {output_path}: {e}")
        return False

def process_video_sample(sample, output_dir=None):
    """
    Process a single video sample to create merged comparison video.
    
    Args:
        sample (dict): Dictionary containing video paths
        output_dir (str, optional): Override output directory
        
    Returns:
        bool: Success status
    """
    src_path = sample['src_path']
    tgt_path = sample['tgt_path']
    gen_path = sample['gen_path']
    
    logging.info(f"Processing sample:")
    logging.info(f"  Source: {src_path}")
    logging.info(f"  Target: {tgt_path}")
    logging.info(f"  Generated: {gen_path}")
    
    # Read all three videos
    src_frames, src_fps, src_size = read_video(src_path)
    tgt_frames, tgt_fps, tgt_size = read_video(tgt_path)
    gen_frames, gen_fps, gen_size = read_video(gen_path)
    
    # Check if all videos were loaded successfully
    if None in [src_frames, tgt_frames, gen_frames]:
        logging.error("Failed to load one or more videos")
        return False
    
    # Use generated video size as target size
    target_size = gen_size
    target_fps = gen_fps if gen_fps > 0 else 30.0
    
    logging.info(f"Target size: {target_size}, FPS: {target_fps}")
    
    # Resize source and target videos to match generated video
    src_frames_resized = resize_video_frames(src_frames, target_size)
    tgt_frames_resized = resize_video_frames(tgt_frames, target_size)
    
    # Merge videos horizontally
    merged_frames = merge_videos_horizontally(src_frames_resized, tgt_frames_resized, gen_frames)
    
    if not merged_frames:
        logging.error("Failed to merge videos")
        return False
    
    # Determine output path
    if output_dir:
        gen_filename = os.path.basename(gen_path)
        output_path = os.path.join(output_dir, gen_filename.replace('.mp4', '_reformat.mp4'))
    else:
        output_path = gen_path.replace('.mp4', '_reformat.mp4')
    
    # Save merged video
    success = save_video(merged_frames, output_path, target_fps)
    
    return success

## add a function to concat two videos horizontally and save the result
def process_video_couple(sample, output_dir=None):
    src_path = sample['src_path']
    gen_path = sample['gen_path']

    src_frames, src_fps, src_size = read_video(src_path)
    gen_frames, gen_fps, gen_size = read_video(gen_path)

    if None in [src_frames, gen_frames]:
        logging.error("Failed to load one or more videos")
        return False

    target_size = gen_size
    target_fps = gen_fps if gen_fps > 0 else 30.0

    src_frames_resized = resize_video_frames(src_frames, target_size)

    merged_frames = merge_videos_horizontally_couple(src_frames_resized, gen_frames)

    if not merged_frames:
        logging.error("Failed to merge videos")
        return False
    
    if output_dir:
        gen_filename = os.path.basename(gen_path)
        output_path = os.path.join(output_dir, gen_filename.replace('.mp4', '_couple.mp4'))
    else:
        output_path = gen_path.replace('.mp4', '_couple.mp4')
    
    success = save_video(merged_frames, output_path, target_fps)
    return success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Reformat video samples from log files')
    parser.add_argument('--input_log', type=str, required=True,
                       help='Path to input log file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (optional, defaults to same as generated videos)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Check if input log file exists
    if not os.path.exists(args.input_log):
        logging.error(f"Input log file not found: {args.input_log}")
        return
    
    # Parse log file
    video_samples = parse_log_file(args.input_log)
    
    if not video_samples:
        logging.error("No video samples found in log file")
        return
    
    # Limit samples if specified
    if args.max_samples:
        video_samples = video_samples[:args.max_samples]
        logging.info(f"Processing first {len(video_samples)} samples")
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Output directory: {args.output_dir}")
    
    # Process each video sample
    success_count = 0
    total_count = len(video_samples)
    
    for i, sample in enumerate(tqdm(video_samples, desc="Processing videos")):
        logging.info(f"\n--- Processing sample {i+1}/{total_count} ---")
        
        try:
            success = process_video_sample(sample, args.output_dir)
            if success:
                success_count += 1
            else:
                logging.error(f"Failed to process sample {i+1}")
        except Exception as e:
            logging.error(f"Error processing sample {i+1}: {e}")
    
    logging.info(f"\n=== Processing Complete ===")
    logging.info(f"Successfully processed: {success_count}/{total_count} samples")
    logging.info(f"Failed: {total_count - success_count}/{total_count} samples")

if __name__ == "__main__":
    main()

