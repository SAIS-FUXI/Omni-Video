"""
VLM (Vision-Language Model) Helper Functions for OmniVideo Inference

This module provides functions for:
- Loading Qwen3-VL model and processor
- Generating video captions
- Extracting VLM features for conditioning
- Model offloading utilities
"""

import gc
import logging
import os
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


# ============================================================================
# Model Loading
# ============================================================================

def load_qwen3vl_model_and_processor(
    model_path: str,
    device: str = 'cuda',
    dtype: str = 'bf16',
    flash_attn: bool = False,
    target_short_side: int = 480,
    video_nframes: int = 6,
    device_map: str = 'auto',
) -> Tuple[Any, Any]:
    """
    Load Qwen3-VL model and processor.
    
    Args:
        model_path: Path to the Qwen3-VL model
        device: Target device ('cuda' or 'cpu')
        dtype: Data type ('fp32', 'fp16', 'bf16')
        flash_attn: Whether to use flash attention
        target_short_side: Target short side for video processing
        video_nframes: Number of frames to sample from video
        device_map: Device map for model parallelism
        
    Returns:
        Tuple of (model, processor)
    """
    logging.info(f"Loading Qwen3-VL model from {model_path}...")
    
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    torch_dtype = dtype_map[dtype]
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    if hasattr(processor, 'image_processor'):
        target_short = target_short_side
        target_long = target_short * 4
        min_pixels = target_short * target_short
        max_pixels = target_short * target_long
        
        processor.image_processor.size = {
            'shortest_edge': min_pixels,
            'longest_edge': max_pixels
        }
        processor.image_processor.min_pixels = min_pixels
        processor.image_processor.max_pixels = max_pixels
    
    if hasattr(processor, 'video_processor'):
        processor.video_processor.num_frames = video_nframes
        processor.video_processor.fps = None
        
        target_short = target_short_side
        target_long = target_short * 4
        min_pixels = target_short * target_short
        max_pixels = target_short * target_long
        processor.video_processor.size = {
            'shortest_edge': min_pixels,
            'longest_edge': max_pixels
        }
        processor.video_processor.min_pixels = min_pixels
        processor.video_processor.max_pixels = max_pixels
    
    actual_device_map = device_map if device_map else ('auto' if device == 'cuda' else device)
    
    model_kwargs = {
        'torch_dtype': torch_dtype,
        'device_map': actual_device_map,
    }
    
    if flash_attn:
        model_kwargs['attn_implementation'] = 'flash_attention_2'
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        **model_kwargs
    )
    model.eval()
    
    return model, processor


# ============================================================================
# Caption Generation
# ============================================================================

def generate_source_video_caption(
    model,
    processor,
    video_path: str,
    system_prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    video_max_duration: float = 5.0,
) -> str:
    """
    Generate fine-grained caption for source video using Qwen3-VL.
    
    Args:
        model: Qwen3-VL model
        processor: Qwen3-VL processor
        video_path: Path to source video
        system_prompt: System prompt for caption generation
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        video_max_duration: Maximum video duration in seconds
        
    Returns:
        Generated caption string
    """
    if not os.path.exists(video_path):
        return ""
    
    device = next(model.parameters()).device
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {
                    "type": "text",
                    "text": "Please provide a detailed, fine-grained caption for this video."
                }
            ]
        }
    ]
    
    chat_template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if hasattr(processor, 'video_processor'):
        if processor.video_processor.fps is None:
            chat_template_kwargs["num_frames"] = getattr(processor.video_processor, 'num_frames', 6) or 6
            chat_template_kwargs["do_sample_frames"] = True
        if hasattr(processor.video_processor, 'size'):
            chat_template_kwargs["size"] = processor.video_processor.size
    
    inputs = processor.apply_chat_template(messages, **chat_template_kwargs)
    
    model_dtype = next(model.parameters()).dtype
    processed_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if 'pixel_values' in k and v.dtype != model_dtype:
                v = v.to(model_dtype)
            processed_inputs[k] = v
        else:
            processed_inputs[k] = v
    inputs = processed_inputs
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
    
    input_len = inputs['input_ids'].shape[1]
    generated_ids_trimmed = generated_ids[:, input_len:]
    
    generated_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return generated_text.strip()


def predict_target_video_caption(
    model,
    processor,
    source_video_caption: str,
    edit_prompt: str,
    system_prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Predict target video caption based on source caption and edit prompt.
    
    Args:
        model: Qwen3-VL model
        processor: Qwen3-VL processor
        source_video_caption: Caption of the source video
        edit_prompt: Edit instruction
        system_prompt: System prompt for caption prediction
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Predicted target video caption
    """
    device = next(model.parameters()).device
    
    user_text = f"""Source video: {source_video_caption}

Edit: {edit_prompt}

Now directly describe the edited video (do not mention what was changed, just describe the final video):"""
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
        )
    
    input_len = inputs['input_ids'].shape[1]
    generated_ids_trimmed = generated_ids[:, input_len:]
    
    generated_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return generated_text.strip()


# ============================================================================
# Feature Extraction
# ============================================================================

def compute_system_prompt_drop_idx(processor, system_prompt: str) -> int:
    """
    Compute the number of system prompt tokens to drop.
    
    Args:
        processor: Qwen3-VL processor
        system_prompt: System prompt string
        
    Returns:
        Number of tokens to drop
    """
    system_prefix = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"
    tokens = processor.tokenizer(system_prefix, return_tensors="pt", add_special_tokens=False)
    drop_idx = tokens.input_ids.shape[1]
    return drop_idx


def extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
    """
    Extract valid hidden states based on attention mask.
    
    Args:
        hidden_states: Hidden states tensor
        mask: Attention mask tensor
        
    Returns:
        List of valid hidden state tensors
    """
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
    return list(split_result)


def extract_qwen3vl_features(
    model,
    processor,
    source_video_path: str,
    edit_prompt: str,
    system_prompt: str,
    video_max_duration: float = 5.0,
) -> Dict[str, Any]:
    """
    Extract Qwen3-VL features (vlm_last_hidden_states).
    
    Args:
        model: Qwen3-VL model
        processor: Qwen3-VL processor
        source_video_path: Path to source video
        edit_prompt: Edit instruction
        system_prompt: System prompt for feature extraction
        video_max_duration: Maximum video duration in seconds
        
    Returns:
        Dictionary containing extracted features
    """
    device = next(model.parameters()).device
    
    drop_idx = compute_system_prompt_drop_idx(processor, system_prompt)
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": source_video_path,
                },
                {
                    "type": "text",
                    "text": edit_prompt
                }
            ]
        }
    ]
    
    chat_template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if hasattr(processor, 'video_processor'):
        if processor.video_processor.fps is None:
            chat_template_kwargs["num_frames"] = getattr(processor.video_processor, 'num_frames', 6) or 6
            chat_template_kwargs["do_sample_frames"] = True
        feature_extraction_size = {
            'shortest_edge': 480,
            'longest_edge': 1920
        }
        chat_template_kwargs["size"] = feature_extraction_size
    
    inputs = processor.apply_chat_template(messages, **chat_template_kwargs)
    
    model_dtype = next(model.parameters()).dtype
    processed_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if 'pixel_values' in k and v.dtype != model_dtype:
                v = v.to(model_dtype)
            processed_inputs[k] = v
        else:
            processed_inputs[k] = v
    inputs = processed_inputs
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    hidden_states = outputs.hidden_states[-1]
    
    if 'attention_mask' in inputs:
        attention_mask = inputs['attention_mask']
    else:
        attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.long, device=device)
    
    split_hidden_states = extract_masked_hidden(hidden_states, attention_mask)
    valid_hidden = split_hidden_states[0]
    
    if drop_idx > 0 and valid_hidden.size(0) > drop_idx:
        valid_hidden = valid_hidden[drop_idx:]
    
    seq_len = valid_hidden.size(0)
    out_attention_mask = torch.ones(seq_len, dtype=torch.long)
    
    result = {
        "source_video_path": source_video_path,
        "edit_prompt": edit_prompt,
        "vlm_last_hidden_states": valid_hidden.cpu(),
        "attention_mask": out_attention_mask.cpu(),
        "hidden_dim": valid_hidden.size(-1),
        "seq_len": seq_len,
    }
    
    return result


def generate_caption_and_extract_features(
    model,
    processor,
    source_video_path: str,
    edit_prompt: str,
    source_caption_system_prompt: str,
    target_caption_system_prompt: str,
    feature_extraction_system_prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    video_max_duration: float = 5.0,
) -> Tuple[str, Dict[str, Any]]:
    """
    One-stop function: input source video and edit prompt, output target caption and Qwen3-VL features.
    
    Args:
        model: Qwen3-VL model
        processor: Qwen3-VL processor
        source_video_path: Path to source video
        edit_prompt: Edit instruction
        source_caption_system_prompt: System prompt for source video captioning
        target_caption_system_prompt: System prompt for target caption prediction
        feature_extraction_system_prompt: System prompt for feature extraction
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        video_max_duration: Maximum video duration in seconds
        
    Returns:
        Tuple of (target_video_caption, features_dict)
    """
    logging.info(f"Processing video: {source_video_path}")
    
    source_video_caption = generate_source_video_caption(
        model, processor, source_video_path,
        system_prompt=source_caption_system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        video_max_duration=video_max_duration
    )
    
    if not source_video_caption:
        return "", {}
    
    target_video_caption = predict_target_video_caption(
        model, processor,
        source_video_caption, edit_prompt,
        system_prompt=target_caption_system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    
    features = extract_qwen3vl_features(
        model, processor,
        source_video_path, edit_prompt,
        system_prompt=feature_extraction_system_prompt,
        video_max_duration=video_max_duration
    )
    
    features['source_video_caption'] = source_video_caption
    features['target_video_caption'] = target_video_caption
    
    return target_video_caption, features


# ============================================================================
# Offload Helpers
# ============================================================================

def offload_qwen3vl_to_cpu(model):
    """
    Offload Qwen3-VL model to CPU to free GPU memory for OmniVideo.
    
    Args:
        model: Qwen3-VL model to offload
    """
    if model is not None:
        logging.info("[Offload] Moving Qwen3-VL to CPU...")
        model.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()


def load_qwen3vl_to_gpu(model, device_map='auto'):
    """
    Move Qwen3-VL model back to GPU(s).
    
    Args:
        model: Qwen3-VL model to load
        device_map: Device map for model parallelism
    """
    if model is not None:
        logging.info(f"[Offload] Moving Qwen3-VL back to GPU (device_map={device_map})...")
        # If it was loaded with a device_map, moving it to 'cuda' usually 
        # respects the map or re-distributes if it's 'auto'.
        model.to('cuda') 
        gc.collect()
        torch.cuda.empty_cache()
