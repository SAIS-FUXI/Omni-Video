"""
OmniVideo Finetuning Script

A professional training script for the OmniVideo mixed condition model supporting
multiple tasks (T2I, I2I, T2V) with DeepSpeed optimization and comprehensive logging.

Author: OmniVideo Team
License: Apache 2.0
"""

import argparse
import json
import logging
import math
import os
import pickle as pkl
import sys
import types
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import deepspeed
import torch
import torch.distributed as dist
import yaml
from easydict import EasyDict
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

# Project imports
from nets.omni.datasets.omnivideo_dataset_patched import create_omnivideo_dataloader
from nets.omni.modules.omni_video_model import OmniVideoMixedConditionModel
from nets.omni.modules.schedulers.flow_match import FlowMatchScheduler
from nets.third_party.wan.utils.utils import str2bool

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
SPECIAL_TOKENS_SUBDIR = "special_tokens"
UNCOND_CONTEXT_SUBDIR = "unconditioned_context"

def _init_logging(rank: int, args: Optional[Any] = None) -> None:
    """
    Initialize distributed logging configuration.
    
    Args:
        rank: Process rank (only rank 0 logs INFO level)
        args: Optional arguments (reserved for future use)
    """
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [Rank %(rank)d] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
            force=True
        )
        # Add rank to all log records
        logging.getLogger().addFilter(lambda record: setattr(record, 'rank', rank) or True)
    else:
        logging.basicConfig(level=logging.ERROR, force=True)

def str2tuple(v: str) -> Tuple[int, ...]:
    """
    Convert string representation to tuple of integers.
    
    Args:
        v: String representation (e.g., '1,2,2' or '(1,2,2)')
        
    Returns:
        Tuple of integers
        
    Examples:
        >>> str2tuple('1,2,2')
        (1, 2, 2)
        >>> str2tuple('(1,2,2)')
        (1, 2, 2)
    """
    v = v.strip()
    if v.startswith('(') and v.endswith(')'):
        v = v[1:-1]
    return tuple(int(x.strip()) for x in v.split(','))

def load_and_merge_config(yaml_path: str, cmd_args: Optional[argparse.Namespace] = None) -> EasyDict:
    """
    Load YAML configuration and merge with command line arguments.
    
    Args:
        yaml_path: Path to the YAML configuration file
        cmd_args: Command line arguments to override config values
        
    Returns:
        Merged configuration as EasyDict
        
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    # Check if file exists
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    # Add the parent directory to sys.path
    parent_dir = os.path.dirname(os.path.dirname(yaml_path))  # Go up two levels
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        # Load YAML file
        with open(yaml_path, 'r') as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML file: {e}")
        
        # Convert to EasyDict
        def dict_to_easydict(d):
            if isinstance(d, dict):
                return EasyDict({k: dict_to_easydict(v) for k, v in d.items()})
            elif isinstance(d, list):
                return [dict_to_easydict(item) if isinstance(item, (dict, list)) else item for item in d]
            elif isinstance(d, str):
                # Try to convert string to number
                try:
                    # Handle scientific notation (e.g., 1e-4)
                    if 'e' in d.lower():
                        return float(d)
                    # Handle regular floats
                    elif '.' in d:
                        return float(d)
                    # Handle integers
                    elif d.isdigit():
                        return int(d)
                    # Handle negative integers
                    elif d.startswith('-') and d[1:].isdigit():
                        return int(d)
                    # Handle negative floats
                    elif d.startswith('-') and '.' in d[1:]:
                        return float(d)
                    # Handle negative scientific notation
                    elif d.startswith('-') and 'e' in d[1:].lower():
                        return float(d)
                except ValueError:
                    pass
            return d
        
        config = dict_to_easydict(config_dict)
        
        # If no command line arguments provided, return just the YAML config
        if cmd_args is None:
            return config
        
        # Only merge basic command line arguments
        basic_args = ['output_dir', 'resume_from', 'ckpt_dir']
        for arg_name in basic_args:
            arg_value = getattr(cmd_args, arg_name, None)
            if arg_value is not None:
                # Ensure the attribute exists in the config
                if not hasattr(config, arg_name):
                    setattr(config, arg_name, None)
                # Set the value
                setattr(config, arg_name, arg_value)
                print(f"Overriding config value for {arg_name} with command-line value: {arg_value}")
            else:
                # If the argument is not provided, ensure it exists in config with None
                if not hasattr(config, arg_name):
                    setattr(config, arg_name, None)
        
        return config
    finally:
        # Clean up sys.path
        if parent_dir in sys.path:
            sys.path.remove(parent_dir)

def save_config_to_yaml(args: EasyDict, output_path: str) -> None:
    """
    Save configuration to YAML file for reproducibility.
    
    Args:
        args: Configuration object to save
        output_path: Path where to save the YAML file
    """
    def easydict_to_dict(ed: Union[EasyDict, Any]) -> Any:
        """Recursively convert EasyDict to regular dict."""
        if isinstance(ed, EasyDict):
            return {key: easydict_to_dict(value) for key, value in ed.items()}
        elif isinstance(ed, (list, tuple)):
            return [easydict_to_dict(item) for item in ed]
        elif isinstance(ed, dict):
            return {k: easydict_to_dict(v) for k, v in ed.items()}
        elif isinstance(ed, (int, float, str, bool, type(None))):
            return ed
        else:
            return str(ed)  # Convert other types to string
    
    try:
        config_dict = easydict_to_dict(args)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, 
                     default_flow_style=False, 
                     sort_keys=False,
                     allow_unicode=True,
                     indent=2)
        
        logging.info(f"Training configuration saved to {output_path}")
    except Exception as e:
        logging.warning(f"Failed to save configuration: {e}")

def _parse_args() -> EasyDict:
    """
    Parse command line arguments and merge with YAML configuration.
    
    Returns:
        Merged configuration as EasyDict
        
    Raises:
        ValueError: If required parameters are missing
    """
    parser = argparse.ArgumentParser(
        description="Train OmniVideoMixedConditionModel with multiple tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints and logs. Overrides config if set."
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from. Overrides config if set."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Directory containing base WAN model checkpoints. Overrides config if set."
    )

    cmd_args = parser.parse_args()
    
    # Load and merge configurations
    args = load_and_merge_config(cmd_args.config, cmd_args)
    
    # Ensure required parameters exist
    if not hasattr(args, 'output_dir') or args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output/training_{timestamp}"
    
    if not hasattr(args, 'ckpt_dir') or args.ckpt_dir is None:
        raise ValueError("ckpt_dir must be specified either in config or via command line.")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get the original YAML filename and create the output path
    original_yaml_name = os.path.basename(cmd_args.config)
    args_file = os.path.join(args.output_dir, original_yaml_name)
    
    # Save the configuration
    save_config_to_yaml(args, args_file)
    
    return args

def load_uncond_feature(
    unconditioned_context_path: str, 
    precision_dtype: torch.dtype, 
    device: torch.device
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load unconditioned context features for classifier-free guidance.
    
    Args:
        unconditioned_context_path: Path to pickle file containing unconditioned context
        precision_dtype: Target data type for tensor conversion
        device: Target device for tensors
        
    Returns:
        Dictionary with 'uncond_context' and 'uncond_ar_vision' keys, or None if loading fails
    """
    try:
        with open(unconditioned_context_path, 'rb') as f:
            pstate = pkl.load(f)
        
        # Validate required keys
        required_keys = ['text_emb', 'vlm_last_hidden_states']
        for key in required_keys:
            if key not in pstate:
                raise KeyError(f"Required key '{key}' not found in {unconditioned_context_path}")
        
        # Process text embeddings
        unconditioned_t5 = pstate['text_emb'][0].to(precision_dtype).to(device)
        if unconditioned_t5.dim() < 2:
            unconditioned_t5 = unconditioned_t5.unsqueeze(0)
        
        # Process vision embeddings
        uncond_ar_vision = pstate['vlm_last_hidden_states'].to(precision_dtype).to(device)
        
        unconditioned_context = {
            'uncond_context': unconditioned_t5, 
            'uncond_ar_vision': uncond_ar_vision
        }
        
        logging.info(
            f"Loaded unconditioned context - T5: {unconditioned_t5.shape}, "
            f"Vision: {uncond_ar_vision.shape}"
        )
        
        return unconditioned_context
        
    except Exception as e:
        logging.error(f"Failed to load unconditioned context from {unconditioned_context_path}: {e}")
        return None


def create_dataloaders(
    args: EasyDict, 
    rank: int, 
    world_size: int
) -> Tuple[Dict[str, Any], Dict[str, int], int]:
    """
    Create dataloaders for all configured tasks.
    
    Args:
        args: Configuration containing dataloader specifications
        rank: Process rank for distributed training
        world_size: Total number of processes
        
    Returns:
        Tuple of (dataloaders dict, dataset sizes dict, total batch size)
        
    Raises:
        ValueError: If no dataloaders are configured
    """
    if not hasattr(args.training, 'dataloaders') or not args.training.dataloaders:
        raise ValueError("Configuration must contain 'dataloaders' for multi-task training")

    dataloaders = {}
    dataset_sizes = {}
    total_batch_size = 0

    for task_name, task_config in args.training.dataloaders.items():
        data_file = task_config.get('data_file')
        if not data_file or not os.path.exists(data_file):
            logging.warning(f"Data file not found for task '{task_name}': {data_file}")
            continue
            
        batch_size = task_config.get('batch_size', args.training.hyperparameters.batch_size)
        num_workers = task_config.get('num_workers', 4)
        shuffle = task_config.get('shuffle', True)
        
        logging.info(f"Creating dataloader for task '{task_name}' - file: {data_file}")
        
        try:
            dataloader = create_omnivideo_dataloader(
                file_path=data_file,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                distributed=dist.is_initialized(),
                rank=rank, 
                world_size=world_size
            )
            
            dataloaders[task_name] = dataloader
            dataset_sizes[task_name] = len(dataloader)
            total_batch_size += batch_size
            
            logging.info(
                f"Created dataloader for '{task_name}': {len(dataloader)} batches, "
                f"batch_size={batch_size}"
            )
            
        except Exception as e:
            logging.error(f"Failed to create dataloader for task '{task_name}': {e}")
            continue
    
    if not dataloaders:
        raise ValueError("No valid dataloaders were created")
    
    return dataloaders, dataset_sizes, total_batch_size

def process_batch(
    batch: Dict[str, Any], 
    task_name: str, 
    device: torch.device, 
    args: Optional[EasyDict] = None, 
    rank: int = 0, 
    epoch: int = 0, 
    step_idx: int = 0
) -> Tuple[Optional[torch.Tensor], ...]:
    """
    Process a batch for a specific task, handling different data formats.
    
    Args:
        batch: Raw batch data from dataloader
        task_name: Name of the task (t2i, i2i, t2v, etc.)
        device: Target device for tensors
        args: Configuration object
        rank: Process rank for logging
        epoch: Current epoch for logging
        step_idx: Current step for logging
        
    Returns:
        Tuple of processed tensors: (videos, context, aligned_emb, ar_vision_input, visual_emb, ref_images)
    """
    # Extract aligned embeddings (SIGLIP features)
    aligned_emb = None
    for key in ['siglip2_img_pooled_output_tgt', 'aligned_emb', 'siglip2_feature']:
        if key in batch:
            aligned_emb = batch[key]
            break

    # Extract AR vision input
    ar_vision_input = batch.get('vlm_last_hidden_states', None)
    if ar_vision_input is not None:
        ar_vision_input = [it.to(device) for it in ar_vision_input]
    
    # Extract text context and prompts
    context = batch.get('text_emb', None)
    prompts = batch.get('prompt', ['N/A'])
    
    # Extract video latents
    tgt_videos = batch.get('latent_feature_tgt', None)
    src_videos = batch.get('latent_feature', None)
    ref_images = batch.get('ref_images', None)
    
    # Move to device if not None
    if tgt_videos is not None:
        tgt_videos = tgt_videos.to(device)
    if src_videos is not None:
        src_videos = src_videos.to(device)
    if ref_images is not None:
        ref_images = ref_images.to(device)
    
    # Task-specific logic for determining input/output videos
    videos = src_videos if tgt_videos is None else tgt_videos
    visual_emb = src_videos if tgt_videos is not None else None
    
    # Handle i2v tasks - use first frame as reference if not provided
    if ref_images is None and 'i2v' in task_name and videos is not None:
        ref_images = videos[:, :, 0:1]  # [B, C, 1, H, W]
    
    # Validate we have valid video data
    if videos is None:
        logging.warning(f"Task {task_name} batch has no video data, skipping")
        return None, None, None, None, None, None
    
    # Log batch info on first step of first epoch (rank 0 only)
    if step_idx == 0 and epoch == 0 and rank == 0:
        logging.info(f"Task '{task_name}' batch info:")
        logging.info(f"  Batch size: {len(prompts)}")
        logging.info(f"  Sample prompt: {prompts[0]}")
        logging.info(f"  Video shape: {videos.shape}")
        logging.info(f"  Context: {len(context) if context else 'None'}")
        logging.info(f"  Aligned emb: {aligned_emb.shape if aligned_emb is not None else 'None'}")
    
    # Move remaining data to device and handle context replacement
    videos = videos.to(device)
    
    # Check if we should replace context with adapter output
    if args and hasattr(args.training, 'model_settings'):
        replace_context = getattr(args.training.model_settings, 'replace_context_with_adapter', False)
        if replace_context:
            context = None
    
    # Move context to device
    if context is not None:
        context = [it.to(device) for it in context]
    
    # Move aligned embeddings to device
    if aligned_emb is not None:
        if isinstance(aligned_emb, list):
            aligned_emb = [it.to(device) for it in aligned_emb]
        else:
            aligned_emb = aligned_emb.to(device)
    
    return videos, context, aligned_emb, ar_vision_input, visual_emb, ref_images

def train(args: EasyDict) -> None:
    """
    Main training function for multi-task OmniVideo model.
    
    Args:
        args: Configuration object containing all training parameters
    """
    # Initialize distributed training if needed
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    # Setup device
    device = torch.device(f"cuda:{local_rank}")
    print(f'Using device {device}', flush=True)
        
    # Initialize distributed environment
    deepspeed.init_distributed()
    
    # Log environment information
    logging.info("=== Training Environment ===")
    logging.info(f"Rank: {rank}/{world_size}, Local Rank: {local_rank}")
    logging.info(f"Device: {device}")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU count: {torch.cuda.device_count()}")
    
    # Extract model configuration
    num_train_timesteps = args.model.num_train_timesteps
    param_dtype = getattr(torch, args.model.param_dtype)
    patch_size = args.model.transformer.get('patch_size', (1, 2, 2))
    
    # Determine precision type
    precision_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32
    }
    precision_dtype = precision_map.get(
        args.training.precision.mixed_precision, 
        torch.float32
    )
    
    logging.info(f"Training precision: {precision_dtype}")
    logging.info(f"Parameter dtype: {param_dtype}")
    logging.info(f"Patch size: {patch_size}")
    
    # Initialize OmniVideoMixedConditionModel with updated parameters
    model = OmniVideoMixedConditionModel.from_pretrained(
        wan_ckpt_dir=args.ckpt_dir,
        adapter_ckpt_dir=None,
        vision_head_ckpt_dir=None, 
        learnable_query_length=args.model.get('ar_vision_head', {}).get('learnable_query_length', 4),
        adapter_in_channels=args.model.adapter.in_channels,
        adapter_out_channels=args.model.adapter.out_channels,
        adapter_query_length=args.model.adapter.query_length,
        device_id=local_rank,
        rank=rank,
        dit_fsdp=args.training.model_settings.dit_fsdp,
        use_usp=args.training.model_settings.use_usp,
        use_visual_context_adapter=args.training.model_settings.train_visual_context_adapter,
        visual_context_adapter_patch_size=args.model.visual_context_adapter.visual_context_adapter_patch_size,  # Updated to use new config path
        max_context_len=args.training.model_settings.max_context_len,
    )
    
    # Set training mode for model components
    if args.training.model_settings.train_wan_model:
        model.wan_model.train().requires_grad_(True)
        logging.info("WanModel parameters are unfrozen and will be trained")
    else:
        model.wan_model.eval().requires_grad_(False)
        logging.info("WanModel parameters are frozen and will not be trained")
    
    if args.training.model_settings.train_adapter:
        model.adapter.train().requires_grad_(True)
        logging.info("Adapter parameters are unfrozen and will be trained")
    else:
        model.adapter.eval().requires_grad_(False)
        logging.info("Adapter parameters are frozen and will not be trained")

    if args.training.model_settings.train_visual_context_adapter and model.visual_context_adapter is not None:
        model.visual_context_adapter.train().requires_grad_(True)
        logging.info("Visual Context Adapter parameters are unfrozen and will be trained")
    elif model.visual_context_adapter is not None:
        model.visual_context_adapter.eval().requires_grad_(False)
        logging.info("Visual Context Adapter parameters are frozen and will not be trained")
    
    if args.training.model_settings.train_ar_vision_head: 
        model.ar_vision_head.train().requires_grad_(True)
        logging.info("AR Vision Head parameters are unfrozen and will be trained")
    else:
        model.ar_vision_head.eval().requires_grad_(False)
        logging.info("AR Vision Head parameters are frozen and will not be trained")

    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        total_params = sum(p.numel() for p in trainable_params)
        logging.info(f"Total trainable parameters: {total_params:,}")
        # adapter parameters
        adapter_params = [p for p in model.adapter.parameters()]
        total_adapter_params = sum(p.numel() for p in adapter_params)
        logging.info(f"Total adapter parameters: {total_adapter_params:,}")

    # Load special token embeddings
    special_tokens = None
    if args.training.special_tokens.enabled:
        special_tokens_path = os.path.join(args.ckpt_dir, SPECIAL_TOKENS_SUBDIR, 'tokens.pkl')
        if os.path.exists(special_tokens_path):
            try:
                with open(special_tokens_path, 'rb') as f:
                    special_tokens = pkl.load(f)
                assert isinstance(special_tokens, dict), "Special tokens must be a dictionary"
                
                for key, value in special_tokens.items():
                    special_tokens[key] = value.to(precision_dtype).to(device)
                
                logging.info(f"Loaded special token embeddings: {list(special_tokens.keys())}")
            except Exception as e:
                logging.warning(f"Failed to load special tokens: {e}")
    
    # Load unconditioned context for classifier-free guidance
    unconditioned_context = None
    if args.training.classifier_free.ratio > 0:
        unconditioned_context_path = os.path.join(args.ckpt_dir, UNCOND_CONTEXT_SUBDIR, 'context.pkl')
        if os.path.exists(unconditioned_context_path):
            unconditioned_context = load_uncond_feature(
                unconditioned_context_path, precision_dtype, device
            )
            if unconditioned_context is None:
                args.training.classifier_free.ratio = 0.0
                logging.warning("Disabling classifier-free guidance due to loading failure")
        else:
            args.training.classifier_free.ratio = 0.0
            logging.warning(f"Unconditioned context not found: {unconditioned_context_path}")
    
    if args.training.classifier_free.ratio > 0:
        logging.info(f"Classifier-free guidance enabled with ratio: {args.training.classifier_free.ratio}")
    else:
        logging.info("Classifier-free guidance disabled")

    if args.training.optimization.gradient_checkpointing and args.training.model_settings.train_wan_model:
        model.enable_gradient_checkpointing()

    # Initialize Flow Match scheduler for training
    flow_scheduler = FlowMatchScheduler(
        num_train_timesteps=args.model.num_train_timesteps,
        num_inference_steps=args.model.num_train_timesteps,
        shift=args.training.model_settings.flow_shift,
        sigma_min=0.0,
        extra_one_step=True,
        is_training=True)
    
    model.to(device)

    sp_size = 1
    
    # Create dataloaders for all tasks
    dataloaders, dataset_sizes, total_batch_size = create_dataloaders(args, rank, world_size)
    if not dataloaders:
        raise ValueError("No dataloaders were created. Check dataloaders configuration.")
    
    # Extract weights for each task (now used for loss weighting, not sampling)
    task_weights = {task: config.get('weight', 1.0) 
                   for task, config in args.training.dataloaders.items()}
    
    # Calculate steps per epoch
    steps_per_epoch = min(dataset_sizes.values())
    total_training_steps = args.training.hyperparameters.num_epochs * steps_per_epoch
    
    logging.info("=== Training Schedule ===")
    logging.info(f"Steps per epoch: {steps_per_epoch}")
    logging.info(f"Total epochs: {args.training.hyperparameters.num_epochs}")
    logging.info(f"Total training steps: {total_training_steps}")
    logging.info(f"Gradient accumulation steps: {args.training.hyperparameters.gradient_accumulation_steps}")
    
    # Initialize optimizer
    optimizer = AdamW(
        trainable_params, 
        lr=float(args.training.hyperparameters.learning_rate),
        weight_decay=float(args.training.hyperparameters.weight_decay)
    )
    
    logging.info(f"Optimizer: AdamW (lr={args.training.hyperparameters.learning_rate}, "
                f"weight_decay={args.training.hyperparameters.weight_decay})")

    # Initialize learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.training.hyperparameters.num_warmup_steps,
        num_training_steps=total_training_steps
    )
    
    logging.info(f"Scheduler: Cosine with warmup ({args.training.hyperparameters.num_warmup_steps} warmup steps)")
    
    # Setup for DeepSpeed
    if args.training.deepspeed.config_path is not None and os.path.exists(args.training.deepspeed.config_path):
        with open(args.training.deepspeed.config_path, 'r') as f:
            ds_config = json.load(f)
        logging.info(f"Loaded DeepSpeed config from {args.training.deepspeed.config_path}")
        # Update the DeepSpeed config with the total batch size
        ds_config['train_micro_batch_size_per_gpu'] = total_batch_size
        ds_config['train_batch_size'] = total_batch_size * dist.get_world_size() * args.training.hyperparameters.gradient_accumulation_steps if dist.is_initialized() else total_batch_size * args.training.hyperparameters.gradient_accumulation_steps
        
        if precision_dtype == torch.bfloat16:
            if 'bf16' not in ds_config:
                ds_config['bf16'] = {'enabled': True}
            else:
                ds_config['bf16']['enabled'] = True 
        elif precision_dtype == torch.float16:
            if 'fp16' not in ds_config:
                ds_config['fp16'] = {'enabled': True}
            else:
                ds_config['fp16']['enabled'] = True
        
    # Save the generated config for reference
    if rank == 0:
        config_path = os.path.join(args.output_dir, "ds_config.json")
        with open(config_path, 'w') as f:
            json.dump(ds_config, f, indent=4)
        logging.info(f"Generated DeepSpeed config saved to {config_path}")
    
    if args.resume_from is not None and args.resume_from.endswith('.pt'):
        logging.info("Loading from a .pt file, so we use pytorch.")
        state_dict = torch.load(args.resume_from, map_location='cpu')
        if 'module' in state_dict:
            state_dict = state_dict['module']
        model.load_state_dict(state_dict)
        logging.info(f"Loaded checkpoint from {args.resume_from}")

    # Initialize DeepSpeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        model_parameters=None,
        config=ds_config
    )
            
    start_epoch = 0
    global_step = 0
    
    # Initialize TensorBoard writer
    writer = None
    if rank == 0 and args.training.logging.use_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
        logging.info(f"TensorBoard logging enabled: {args.output_dir}/tensorboard")
    
    # Start training
    logging.info("Starting mixed tasks training with all tasks in each iteration...")
    for epoch in range(start_epoch, args.training.hyperparameters.num_epochs):
        # Set epoch for all samplers
        for task_name, dataloader in dataloaders.items():
            if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
                dataloader.sampler.set_epoch(epoch)
                logging.info(f"Set epoch {epoch} for dataloader {task_name}")
                
        # Create iterators for all dataloaders
        iterators = {task_name: iter(dataloader) for task_name, dataloader in dataloaders.items()}
        
        # Create a progress bar based on steps_per_epoch
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{args.training.hyperparameters.num_epochs}") if rank == 0 else range(steps_per_epoch)
        
        # Track losses per task for logging
        task_losses = {task_name: [] for task_name in dataloaders.keys()}
        
        for step_idx in pbar:
            # Process a batch from each task and accumulate losses
            all_task_losses = {}
            weighted_loss = 0.0
            valid_task_count = 0
            
            for task_name, iterator in iterators.items():
                # Get a batch for the current task
                batch = next(iterator)
                
                # Process the batch to get necessary inputs
                videos, context, aligned_emb, ar_vision_input, visual_emb, ref_images = process_batch(
                    batch, task_name, device, args, rank, epoch, step_idx
                )
                
                # Skip if video data is invalid
                if videos is None:
                    continue
                
                batch_size, _, frames, height, width = videos.shape
                seq_len = math.ceil((height * width) / 
                                (patch_size[1] * patch_size[2]) * 
                                frames / sp_size) * sp_size
                
                with torch.autocast(
                    device_type='cuda', 
                    dtype=precision_dtype, 
                    enabled=args.training.precision.mixed_precision != "fp32"
                ):
                    # Uniformly sample timesteps
                    timestep = torch.randint(0, flow_scheduler.num_train_timesteps, (batch_size,))
                    t = flow_scheduler.timesteps[timestep].to(dtype=param_dtype, device=device)            

                    # Generate noise
                    noise = torch.randn_like(videos)

                    # Add noise to target video using flow matching scheduler
                    noisy_video = flow_scheduler.add_noise(videos, noise, t)

                    # Add visual embeddings as input if specified
                    if args.training.model_settings.use_visual_as_input and visual_emb is not None:
                        noisy_video = noisy_video + visual_emb

                    # Get training target (velocity field)
                    target = flow_scheduler.training_target(videos, noise, t)

                    # Get training weights for current timesteps
                    weights = flow_scheduler.training_weight(t)

                    # Forward pass using OmniVideoMixedConditionModel
                    velocity_pred = model_engine(
                        noisy_video, 
                        t=t, 
                        context=context, 
                        aligned_emb=aligned_emb,
                        ar_vision_input=ar_vision_input,
                        visual_emb=visual_emb,
                        ref_images=ref_images,
                        seq_len=seq_len,
                        special_token_dict=special_tokens,
                        classifier_free_ratio=args.training.classifier_free.ratio,
                        unconditioned_context=unconditioned_context,
                        condition_mode=args.training.model_settings.condition_mode,
                    )
                    
                    if isinstance(velocity_pred, list):
                        velocity_pred = torch.stack(velocity_pred, dim=0)

                    # Flow matching loss: weighted MSE between predicted and target velocity
                    if weights.ndim > 0:  # If weights are per-sample
                        weights = weights.view(-1, 1, 1, 1, 1).to(device)
                        task_loss = torch.mean(weights * (velocity_pred - target) ** 2)
                    else:  # If weights are scalar
                        task_loss = torch.nn.functional.mse_loss(velocity_pred, target)
                    
                    # Apply task weight to the loss
                    task_weight = task_weights.get(task_name, 1.0)
                    weighted_task_loss = task_loss * task_weight
                    model_engine.backward(weighted_task_loss)
                    
                    # Accumulate losses
                    all_task_losses[task_name] = task_loss.item()
                    weighted_loss += weighted_task_loss
                    valid_task_count += 1
                    
                    # Track loss per task for monitoring
                    task_losses[task_name].append(task_loss.item())
            
            # Skip optimization if no valid tasks were processed
            if valid_task_count == 0:
                logging.warning(f"Rank {rank}: No valid tasks processed in step {step_idx}, skipping.")
                continue
                
            # Backward and optimize with DeepSpeed
            model_engine.step()
            
            # Update learning rate scheduler
            if global_step % args.training.hyperparameters.gradient_accumulation_steps == 0:
                scheduler.step()
            
            global_step += 1
                        
            # Compute and synchronize average loss across all ranks every log_interval steps
            if step_idx % args.training.logging.log_interval == 0:
                # Compute average loss for each task
                avg_task_losses = {}
                for task, losses in task_losses.items():
                    if losses:  # If we have losses for this task
                        avg_task_losses[task] = sum(losses) / len(losses)
                
                # Compute average overall loss on current device
                local_avg_loss = weighted_loss.detach() / valid_task_count
                
                # Synchronize loss across all processes
                if dist.is_initialized():
                    dist.all_reduce(local_avg_loss, op=dist.ReduceOp.SUM)
                    local_avg_loss = local_avg_loss / dist.get_world_size()
                
                # Only print log on main process
                if rank == 0:
                    task_loss_str = ", ".join([f"{t}: {l:.4f}" for t, l in avg_task_losses.items()])
                    logging.info(f"Epoch {epoch}, Step {step_idx}, "
                                f"Avg Loss: {local_avg_loss.item():.4f}, Task Losses: {task_loss_str}, "
                                f"LR: {scheduler.get_last_lr()[0]:.6f}")
                    
                    # Reset task losses after logging
                    task_losses = {task_name: [] for task_name in dataloaders.keys()}
            
            # Update progress bar
            if isinstance(pbar, tqdm):
                individual_losses = {task: f"{loss:.4f}" for task, loss in all_task_losses.items()}
                pbar.set_postfix({"loss": weighted_loss.item() / valid_task_count, **individual_losses, "lr": f"{scheduler.get_last_lr()[0]:.6f}"})
            
            # Log to tensorboard
            if rank == 0 and args.training.logging.use_tensorboard:
                for task_name, loss in all_task_losses.items():
                    writer.add_scalar(f"Loss/{task_name}", loss, global_step)
                writer.add_scalar("Loss/overall", weighted_loss.item() / valid_task_count, global_step)
                writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)

            # Save checkpoint using DeepSpeed
            if (step_idx + 1) % args.training.logging.save_interval == 0 or (step_idx + 1) >= steps_per_epoch:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}_step_{step_idx}")
                # Save model and optimizer state using DeepSpeed's save_checkpoint method
                client_state = {
                    'epoch': epoch,
                    'step': step_idx
                }
                # all processes must call this
                model_engine.save_checkpoint(
                    save_dir=checkpoint_dir,
                    client_state=client_state,
                    tag=f"epoch_{epoch}_step_{step_idx}"
                )
                if rank == 0:
                    logging.info(f"DeepSpeed checkpoint saved to {checkpoint_dir}")
    
    logging.info("Training completed successfully!")


if __name__ == "__main__":
    rank = int(os.getenv("RANK", 0))
    _init_logging(rank)
    args = _parse_args()
    
    # Train model
    train(args)