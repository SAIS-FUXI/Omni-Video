import os
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import einops
from typing import List, Dict, Optional, Union, Tuple
import random

from .model import WanModel, WanRMSNorm
from .visual_context_adapter import VisualContextAdapter

class UnifiedWanWithMixedConditionModel(nn.Module):
    """
    A hyper model that combines DM_Adapter and WanModel.
    
    This version implements "Tight Concatenation": it only concatenates the 
    actual tokens present in the conditions without any fixed-length padding 
    or buffers within the sequence.
    """
    
    def __init__(
        self,
        wan_model_or_ckpt_dir: Union[WanModel, str, None] = None,
        wan_subfolder: Optional[str] = None,
        vlm_in_dim: int = 4096,
        precision_dtype: torch.dtype = torch.float32,
        device_id: int = 0,
        rank: int = 0,
        dit_fsdp: bool = False,
        use_usp: bool = False,
        use_visual_context_adapter: bool = False,
        visual_context_adapter_patch_size: tuple = None,
        max_context_len: int = None, 
        eps: float = 1e-6,
        wan_config: Optional[Dict] = None,
        skip_init: bool = False,
    ):
        super().__init__()
        
        # Handle WanModel initialization
        if isinstance(wan_model_or_ckpt_dir, str):
            self.wan_model = WanModel.from_pretrained(wan_model_or_ckpt_dir, subfolder=wan_subfolder)
        elif isinstance(wan_model_or_ckpt_dir, WanModel):
            self.wan_model = wan_model_or_ckpt_dir
        elif wan_config is not None:
            # Initialize from config with random weights
            logging.info(f"Initializing WanModel with random weights from config (skip_init={skip_init})")
            # Extract WanModel relevant params from the full config if needed
            model_params = {
                'patch_size': wan_config.get('patch_size', (1, 2, 2)),
                'dim': wan_config.get('dim', 2048),
                'ffn_dim': wan_config.get('ffn_dim', 8192),
                'freq_dim': wan_config.get('freq_dim', 256),
                'num_heads': wan_config.get('num_heads', 16),
                'num_layers': wan_config.get('num_layers', 32),
                'window_size': wan_config.get('window_size', (-1, -1)),
                'qk_norm': wan_config.get('qk_norm', True),
                'cross_attn_norm': wan_config.get('cross_attn_norm', True),
                'eps': wan_config.get('eps', 1e-6),
            }
            # Add other params that might be in shared_cfg or elsewhere
            if 'model_type' in wan_config: model_params['model_type'] = wan_config['model_type']
            if 'text_len' in wan_config: model_params['text_len'] = wan_config['text_len']
            if 'in_dim' in wan_config: model_params['in_dim'] = wan_config['in_dim']
            if 'text_dim' in wan_config: model_params['text_dim'] = wan_config['text_dim']
            if 'out_dim' in wan_config: model_params['out_dim'] = wan_config['out_dim']
            
            self.wan_model = WanModel(**model_params, skip_init=skip_init)
        else:
            raise ValueError("Either wan_model_or_ckpt_dir or wan_config must be provided")

        if max_context_len is not None:
            self.wan_model.text_len = max_context_len
        self.max_context_len = self.wan_model.text_len
        
        # VLM Features Projection (Simple norm and transform like Qwen-Image)
        # Note: RMSNorm and Linear will still do their own default init, but they are small
        self.vlm_norm = WanRMSNorm(vlm_in_dim, eps=eps)
        self.vlm_proj = nn.Linear(vlm_in_dim, self.wan_model.text_dim)

        # Visual Context Adapter
        if use_visual_context_adapter:
            self.visual_context_adapter = VisualContextAdapter(
                patch_size=self.wan_model.patch_size if visual_context_adapter_patch_size is None else visual_context_adapter_patch_size ,
                in_channels=self.wan_model.in_dim,
                hidden_dim=self.wan_model.dim,
                out_dim=self.wan_model.text_dim,
                eps=eps,
            )
            # VisualContextAdapter might have its own init_weights, but it's usually small
        else:
            self.visual_context_adapter = None

    @classmethod
    def from_pretrained(
        cls,
        wan_ckpt_dir: str,
        wan_subfolder: Optional[str] = None,
        vlm_in_dim: int = 4096,
        precision_dtype: torch.dtype = torch.float32,
        device_id: Union[int, str] = 0,
        rank: int = 0,
        dit_fsdp: bool = False,
        use_usp: bool = False,
        use_visual_context_adapter: bool = False,
        visual_context_adapter_patch_size: tuple = None,
        max_context_len: int = None, 
        eps: float = 1e-6,
        wan_config: Optional[Dict] = None,
        skip_init: bool = False,
    ):
        return cls(
            wan_model_or_ckpt_dir=wan_ckpt_dir,
            wan_subfolder=wan_subfolder,
            vlm_in_dim=vlm_in_dim,
            precision_dtype=precision_dtype,
            device_id=device_id,
            rank=rank,
            dit_fsdp=dit_fsdp,
            use_usp=use_usp,
            use_visual_context_adapter=use_visual_context_adapter,
            visual_context_adapter_patch_size=visual_context_adapter_patch_size,
            max_context_len=max_context_len,
            eps=eps,
            wan_config=wan_config,
            skip_init=skip_init,
        )

    def enable_gradient_checkpointing(self):
        self.wan_model.enable_gradient_checkpointing()
        if hasattr(self.visual_context_adapter, 'enable_gradient_checkpointing'):
            self.visual_context_adapter.enable_gradient_checkpointing()

    def reset_wan_text_len(self, new_len: int):
        self.wan_model.text_len = new_len
    
    def forward(
        self, 
        x: List[torch.Tensor], 
        t: torch.Tensor, 
        context: List[torch.Tensor] = None, 
        aligned_emb: Optional[torch.Tensor] = None,
        ar_vision_input: Optional[List[torch.Tensor]] = None,
        visual_emb: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
        special_token_dict: Optional[Dict[str, torch.Tensor]] = None,
        classifier_free_ratio: Optional[float] = None,
        unconditioned_context: Optional[Dict[str, torch.Tensor]] = None,
        condition_mode: str = "auto",
    ) -> torch.Tensor:
        if condition_mode == "auto":
            condition_mode = "full"
        
        batch_size = x.size(0) if isinstance(x, torch.Tensor) else len(x)

        # 1. Process CFG logic
        if condition_mode == "full":
            # Sample all random numbers upfront to ensure RNG states stay synchronized
            # across ranks even if branching logic differs.
            cfg_rands = [random.random() for _ in range(batch_size)]
            ar_only_rands = [random.random() for _ in range(batch_size)]

            for idx in range(batch_size):
                r = cfg_rands[idx]
                # 1.1 Standard CFG: Drop everything (including AR vision)
                if classifier_free_ratio is not None and classifier_free_ratio > 0 and r < classifier_free_ratio:
                    if unconditioned_context is None:
                        raise ValueError("unconditioned_context must be provided when classifier_free_ratio > 0")

                    if isinstance(unconditioned_context, dict):
                        # if ar_vision_input is not None and ar_vision_input[idx] is not None and 'uncond_ar_vision' in unconditioned_context:
                        #     ar_vision_input[idx] = unconditioned_context['uncond_ar_vision']

                        ## temperally use zero tensor as unconditioned ar_vision_input
                        if ar_vision_input is not None and ar_vision_input[idx] is not None:
                            # Use zero tensor with 2 tokens as null embedding
                            vlm_dim = ar_vision_input[idx].size(1)
                            ar_vision_input[idx] = torch.zeros(2, vlm_dim, device=ar_vision_input[idx].device, dtype=ar_vision_input[idx].dtype) + 1e-6

                        if context is not None and context[idx] is not None and 'uncond_context' in unconditioned_context:
                            context[idx] = unconditioned_context['uncond_context']
                
                # 1.2 New AR-Vision-Only Condition: 20% prob (if AR vision exists)
                elif ar_vision_input is not None and ar_vision_input[idx] is not None and ar_only_rands[idx] < 0.00:
                    # Keep AR vision and visual_emb (if exists), but drop context
                    # This encourages the model to generate based on VLM features and visual cues
                    if context is not None:
                        if unconditioned_context is not None and 'uncond_context' in unconditioned_context:
                            context[idx] = unconditioned_context['uncond_context']
                        else:
                            context[idx] = None
                        

            # 2. Process VLM Features (Simple norm and transform like Qwen-Image)
            vlm_output = None if ar_vision_input is None else [None] * batch_size
            if ar_vision_input is not None:
                if isinstance(ar_vision_input, list):
                    for idx in range(batch_size):
                        if ar_vision_input[idx] is not None:
                            h = self.vlm_norm(ar_vision_input[idx])
                            h = self.vlm_proj(h)
                            vlm_output[idx] = h
                else:
                    vlm_output = self.vlm_norm(ar_vision_input)
                    vlm_output = self.vlm_proj(vlm_output)
            
            # 3. Process Visual Context
            processed_visual_emb = None if visual_emb is None else [None] * batch_size
            
            if visual_emb is not None and self.visual_context_adapter is not None:
                if isinstance(visual_emb, list):
                    for idx in range(batch_size):
                        if visual_emb[idx] is not None:
                            processed_visual_emb[idx] = self.visual_context_adapter(visual_emb[idx]).squeeze(0)
                else:
                    processed_visual_emb = self.visual_context_adapter(visual_emb)
            
            # 4. Tight Concatenation
            mixed_context = []
            for idx in range(batch_size):
                components = []
                
                # Extract 2D items
                def get_item(item, idx=None):
                    if item is None:
                        return None
                    if isinstance(item, list) and idx is not None:
                        item = item[idx]
                    
                    if isinstance(item, torch.Tensor):
                        # If we have a batch tensor [B, L, D], take the sample
                        if idx is not None and item.dim() >= 3 and item.size(0) == batch_size:
                            item = item[idx]
                        
                        # Ensure the tensor is 2D [L, D]
                        if item.dim() == 3:
                            item = item[0]
                        elif item.dim() == 1:
                            item = item.unsqueeze(0)
                    return item

                vlm_item = get_item(vlm_output, idx)
                context_item = get_item(context, idx)
                visual_item = get_item(processed_visual_emb, idx)

                if special_token_dict is not None:
                    # Order: VLM -> Text -> Visual
                    img_st = get_item(special_token_dict.get('<img_st>'))
                    img_ed = get_item(special_token_dict.get('<img_ed>'))
                    prp_st = get_item(special_token_dict.get('<prp_st>'))
                    prp_ed = get_item(special_token_dict.get('<prp_ed>'))

                    # 1. VLM Features
                    if vlm_item is not None:
                        components.append(vlm_item)
                    # 2. Text Prompt
                    if context_item is not None:
                        components.extend([prp_st, context_item, prp_ed])
                    # 3. Visual
                    if visual_item is not None:
                        components.extend([img_st, visual_item, img_ed])
                else:
                    for item in [vlm_item, context_item, visual_item]:
                        if item is not None: components.append(item)

                if components:
                    # Filter and Concat
                    components = [c for c in components if c is not None]
                    new_context = torch.cat(components, dim=0)
                    if self.max_context_len is not None and new_context.shape[0] > self.max_context_len:
                        new_context = new_context[:self.max_context_len]
                    mixed_context.append(new_context)
                else:
                    # Safety fallback
                    mixed_context.append(torch.zeros(1, self.wan_model.text_dim, device=x[0].device, dtype=x[0].dtype))

            return self.wan_model(x, t=t, context=mixed_context, seq_len=seq_len)

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.wan_model.save_pretrained(os.path.join(save_dir, "wan_model"))
        if self.visual_context_adapter: torch.save(self.visual_context_adapter.state_dict(), os.path.join(save_dir, "visual_context_adapter_pytorch_model.bin"))
        torch.save(self.vlm_norm.state_dict(), os.path.join(save_dir, "vlm_norm_pytorch_model.bin"))
        torch.save(self.vlm_proj.state_dict(), os.path.join(save_dir, "vlm_proj_pytorch_model.bin"))

    def to(self, *args, **kwargs):
        self.wan_model.to(*args, **kwargs)
        self.vlm_norm.to(*args, **kwargs)
        self.vlm_proj.to(*args, **kwargs)
        if self.visual_context_adapter: self.visual_context_adapter.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def enable_eval(self):
        self.eval()
        self.wan_model.eval()
        self.vlm_norm.eval()
        self.vlm_proj.eval()
        if self.visual_context_adapter: self.visual_context_adapter.eval()

    def enable_train(self):
        self.train()
        self.wan_model.train()
        self.vlm_norm.train()
        self.vlm_proj.train()
        if self.visual_context_adapter: self.visual_context_adapter.train()
