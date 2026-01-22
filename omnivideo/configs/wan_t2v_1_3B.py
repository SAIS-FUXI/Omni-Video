from easydict import EasyDict

from .shared_config import omnivideo_shared_cfg

#------------------------ OmniVideo T2V 1.3B ------------------------#

t2v_1_3B = EasyDict(__name__='Config: OmniVideo T2V 1.3B')
t2v_1_3B.update(omnivideo_shared_cfg)

# t5
t2v_1_3B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
t2v_1_3B.t5_tokenizer = 'google/umt5-xxl'

# vae
t2v_1_3B.vae_checkpoint = 'Wan2.1_VAE.pth'
t2v_1_3B.vae_stride = (4, 8, 8)

# transformer
t2v_1_3B.patch_size = (1, 2, 2)
t2v_1_3B.dim = 1536
t2v_1_3B.ffn_dim = 8960
t2v_1_3B.freq_dim = 256
t2v_1_3B.num_heads = 12
t2v_1_3B.num_layers = 30
t2v_1_3B.window_size = (-1, -1)
t2v_1_3B.qk_norm = True
t2v_1_3B.cross_attn_norm = True
t2v_1_3B.eps = 1e-6

# inference
t2v_1_3B.sample_shift = 12.0
t2v_1_3B.sample_steps = 40
t2v_1_3B.boundary = 0.875
t2v_1_3B.sample_guide_scale = (3.0, 4.0)  # low noise, high noise

# model adapter settings
t2v_1_3B.use_visual_context_adapter = True
t2v_1_3B.visual_context_adapter_patch_size = (1, 4, 4)
t2v_1_3B.condition_mode = "full"
t2v_1_3B.vlm_in_dim = 2048  # Qwen3-VL hidden dimension
