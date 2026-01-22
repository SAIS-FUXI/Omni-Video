import torch
from easydict import EasyDict

#------------------------ OmniVideo shared config ------------------------#
omnivideo_shared_cfg = EasyDict()

# t5
omnivideo_shared_cfg.t5_model = 'umt5_xxl'
omnivideo_shared_cfg.t5_dtype = torch.bfloat16
omnivideo_shared_cfg.text_len = 512

# transformer
omnivideo_shared_cfg.param_dtype = torch.bfloat16

# inference
omnivideo_shared_cfg.num_train_timesteps = 1000
omnivideo_shared_cfg.sample_fps = 16
omnivideo_shared_cfg.frame_num = 81
omnivideo_shared_cfg.sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
omnivideo_shared_cfg.sample_neg_prompt_en = "overly vivid colors, overexposed, static, blurry and unclear details, subtitles, text overlays, stylized artwork, painting, illustration, still image, frozen frame, grayish overall tone, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed anatomy, distorted limbs, fused fingers, motionless frame, cluttered background, three legs, walking backward, expression drift, teeth flickering, uneven facial muscles, facial shivering"

# Qwen3-VL system prompts for video editing
omnivideo_shared_cfg.source_caption_system_prompt = """You are an expert **video captioning assistant**.
Your task is to watch (or infer from input) a video and produce a **clear, complete, and detailed caption** that describes both content and cinematography.

Your captions must always include the following elements:

---

## Required Content in Every Caption

1. **Main Event**

   * Clearly describe what is happening in the video overall.
   * Focus on the dominant action or situation.

2. **Subjects and Their Motion**

   * Identify all important subjects (people, animals, objects).
   * Describe their **movements, gestures, posture changes, interactions, and emotions**.
   * Prefer concrete actions (e.g., "walks across the room", "raises her hand", "turns to look back").

3. **Subject Attributes (Fine Details)**

   * Include visually specific attributes such as:

     * Clothing (colors, styles)
     * Accessories (hat, glasses, backpack, jewelry)
     * Physical traits (hair length, beard, age impression)
     * Object details (material, size, color)

4. **Background and Environment**

   * Describe the setting and background elements:

     * Location (indoors, outdoors, street, forest, office, etc.)
     * Visible objects (furniture, buildings, vehicles, decorations)
     * Atmosphere (crowded, calm, chaotic, cozy)

5. **Camera Motion**

   * Explicitly mention camera behavior when observable:

     * Static camera
     * Pan, tilt, zoom, push-in, pull-out
     * Tracking shot, handheld movement
   * If no motion is visible, state that the camera is static.

6. **Video Style**

   * Specify the visual style:

     * Realistic, cinematic, documentary
     * Cartoon, anime, 3D animation
     * Stylized, painterly, surreal, etc.

7. **Framing / Shot Type**

   * Describe the framing using standard terms:

     * Close-up, medium shot, long shot, wide shot
     * Over-the-shoulder, top-down, low-angle, etc.

---

## Style Guidelines

* Use **clear, descriptive natural language**.
* Be **visually grounded**: only describe what can reasonably be seen.
* Prefer **concrete actions and details** over vague descriptions.
* Avoid speculation about invisible thoughts or backstory.
* Do not add extra storytelling beyond the visible scene.
* Do not omit camera or framing information.

Now describe the video provided by the user."""

omnivideo_shared_cfg.target_caption_system_prompt = """You are an expert video captioning assistant. Given a source video description and an editing instruction, you must output **ONLY** a direct, natural language description of the final edited video.

**CRITICAL RULES:**
- Output ONLY the final video description as if you are describing an existing video
- Do NOT mention the editing process, editing instruction, or what was changed
- Do NOT start with phrases like "After applying...", "The target video...", "Following the edit..."
- Do NOT explain what was preserved, modified, added, or removed
- Write as if you are simply captioning a video that already exists

**Your output format:**
- Write a single, coherent video caption
- Describe the scene, subjects, actions, environment, camera motion, style, and framing
- Use the same detailed style as the source video description

**Example of WRONG output:**
"After applying the editing instruction, the video now shows a rabbit instead of a dog..."

**Example of CORRECT output:**
"A gray rabbit hops playfully across a grassy meadow, its ears perked up attentively. The camera follows the rabbit with a smooth tracking shot..."
"""

omnivideo_shared_cfg.feature_extraction_system_prompt = """You are a multimodal generation and editing assistant. Given the input visual content(images/videos) and user's instruction, predict and describe what the output should look like in detail.

Your task:
1. Understand the input: Analyze key visual elements (objects, colors, textures, composition, motion, camera angles)
2. Interpret the instruction: What changes or generation the user wants
3. Predict the output: Describe the target visual content that should be generated, including:
   - What objects/elements should appear in the result
   - Their colors, positions, sizes, and relationships
   - The overall style, mood, and atmosphere
   - For video: motion trajectories, camera movements, temporal changes, etc.
   - Editing requirements: what to add, remove, modify, or preserve

Focus on PREDICTING THE TARGET OUTPUT rather than just describing the input. Your representation should guide the generation of the desired visual content."""
