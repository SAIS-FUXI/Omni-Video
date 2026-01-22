# Data Preparation for OmniVideo

This document describes the data preparation pipeline for training OmniVideo models. The process involves extracting latent features from videos and text prompts to create efficient training datasets.

## Overview

To finetune OmniVideo models, input videos and their corresponding prompts need to be preprocessed into latent features and multimodal language model (MLM) features. This offline feature extraction approach significantly reduces GPU memory requirements during training and improves overall training efficiency.

The data preparation consists of two main steps:

## Step 1: VAE and T5 Feature Extraction

### Overview
Extract video latent features using a VAE (Variational Autoencoder) and text embeddings using T5 encoder.

### Usage
```bash
bash tools/data_prepare/run_vae_feature.sh
```

### Input Format
The input should be a JSON file containing video paths and corresponding prompts. Each entry should follow this structure:

```json
{
    "video": "/path/to/video.mp4",
    "conversations": [
        {
            "from": "human",
            "value": "<video>\nDescribe the events in the video shown by these frames in at least three sentences."
        },
        {
            "from": "gpt",
            "value": "In the video, a man is seen standing on a boat in the middle of the ocean. He is wearing a black jacket and a black cap. The man is holding a walkie-talkie in his hand and appears to be speaking into it. The ocean is calm with small waves, and the sky is overcast. The man seems to be communicating with someone on the boat or on the shore. The video captures the serene environment of the ocean and the man's interaction with the walkie-talkie."
        }
    ]
}
```

### Important Notes
- Only the `video` field and the `value` field from the `gpt` conversation are required
- Other fields in the JSON structure are ignored during processing
- Each video-prompt pair will be saved as a separate pickle file containing the extracted features

### Output
The script generates pickle files containing:
- VAE latent features for video frames
- T5 text embeddings for prompts
- Metadata including video path and frame information

## Step 2: AR Model Feature Extraction

### Overview
Extract autoregressive (AR) model features from the VAE features generated in Step 1.

### Usage
```bash
bash tools/data_prepare/run_ar_feature.sh
```

### Input
The script uses the pickle file list generated from Step 1 as input. Set the `$DATA_FILE` variable in `run_ar_feature.sh` to point to your VAE feature file list.

### Output
Final pickle files containing all features required for OmniVideo training:
- VAE latent features
- T5 text embeddings  
- AR model features
- Complete metadata

## Configuration

### Key Parameters
- **Frame Count**: Number of frames to extract (default: 81, must be 4n+1)
- **Sampling Rate**: Frame sampling interval (default: 3)
- **Target Size**: Output resolution (default: 480x832)
- **Skip Frames**: Number of initial frames to skip (default: 0)

## Integration with Training

The generated pickle files can be directly used with the OmniVideo training pipeline. The feature extraction process ensures optimal memory usage and training efficiency.

For detailed training instructions, refer to the main training documentation and `finetune_model.py`.
