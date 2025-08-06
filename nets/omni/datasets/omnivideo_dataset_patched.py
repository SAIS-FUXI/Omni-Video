import os
import pickle
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class OmniVideoDataset(Dataset):
    """
    Dataset for loading pre-encoded features from pickle files for OmniVideo model training.
    
    Each pickle file contains a dictionary with pre-encoded text embeddings,
    video latent features, and other metadata needed for training.
    """
    
    def __init__(self, file_path):
        """
        Initialize the dataset with a list of pickle file paths.
        
        Args:
            file_paths (list): List of paths to pickle files containing pre-encoded features
        """
        with open(file_path, 'r') as ins:
            self.file_paths = [it.strip(' \t\n') for it in ins]
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        Load and return a sample from the dataset at the given index.
        
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            dict: Dictionary containing all required inputs for the model:
                - 'text_emb': Pre-encoded text embeddings
                - 'prompt': Original text prompt (for logging)
                - 'latent_feature': Video latent features
                - Additional metadata if available
        """
        # Load the pickle file
        real_idx = idx

        # try several times to load a right pickle file and process it until success
        try_idx = 0
        while try_idx < 20:
            try:
            # if True:
                file_path = self.file_paths[real_idx]

                # import pdb; pdb.set_trace();
                # print(f'file_path is {file_path}', flush=True)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                
                # Ensure required keys are present
                required_keys = [] #['prompt']
                for key in required_keys:
                    if key not in data:
                        raise KeyError(f"Required key '{key}' not found in data from {file_path}")
                
                # change the key of text emb to 'text_emb'
                if 't5_emb' in data:
                    data['text_emb'] = data.pop('t5_emb')
                
                # [1, 267, 4096]
                if 'vlm_last_hidden_states' in data:
                    data['vlm_last_hidden_states'] = data.pop('vlm_last_hidden_states')
                                            
                # Ensure 'prompt' is available for logging, or use a placeholder
                if 'prompt' not in data:
                    data['prompt'] = f"Unknown (from {os.path.basename(file_path)})"

                # Convert numpy arrays to tensors if needed
                for key in data:
                    ## 这里注意，抽取的txt_emb可能是只包含一个元素的list
                    if key == 'text_emb' and isinstance(data['text_emb'], list) and len(data['text_emb']) == 1:
                        data[key] = data['text_emb'][0]

                    if hasattr(data[key], 'shape') and not isinstance(data[key], torch.Tensor):
                        data[key] = torch.tensor(data[key])

                return data

            except Exception as e:
                logging.warning(f"Error loading {file_path}: {e}")
                logging.info(f"Retrying with another file (try {try_idx + 1})")
                try_idx += 1
                real_idx = np.random.randint(0, len(self.file_paths))
                

def omnivideo_collate_fn(batch):
    """
    Custom collate function for OmniVideoDataset.
    
    Combines tensors into batched tensors with first dimension as batch size.
    Text embeddings are handled specially based on their structure.
    Prompts are collected into a list.
    For spatial tensors (like video features), pads smaller tensors to match the largest dimensions.
    
    Args:
        batch (list): List of dictionaries from dataset __getitem__
        
    Returns:
        dict: Batched data with properly combined tensors and lists
    """
    elem = batch[0]
    result = {}
    
    for key in elem:
        if key == 'prompt':
            # Collect prompts into a list
            result[key] = [d[key] for d in batch]
        elif key == 'text_emb' or key == 'vlm_last_hidden_states':
            # Handle text embeddings based on their structure
            if isinstance(elem[key], list):
                # If text_emb is a list of tensors (like from T5 encoder)
                result[key] = [torch.stack([d[key][i] for d in batch], dim=0) 
                              for i in range(len(elem[key]))]
            elif isinstance(elem[key], torch.Tensor):
                # If text_emb is a single tensor
                # result[key] = torch.stack([d[key] for d in batch], dim=0)
                result[key] = [d[key] for d in batch] # return list[tensor]
            else:
                # Fallback for other types
                result[key] = [d[key] for d in batch]
        elif isinstance(elem[key], torch.Tensor):
            # Handle tensors that might have different spatial dimensions
            tensors = [d[key] for d in batch]
            
            # Check if all tensors have the same shape
            shapes = [t.shape for t in tensors]
            if len(set(shapes)) == 1:
                # All tensors have the same shape, stack normally
                result[key] = torch.stack(tensors, dim=0)
            else:
                # Tensors have different shapes, need to handle spatial dimensions
                try:
                    # Try to stack first (in case the difference is not in spatial dims)
                    result[key] = torch.stack(tensors, dim=0)
                except RuntimeError as e:
                    if "stack expects each tensor to be equal size" in str(e):
                        # Handle spatial dimension mismatch by padding
                        result[key] = pad_tensors_to_same_size(tensors, key, batch)
                    else:
                        raise e
        else:
            # Fallback for other types
            result[key] = [d[key] for d in batch]
    
    return result

def pad_tensors_to_same_size(tensors, key_name, batch):
    """
    Replace tensors with different shapes to match the most common shape in the batch.
    Uses copies of other tensors from the same batch instead of zero tensors.
    
    Args:
        tensors (list): List of tensors with potentially different spatial dimensions
        key_name (str): Name of the key for logging purposes
        batch (list): List of dictionaries from dataset __getitem__
        
    Returns:
        torch.Tensor: Stacked tensor with all tensors having the same shape
    """
    # Get all shapes and find the most common one
    shapes = [t.shape for t in tensors]
    
    # Count occurrences of each shape
    from collections import Counter
    shape_counts = Counter(shapes)
    
    # Find the most common shape
    target_shape = shape_counts.most_common(1)[0][0]
    target_count = shape_counts.most_common(1)[0][1]
    
    # If there are multiple different shapes, print debugging information
    if len(shape_counts) > 1:
        logging.error(f"Shape mismatch detected for key '{key_name}'!")
        logging.error(f"File paths and corresponding shapes in this batch:")
        
        for i, (tensor, data_dict) in enumerate(zip(tensors, batch)):
            file_path = data_dict.get('video_path_tgt', data_dict.get('video_path', f'Unknown_file_{i}'))
            logging.error(f"  [{i}] {file_path} -> shape: {tensor.shape}")
        
        logging.error(f"Target shape chosen: {target_shape} (appears {target_count}/{len(tensors)} times)")
        logging.error(f"All shapes in batch: {dict(shape_counts)}")
    
    logging.info(f"For key '{key_name}': target shape {target_shape} appears {target_count}/{len(tensors)} times")
    
    # Find tensors that match the target shape (to use as replacements)
    valid_tensors = [tensor for tensor in tensors if tensor.shape == target_shape]
    
    if not valid_tensors:
        logging.error(f"No valid tensors found for key '{key_name}' with target shape {target_shape}")
        # Fallback: create zero tensors
        fallback_tensors = [torch.zeros(target_shape, dtype=tensors[0].dtype, device=tensors[0].device) 
                           for _ in range(len(tensors))]
        return torch.stack(fallback_tensors, dim=0)
    
    # Replace tensors that don't match the target shape
    corrected_tensors = []
    replacements_made = 0
    replacement_idx = 0  # Index to cycle through valid tensors
    
    for i, tensor in enumerate(tensors):
        if tensor.shape == target_shape:
            # Shape matches, keep the original tensor
            corrected_tensors.append(tensor)
        else:
            # Shape doesn't match, replace with a copy of a valid tensor from the batch
            replacement_tensor = valid_tensors[replacement_idx % len(valid_tensors)].clone()
            corrected_tensors.append(replacement_tensor)
            replacements_made += 1
            replacement_idx += 1
            logging.info(f"Replaced tensor {i} for key '{key_name}': {tensor.shape} → {target_shape} (using copy of valid tensor)")
    
    if replacements_made > 0:
        logging.warning(f"Replaced {replacements_made}/{len(tensors)} tensors for key '{key_name}' with copies from the same batch")
    
    # Stack the corrected tensors
    try:
        result = torch.stack(corrected_tensors, dim=0)
        return result
    except Exception as e:
        logging.error(f"Failed to stack corrected tensors for key '{key_name}': {e}")
        # Fallback: return copies of the first valid tensor
        fallback_tensors = [valid_tensors[0].clone() for _ in range(len(tensors))]
        return torch.stack(fallback_tensors, dim=0)

def create_omnivideo_dataloader(file_path, batch_size=1, shuffle=True, num_workers=4, distributed=False, rank=0, world_size=1):
    """    
    Args:
        file_path (str): Path to a text file containing paths to pickle files
        batch_size (int, optional): Batch size for training. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.
        distributed (bool, optional): Whether to use distributed sampling. Defaults to False.
        rank (int, optional): Rank of the current process in distributed training. Defaults to 0.
        world_size (int, optional): Total number of processes in distributed training. Defaults to 1.
        
    Returns:
        DataLoader: PyTorch DataLoader for the OmniVideoDataset
    """
    dataset = OmniVideoDataset(file_path)
    
    # 为分布式训练创建采样器
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        shuffle = False  # 当使用DistributedSampler时，DataLoader的shuffle必须为False
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=omnivideo_collate_fn
    )
    
    return dataloader

def create_dummy_dataset(output_dir, num_samples=128, frames=21, height=32, width=32, emb_dim=4096, hidden_dim=1024):
    """
    Creates a dummy dataset of pickle files for testing the OmniVideoDataset.
    
    Args:
        output_dir (str): Directory to save the dummy pickle files
        num_samples (int, optional): Number of samples to generate. Defaults to 10.
        frames (int, optional): Number of frames in each video. Defaults to 21.
        height (int, optional): Height of each frame. Defaults to 32.
        width (int, optional): Width of each frame. Defaults to 32.
        emb_dim (int, optional): Dimension of text embeddings. Defaults to 4096.
        hidden_dim (int, optional): Hidden dimension for text features. Defaults to 1024.
        
    Returns:
        str: Path to the text file containing paths to the generated pickle files
    """
    os.makedirs(output_dir, exist_ok=True)
    file_paths = []
    
    for i in range(num_samples):
        # Create dummy data
        dummy_data = {
            # T5 encoder typically outputs a list of tensors
            'text_emb': torch.randn(200, emb_dim),  # First tensor (hidden states)
            'aligned_emb': torch.randn(1, hidden_dim), # Second tensor for aligned embeddings
            'prompt': f"This is a dummy prompt for sample {i}",
            'latent_feature': torch.randn(16, frames, height, width),  # [C, F, H, W]
            'vlm_last_hidden_states': torch.randn(1, 267, 4096),
        }
        
        # Save to pickle file
        file_path = os.path.join(output_dir, f"dummy_sample_{i}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(dummy_data, f)
        
        file_paths.append(file_path)
    
    # Create a text file with the file paths
    txt_file_path = os.path.join(output_dir, "file_paths.txt")
    with open(txt_file_path, 'w') as f:
        for path in file_paths:
            f.write(f"{path}\n")
    
    return txt_file_path

def test_omnivideo_dataloader(output_dir=None, batch_size=2):
    """
    Test function for the OmniVideoDataset and dataloader.
    
    Args:
        output_dir (str, optional): Directory to save dummy data. If None, uses a temporary directory.
        batch_size (int, optional): Batch size for testing. Defaults to 2.
        
    Returns:
        bool: True if test passes, False otherwise
    """
    import tempfile
    import shutil
    
    # Create temporary directory if output_dir is not provided
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
        cleanup = True
    else:
        os.makedirs(output_dir, exist_ok=True)
        cleanup = False
    
    try:
        # Create dummy dataset and get the path to the text file
        txt_file_path = create_dummy_dataset(output_dir, hidden_dim=1152)
        
        # Create dataloader
        dataloader = create_omnivideo_dataloader(txt_file_path, batch_size=batch_size, num_workers=0)
        
        # Test iteration
        for batch_idx, batch in enumerate(dataloader):
            # Check if batch contains required keys
            assert 'text_emb' in batch, "Batch missing 'text_emb'"
            assert 'prompt' in batch, "Batch missing 'prompt'"
            # assert 'latent_feature' in batch, "Batch missing 'latent_feature'"
            
            # Check shapes
            assert batch['latent_feature'].shape[0] == batch_size, f"Expected batch size {batch_size}, got {batch['latent_feature'].shape[0]}"
            
            # Print batch info
            print(f"Batch {batch_idx}:")
            print(f"  text_emb shape: {batch['text_emb'][0].shape}")
            print(f"  latent_feature shape: {batch['latent_feature'].shape}")
            print(f"  prompt: {batch['prompt'][0][:30]}...")
            print(f"  aligned_emb shape: {batch['aligned_emb'][0].shape}")
            print(f"  vlm_last_hidden_states shape: {batch['vlm_last_hidden_states'][0].shape}")
            print("  ---------------------")
            
            # Only test first batch
            if batch_idx == 0:
                break
        
        print("Test passed successfully!")
        return True
    
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False
    
    finally:
        # Clean up temporary directory
        if cleanup:
            shutil.rmtree(output_dir)

if __name__ == "__main__":
    # Run test if script is executed directly
    output_dir = 'output/dataset'
    os.makedirs(output_dir, exist_ok=True)
    test_omnivideo_dataloader(output_dir)