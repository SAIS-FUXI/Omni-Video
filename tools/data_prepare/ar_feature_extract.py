import argparse
import re
import os
import json
import glob
import random
import torch
import hashlib
import pickle
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL
from tqdm import tqdm
from decord import VideoReader, cpu
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

disable_torch_init()

class VideoDataset(Dataset):
    def __init__(self, data_file, result_folder):
        result_files = list(glob.glob(os.path.join(result_folder, '*.pkl')))
        processed_results = {os.path.splitext(os.path.basename(f))[0]: 1 for f in result_files}

        data = []
        if not os.path.exists(result_folder):
            os.makedirs(result_folder, exist_ok=True)
        # data = [os.path.join(data_file, item) for item in os.listdir(data_file)]
        data = [line.strip() for line in open(data_file)]

        self.data = []
        for item in tqdm(data):
            item_filename = os.path.splitext(os.path.basename(item))[0]
            if item_filename not in processed_results:
                self.data.append(item)

        print(f"Total {len(data)} samples, left {len(self.data)} to process")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        try:
            item = pickle.load(open(self.data[index], mode='rb'))
            item_id = os.path.splitext(os.path.basename(self.data[index]))[0]
            item["id"] = item_id
        except:
            return self.__getitem__(index+1)

        return item
    
def collate_fn(samples):
    data_dict_item = {
        "ids": [item["id"] for item in samples],
        'prompts': [item["prompt"] for item in samples],
        "videos": [item["video_path"] for item in samples],
        "latent_features": [item["latent_feature"] for item in samples],
        "text_embs": [item["text_emb"] for item in samples],
        "frame_nums":[item["frame_num"] for item in samples],
    }
    
    return data_dict_item

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def load_video(video_path, max_frames_num):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())
        frame_idx = np.linspace(0, total_frame_num - 2, max_frames_num, dtype=int)
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return [Image.fromarray(img) for img in spare_frames]
    except Exception as e:
        print(f"Failed to load video {video_path} with error: {e}")
        return [Image.new("RGB", (448, 448), (0, 0, 0))] * max_frames_num


def format_features(featurefile, vlm_last_hidden_state):
    data = pickle.load(open(featurefile, mode='rb'))
    new_data ={
        "video_path_tgt": data["video_path"], 
        "prompt": data["prompt"],
        "latent_feature_tgt": data["latent_feature"],
        "t5_emb": data["text_emb"],
        "frame_num":data["frame_num"],
        "vlm_last_hidden_states":vlm_last_hidden_state,
    }
    return new_data

@torch.inference_mode()
def main(args):
    if args.pdb_debug:
        import pdb; pdb.set_trace()
        args.num_workers = 0
    
    if not args.pdb_debug:
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=int(os.getenv('WORLD_SIZE', '1')),
            rank=int(os.getenv('RANK', '0')),
        )
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    
    device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
    # ======================================================
    # 1. read video dataset
    # ======================================================
    dataset = VideoDataset(args.data_file, args.result_folder)
    if args.pdb_debug or torch.cuda.device_count() == 1:
        sampler = None 
    else:
        sampler = InferenceSampler(len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )

    # ======================================================
    # 2. load model and prepare inputs
    # ======================================================
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base, device=device)

    # ======================================================
    # 3. generate captions
    # ======================================================
    if args.error_report_file == None:
        current_directory = os.getcwd()
        args.error_report_file = os.path.join(current_directory, "failed_captioning_error.log")
    print("Setting error logging path: {}".format(args.error_report_file))
    error_logger = open(args.error_report_file, mode='a', encoding='utf-8')



    for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        data_id = data["ids"][0] # the batch must be 1
        prompt = data["prompts"][0]
        vid_path = data["videos"][0]
        latent_feature = data["latent_features"][0]
        text_emb = data["text_embs"][0]
        frame_num = data["frame_nums"][0]
        answer = "[GEN_VID]"

        #images = [PIL.Image.open(img_path).convert("RGB")]
        # images = process_images(images, image_processor, model.config).half().cuda()
        images = None
        data_id = data["ids"][0]

        qs = "Generate a video about:\n" + prompt
        # if model.config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        # else:
        #     qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

        conv = conv_templates["llama_3"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        attention_masks = input_ids.ne(pad_token_ids).long().cuda()
        inputs = {  
                "input_ids":input_ids,
                "images":images,
                "labels":input_ids,
                "attention_mask":attention_masks,
        }
        #import pdb; pdb.set_trace()
        with torch.inference_mode():
            vlm_outputs = model(output_hidden_states=True, **inputs)
            vlm_last_hidden_states = vlm_outputs.hidden_states[-1]
            # print(vlm_last_hidden_states.shape)
            # print(data_id)

        saved_feat = {
            # "id": data["ids"][0],
            "video_path_tgt": data["videos"][0], 
            "prompt": data["prompts"][0],
            "latent_feature_tgt": data["latent_features"][0],
            "t5_emb": data["text_embs"][0],
            "frame_num":data["frame_nums"][0],
            "vlm_last_hidden_states":vlm_last_hidden_states.detach().cpu(),
        }

        output_path = os.path.join(args.result_folder, data_id+".pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(saved_feat, f)


                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str)
    parser.add_argument("--result-folder", type=str)
    parser.add_argument("--error-report-file", type=str, default=None )
    parser.add_argument('--local-rank', type=int, default=0)    
    parser.add_argument("--model-path", type=str, default="Fr0zencr4nE/Cockatiel-13B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num-video-frames", type=int, default=8)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--pdb_debug", action="store_true")
    args = parser.parse_args()

    main(args)