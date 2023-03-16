import horovod.torch as hvd
import torch
from torch.utils.data import Dataset
import random
import os
import json
from torch.utils.data.dataloader import default_collate
from src.utils.logger import LOGGER
from src.utils.basic_utils import flat_list_of_lists
from src.datasets.data_utils import mask_batch_text_tokens, img_collate
from src.datasets.dataloader import init_transform_dict, init_transform_dict_simple
import decord
from decord import VideoReader
from decord import cpu, gpu
decord.bridge.set_bridge("torch")
import math
import torch.nn.functional as F
import numpy as np
import cv2
import lmdb
import glob
import src.utils.stop_words as stop_words
from PIL import Image
from src.datasets.sample_frames import SampleFrames

class HDVILAPretrainDataset(Dataset):
    """
    vis_dir: video data root path
    anno_path: video meta data path (json, jsonl, lmdb)
    vid_cap_path: generated video caption path
    vid_txt: "subtitle" "caption" "caption_subtitle_concat" "caption_subtitle_random"
    vis_format: ["video", "videoframe"]
    """

    def __init__(self, cfg, vis_dir, anno_path, \
        vid_cap_path="", img_dir="", cap_path="", \
        img_source="", img_ratio=0, vid_txt='subtitle', vis_format='video', mode="train"):
        self.vis_format = vis_format

        self.cfg = cfg
        self.vis_dir = vis_dir
        self.anno_path = anno_path
        self.vid_cap_path = vid_cap_path
        
        self.vid_txt = vid_txt

        self.mode = mode
        self.n_clips = cfg.train_n_clips if mode == "train" else cfg.test_n_clips
        self.num_frm = cfg.train_num_frms if mode == "train" else cfg.test_num_frms
        self.sample_rate = cfg.sample_rate
        self.transform = init_transform_dict_simple(video_res=cfg.video_res,
                                             input_res=cfg.input_res)[mode]
        self.frame_sampler = SampleFrames(clip_len=self.num_frm, 
                                          frame_interval=self.sample_rate, 
                                          num_clips=self.n_clips, 
                                          temporal_jitter=True)
        
        self.video_init_dataset()
    
        if self.vis_format == "videoframe" or "caption" in self.vid_txt:
            self.caption_init_dataset()
    
    def caption_init_dataset(self):
        self.caption_use_lmdb = False
        json_type = os.path.splitext(self.vid_cap_path)[-1]
        assert json_type in ['.json', '.jsonl', '']
        if json_type == '':
            print('begin cap lmdb open')
            cap_env = lmdb.open(self.vid_cap_path, readonly=True, create=False)
            print('finish cap lmdb open')
            self.cap_txn = cap_env.begin()
            print('finish cap env begin')
            self.caption_use_lmdb = True
        else:
            captions = json.load(open(self.vid_cap_path))
            cap_dict = {}
            for item in captions:
                cap_dict[item["image_id"]] = item["caption"]
            del captions
            self.cap_dict = cap_dict

    def video_init_dataset(self):
        self.use_lmdb = False
        json_type = os.path.splitext(self.anno_path)[-1]
        assert json_type in ['.json', '.jsonl', '']

        if json_type == '':
            self.use_lmdb = True
            print('begin lmdb open')
            env = lmdb.open(self.anno_path, readonly=True, create=False)
            print('finish lmdb open')
            self.txn = env.begin()
            print('finish env begin')
            self.anno_path = self.anno_path + ".jsonl"
            json_type = os.path.splitext(self.anno_path)[-1]

        if json_type == '.jsonl':
            data = []
            with open(self.anno_path) as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            data = json.load(open(self.anno_path))
        self.datalist = data

    def video_idx2path(self, index):
        if 'clip_id' in self.datalist[index]:
            vis_id = self.datalist[index]['clip_id']
        elif 'video_id' in self.datalist[index]:
            vis_id = self.datalist[index]['video_id']
        clip_name = vis_id
        video_name = vis_id.split('.')[0]

        return os.path.join(self.vis_dir, video_name, clip_name)
    
    def video_get_text(self, index):
        if self.use_lmdb:
            texts = self.txn.get(self.datalist[index]['clip_id'].encode()).decode()
        else:
            texts = self.datalist[index]['text']
        if "caption" in self.vid_txt:
            if self.caption_use_lmdb:
                caption = self.cap_txn.get(self.datalist[index]['clip_id'].encode(), None)
                if caption:
                    caption = caption.decode()
            else:
                caption = self.cap_dict.get(self.datalist[index]['clip_id'], None)
            if caption and self.vid_txt == "caption":
                texts = caption
            elif caption and self.vid_txt == "caption_subtitle_concat":
                texts = caption + ". with subtitle: " + texts
            elif caption and self.vid_txt == "caption_subtitle_random":
                if_caption = random.randint(0, 1)
                if if_caption:
                    texts = caption

        if self.vis_format == "videoframe":
            if self.caption_use_lmdb:
                mid_caption = self.cap_txn.get(self.datalist[index]['clip_id'].encode(), None)
                if mid_caption:
                    mid_caption = mid_caption.decode()
                else:
                    mid_caption = "None"
                    LOGGER.info(f"Failed to load caption of video: {self.datalist[index]['clip_id']}. "
                                f"The caption will be None.")
            else:
                mid_caption = self.cap_dict.get(self.datalist[index]['clip_id'], "None")
        else:
            mid_caption = None
            
        return [texts], [mid_caption]
    
    def __len__(self):
        return len(self.datalist)

    def get_sample_idx(self, total_frame_num):
        """
        sample rate > 0: use SampleFrames, loop default
        sample rate = 0: uniform sampling, temporal jittering
        """
        if self.sample_rate > 0:
            results = {"total_frames": total_frame_num,
                    "start_index": 0}
            results = self.frame_sampler(results)
            return results["frame_inds"]
        elif self.sample_rate == 0:
            if hasattr(self.cfg, "sample_jitter") and self.cfg.sample_jitter and self.mode == "train":
                interval = int(total_frame_num / (self.n_clips*self.num_frm - 1))
                start = np.random.randint(0, interval+1)
                end = np.random.randint(total_frame_num-1-interval, total_frame_num)
                return np.linspace(start, end, self.n_clips*self.num_frm).astype(int)
            else:
                return np.linspace(0, total_frame_num-1, self.n_clips*self.num_frm).astype(int)

    def load_video(self, vis_path):
        vr = VideoReader(vis_path, ctx=cpu(0))
        total_frame_num = len(vr)

        frame_idx = self.get_sample_idx(total_frame_num)
        img_array = vr.get_batch(frame_idx) # (n_clips*num_frm, H, W, 3)

        img_array = img_array.permute(0, 3, 1, 2).float() / 255.
        img_array = self.transform(img_array)

        if self.vis_format == "videoframe":
            mid_frame = vr.get_batch([int(total_frame_num/2)])
            mid_frame = mid_frame.permute(0, 3, 1, 2).float() / 255.
            mid_frame = self.transform(mid_frame)
        else:
            mid_frame = None

        return img_array, mid_frame

    def __getitem__(self, index):
        
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                vis_path = self.video_idx2path(index)
                video, mid_frame = self.load_video(vis_path)
                texts, mid_caption = self.video_get_text(index)
            
            except:
                LOGGER.info(f"Failed to load examples with video: {os.path.basename(vis_path)}. "
                            f"Will select an random sample as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            
            else:
                break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        
        if self.vis_format == "videoframe":
            image = mid_frame
            captions = mid_caption
        else:
            image = None
            captions = None

        return dict(
            video = video,  # [clips*num_frm, C, H_crop, W_crop]
            texts = texts,
            image = image,
            captions = captions
        )



class PretrainCollator(object):
    def __init__(self, tokenizer, max_length=40, is_train=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

    def collate_batch(self, batch):
        batch_size = len(batch)
        video = default_collate([d["video"] for d in batch])

        if "image" in batch[0] and batch[0]["image"] is not None:
            image = default_collate([d["image"] for d in batch])
        else:
            image = None

        # subtitle data
        text_str_list = flat_list_of_lists([d["texts"] for d in batch])
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)

        # caption data
        if "captions" in batch[0] and batch[0]["captions"] is not None:
            text_str_list = flat_list_of_lists([d["captions"] for d in batch])
            batch_enc = self.tokenizer.batch_encode_plus(
                text_str_list,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            caption_ids = batch_enc.input_ids  # (B*img_num, L)
            caption_masks = batch_enc.attention_mask  # (B*img_num, L)
            caption_ids = caption_ids.view(batch_size, -1, caption_ids.shape[-1])
            caption_masks = caption_masks.view(batch_size, -1, caption_masks.shape[-1])
        else:
            caption_ids = None
            caption_masks = None

        collated_batch = dict(
            video=video,   # [B, n_clips*num_frms, C, H, W]
            text_input_ids=text_input_ids,  # [B, L]
            text_input_mask=text_input_mask,
            image=image,   # [B, img_num, C, H, W]
            caption_ids=caption_ids,   # [B, img_num, L]
            caption_masks=caption_masks
        )

        return collated_batch

    


    