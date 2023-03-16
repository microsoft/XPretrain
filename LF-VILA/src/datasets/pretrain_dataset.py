from collections import defaultdict
import os
import random
from abc import abstractmethod

import cv2

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms
import jsonlines
import decord
from decord import VideoReader, cpu
from easydict import EasyDict
import operator
import functools
from transformers import BertTokenizer
import traceback
import lmdb

from src.utils.logger import LOGGER

decord.bridge.set_bridge("torch")

class PreTrainDataset(Dataset):

    def __init__(self,
                 cfg,
                 metadata_dir,
                 video_path,
                 sample_frame,
                 sample_clip,
                 tokenizer,
                 transform=None,
                 return_rawtext=False,
                 return_index=False,
                 is_train=True
                 ):
        self.cfg =  cfg
        self.metadata_dir = metadata_dir
        self.transform = transform
        self.video_path = video_path
        self.return_rawtext = return_rawtext
        self.return_index = return_index
        self.reliable_idx_list = [(0,0)]
        self.sample_frame = sample_frame
        self.sample_clip = sample_clip
        if is_train and cfg.DATA.use_lmdb_train_data:
            self.use_lmdb = True
            env = lmdb.open(self.metadata_dir, readonly=True, create=False)
            LOGGER.info(f"finish lmdb open")
            self.txn = env.begin()
        else:
            self.metadata = self._load_metadata()
            self.use_lmdb = False
        self.tokenizer = tokenizer
        self.is_train= is_train

    def _get_video_path(self, clip_id):
        '''
        convert clip_id to video clip path
        '''
        folder = self.video_path
        video_name = clip_id.split('.')[0]
        return os.path.join(folder, video_name, clip_id)

    def _load_metadata(self):
        '''
        metadata:[ list of {'clip_id':xxx, 'text':xxx}] 
        '''
        data = []
        with open(self.metadata_dir) as f:
            for l in jsonlines.Reader(f):
                data.append(l)
        return data
        

    def _read_videos(self, clip_ids, sample_frame_num, uniform = True):
        '''
        args:
            clip_ids: list,
            sample_frame_num: total frame num

        return: 
            img_arrays: [num_frm, 3, H, W]
            chunk_mask: [num_frm, n_clip], , mask for indicating frames belong to each clip
        '''
        vrs = []
        for clip_id in clip_ids:
            video_path = self._get_video_path(clip_id)
            vr = VideoReader(video_path, ctx=cpu(0))
            vrs.append(vr) 

        frame_nums = [len(x) for x in vrs]
        chunk = self._split_video_chunk(frame_nums, sample_frame_num, uniform)

        if min([len(v) for v in chunk]) == 0:
            raise ValueError(f"Need to sample at least 1 frame.")

        img_arrays = []
        for i in range(len(vrs)):
            img_array = vrs[i].get_batch(chunk[i])
            img_arrays.append(img_array)
        img_arrays = torch.cat(img_arrays).float() / 255 # num_frm, H, W, 3

   
        img_arrays = img_arrays.permute(0, 3, 1, 2) # num_frm,3,H,W

        return img_arrays

    def _split_video_chunk(self, frame_nums, sample_frame_num, uniform):
        '''
        split whole frames to chunks
        '''
        if not uniform:
            frame_idx = np.linspace(0, sum(frame_nums)-1, num=sample_frame_num).astype(int)
            chunks = defaultdict(list)
            frame_cumsum = np.cumsum(frame_nums)
        
            j = 0
            for i in frame_idx:
                if i >= frame_cumsum[j]:
                    j+=1
                if j==0:
                    chunks[j].append(i)
                else:
                    chunks[j].append(i-frame_cumsum[j-1])
                if j > len(frame_cumsum):
                    break
            chunks = [chunks[k] for k in range(len(frame_cumsum))]
        else:
            sample_num_each = int(sample_frame_num / len(frame_nums))
            chunks = [list(np.linspace(0, i, num = sample_num_each+1)[:sample_num_each]) for i in frame_nums]
        return chunks

    def tokenize(self, texts, total_chunk = 8, max_length = 50):
        '''
        tokenizing text for pretraining
        args:
            texts: list of text segment
            total_chunk: num of text segments
        return:
            text_ids: sequence of token id
            attention_mask: segment_ids to distinguish the sentences
            chunk: index of [CLS]

        '''

        encoded = [self.tokenizer(x, padding='max_length', truncation=True, max_length=max_length) for x in texts]

        text_ids = [x.input_ids for x in encoded]
        attention_mask = [x.attention_mask for x in encoded]

        if len(texts) < total_chunk:
            for i in range(total_chunk-len(texts)):
                text_ids.append([0 for x in range(max_length)])
                attention_mask.append([0 for x in range(max_length)])
        
        return text_ids, attention_mask

    def __len__(self):
        if self.use_lmdb:
            return self.cfg.DATA.len_lmdb_train_data
        else:
            return len(self.metadata)
        
    def __getitem__(self, index):

        num_retries = 10
        for j in range(num_retries):

            if self.use_lmdb:
                item = eval(self.txn.get(str(index).encode()).decode())
            else:
                item = self.metadata[index]
            if j == 0:
                if self.is_train:
                    if len(item) > self.sample_clip:
                        start_index = random.choice(list(range(len(item)-self.sample_clip)))
                    else:
                        start_index = 0
                else:
                    start_index = 0

            clip_ids = [x['clip_id'] for x in item][start_index:start_index+self.sample_clip]
            try:

                video = self._read_videos(clip_ids, self.sample_frame)

                rawtext = [x['text'] for x in item][start_index:start_index+self.sample_clip]
                text_ids, attention_mask = self.tokenize(rawtext, total_chunk=self.sample_clip)

                if self.transform is not None:
                    video = self.transform(video) # N, C, H, W
                video = video.permute(1, 0, 2, 3) # C, N, H, W

            except Exception as e:
                traceback.print_exc()
                LOGGER.info(f"Failed to load examples with video: {clip_ids[0]}. "
                                f"Will select an example from reliable list as a replacement.")
                index, start_index = random.choice(self.reliable_idx_list)
                continue
            else:
                if len(self.reliable_idx_list) < 10000:
                    self.reliable_idx_list.append((index,start_index))
                break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        data = {
                'video_frames': video, # C, N, H, W
                'text_ids': torch.tensor(text_ids), # Seq_len
                'attention_mask': torch.tensor(attention_mask),
                }

        if self.return_rawtext:
            data['rawtext'] = rawtext

        if self.return_index:
            data['index'] = index

        return data
     
if __name__ == '__main__':

    import pdb; pdb.set_trace()
