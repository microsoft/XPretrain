import os
import random
import jsonlines
import decord
import lmdb
from decord import VideoReader, cpu
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from src.utils.logger import LOGGER

decord.bridge.set_bridge("torch")

class RetrievalDataset(Dataset):

    def __init__(self,
                 cfg,
                 metadata_dir,
                 video_path,
                 sample_frame,
                 sample_clip,
                 tokenizer,
                 transform=None,
                 is_train = True,
                 return_rawtext=False,
                 return_index=False,
                 **kwargs
                 ):
        self.cfg = cfg
        self.metadata_dir = metadata_dir
        self.transform = transform
        self.video_path = video_path
        self.return_rawtext = return_rawtext
        self.return_index = return_index
        self.reliable_idx_list = []
        self.sample_frame = sample_frame
        self.sample_clip = sample_clip

        self.is_train = is_train
        self._load_metadata()
        self.tokenizer = tokenizer
        

    def _load_metadata(self):
        data = []
        with open(self.metadata_dir) as f:
            for l in jsonlines.Reader(f):
                data.append(l)
        self.metadata = data

    def _read_video(self, video_id, sample_frame_num):
        '''
        read frames from long video
        args:
            video_id: str,
            sample_frame_num: frames used
        return: 
            img_arrays: [num_frm, 3, H, W]
            chunk_mask: [num_frm, n_clip], , mask for indicating frames belong to each clip

        '''

        video_path = os.path.join(self.video_path, video_id + '.mp4')
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frame = len(vr)
        if self.is_train:
            interval = int(num_frame / (sample_frame_num - 1))
            start = np.random.randint(0, interval+1)
            end = np.random.randint(num_frame-1-interval, num_frame)
            frame_idx =  np.linspace(start, end, sample_frame_num).astype(int)
        else:
            frame_idx = np.linspace(0, num_frame-1, sample_frame_num).astype(int)

        img_arrays = vr.get_batch(frame_idx)

        img_arrays = img_arrays.float() / 255
  
        img_arrays = img_arrays.permute(0, 3, 1, 2) # N,C,H,W

        return img_arrays

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
        if (not hasattr(self.cfg.DATA, 'use_split')) or self.cfg.DATA.use_split:

            def merge(texts, tolen=8):
                if len(texts) <= tolen:
                    return texts
                else:
                    while len(texts) > tolen:
                        texts_2g = [len(texts[i])+len(texts[i+1]) for i in range(len(texts)-1)]
                        min_index = texts_2g.index(min(texts_2g))
                        texts_group = []
                        for i in range(len(texts)):
                            if i != min_index and i != min_index+1:
                                texts_group.append(texts[i])
                            elif i == min_index:
                                texts_group.append(' '.join(texts[i:i+2]))
                            else:
                                continue
                        texts = texts_group
                    return texts

            if len(texts) > total_chunk:
                texts = merge(texts, tolen=total_chunk)

            encoded = [self.tokenizer(x, padding='max_length', truncation=True, max_length=max_length) for x in texts]

            text_ids = [x.input_ids for x in encoded]
            attention_mask = [x.attention_mask for x in encoded]

            if len(texts) < total_chunk:
                for i in range(total_chunk-len(texts)):
                    text_ids.append([0 for x in range(max_length)])
                    attention_mask.append([0 for x in range(max_length)])

        else:
            texts = ' '.join(texts)
            encoded = [self.tokenizer(x, padding='max_length', truncation=True, max_length=max_length) for x in [texts]]

            text_ids = [x.input_ids for x in encoded]
            attention_mask = [x.attention_mask for x in encoded]

        return text_ids, attention_mask

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        num_retries = 10
        for j in range(num_retries):

            try:

                item = self.metadata[index]

                clip_id = item['clip_id']

                video= self._read_video(clip_id, self.sample_frame)

                rawtext = item['text']
                if not isinstance(rawtext, list):
                    rawtext = [rawtext]
                text_ids, attention_mask = self.tokenize(rawtext, total_chunk=self.sample_clip)

                if self.transform is not None:
                    video = self.transform(video) # N, C, H, W
                video = video.permute(1, 0, 2, 3) # C, N, H, W
            
            except Exception as e:
                LOGGER.info(f"Failed to load examples with video: {clip_id}. "
                                f"Will try again.")
                continue
            else:
                break

        data = {
                    'video_frames': video, # C, N, H, W
                    'text_ids': torch.tensor(text_ids), # Seq_len
                    'attention_mask': torch.tensor(attention_mask),
                    }

        if self.return_rawtext:
            data['rawtext'] = rawtext

        if self.return_index:
            data['index'] = torch.tensor(index)

        return data

