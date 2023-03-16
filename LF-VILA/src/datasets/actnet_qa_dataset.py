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

class ActnetQADataset(Dataset):
    def __init__(self,
                 cfg,
                 metadata_dir,
                 video_path,
                 sample_frame,
                 sample_clip,
                 tokenizer,
                 transform=None,
                 is_train=True,
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

        self._load_metadata()
        self.tokenizer = tokenizer
        self.is_train = is_train

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
            frame_idx = np.linspace(start, end, num=sample_frame_num).astype(int)
        else:
            frame_idx = np.linspace(0, num_frame-1, num=sample_frame_num).astype(int)

        img_arrays = vr.get_batch(frame_idx)

        img_arrays = img_arrays.float() / 255
  
        img_arrays = img_arrays.permute(0, 3, 1, 2) # N,C,H,W

        return img_arrays

    def tokenize(self, text_q, max_length = 50):
        text_q = [text_q]
        
        encoded_qa = [self.tokenizer(x, padding='max_length', truncation=True, max_length=max_length) for x in text_q]

        text_ids = torch.tensor([x.input_ids for x in encoded_qa])
        attention_mask = torch.tensor([x.attention_mask for x in encoded_qa])
        return text_ids, attention_mask

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, index):
        num_retries = 10
        for j in range(num_retries):
            try:
                item = self.metadata[index]

                clip_id = item['video_name']

                video = self._read_video(clip_id, self.sample_frame)

                rawtext_q = item['question']
                label_a = item['answer']

                text_ids, attention_mask = self.tokenize(rawtext_q)
                

                if self.transform is not None:
                    video = self.transform(video)
                video = video.permute(1, 0, 2, 3)

                data = {
                        'video_frames': video,
                        'text_ids': text_ids,
                        'attention_mask': attention_mask,
                        'label': label_a
                        }
            except:
                index = random.randint(0, len(self) - 1)
                continue
            else:
                break

        if self.return_rawtext:
            data['rawtext'] = rawtext_q

        if self.return_index:
            data['index'] = torch.tensor(index)

        return data


