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

class VideoClassificationDataset(Dataset):
    def __init__(self,
                 cfg,
                 metadata_dir,
                 video_path,
                 sample_frame,
                 sample_clip,
                 tokenizer,
                 transform=None,
                 return_index=False,
                 is_train=True,
                 **kwargs
                 ):
        self.cfg = cfg
        self.metadata_dir = metadata_dir
        self.transform = transform
        self.video_path = video_path
        self.return_index = return_index
        self.reliable_idx_list = []
        self.sample_frame = sample_frame

        self._load_metadata()
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

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        num_retries = 10
        for j in range(num_retries):
            try:
                item = self.metadata[index]

                video_id = item['video_id']

                video = self._read_video(video_id, self.sample_frame)


                label = int(item['recipe_type'])

                if self.transform is not None:
                    video = self.transform(video) # N, C, H, W
                video = video.permute(1, 0, 2, 3) # C, N, H, W

                data = {
                        'video_frames': video, # C, N, H, W
                        'label': torch.tensor(label)
                        }
            except:
                index = random.randint(0, len(self) - 1)
                continue
            else:
                break

        if self.return_index:
            data['index'] = torch.tensor(index)

        return data


