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

class How2QADataset(Dataset):
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

        video_path = os.path.join(self.video_path, video_id + '.mp4')
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frame = len(vr)

        frame_span = [0, num_frame-1]

        if self.is_train:
            interval = int(num_frame / (sample_frame_num - 1))
            start = np.random.randint(0, interval+1)
            end = np.random.randint(num_frame-1-interval, num_frame)
            frame_idx = np.linspace(start, end, num=sample_frame_num).astype(int)
        else:
            frame_idx = np.linspace(frame_span[0], frame_span[1], num=sample_frame_num).astype(int)

        img_arrays = vr.get_batch(frame_idx)
        img_arrays = np.array(img_arrays) 
        img_arrays = torch.from_numpy(img_arrays).float() / 255
        
        img_arrays = img_arrays.permute(0, 3, 1, 2) 

        return img_arrays,num_frame 

    def tokenize(self, texts_q, texts_a, text_s, subtitle_spans, num_frame, max_length = 50, max_num_subtitle = 2):

        def merge(texts, spans, tolen=8):
            if len(texts) <= tolen:
                return texts, spans
            else:
                while len(texts) > tolen:
                    texts_2g = [len(texts[i])+len(texts[i+1]) for i in range(len(texts)-1)]
                    min_index = texts_2g.index(min(texts_2g))
                    texts_group = []
                    spans_group = []
                    for i in range(len(texts)):
                        if i != min_index and i != min_index+1:
                            texts_group.append(texts[i])
                            spans_group.append(spans[i])
                        elif i == min_index:
                            texts_group.append(' '.join(texts[i:i+2]))
                            spans_group.append((spans[i][0], spans[i+1][1]))
                        else:
                            continue
                    texts = texts_group
                    spans = spans_group
                return texts, spans

        if len(text_s) > max_num_subtitle:
            text_s, subtitle_spans = merge(text_s, subtitle_spans, tolen=max_num_subtitle)
        

        encoded_q = self.tokenizer(texts_q,padding='max_length', truncation=True, max_length=max_length)
        encoded_a = [self.tokenizer(x,padding='max_length', truncation=True, max_length=max_length) for x in texts_a]
        encoded_s = [self.tokenizer(x,padding='max_length', truncation=True, max_length=max_length) for x in text_s]

        text_ids_q = torch.tensor(encoded_q.input_ids)
        attention_mask_q = torch.tensor(encoded_q.attention_mask)

        text_ids_a = torch.tensor([x.input_ids for x in encoded_a])
        attention_mask_a = torch.tensor([x.attention_mask for x in encoded_a])

        text_ids_s = [x.input_ids for x in encoded_s]
        attention_mask_s = [x.attention_mask for x in encoded_s]

        if len(text_s) < max_num_subtitle:
            for i in range(max_num_subtitle-len(text_s)):
                text_ids_s.append([0 for x in range(max_length)])
                attention_mask_s.append([0 for x in range(max_length)])

        text_ids_s = torch.tensor(text_ids_s)
        attention_mask_s = torch.tensor(attention_mask_s)

        text_ids =  torch.cat([text_ids_q.repeat(4,1,1), text_ids_a.unsqueeze(1), text_ids_s.repeat(4,1,1)], dim=1)
        attention_mask = torch.cat([attention_mask_q.repeat(4,1,1), attention_mask_a.unsqueeze(1), attention_mask_s.repeat(4,1,1)], dim=1)

        return text_ids, attention_mask 

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, index):

        item = self.metadata[index]

        clip_id = item['clip_id']
        span = item['span']

        rawtext_q = item['text_q']
        rawtext_a = item['text_a']
        rawtext_s = item['text_s']
        label = item['answer_idx']

        subtitle_spans = [(x['start'], x['end']) for x in rawtext_s]

        rawtext_s = [x['text'] for x in rawtext_s]

        video, num_frame = self._read_video(clip_id, self.sample_frame)

        if self.transform is not None:
            video = self.transform(video) 
        video = video.permute(1, 0, 2, 3) 

        text_ids, attention_mask = self.tokenize(rawtext_q, rawtext_a,rawtext_s,subtitle_spans,num_frame,max_num_subtitle=self.cfg.DATA.max_num_subtitle)

        temporal_label, temporal_label_weight = get_temporal_loss_label(span, num_frame, num_labels=self.sample_frame)
        temporal_label = np.array(temporal_label)  
        temporal_label = torch.from_numpy(temporal_label)
        temporal_label_weight = np.array(temporal_label_weight)  
        temporal_label_weight = torch.from_numpy(temporal_label_weight)

        data = {
                'video_frames': video, 
                'text_ids': text_ids, 
                'attention_mask': attention_mask,
                'label': label,
                'temporal_label': temporal_label,
                'temporal_label_weight': temporal_label_weight
                }

        return data

def get_temporal_loss_label(span, num_frame, num_labels=32, fps=3):
    def clamp(minimum, x, maximum):
        return max(minimum, min(x, maximum))
    label = [0] * num_labels
    total_time = num_frame / fps
    if span[0] == span[0] and span[1] == span[1]:
        start = span[0] / total_time
        end = span[1] / total_time
        start = clamp(0, start, 1)
        end = clamp(0, end, 1)
    else:
        start = 0
        end = 1
    start = int(round(start * num_labels))
    end = int(round(end * num_labels))
    start = clamp(0, start, num_labels-1)
    end = clamp(start+1, end, num_labels)
    label[start:end] = [1]*(end-start)
    N1 = end-start
    N0 = max(1, num_labels - N1)
    label_weight = [num_labels*0.5/N0] * num_labels
    label_weight[start:end] = [num_labels*0.5/N1]*(end-start)
    return label, label_weight
