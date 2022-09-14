import torch
from torch.utils.data import Dataset
import random
import os
import json
from torch.utils.data.dataloader import default_collate
from src.utils.logger import LOGGER
from src.utils.basic_utils import flat_list_of_lists
from src.datasets.data_utils import mask_batch_text_tokens, img_collate
from decord import VideoReader
from decord import cpu, gpu
import math
import torch.nn.functional as F
import numpy as np
import cv2
import lmdb

class HDVILAPretrainDataset(Dataset):
    def __init__(self, cfg, tokenizer, vis_dir, anno_path, vis_format="video", is_train=True):
        assert vis_format == "video"
        self.tokenizer = tokenizer
        self.vis_dir = vis_dir
        self.anno_path = anno_path
        self.cfg = cfg
        self.is_train = is_train
        self.itm_neg_prob = cfg.itm_neg_prob
        self.use_itm = cfg.use_itm
        self.reliable_idx_list = []
        self.init_dataset_process()

    def init_dataset_process(self):
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
                    data.append(json.loads(line, strict=False))
        else:
            data = json.load(open(self.anno_path))
        self.datalist = data


    def id2path(self, id):
        clip_name = id
        video_name = id.split('.')[0]
        if 'msrvtt' in self.vis_dir:
            return os.path.join(self.vis_dir, clip_name+".mp4")
        return os.path.join(self.vis_dir, video_name, clip_name)

    def __len__(self):
        return len(self.datalist)

    def load_video(self, vis_path):
        vr = VideoReader(vis_path, ctx=cpu(0))
        total_frame_num = len(vr)
        n_clips = self.cfg.train_n_clips
        num_frm = self.cfg.num_frm
        assert num_frm % 2 == 1
        neighbor = int((num_frm - 1) / 2)
        sample_rate = self.cfg.sample_rate
        if total_frame_num < 2*neighbor*sample_rate + n_clips:
            sample_rate = math.floor((total_frame_num - n_clips)/(2*neighbor))
        middle_idx = random.sample(list(range(neighbor * sample_rate, total_frame_num - neighbor * sample_rate)), n_clips)
        middle_idx.sort()
        frame_idx = []
        for m in middle_idx:
            frame_idx.extend(list(range(m-neighbor*sample_rate, m+neighbor*sample_rate+1, sample_rate)))
        img_array = vr.get_batch(frame_idx).asnumpy() # (n_clips*num_frm, H, W, 3)

        if hasattr(self.cfg, "360p"):
            h, w = 360, 640
        elif hasattr(self.cfg, "180p"):
            h, w = 180, 320
        else:
            h, w = 720, 1280
        # some video are smaller than 720p
        if img_array.shape[-3] != h or img_array.shape[-2] != w:
            img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
            img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
            img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
        img_array = img_array.reshape((n_clips, num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
        return img_array

    def get_images(self, img_array, down4=False, up4=False):
        img_array_copy = img_array.reshape((-1, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
        output = []
        for i in range(img_array_copy.shape[0]):
            if down4:
                up = cv2.resize(img_array_copy[i], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
            if up4:
                up = cv2.resize(img_array_copy[i], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            else:
                up = img_array_copy[i]
            output.append(up)
        return np.asarray(output)

    def process_img_array(self, img_array):
        """
        Input img_array: [clips, num_frm, H, W, C]
        return: middle_images: [clips, 1, C, H_crop, W_crop]
                sample_patches: [clips, num_frm, patch_num, C, kernel, kernel]
                sample_patches_gt: [clips, 1, patch_num, C, kernel, kernel]
                sample_idx: [clips, patch_num]
        """
        n_clips = img_array.shape[0]
        num_frm = img_array.shape[1]
        neighbor = int((num_frm - 1) / 2)
        h = img_array.shape[-3]
        w = img_array.shape[-2]
        crop_h, crop_w = self.cfg.crop_size[0], self.cfg.crop_size[1]
        x = random.randint(0, h - crop_h)
        y = random.randint(0, w - crop_w)
        img_array = img_array[:, :, x:x+crop_h, y:y+crop_w, :]  # [clips, num_frm, H_crop, W_crop, C]

        other_mask = np.ones(img_array.shape, dtype=bool)
        shape = list(img_array.shape)
        shape[1] = shape[1] - 1
        other_mask[:, neighbor] = False

        if hasattr(self.cfg, "180p"):
            img_middle = self.get_images(img_array[:, neighbor], up4=True)  # [clips, H_crop, W_crop, C]
            img_other = self.get_images(img_array[other_mask].reshape(shape))
        else:
            img_middle = self.get_images(img_array[:, neighbor])  # [clips, H_crop, W_crop, C]
            img_other = self.get_images(img_array[other_mask].reshape(shape), down4=True)

        img_other = img_other.reshape((n_clips, num_frm-1, img_other.shape[-3], img_other.shape[-2], img_other.shape[-1]))
        img_middle = torch.from_numpy(img_middle).permute(0, 3, 1, 2)   # [clips, C, H_crop, W_crop]
        img_other = torch.from_numpy(img_other).permute(0, 1, 4, 2, 3)  # [clips, num_frm-1, C, H_crop, W_crop]

        return img_middle, img_other

    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                vis_id = self.datalist[index]['clip_id']
                if self.use_lmdb:
                    texts = [self.txn.get(self.datalist[index]['clip_id'].encode()).decode()]
                else:
                    texts = [self.datalist[index]['text']]
                vis_path = self.id2path(vis_id)
            
                img_array = self.load_video(vis_path)
            except:
                LOGGER.info(f"Failed to load examples with video: {vis_id}. "
                            f"Will select an example from reliable list as a replacement.")
                if len(self.reliable_idx_list) > 0:
                    index = random.choice(self.reliable_idx_list)
                else:
                    index = random.randint(0, len(self) - 1)
                continue
            else:
                if len(self.reliable_idx_list) < 1000:
                    self.reliable_idx_list.append(index)
                break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        examples = [self._get_single_example(e, index) for e in texts]
        img_middle, img_other = self.process_img_array(img_array)

        return dict(
            img_middle=img_middle,  # [clips, C, H_crop, W_crop]
            img_other=img_other,  # [clips, num_frm-1, C, H_crop/4, W_crop/4]
            examples=examples,
            n_examples=len(examples)  # used to create image feature copies.
        )

    def _get_single_example(self, text, index):
        # sample itm
        # random.random is uniform distributed in [0.0, 1.0)
        if self.use_itm and random.random() < self.itm_neg_prob:
            text_str = self._get_random_negative_caption(index)
            itm_label = 0  # negative pair
        else:
            text_str = text
            itm_label = 1  # positive pair
        return dict(
            text_str=text_str,
            itm_label=itm_label
        )

    def _get_random_negative_caption(self, gt_index):
        max_trials = 5
        while max_trials > 0:
            neg_index = int(random.random() * len(self))
            if self.use_lmdb:
                text = self.txn.get(self.datalist[neg_index]['clip_id'].encode()).decode()
            else:
                text = self.datalist[neg_index]['text']
            if neg_index == gt_index:
                max_trials -= 1
                continue
            else:
                break

        if max_trials == 0:
            raise Warning(f"The negative sampler cannot sample a true negative within 5 trials")
        neg_txt = text
        return neg_txt


class PretrainCollator(object):
    """is_train is kept here if we want to remove
    the randomness during validation of MLM accuracy.
    In that case, instantiate two PretrainCollator"""
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15,
                 max_length=20, is_train=True):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.is_train = is_train

    def collate_batch(self, batch):
        if isinstance(batch[0]["img_middle"], torch.Tensor):
            v_collate = default_collate
        else:
            v_collate = img_collate
        img_middle = v_collate([d["img_middle"] for d in batch])

        if isinstance(batch[0]["img_other"], torch.Tensor):
            v_collate = default_collate
        else:
            v_collate = img_collate
        img_other = v_collate([d["img_other"] for d in batch])

        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        batch_enc = self.tokenizer.batch_encode_plus(
            [d["text_str"] for d in text_examples],
            max_length=self.max_length,
            pad_to_max_length=True,
            return_tensors="pt"
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        if self.mlm:
            text_input_ids, mlm_labels = mask_batch_text_tokens(
                text_input_ids, self.tokenizer,
                is_train=self.is_train)  # make mlm data
        else:
            text_input_ids, mlm_labels = text_input_ids, None
        text_input_mask = batch_enc.attention_mask  # (B, L)
        itm_labels = default_collate(
            [d["itm_label"] for d in text_examples])  # (B, )
        return dict(
            img_middle=img_middle.float(),
            img_other=img_other.float(),   # [B, clips, num_frm-1, C, H_crop/4, W_crop/4]
            text_input_ids=text_input_ids,
            mlm_labels=mlm_labels,
            text_input_mask=text_input_mask,
            itm_labels=itm_labels,
        )

