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
import glob
from PIL import Image

class HDVILAMSRVTTMCEvalDataset(Dataset):
    """
    datalist
    """

    def __init__(self, cfg, tokenizer, vis_dir, anno_path, vis_format='video',is_train=True,is_inference = False):
        assert vis_format in ["video","frame"]
        self.tokenizer = tokenizer
        self.vis_dir = vis_dir
        self.anno_path = anno_path
        self.cfg = cfg
        self.is_train = is_train
        self.is_inference = is_inference
        self.vis_format = vis_format
        self.n_clips = self.cfg.train_n_clips if not self.is_inference else self.cfg.inference_n_clips
        self.num_frm = self.cfg.num_frm
        assert self.num_frm % 2 == 1
        self.neighbor = int((self.num_frm - 1) / 2)
        self.sample_rate = self.cfg.sample_rate
        self.init_dataset_process()
        self.id2answer = {d["qid"]: int(d["answer"]) for d in self.datalist}

    def init_dataset_process(self):
        json_type = os.path.splitext(self.anno_path)[-1]
        assert json_type in ['.json', '.jsonl']

        if json_type == '.jsonl':
            data = []
            with open(self.anno_path) as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            data = json.load(open(self.anno_path))
        self.datalist = data

    def id2path(self, id):
        clip_name = id
        if self.vis_format == 'video':
            name = os.path.join(self.vis_dir, clip_name.split('/')[-1]+".mp4")
        else:
            name = os.path.join(self.vis_dir, clip_name)
        return name

    def __len__(self):
        return len(self.datalist)

    def get_sample_idx(self, total_frame_num):
        sample_rate = self.sample_rate
        if total_frame_num < 2 * self.neighbor * self.sample_rate + self.n_clips:
            sample_rate = math.floor((total_frame_num - self.n_clips) / (2 * self.neighbor))
        if not self.is_inference:
            middle_idx = random.sample(
                list(range(self.neighbor * sample_rate, total_frame_num - self.neighbor * sample_rate)), self.n_clips)
        else:
            middle_idx = list(range(self.neighbor * sample_rate, total_frame_num - self.neighbor * sample_rate))
            middle_idx = middle_idx[::int(len(middle_idx) / self.n_clips)][:self.n_clips]
        middle_idx.sort()
        frame_idx = []
        other_idx = []
        for m in middle_idx:
            frame_idx.extend(
                list(range(m - self.neighbor * sample_rate, m + self.neighbor * sample_rate + 1, sample_rate)))
            other_idx.extend(list(range(m - self.neighbor * sample_rate, m, sample_rate)) + list(
                range(m + sample_rate, m + self.neighbor * sample_rate + 1, sample_rate)))

        return middle_idx, other_idx, frame_idx

    def load_video(self, vis_path):
        vr = VideoReader(vis_path, ctx=cpu(0))
        total_frame_num = len(vr)

        middle_idx, other_idx, frame_idx = self.get_sample_idx(total_frame_num)
        img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)

        h = img_array.shape[-3]
        w = img_array.shape[-2]
        # some video are smaller than 720p
        if h != 180 or w != 288:
            img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
            img_array = torch.nn.functional.interpolate(img_array, size=(180, 288))
            img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

        h = img_array.shape[-3]
        w = img_array.shape[-2]

        crop_h, crop_w = self.cfg.crop_size[0], self.cfg.crop_size[1]
        if self.is_train:
            x = random.randint(0, h - crop_h)
            y = random.randint(0, w - crop_w)
            if_flip = random.choice([False, False, True])
            if if_flip:
                img_array = img_array[:, :, ::-1, :].copy()
        else:
            x = int((h - crop_h) / 2)
            y = int((w - crop_w) / 2)
        img_array = img_array[:, x: x + crop_h, y: y + crop_w, :]
        img_array = img_array.reshape(
            (self.n_clips, self.num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

        other_mask = np.ones(img_array.shape, dtype=bool)
        shape = list(img_array.shape)
        shape[1] = shape[1] - 1
        other_mask[:, self.neighbor] = False

        middle_frames = img_array[:, self.neighbor]
        other_frames = img_array[other_mask]

        middle_frames = np.asarray([cv2.resize(middle_frames[i], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC) \
                                    for i in range(middle_frames.shape[0])])

        other_frames = torch.from_numpy(other_frames).reshape(self.n_clips, self.num_frm - 1, crop_h, crop_w,
                                                              3).permute(0, 1, 4, 2, 3).float()
        middle_frames = torch.from_numpy(middle_frames).reshape(self.n_clips, 4 * crop_h, 4 * crop_w, 3).permute(0, 3,
                                                                                                                 1,
                                                                                                                 2).float()

        return middle_frames, other_frames


    def __getitem__(self, index):

        num_retries = 10  # skip error videos

        # for _ in range(num_retries):

        #     try:
        vis_id = self.datalist[index]['clip_name']
        texts = self.datalist[index]["options"]
        qid = self.datalist[index]["qid"]
            
        vis_path = self.id2path(vis_id)
        middle_frames, other_frames = self.load_video(vis_path)
        #         break

        #     except:
        #         LOGGER.info(f"Failed to load examples with video: {vis_id}. "
        #                         f"Will select an example from reliable list as a replacement.")
        #         index = random.randint(0, len(self) - 1)
        #         continue
        # else:
        #     raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        # img_middle_l, img_middle_h, img_other_l = self.process_img_array(img_array)
        # print('img_middle_l',img_middle_l.shape)
        # print('img_middle_h',img_middle_h.shape)
        # print('img_other_l',img_other_l.shape)

        return dict(
            img_middle_l=middle_frames,  # [clips, C, H_crop, W_crop]
            img_other_l=other_frames,  # [clips, num_frm-1, C, H_crop/4, W_crop/4]
            texts = texts,
            qid=qid
        )

    def evaluate_qa_accuracy(self, pred_id2answer, force_same=True):
        """
        Args:
            pred_id2answer: dict, {id: pred_answer_idx}
            force_same: bool, if True, the predictions should contain the same set of ids as the GT.
        """
        gt_ids = list(self.id2answer.keys())
        pred_ids = list(pred_id2answer.keys())
        print(f"There are {len(gt_ids)} gt ids, {len(pred_ids)} pred ids.")
        if force_same:
            assert set(gt_ids) == set(pred_ids)
            shared_ids = list(set(gt_ids) & set(pred_ids))
        else:
            shared_ids = pred_ids

        gt_answers = np.array([self.id2answer[k] for k in shared_ids])
        pred_answers = np.array([pred_id2answer[k] for k in shared_ids])
        acc = np.mean(gt_answers == pred_answers)
        return dict(mc_accuracy=f"{100 * acc:.2f}")



class MSRVTTMCCollator(object):
    def __init__(self, tokenizer, max_length=40, is_train=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

    def collate_batch(self, batch):
        if isinstance(batch[0]["img_middle_l"], torch.Tensor):
            v_collate = default_collate
        else:
            v_collate = img_collate
        img_middle_l = v_collate([d["img_middle_l"] for d in batch])

        # if isinstance(batch[0]["img_middle_h"], torch.Tensor):
        #     v_collate = default_collate
        # else:
        #     v_collate = img_collate
        # img_middle_h = v_collate([d["img_middle_h"] for d in batch])

        if isinstance(batch[0]["img_other_l"], torch.Tensor):
            v_collate = default_collate
        else:
            v_collate = img_collate
        img_other_l = v_collate([d["img_other_l"] for d in batch])


        # group data
        text_examples = flat_list_of_lists([d["texts"] for d in batch])
        question_ids = [d["qid"] for d in batch]

        # group elements data
        # directly concatenate question and option as a single seq.
        text_str_list = [d for d in text_examples]  # (B, )
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_tensors="pt"
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)

        collated_batch = dict(
            img_middle=img_middle_l.float(), # [B, clips, C, H_crop, W_crop]
            img_other=img_other_l.float(),   # [B, clips, num_frm-1, C, H_crop/4, W_crop/4]
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask,
            question_ids=question_ids,

        )

        return collated_batch
