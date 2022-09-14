import torch
import random
import numpy as np
import copy
from torch.utils.data.dataloader import default_collate
from src.utils.basic_utils import flat_list_of_lists
from src.utils.load_save import LOGGER
from PIL import Image
from PIL import GifImagePlugin
import os
from decord import VideoReader
from decord import cpu, gpu
import math
import torch.nn.functional as F
import cv2
from src.datasets.data_utils import (
    ImageResize, ImagePad, image_to_tensor)

class HDVILAVideoQADataset():
    """ This should work for both train and test (where labels are not available).
    task_type: str, one of [action, frameqa, transition]
        where action and transition are multiple-choice QA,
            frameqa is opened QA similar to VQA.
    datalist: list(tuples)  each tuple is (img_id, list(dicts)),
        each dict
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    itm_neg_prob: float [0, 1] set to 0 will disable itm.
    return_label: bool, whether return label in __getitem__
    random_sample_clips:
    """
    open_ended_qa_names = ["frameqa", "msrvtt_qa"]

    def __init__(self, cfg, task_type, datalist, tokenizer, img_lmdb_dir,
                 fps=3, num_frm=3, frm_sampling_strategy="rand",
                 max_img_size=1000, max_txt_len=20, ans2label=None,
                 ensemble_n_clips=1, return_label=True, is_train=True, random_sample_clips=True, img_aug=0):
        self.cfg = cfg
        self.datalist = datalist
        self.img_lmdb_dir = img_lmdb_dir
        if "gif" in img_lmdb_dir:
            self.use_gif = True
        else:
            self.use_gif = False
        self.is_inference = cfg.do_inference
        self.n_clips = self.cfg.train_n_clips if not self.is_inference else self.cfg.inference_n_clips
        self.num_frm = num_frm
        self.sample_rate = self.cfg.sample_rate
        self.neighbor = int((num_frm - 1) / 2)
        self.return_label = return_label
        self.is_train = is_train
        self.task_type = task_type
        self.ans2label = ans2label
        self.num_labels = len(ans2label)

        self.label2ans = {v: k for k, v in ans2label.items()}
        self.qid2data = {d["question_id"]: d for group in datalist for d in group[1]}

        self.img_resize = ImageResize(
            max_img_size,
            "bilinear")  # longer side will be resized to 1000
        if hasattr(cfg, "pad_value"):
            pad_value = int(255/2)
        else:
            pad_value = 0
        self.img_pad = ImagePad(
            max_img_size, max_img_size, fill=pad_value)
        self.max_img_size = max_img_size

    def __len__(self):
        return len(self.datalist)

    def id2path(self, vid):
        return os.path.join(self.img_lmdb_dir, vid + ".mp4")

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
            if sample_rate > 0:
                frame_idx.extend(
                    list(range(m - self.neighbor * sample_rate, m + self.neighbor * sample_rate + 1, sample_rate)))
                other_idx.extend(list(range(m - self.neighbor * sample_rate, m, sample_rate)) + list(
                    range(m + sample_rate, m + self.neighbor * sample_rate + 1, sample_rate)))
            else:
                frame_idx.extend([m] * self.num_frm)
                other_idx.extend([m] * (self.num_frm - 1))

        return middle_idx, other_idx, frame_idx

    def load_video(self, vid):
        vis_path = self.id2path(vid)
        vr = VideoReader(vis_path, ctx=cpu(0))
        total_frame_num = len(vr)

        middle_idx, other_idx, frame_idx = self.get_sample_idx(total_frame_num)
        img_array = vr.get_batch(frame_idx).asnumpy() # (n_clips*num_frm, H, W, 3)

        h = img_array.shape[-3]
        w = img_array.shape[-2]

        if self.use_gif:
            img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
            img_array = self.img_pad(self.img_resize(img_array))
            img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
            h, w = self.max_img_size, self.max_img_size
            crop_h, crop_w = self.max_img_size, self.max_img_size

        else:
            if h != self.cfg.reshape_size[0] or w != self.cfg.reshape_size[1]:
                img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
                img_array = torch.nn.functional.interpolate(img_array, size=(self.cfg.reshape_size[0], self.cfg.reshape_size[1]))
                img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
            h, w = self.cfg.reshape_size[0], self.cfg.reshape_size[1]
            crop_h, crop_w = self.cfg.crop_size[0], self.cfg.crop_size[1]

        if self.is_train:
            x = random.randint(0, h - crop_h)
            y = random.randint(0, w - crop_w)
            if_flip = random.choice([False, False, True])
            if if_flip:
                img_array = img_array[:, :, ::-1, :].copy()
        else:
            x = int((h-crop_h)/2)
            y = int((w-crop_w)/2)
        img_array = img_array[:, x: x + crop_h, y: y + crop_w,:]
        img_array = img_array.reshape((self.n_clips, self.num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

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
        middle_frames = torch.from_numpy(middle_frames).reshape(self.n_clips, 4*crop_h, 4*crop_w, 3).permute(0,3,1,2).float()

        return middle_frames, other_frames

    def __getitem__(self, index):
        # skip error videos:
        num_retries = 5
        for _ in range(num_retries):
            vid_id, examples = self.datalist[index]  # one video with multiple examples

            try:
                middle_frames, other_frames = self.load_video(vid_id)

            # Select a random video if the current video was not able to access.
            except:
                LOGGER.info(f"Failed to load examples with video: {vid_id}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            examples = [self._get_single_example(e) for e in examples]

            return dict(
                img_middle=middle_frames,
                img_other=other_frames,
                examples=examples,
                n_examples=len(examples)  # used to create image feature copies.
            )
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

    def _get_single_example(self, data):
        example = dict(
            q_str=data["question"],
            question_id=data["question_id"],
            label=data["answer"]
        )
        if self.task_type in ["action", "transition"]:
            example["options_str_list"] = data["options"]
        elif self.task_type in self.open_ended_qa_names:
            if self.return_label:
                example["label"] = self.ans2label[example["label"]]
        if not self.return_label:
            example["label"] = None
        return example

    def evaluate_tgif_qa(self, results):
        """
        Args:
            results: list(dict),
              each dict is
                {
                    "question_id": int,
                    "answer": int or float, either answer_idx (int)
                }
        Returns:
            TGIF-QA score
        """
        preds = []
        gts = []
        # for frameQA
        answer_types = []
        answer_type2idx = dict(
            frameqa={"object": 0, "number": 1, "color": 2, "location": 3},
            msrvtt_qa={k: idx for idx, k in enumerate(["what", "who", "how", "where", "when"])},
        )

        qid2pred_ans = {r["question_id"]: r["answer"] for r in results}
        if self.task_type in self.open_ended_qa_names:  # convert ans_idx, int --> str
            qid2pred_ans = {k: self.label2ans[v] for k, v in qid2pred_ans.items()}

        for qid, pred_ans in qid2pred_ans.items():
            preds.append(pred_ans)

            gt_data = self.qid2data[qid]
            gt_ans = gt_data["answer"]
            if self.task_type in self.open_ended_qa_names:
                answer_types.append(answer_type2idx[self.task_type][gt_data["answer_type"]])
            gts.append(gt_ans)

        preds = np.array(preds)
        gts = np.array(gts)
        metrics = dict()
        # preds and gts are array of strings
        metrics["overall_acc"] = float(np.mean(preds == gts))
        if self.task_type in self.open_ended_qa_names:
            answer_types = np.array(answer_types)
            ratios = dict()
            for ans_type, ans_type_idx in answer_type2idx[self.task_type].items():
                answer_type_mask = answer_types == ans_type_idx
                answer_type_corrects = (
                        preds[answer_type_mask] == gts[answer_type_mask])
                metrics[f"{ans_type}_acc"] = float(
                    np.mean(answer_type_corrects)) if len(answer_type_corrects) != 0 else 0
                ratios[f"{ans_type}_ratio"] = [
                    1. * len(answer_type_corrects) / len(answer_types),
                    len(answer_type_corrects)]
            metrics["ratios"] = ratios
        return metrics


class VideoQACollator(object):
    def __init__(self, tokenizer, max_length=20, task_type="action", n_options=5):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.n_options = n_options

    def collate_batch(self, batch):
        v_collate = default_collate
        img_middle = v_collate([d["img_middle"] for d in batch])
        img_other = v_collate([d["img_other"] for d in batch])
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        # directly concatenate question and option as a single seq.
        if self.task_type in ["action", "transition"]:
            text_str_list = flat_list_of_lists(
                [[d["q_str"] + " " + d["options_str_list"][i] for i in range(self.n_options)]
                 for d in text_examples]
            )  # (B * n_options, )
        else:
            text_str_list = [d["q_str"] for d in text_examples]  # (B, )
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_tensors="pt"
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)

        labels = default_collate([int(d["label"]) for d in text_examples]) \
            if text_examples[0]["label"] is not None else None  # (B, #ans)
        question_ids = [d["question_id"] for d in text_examples]

        return dict(
            img_middle=img_middle.float(),  # [B, clips, C, H, W]
            img_other=img_other.float(),  # [B, clips, num_frm-1, C, H/4, W/4]
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask,
            question_ids=question_ids,
            labels=labels,
            # n_examples_list=n_examples_list  # used to create image feature copies.
        )
