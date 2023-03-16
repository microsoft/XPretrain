import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import transforms


from .pretrain_dataset import PreTrainDataset
from .retrieval_dataset import RetrievalDataset

from .actnet_qa_dataset import ActnetQADataset
from .how2qa_dataset import How2QADataset
from .violin_dataset import ViolinDataset
from .video_classification_dataset import VideoClassificationDataset

from transformers import BertTokenizer
from transformers import CLIPTokenizer

from src.utils.logger import LOGGER
from src.utils.data import mask_batch_text_tokens

from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import default_collate

from src.utils.dist import SequentialDistributedSampler


class PretrainCollator(object):
    """is_train is kept here if we want to remove
    the randomness during validation of MLM accuracy.
    In that case, instantiate two PretrainCollator"""
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability

    def collate_batch(self, batch):

        video_frames = default_collate([d["video_frames"] for d in batch])

        if 'attention_mask' in batch[0]:
            attention_mask = default_collate([d["attention_mask"] for d in batch])
        else:
            attention_mask = None

        if 'index' in batch[0]:
            index = default_collate([d["index"] for d in batch])
        else:
            index = None

        # group data
        if 'text_ids' in batch[0]:
            text_ids = default_collate([d["text_ids"] for d in batch]) # (B, L)
            if self.mlm:
                B,M,L = text_ids.shape
                text_ids, mlm_labels = mask_batch_text_tokens(
                    text_ids.view(B, M*L), self.tokenizer)  # make mlm data
                text_ids = text_ids.view(B,M,L)
                mlm_labels = mlm_labels.view(B, M*L)
            else:
                text_ids, mlm_labels = text_ids, None

        else:
            text_ids, mlm_labels = None, None

        if 'label' in batch[0]:
            labels = default_collate([d["label"] for d in batch])
        else:
            labels = None

        if 'temporal_label' in batch[0]:
            temporal_labels = default_collate([d["temporal_label"] for d in batch])
        else:
            temporal_labels = None

        if 'temporal_label_weight' in batch[0]:
            temporal_label_weights = default_collate([d["temporal_label_weight"] for d in batch])
        else:
            temporal_label_weights = None


        return dict(
                video_frames=video_frames, # C, N, H, W
                text_ids=text_ids, # Seq_len
                mlm_labels=mlm_labels,
                attention_mask=attention_mask,
                labels=labels,
                temporal_labels = temporal_labels,
                temporal_label_weights=temporal_label_weights,
                index = index
                )


def init_transform_dict(input_res=(192, 320),
                        center_crop=200,
                        randcrop_scale=(0.8, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225)):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    transform_dict = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_res, scale=randcrop_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize([240, 428]),
            transforms.CenterCrop([int(240*0.9), int(428*0.9)]),
            transforms.Resize(input_res),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize([240, 428]),
            transforms.CenterCrop([int(240*0.9), int(428*0.9)]),
            transforms.Resize(input_res),
            normalize,
        ])
    }
    return transform_dict

def build_dataset(args,config,tokenizer,split='train'):
    transform=init_transform_dict(config.DATA.input_res, config.DATA.center_crop)[split]

    dataset_dicts = config.DATA.DATASET_train if split=='train' else config.DATA.DATASET_val
    if isinstance(dataset_dicts, dict):
        dataset_dicts = [dataset_dicts]
    datasets = {}
    for dataset_dict in dataset_dicts:
        name = dataset_dict['name']
        metadata_dir=os.path.join(args.blob_mount_dir, dataset_dict['metadata_dir'])
        video_path=os.path.join(args.blob_mount_dir, dataset_dict['video_path'])
        sample_frame = config.DATA.sample_frame
        sample_clip = config.DATA.sample_clip

        if hasattr(config.TRAINING, 'save_feats'):
            return_index = config.TRAINING.save_feats
        else:
            return_index = False

        dataset = globals()[dataset_dict['type']](config,
                                metadata_dir,
                                video_path,
                                sample_frame,
                                sample_clip,
                                tokenizer,
                                transform=transform,
                                is_train=True if split=='train' else False,
                                return_index=return_index)
  
        LOGGER.info(f'build dataset: {name}, {len(dataset)}')

        datasets[name] = dataset
    return datasets



def build_dataloader(args, config):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    if hasattr(config.DATA, "use_subtitle_span"):
        tokenizer._additional_special_tokens = ["[unused%d]"%(i+1) for i in range(config.DATA.sample_frame)]

    dataset_trains = build_dataset(args, config,tokenizer, split='train')

    dataset_vals = build_dataset(args, config, tokenizer,split='val')

    data_collator = PretrainCollator(tokenizer=tokenizer,
                                     mlm=config.stage == 2 and config.TRAINING.use_mlm,
                                     mlm_probability=0.15)

    sampler_train, sampler_val = None, None

    if args.distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()

        LOGGER.info(f'using dist training, build sampler')
        
    data_loader_trains = {}
    for k,dataset_train in dataset_trains.items():
        if args.distributed:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=config.DATA.BATCH_SIZE_per_gpu,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            collate_fn = data_collator.collate_batch,
            drop_last=True,
        )

        data_loader_trains[k] = data_loader_train

    data_loader_vals = {}
    for k,dataset_val in dataset_vals.items():

        if args.distributed:
            sampler_val = SequentialDistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=config.DATA.BATCH_SIZE_per_gpu,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            collate_fn = data_collator.collate_batch,
            drop_last=False
        )
        data_loader_vals[k] = data_loader_val

    LOGGER.info(f'build dataloader done!')
    LOGGER.info(f'dataloader_train: {len(data_loader_train)}')
    for k,v in data_loader_vals.items():
        LOGGER.info(f'data_loader_val {k}: {len(v)}')
    return dataset_trains, dataset_vals, data_loader_trains, data_loader_vals

