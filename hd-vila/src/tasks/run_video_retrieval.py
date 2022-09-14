import torch
import os
import time
import random
import math
from transformers import BertConfig, BertTokenizerFast
from src.modeling.modeling_stage import HDVILAForPreTraining

from src.modeling.e2e_model import HDVILA

from src.datasets.dataset_pretrain import HDVILAPretrainDataset, PretrainCollator

from src.datasets.dataset_video_retrieval import (
    HDVILAVideoRetrievalDataset, VideoRetrievalCollator)
from src.datasets.dataloader import InfiniteIterator, PrefetchLoader
from src.datasets.data_utils import ImageNorm, mk_input_group
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from src.configs.config import shared_configs
from src.utils.misc import set_random_seed, NoOp, zero_none_grad
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter
from src.utils.basic_utils import (
    load_jsonl, load_json, save_json, get_rounded_percentage)
from src.utils.load_save import (ModelSaver,
                                 BestModelSaver,
                                 save_training_meta,
                                 load_state_dict_with_mismatch)
from src.utils.load_save import E2E_TrainingRestorer as TrainingRestorer
from src.optimization.sched import get_lr_sched
from src.optimization.utils import setup_e2e_optimizer
from src.optimization.loss import build_loss_func

import numpy as np
from tqdm import tqdm
from os.path import join, exists
from easydict import EasyDict as edict
from apex import amp
from torch.utils.data.distributed import DistributedSampler
import horovod.torch as hvd
from src.utils.distributed import all_gather_list, SequentialDistributedSampler
from collections import defaultdict
from src.utils.metrics import cal_cossim, compute_metrics
import torchvision.utils as tutils



def mk_video_ret_dataloader(dataset_name, vis_format, anno_path, vis_dir, cfg, tokenizer, is_train=True,is_inference = False):
    """"""

    dataset = HDVILAVideoRetrievalDataset(
        cfg=cfg,
        tokenizer=tokenizer,
        vis_dir=vis_dir,
        anno_path=anno_path,
        vis_format=vis_format,
        is_train=is_train,
        is_inference = is_inference
    )
    LOGGER.info(f"[{dataset_name}] is_train {is_train} "
                f"dataset size {len(dataset)}, ")

    batch_size = cfg.train_batch_size if is_train else cfg.val_batch_size
    if is_train:
        sampler = DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank(),
            shuffle=is_train)
    else:

        sampler = SequentialDistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    vret_collator = VideoRetrievalCollator(
        tokenizer=tokenizer, max_length=cfg.max_txt_len, is_train=is_train)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=vret_collator.collate_batch)
    return dataloader



def setup_dataloaders(cfg, tokenizer):
    LOGGER.info("Init. train_loader and val_loader...")

    db = cfg.train_datasets
    train_loader = mk_video_ret_dataloader(
        dataset_name=db.name, vis_format=db.vis_format,
        anno_path=db.txt, vis_dir=db.vis,
        cfg=cfg, tokenizer=tokenizer, is_train=True
    )


    val_loaders = {}
    for db in cfg.val_datasets:
        val_loaders[db.name] = mk_video_ret_dataloader(
            dataset_name=db.name, vis_format=db.vis_format,
            anno_path=db.txt, vis_dir=db.vis,
            cfg=cfg, tokenizer=tokenizer, is_train=False
        )

    inference_loaders = {}
    for db in cfg.inference_datasets:
        inference_loaders[db.name] = mk_video_ret_dataloader(
            dataset_name=db.name, vis_format=db.vis_format,
            anno_path=db.txt, vis_dir=db.vis,
            cfg=cfg, tokenizer=tokenizer, is_train=False, is_inference=True
        )
    return train_loader, val_loaders, inference_loaders


def setup_model(cfg, device=None):
    LOGGER.info("Setup model...")
    # has to be a BertConfig instance
    model_cfg = load_json(cfg.model_config)
    model_cfg = BertConfig(**model_cfg)
    # add downstream model config
    # add model-specific config
    add_attr_list = [
        "pixel_random_sampling_size",
        "use_itc",
        "score_agg_func",
        "backbone_channels",
        "resnet_depth",
        "resnet_frozen_stage",
        "timesformer_depth",
        "timesformer_heads",
        "timesformer_type",
        "bert_mean",
        "bert_frozen_stage",
    ]
    for k in add_attr_list:
        setattr(model_cfg, k, cfg[k])
    

    # we separate the CNN and the transformer in order to use different optimizer for each
    # transformer still has a CNN layer inside, used to down sample grid.
    LOGGER.info("setup e2e model")
    model = HDVILA(config=model_cfg,
                     transformer_cls=HDVILAForPreTraining,
                     stage=1)

    if cfg.e2e_weights_path:
        LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
        load_state_dict_with_mismatch(model, cfg.e2e_weights_path)
    else:
        LOGGER.info(f"Loading cnn weights from {cfg.mmdetection_weights_path}")
        LOGGER.info(f"Loading bert weights from {cfg.bert_weights_path}")
        model.load_separate_ckpt(
            cnn_weights_path=cfg.mmdetection_weights_path,
            bert_weights_path=cfg.bert_weights_path)
    if cfg.freeze_cnn:
        model.freeze_cnn_backbone()
    model.to(device)

    LOGGER.info("Setup model done!")
    return model


def forward_step(model, batch, cfg):
    """shared for training and validation"""
    outputs = model(batch)  # dict
    return outputs


@torch.no_grad()
def validate(model, val_loaders, cfg):
    """use eval_score=False when doing inference on test sets where answers are not available"""
    model.eval()

    st = time.time()
    
    for loader_name, val_loader in val_loaders.items():
        LOGGER.info(f"Loop val_loader {loader_name}.")
        val_log = {f'valid/{loader_name}_t2v_recall_1': 0,
                   f'valid/{loader_name}_t2v_recall_5': 0,
                   f'valid/{loader_name}_t2v_recall_10': 0,
                   f'valid/{loader_name}_t2v_recall_50': 0,
                   f'valid/{loader_name}_t2v_recall_median': 0,
                   f'valid/{loader_name}_t2v_recall_mean': 0,
                   f'valid/{loader_name}_v2t_recall_1': 0,
                   f'valid/{loader_name}_v2t_recall_5': 0,
                   f'valid/{loader_name}_v2t_recall_10': 0,
                   f'valid/{loader_name}_v2t_recall_50': 0,
                   f'valid/{loader_name}_v2t_recall_median': 0,
                   f'valid/{loader_name}_v2t_recall_mean': 0}

        valid_len = len(val_loader.loader.dataset)
        text_feats = []
        vis_feats = []
        for val_step, batch in enumerate(val_loader):
            # use iter to reset MetaLoader
            # forward pass
            # print(val_step)

            feats = model(**batch)  # dict
            # # print('feats vis_features', feats['vis_features'].shape)
            # vis_feat = hvd.allgather(feats['vis_features'])
            # text_feat = hvd.allgather(feats['text_features'])

            # print('allgather vis_features', vis_feat.shape)

            vis_feats.append(feats['vis_features'])
            text_feats.append(feats['text_features'])

            if cfg.vis_steps>0 and (val_step % cfg.vis_steps == 0) and hvd.rank() == 0:
                other_frames = batch["img_other_l"] / 255.
                other_frames = other_frames.reshape(-1, other_frames.shape[-3], other_frames.shape[-2], other_frames.shape[-1])
                os.makedirs(join(cfg.output_dir, 'image'), exist_ok=True)
                tutils.save_image(other_frames, join(cfg.output_dir, 'image/val_step%d.png' % val_step), nrow=4,
                                normalize=True, range=(0.0, 1.0))
                LOGGER.info(f'Step {val_step}: Image saved')

            # if  hvd.rank() == 0:
            #     print('finished', val_step)

        vis_feats = torch.cat(vis_feats)
        text_feats = torch.cat(text_feats)

        vis_feats = hvd.allgather(vis_feats).cpu().numpy()
        text_feats = hvd.allgather(text_feats).cpu().numpy()
        

        text_feats = np.vstack(text_feats)[:valid_len]
        vis_feats = np.vstack(vis_feats)[:valid_len]

        sim_matrix = cal_cossim(text_feats, vis_feats)

        v2tr1,v2tr5,v2tr10,v2tr50,v2tmedr,v2tmeanr = compute_metrics(sim_matrix.T)
        t2vr1,t2vr5,t2vr10,t2vr50,t2vmedr,t2vmeanr = compute_metrics(sim_matrix)

        if cfg.save_feat and hvd.rank() == 0:
            os.makedirs(join(cfg.output_dir, 'save_feat'), exist_ok=True)
            np.save(join(cfg.output_dir, 'save_feat', 'text_feats.npy'), text_feats)
            np.save(join(cfg.output_dir, 'save_feat', 'vis_feats.npy'), vis_feats)
            np.save(join(cfg.output_dir, 'save_feat', 'sim_matrix.npy'), sim_matrix)


        val_log.update({f'valid/{loader_name}_t2v_recall_1': t2vr1,
                        f'valid/{loader_name}_t2v_recall_5': t2vr5,
                        f'valid/{loader_name}_t2v_recall_10': t2vr10,
                        f'valid/{loader_name}_t2v_recall_50': t2vr50,
                        f'valid/{loader_name}_t2v_recall_median': t2vmedr,
                        f'valid/{loader_name}_t2v_recall_mean': t2vmeanr,
                        f'valid/{loader_name}_v2t_recall_1': v2tr1,
                        f'valid/{loader_name}_v2t_recall_5': v2tr5,
                        f'valid/{loader_name}_v2t_recall_10': v2tr10,
                        f'valid/{loader_name}_v2t_recall_50': v2tr50,
                        f'valid/{loader_name}_v2t_recall_median': v2tmedr,
                        f'valid/{loader_name}_v2t_recall_mean': v2tmeanr
                        })

        TB_LOGGER.log_scalar_dict(val_log)
        LOGGER.info(f"validation finished in {int(time.time() - st)} seconds, "
                    f"validated on {vis_feats.shape[0]} videos"
                    f"{loader_name} t2v recall@1: {val_log['valid/%s_t2v_recall_1'%(loader_name)] * 100:.4f} "
                    f"{loader_name} t2v recall@5: {val_log['valid/%s_t2v_recall_5'%(loader_name)] * 100:.4f} "
                    f"{loader_name} t2v recall@10: {val_log['valid/%s_t2v_recall_10'%(loader_name)] * 100:.4f} "
                    f"{loader_name} t2v recall@50: {val_log['valid/%s_t2v_recall_50'%(loader_name)] * 100:.4f} "
                    f"{loader_name} t2v recall_med: {val_log['valid/%s_t2v_recall_median'%(loader_name)] :.1f} "
                    f"{loader_name} t2v recall_mean: {val_log['valid/%s_t2v_recall_mean'%(loader_name)] :.1f} "
                    f"{loader_name} v2t recall@1: {val_log['valid/%s_v2t_recall_1'%(loader_name)] * 100:.4f} "
                    f"{loader_name} v2t recall@5: {val_log['valid/%s_v2t_recall_5'%(loader_name)] * 100:.4f} "
                    f"{loader_name} v2t recall@10: {val_log['valid/%s_v2t_recall_10'%(loader_name)] * 100:.4f} "
                    f"{loader_name} v2t recall@50: {val_log['valid/%s_v2t_recall_50'%(loader_name)] * 100:.4f} "
                    f"{loader_name} v2t recall_med: {val_log['valid/%s_v2t_recall_median'%(loader_name)] :.1f} "
                    f"{loader_name} v2t recall_mean: {val_log['valid/%s_v2t_recall_mean'%(loader_name)] :.1f} "
                    )
    model.train()
    return val_log, t2vr1


def start_training():
    cfg = shared_configs.get_pretraining_args()
    data_mount(cfg)
    set_random_seed(cfg.seed)

    n_gpu = hvd.size()
    cfg.n_gpu = n_gpu
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    if hvd.rank() != 0:
        LOGGER.disabled = True
    LOGGER.info(f"device: {device} n_gpu: {n_gpu}, "
                f"rank: {hvd.rank()}, 16-bits training: {cfg.fp16}")

    if hvd.rank() != 0:
        LOGGER.disabled = True

    model = setup_model(cfg, device=device)
    model.train()

    optimizer = setup_e2e_optimizer(model, cfg)

    # Horovod: (optional) compression algorithm.compressin
    compression = hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression)

    #  Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    model, optimizer = amp.initialize(
        model, optimizer, enabled=cfg.fp16, opt_level=cfg.amp_level,
        keep_batchnorm_fp32=True if cfg.amp_level=='O2' else None)

    # prepare data
    tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer_dir)
    train_loader, val_loaders, inference_loaders = setup_dataloaders(cfg, tokenizer)

    img_norm = None
    train_loader = PrefetchLoader(train_loader, img_norm)
    val_loaders = {k: PrefetchLoader(v, img_norm)
                   for k, v in val_loaders.items()}
    inference_loaders = {k: PrefetchLoader(v, img_norm)
                   for k, v in inference_loaders.items()}


    # compute the number of steps and update cfg
    total_train_batch_size = int(
        n_gpu * cfg.train_batch_size *
        cfg.gradient_accumulation_steps * cfg.max_n_example_per_group)

    total_n_examples = len(train_loader.dataset) * cfg.max_n_example_per_group 
    print('total_n_examples', total_n_examples)

    cfg.num_train_steps = int(math.ceil(
        1. * cfg.num_train_epochs * total_n_examples / total_train_batch_size))

    cfg.valid_steps = int(math.ceil(
        1. * cfg.num_train_steps / cfg.num_valid /
        cfg.min_valid_steps)) * cfg.min_valid_steps
    actual_num_valid = int(math.floor(
        1. * cfg.num_train_steps / cfg.valid_steps)) + 1

    n_steps_in_epoch = int(math.ceil(1. * total_n_examples / total_train_batch_size))

    # restore
    restorer = TrainingRestorer(cfg, model, optimizer)
    global_step = restorer.global_step
    TB_LOGGER.global_step = global_step
    if hvd.rank() == 0:
        LOGGER.info("Saving training meta...")
        save_training_meta(cfg)
        LOGGER.info("Saving training done...")
        TB_LOGGER.create(join(cfg.output_dir, 'log'))
        # pbar = tqdm(total=cfg.num_train_steps)
        model_saver = ModelSaver(join(cfg.output_dir, "ckpt"))
        best_model_saver = BestModelSaver(join(cfg.output_dir, "ckpt"))
        add_log_to_file(join(cfg.output_dir, "log", "log.txt"))
    else:
        LOGGER.disabled = True
        # pbar = NoOp()
        model_saver = NoOp()
        restorer = NoOp()
        best_model_saver = NoOp()

    if global_step > 0:
        pass # pbar.update(global_step)

    LOGGER.info(cfg)
    LOGGER.info("Starting training...")
    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info(f"  Single-GPU Non-Accumulated batch size = {cfg.train_batch_size}")
    LOGGER.info(f"  max_n_example_per_group = {cfg.max_n_example_per_group}")
    LOGGER.info(f"  Accumulate steps = {cfg.gradient_accumulation_steps}")
    LOGGER.info(f"  Total batch size = #GPUs * Single-GPU batch size * "
                f"max_n_example_per_group * Accumulate steps [Image] = {total_train_batch_size}")
    LOGGER.info(f"  Total #epochs = {cfg.num_train_epochs}")
    LOGGER.info(f"  Total #steps = {cfg.num_train_steps}")
    LOGGER.info(f"  Validate and Save every {cfg.valid_steps} steps, in total {actual_num_valid} times")
    LOGGER.info(f"  Only Validate every {cfg.only_valid_steps} steps")

    # quick hack for amp delay_unscale bug
    with optimizer.skip_synchronize():
        optimizer.zero_grad()
        if global_step == 0:
            optimizer.step()
    debug_step = 3
    running_loss = RunningMeter('train_loss',smooth=0)

    # LOGGER.info(f'Step zero: start validation')
    # validate(model, val_loaders, cfg)

    LOGGER.info(f'Step zero: start inference')
    validate(model, inference_loaders, cfg)

    loss_func = build_loss_func(cfg.loss_config)

    for step, batch in enumerate(InfiniteIterator(train_loader)):
        # forward pass
        # if not cfg.use_itm:
        #     batch['itm_labels'] = None

        outputs = model(**batch)
        vis_feat = hvd.allgather(outputs['vis_features'])
        text_feat = hvd.allgather(outputs['text_features'])
        # t2v = torch.matmul(text_feat, vis_feat.permute(1, 0)) / cfg.temp  # temperature
        # v2t = t2v.permute(1, 0)
        # t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        # v2t_label = t2v_label
        # loss = 0.5 * (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()

        loss = loss_func(vis_feat, text_feat)

        if step%50 == 0:
            LOGGER.info(f'Step {global_step}: loss {loss} ')

        running_loss(loss.item())

        delay_unscale = (step + 1) % cfg.gradient_accumulation_steps != 0
        with amp.scale_loss(
                loss, optimizer, delay_unscale=delay_unscale
                ) as scaled_loss:
            scaled_loss.backward()
            zero_none_grad(model)
            optimizer.synchronize()

        # optimizer
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            global_step += 1
            TB_LOGGER.log_scalar_dict({'vtc_loss': running_loss.val})
            n_epoch = int(1.* cfg.gradient_accumulation_steps *
                          global_step / n_steps_in_epoch)
            # learning rate scheduling transformer
            lr_this_step_transformer = get_lr_sched(
                global_step, cfg.decay, cfg.learning_rate,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                decay_epochs=cfg.step_decay_epochs, multi_step_epoch=n_epoch)

            # learning rate scheduling cnn
            lr_this_step_cnn = get_lr_sched(
                global_step, cfg.cnn_lr_decay, cfg.cnn_learning_rate,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                decay_epochs=cfg.cnn_step_decay_epochs,
                multi_step_epoch=n_epoch)

            # learning rate scheduling low
            lr_this_step_align = get_lr_sched(
                global_step, cfg.decay, cfg.align_learning_rate,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                decay_epochs=cfg.step_decay_epochs,
                multi_step_epoch=n_epoch)

            # Hardcoded param group length
            # assert len(optimizer.param_groups) == 16
            for pg_n, param_group in enumerate(
                    optimizer.param_groups):
                if pg_n in [0, 1]:
                    param_group['lr'] = (
                        cfg.transformer_lr_mul * lr_this_step_transformer)
                elif pg_n in [2, 3]:
                    param_group['lr'] = lr_this_step_transformer
                elif pg_n in [4, 5]:
                    param_group['lr'] = (
                        cfg.cnn_lr_mul * lr_this_step_cnn)
                elif pg_n in [6, 7]:
                    param_group['lr'] = lr_this_step_cnn
                elif pg_n in [8, 9, 10, 11]:
                    param_group['lr'] = lr_this_step_align
                
            TB_LOGGER.add_scalar(
                "train/lr_transformer", lr_this_step_transformer,
                global_step)
            TB_LOGGER.add_scalar(
                "train/lr_cnn", lr_this_step_cnn, global_step)
            TB_LOGGER.add_scalar(
                "train/lr_align", lr_this_step_align, global_step)

            # update model params
            if cfg.grad_norm != -1:
                grad_norm = clip_grad_norm_(
                    amp.master_params(optimizer), cfg.grad_norm)
                TB_LOGGER.add_scalar("train/grad_norm", grad_norm, global_step)
            TB_LOGGER.step()

            # Check if there is None grad
            none_grads = [
                p[0] for p in model.named_parameters()
                if p[1].requires_grad and p[1].grad is None]

            assert len(none_grads) == 0, f"{none_grads}"

            with optimizer.skip_synchronize():
                optimizer.step()
                optimizer.zero_grad()
            restorer.step()

            # checkpoint
            if global_step % cfg.valid_steps == 0:
                LOGGER.info(f'Step {global_step}: start validation and Save')
                _,t2vr1 = validate(model, inference_loaders, cfg)
                model_saver.save(step=global_step, model=model)
                if hvd.rank() == 0 and t2vr1>best_model_saver.bestr1:
                    best_model_saver.save(step=global_step, model=model)
                    best_model_saver.bestr1 = t2vr1
            else:
                if global_step % cfg.only_valid_steps == 0:
                #     LOGGER.info(f'Step {global_step}: start validation')
                #     validate(model, val_loaders, cfg)

                    LOGGER.info(f'Step {global_step}: start inference')
                    _,t2vr1 = validate(model, inference_loaders, cfg)
                    if hvd.rank() == 0 and t2vr1>best_model_saver.bestr1:
                        best_model_saver.save(step=global_step, model=model)
                        best_model_saver.bestr1 = t2vr1

                    # LOGGER.info(f'Step {global_step}: start inference')
                    # validate(model, inference_loaders, cfg)

        if global_step >= cfg.num_train_steps:
            break

    if global_step % cfg.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        _,t2vr1 = validate(model, inference_loaders, cfg)
        # LOGGER.info(f'Step {global_step}: start inference')
        # validate(model, inference_loaders, cfg)

        model_saver.save(step=global_step, model=model)
        if hvd.rank() == 0 and t2vr1>best_model_saver.bestr1:
            best_model_saver.save(step=global_step, model=model)
            best_model_saver.bestr1 = t2vr1

def data_mount(cfg):
    keys = ["e2e_weights_path",
            "mmdetection_weights_path",
            "bert_weights_path",
            "tokenizer_dir",
            "output_dir"]
    for key in keys:
        if cfg[key] is not None:
            cfg[key] = os.path.join(cfg.data_mount_dir, cfg[key])

    db = cfg.train_datasets
    db.txt = os.path.join(cfg.data_mount_dir, db.txt)
    db.vis = os.path.join(cfg.data_mount_dir, db.vis)

    for db in cfg.val_datasets:
        db.txt = os.path.join(cfg.data_mount_dir, db.txt)
        db.vis = os.path.join(cfg.data_mount_dir, db.vis)

    for db in cfg.inference_datasets:
        db.txt = os.path.join(cfg.data_mount_dir, db.txt)
        db.vis = os.path.join(cfg.data_mount_dir, db.vis)



if __name__ == '__main__':
    # Initialize Horovod
    hvd.init()
    start_training()

# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python src/tasks/run_video_retrieval.py --config src/configs/msrvtt_retrieval_local_inference.json  --data_mount_dir /data_mount/
