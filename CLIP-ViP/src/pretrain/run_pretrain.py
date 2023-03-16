import os
import time
import random
import math
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from os.path import join, exists
from easydict import EasyDict as edict
import pprint


import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torchvision.utils as tutils
from torch.utils.data.distributed import DistributedSampler

from apex import amp
import horovod.torch as hvd
from transformers import CLIPTokenizerFast

from src.modeling.VidCLIP import VidCLIP

from src.datasets.dataset_pretrain_stage1_all_source import HDVILAPretrainDataset, PretrainCollator
from src.datasets.dataloader import MetaLoader, PrefetchLoader
from src.datasets.data_utils import ImageNorm, mk_input_group

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
from src.utils.distributed import all_gather_list
from src.utils.metrics import cal_cossim, compute_metrics


def mk_pretrain_dataloader(dataset_name, vis_format, anno_path, vis_dir, vid_cap_path,\
                img_dir, cap_path, img_source, img_ratio, vid_txt, cfg, tokenizer, mode):
    """"""
    is_train = mode == "train"
    dataset = HDVILAPretrainDataset(
        cfg=cfg,
        vis_dir=vis_dir,
        anno_path=anno_path,
        vid_cap_path=vid_cap_path,
        img_dir=img_dir,
        cap_path=cap_path,
        img_source=img_source,
        img_ratio=img_ratio,
        vid_txt=vid_txt,
        vis_format=vis_format,
        mode=mode
    )
    LOGGER.info(f"[{dataset_name}] is_train {is_train} "
                f"dataset size {len(dataset)}, ")

    batch_size = cfg.train_batch_size if is_train else cfg.test_batch_size
    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank(),
        shuffle=is_train)
    data_collator = PretrainCollator(
        tokenizer=tokenizer, max_length=cfg.max_txt_len, is_train=is_train)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=data_collator.collate_batch)
    return dataloader



def setup_dataloaders(cfg, tokenizer):
    LOGGER.info("Init. train_loader and val_loader...")
    train_loaders = {}
    for db in cfg.train_datasets:
        train_loaders[db.name] = mk_pretrain_dataloader(
            dataset_name=db.name, vis_format=db.vis_format,
            anno_path=db.txt, vis_dir=db.vis, vid_cap_path=db.vid_cap_path,
            img_dir=db.img_dir, cap_path=db.cap_path, 
            img_source=db.img_source, img_ratio=db.img_ratio, vid_txt=db.vid_txt,
            cfg=cfg, tokenizer=tokenizer, mode="train"
        )
        if "ratio" in db:
            train_loaders[db.name] = (train_loaders[db.name], db.ratio)

    val_loaders = {}
    for db in cfg.val_datasets:
        val_loaders[db.name] = mk_pretrain_dataloader(
            dataset_name=db.name, vis_format=db.vis_format,
            anno_path=db.txt, vis_dir=db.vis, vid_cap_path="",
            img_dir="", cap_path="", 
            img_source="", img_ratio=0, vid_txt='subtitle',
            cfg=cfg, tokenizer=tokenizer, mode="val"
        )
    return train_loaders, val_loaders

def setup_model(cfg, device=None):
    LOGGER.info("Setup model...")
    
    model = VidCLIP(cfg)

    if cfg.e2e_weights_path:
        LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
        load_state_dict_with_mismatch(model, cfg.e2e_weights_path)
    
    if hasattr(cfg, "freeze_text_model") and cfg.freeze_text_model:
        freeze_text_proj = hasattr(cfg, "freeze_text_proj") and cfg.freeze_text_proj
        LOGGER.info(f"Freeze CLIP text model and the status of freezing proj is: {freeze_text_proj}")
        model.freeze_text_encoder(freeze_text_proj)

    model.to(device)

    LOGGER.info("Setup model done!")
    return model

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
                   f'valid/{loader_name}_t2v_recall_median': 0,
                   f'valid/{loader_name}_t2v_recall_mean': 0,
                   f'valid/{loader_name}_v2t_recall_1': 0,
                   f'valid/{loader_name}_v2t_recall_5': 0,
                   f'valid/{loader_name}_v2t_recall_10': 0,
                   f'valid/{loader_name}_v2t_recall_median': 0,
                   f'valid/{loader_name}_v2t_recall_mean': 0}
        text_feats = []
        vis_feats = []
        for val_step, batch in enumerate(val_loader):
            feats = model(**batch)  # dict
            # print('feats vis_features', feats['vis_features'].shape)
            vis_feat = hvd.allgather(feats['vis_features'])
            text_feat = hvd.allgather(feats['text_features'])

            # print('allgather vis_features', vis_feat.shape)

            text_feats.append(text_feat.cpu().numpy())
            vis_feats.append(vis_feat.cpu().numpy())

        # # Gather across all processes
        # text_feats = all_gather_list(text_feats)
        # vis_feats = all_gather_list(vis_feats)

        text_feats = np.vstack(text_feats)
        vis_feats = np.vstack(vis_feats)

        sim_matrix = cal_cossim(text_feats, vis_feats)

        v2tr1,v2tr5,v2tr10,v2tmedr,v2tmeanr = compute_metrics(sim_matrix.T)
        t2vr1,t2vr5,t2vr10,t2vmedr,t2vmeanr = compute_metrics(sim_matrix)


        val_log.update({f'valid/{loader_name}_t2v_recall_1': t2vr1,
                        f'valid/{loader_name}_t2v_recall_5': t2vr5,
                        f'valid/{loader_name}_t2v_recall_10': t2vr10,
                        f'valid/{loader_name}_t2v_recall_median': t2vmedr,
                        f'valid/{loader_name}_t2v_recall_mean': t2vmeanr,
                        f'valid/{loader_name}_v2t_recall_1': v2tr1,
                        f'valid/{loader_name}_v2t_recall_5': v2tr5,
                        f'valid/{loader_name}_v2t_recall_10': v2tr10,
                        f'valid/{loader_name}_v2t_recall_median': v2tmedr,
                        f'valid/{loader_name}_v2t_recall_mean': v2tmeanr
                        })

        TB_LOGGER.log_scalar_dict(val_log)
        LOGGER.info(f"validation finished in {int(time.time() - st)} seconds, "
                    f"validated on {vis_feats.shape[0]} videos"
                    f"{loader_name} t2v recall@1: {val_log['valid/%s_t2v_recall_1'%(loader_name)] * 100:.4f} "
                    f"{loader_name} t2v recall@5: {val_log['valid/%s_t2v_recall_5'%(loader_name)] * 100:.4f} "
                    f"{loader_name} t2v recall@10: {val_log['valid/%s_t2v_recall_10'%(loader_name)] * 100:.4f} "
                    f"{loader_name} t2v recall_med: {val_log['valid/%s_t2v_recall_median'%(loader_name)] :.1f} "
                    f"{loader_name} t2v recall_mean: {val_log['valid/%s_t2v_recall_mean'%(loader_name)] :.1f} "
                    f"{loader_name} v2t recall@1: {val_log['valid/%s_v2t_recall_1'%(loader_name)] * 100:.4f} "
                    f"{loader_name} v2t recall@5: {val_log['valid/%s_v2t_recall_5'%(loader_name)] * 100:.4f} "
                    f"{loader_name} v2t recall@10: {val_log['valid/%s_v2t_recall_10'%(loader_name)] * 100:.4f} "
                    f"{loader_name} v2t recall_med: {val_log['valid/%s_v2t_recall_median'%(loader_name)] :.1f} "
                    f"{loader_name} v2t recall_mean: {val_log['valid/%s_v2t_recall_mean'%(loader_name)] :.1f} "
                    )
    model.train()
    return val_log, t2vr1

def start_training():
    cfg = shared_configs.get_pretraining_args()
    blob_mount(cfg)
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
    tokenizer = CLIPTokenizerFast.from_pretrained(cfg.clip_config)
    train_loaders, val_loaders = setup_dataloaders(cfg, tokenizer)
    train_loader = MetaLoader(train_loaders,
                              accum_steps=cfg.gradient_accumulation_steps,
                              distributed=n_gpu > 1)
    img_norm = None
    train_loader = PrefetchLoader(train_loader, img_norm)
    val_loaders = {k: PrefetchLoader(v, img_norm)
                   for k, v in val_loaders.items()}

    n_batches_in_epoch = train_loader.n_batches_in_epoch

    # compute the number of steps and update cfg
    total_train_batch_size = int(
        n_gpu * cfg.train_batch_size *
        cfg.gradient_accumulation_steps * cfg.max_n_example_per_group)
    total_n_epochs = cfg.num_train_epochs
    cfg.num_train_steps = int(math.ceil(
        1. * n_batches_in_epoch * total_n_epochs /
        (n_gpu * cfg.gradient_accumulation_steps)))
    cfg.valid_steps = int(math.ceil(
        1. * cfg.num_train_steps / cfg.num_valid /
        cfg.min_valid_steps)) * cfg.min_valid_steps
    actual_num_valid = int(math.floor(
        1. * cfg.num_train_steps / cfg.valid_steps)) + 1

    # restore
    restorer = TrainingRestorer(cfg, model, optimizer)
    global_step = restorer.global_step
    TB_LOGGER.global_step = global_step
    if hvd.rank() == 0:
        LOGGER.info("Saving training meta...")
        save_training_meta(cfg)
        LOGGER.info("Saving training done...")
        if cfg.if_tb_log:
            TB_LOGGER.create(join(cfg.output_dir, 'log'))
        # pbar = tqdm(total=cfg.num_train_steps)
        if cfg.if_model_saver:
            model_saver = ModelSaver(join(cfg.output_dir, "ckpt"))
            best_model_saver = BestModelSaver(join(cfg.output_dir, "ckpt"))
        else:
            model_saver = NoOp()
            restorer = NoOp()
            best_model_saver = NoOp()
            
        if cfg.if_log2file:
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
    LOGGER.info(f"  Total #batches - single epoch = {n_batches_in_epoch}.")
    LOGGER.info(f"  Total #epochs = {total_n_epochs}.")
    LOGGER.info(f"  Total #steps = {cfg.num_train_steps}")
    LOGGER.info(f"  Validate and Save every {cfg.valid_steps} steps, in total {actual_num_valid} times")
    LOGGER.info(f"  Only Validate every {cfg.only_valid_steps} steps")

    # quick hack for amp delay_unscale bug
    with optimizer.skip_synchronize():
        optimizer.zero_grad()
        if global_step == 0:
            optimizer.step()

    tasks = ["contrastive", "logit_scale"]
    task2loss = {t: RunningMeter(f'train_loss/{t}', smooth=0)
                 for t in tasks}
    task2loss["loss"] = RunningMeter('train_loss/loss', smooth=0)

    LOGGER.info(f'Step zero: start inference')
    validate(model, val_loaders, cfg)

    loss_func = build_loss_func(cfg.loss_config)

    train_loader_iter = iter(train_loader)
    step = 0
    while global_step < cfg.num_train_steps:
        if hasattr(cfg, "break_step") and global_step == cfg.break_step:
            break
        
        task, batch = next(train_loader_iter)
        
        # clamp logit_scale
        if hasattr(model, 'module'):
            torch.clamp_(model.module.clipmodel.logit_scale.data, 0, np.log(200))
            logit_scale_ = model.module.clipmodel.logit_scale.data
        else:
            torch.clamp_(model.clipmodel.logit_scale.data, 0, np.log(200))
            logit_scale_ = model.clipmodel.logit_scale.data

        outputs = model(**batch)
        if cfg.loss_config.if_gather:
            vis_feat = hvd.allgather(outputs['vis_features'])
            text_feat = hvd.allgather(outputs['text_features'])
            if hasattr(model, 'module'):
                logit_scale = model.module.clipmodel.logit_scale
            else:
                logit_scale = model.clipmodel.logit_scale
            if "img_features" in outputs and outputs["img_features"] is not None:
                img_feat = hvd.allgather(outputs['img_features'])
            if "cap_features" in outputs and outputs["cap_features"] is not None:
                cap_feat = hvd.allgather(outputs['cap_features'])
            
            if cfg.loss_config.loss_name == "NCELearnableTempLoss":
                loss = loss_func(vis_feat, text_feat, logit_scale)
            elif cfg.loss_config.loss_name in ["VidImgNCELearnableTempLoss", \
                "VidImgDivideNCELearnableTempLoss", "NCELearnableTempLoss_vsc", \
                "NCELearnableTempLoss_vs_vc", "NCELearnableTempLoss_vs_vc_fc", \
                "NCELearnableTempLoss_vsc_fc"]:
                if global_step == 1:
                    print(vis_feat.shape, text_feat.shape, img_feat.shape, cap_feat.shape)
                loss = loss_func(vis_feat, text_feat, img_feat, cap_feat, logit_scale)
            else:
                loss = loss_func(vis_feat, text_feat)
        else:
            loss = outputs['loss']

        task2loss["logit_scale"](logit_scale_.item())
        task2loss["contrastive"](loss.item())
        task2loss["loss"](loss.item())

        delay_unscale = (step + 1) % cfg.gradient_accumulation_steps != 0
        with amp.scale_loss(
                loss, optimizer, delay_unscale=delay_unscale
                ) as scaled_loss:
            scaled_loss.backward()
            # zero_none_grad(model)
            optimizer.synchronize()

        # optimizer
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            global_step += 1
            TB_LOGGER.log_scalar_dict({l.name: l.val
                                       for l in task2loss.values()
                                       if l.val is not None})
            n_epoch = int(1. * n_gpu * cfg.gradient_accumulation_steps *
                          global_step / n_batches_in_epoch)
            # learning rate scheduling transformer
            lr_this_step = get_lr_sched(
                global_step, cfg.decay, cfg.learning_rate,
                cfg.num_train_steps, warmup_ratio=cfg.warmup_ratio,
                decay_epochs=cfg.step_decay_epochs, multi_step_epoch=n_epoch)

            for pg_n, param_group in enumerate(
                    optimizer.param_groups):
                if pg_n in [0, 1]:
                    param_group['lr'] = (
                        cfg.lr_mul * lr_this_step)
                elif pg_n in [2, 3]:
                    param_group['lr'] = lr_this_step
                
            TB_LOGGER.add_scalar(
                "train/lr", lr_this_step,
                global_step)

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
                _, t2vr1 = validate(model, val_loaders, cfg)
                model_saver.save(step=global_step, model=model)
                # if hvd.rank() == 0 and cfg.if_model_saver and t2vr1 > best_model_saver.bestr1:
                #     best_model_saver.save(step=global_step, model=model)
                #     best_model_saver.bestr1 = t2vr1
            else:
                if global_step % cfg.only_valid_steps == 0:
                    LOGGER.info(f'Step {global_step} Training Loss: {loss.item()} LR: {lr_this_step} Logit Scale: {logit_scale_.item()}')
                    LOGGER.info(f'Step {global_step}: start validation')
                    _, t2vr1 = validate(model, val_loaders, cfg)

        step += 1

    if global_step % cfg.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        validate(model, val_loaders, cfg)
        model_saver.save(step=global_step, model=model)
    
def blob_mount(cfg):
    keys = ["e2e_weights_path",
            "output_dir"]
    for key in keys:
        if cfg[key] is not None:
            cfg[key] = os.path.join(cfg.blob_mount_dir, cfg[key])

    for db in cfg.train_datasets:
        db.txt = os.path.join(cfg.blob_mount_dir, db.txt)
        db.vis = os.path.join(cfg.blob_mount_dir, db.vis)
        if hasattr(db, "img_dir") and db.img_dir:
            db.img_dir = os.path.join(cfg.blob_mount_dir, db.img_dir)
        if hasattr(db, "cap_path") and db.cap_path:
            db.cap_path = os.path.join(cfg.blob_mount_dir, db.cap_path)
        if hasattr(db, "vid_cap_path") and db.vid_cap_path:
            db.vid_cap_path = os.path.join(cfg.blob_mount_dir, db.vid_cap_path)

    for db in cfg.val_datasets:
        db.txt = os.path.join(cfg.blob_mount_dir, db.txt)
        db.vis = os.path.join(cfg.blob_mount_dir, db.vis)

if __name__ == '__main__':
    # Initialize Horovod
    hvd.init()
    start_training()

# horovodrun -np 4 python src/pretrain/run_pretrain_contrast_hvd.py --config src/configs/pretrain_contrast.json  --blob_mount_dir /blob_mount/
