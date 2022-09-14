import horovod.torch as hvd
import os
from collections import defaultdict
from tqdm import tqdm
from os.path import join
from apex import amp
import time
import random
import pprint
import math
import numpy as np
from transformers import BertConfig, BertTokenizerFast

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.utils as tutils
from torch.utils.data.distributed import DistributedSampler

from src.modeling.modeling_stage import HDVILAForPreTraining
from src.modeling.e2e_model import HDVILA

from src.datasets.dataset_pretrain import HDVILAPretrainDataset, PretrainCollator
from src.datasets.dataloader import MetaLoader, PrefetchLoader
from src.datasets.data_utils import ImageNorm, mk_input_group

from src.optimization.sched import get_lr_sched
from src.optimization.utils import setup_e2e_optimizer

from src.configs.config import shared_configs
from src.utils.misc import set_random_seed, NoOp, zero_none_grad
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter
from src.utils.basic_utils import load_jsonl, load_json
from src.utils.load_save import (ModelSaver,
                                 save_training_meta,
                                 load_state_dict_with_mismatch)
from src.utils.load_save import E2E_TrainingRestorer as TrainingRestorer
from src.utils.distributed import all_gather_list
from src.utils.metrics import cal_cossim, compute_metrics


def mk_pretrain_dataloader(dataset_name, vis_format, anno_path, vis_dir, cfg, tokenizer, is_train=True):
    dataset = HDVILAPretrainDataset(
        cfg=cfg,
        tokenizer=tokenizer,
        vis_dir=vis_dir,
        anno_path=anno_path,
        vis_format=vis_format,
        is_train=is_train
    )
    LOGGER.info(f"[{dataset_name}] is_train {is_train} "
                f"dataset size {len(dataset)}, ")
    batch_size = cfg.train_batch_size if is_train else cfg.val_batch_size
    
    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank(),
        shuffle=is_train)
    data_collator = PretrainCollator(tokenizer=tokenizer,
                                     mlm=cfg.use_mlm,
                                     mlm_probability=0.15,
                                     max_length=cfg.max_txt_len,
                                     is_train=is_train)
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
            anno_path=db.txt, vis_dir=db.vis,
            cfg=cfg, tokenizer=tokenizer, is_train=True
        )
        if "ratio" in db:
            train_loaders[db.name] = (train_loaders[db.name], db.ratio)

    val_loaders = {}
    for db in cfg.val_datasets:
        val_loaders[db.name] = mk_pretrain_dataloader(
            dataset_name=db.name, vis_format=db.vis_format,
            anno_path=db.txt, vis_dir=db.vis,
            cfg=cfg, tokenizer=tokenizer, is_train=False
        )
    return train_loaders, val_loaders


def setup_model(cfg, device=None):
    LOGGER.info("Setup model...")
    # has to be a BertConfig instance
    model_cfg = load_json(cfg.model_config)
    model_cfg = BertConfig(**model_cfg)

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
        "bert_mean"
    ]
    for k in add_attr_list:
        setattr(model_cfg, k, cfg[k])
    LOGGER.info(f"model_cfg {pprint.pformat(model_cfg.to_dict())}")

    LOGGER.info("setup e2e model")

    model = HDVILA(
        model_cfg,
        transformer_cls=HDVILAForPreTraining,
        )

    LOGGER.info(f"Loading cnn weights from {cfg.mmdetection_weights_path}")
    LOGGER.info(f"Loading bert weights from {cfg.bert_weights_path}")
    model.load_separate_ckpt(
        cnn_weights_path=cfg.mmdetection_weights_path,
        bert_weights_path=cfg.bert_weights_path
    )

    if cfg.e2e_weights_path:
        LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
        load_state_dict_with_mismatch(model, cfg.e2e_weights_path)

    if cfg.freeze_cnn:
        model.freeze_cnn_backbone()

    if hasattr(cfg, "freeze_s1"):
        model.freeze_stage_one_params()
    
    if hasattr(cfg, "freeze_s1_vision"):
        model.freeze_stage_one_vision_params()

    model.to(device)

    LOGGER.info("Setup model done!")
    return model


@torch.no_grad()
def validate_retrieval(model, val_loaders, cfg):
    model.eval()
    st = time.time()

    for loader_name, val_loader in val_loaders.items():
        if loader_name not in ["vllpfull"]:
            continue
        LOGGER.info(f"Loop val_loader {loader_name}.")
        val_log = {f'valid/{loader_name}_t2v_recall_1': 0,
                   f'valid/{loader_name}_t2v_recall_5': 0,
                   f'valid/{loader_name}_t2v_recall_10': 0,
                   f'valid/{loader_name}_v2t_recall_1': 0,
                   f'valid/{loader_name}_v2t_recall_5': 0,
                   f'valid/{loader_name}_v2t_recall_10': 0}
        text_feats = []
        vis_feats = []
        for val_step, batch in enumerate(val_loader):
            # use iter to reset MetaLoader
            # forward pass
            if not cfg.use_itm:
                batch["itm_labels"] = None
            feats = model(**batch)  # dict
            vis_feat = hvd.allgather(feats['vis_features'])
            text_feat = hvd.allgather(feats['text_features'])

            text_feats.append(text_feat.cpu().numpy())
            vis_feats.append(vis_feat.cpu().numpy())

        # # Gather across all processes
        # text_feats = all_gather_list(text_feats)
        # vis_feats = all_gather_list(vis_feats)

        text_feats = np.vstack(text_feats)
        vis_feats = np.vstack(vis_feats)

        sim_matrix = cal_cossim(text_feats, vis_feats)
        v2tr1, v2tr5, v2tr10, medr, meanr = compute_metrics(sim_matrix.T)
        t2vr1, t2vr5, t2vr10, medr, meanr = compute_metrics(sim_matrix)

        val_log.update({f'valid/{loader_name}_t2v_recall_1': t2vr1,
                        f'valid/{loader_name}_t2v_recall_5': t2vr5,
                        f'valid/{loader_name}_t2v_recall_10': t2vr10,
                        f'valid/{loader_name}_v2t_recall_1': v2tr1,
                        f'valid/{loader_name}_v2t_recall_5': v2tr5,
                        f'valid/{loader_name}_v2t_recall_10': v2tr10})

        # TB_LOGGER.log_scalar_dict(val_log)
        LOGGER.info(f"validation finished in {int(time.time() - st)} seconds, "
                    f"validated on {vis_feats.shape[0]} videos"
                    f"{loader_name} t2v recall@1: {val_log['valid/%s_t2v_recall_1' % (loader_name)] * 100:.4f} "
                    f"{loader_name} t2v recall@5: {val_log['valid/%s_t2v_recall_5' % (loader_name)] * 100:.4f} "
                    f"{loader_name} t2v recall@10: {val_log['valid/%s_t2v_recall_10' % (loader_name)] * 100:.4f} "
                    f"{loader_name} v2t recall@1: {val_log['valid/%s_v2t_recall_1' % (loader_name)] * 100:.4f} "
                    f"{loader_name} v2t recall@5: {val_log['valid/%s_v2t_recall_5' % (loader_name)] * 100:.4f} "
                    f"{loader_name} v2t recall@10: {val_log['valid/%s_v2t_recall_10' % (loader_name)] * 100:.4f} "
                    )
    model.train()
    return val_log


@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()

    mlm_loss = 0
    n_mlm_tokens = 0
    n_mlm_corrects = 0
    itm_loss = 0
    n_itm_ex = 0
    n_itm_corrects = 0
    itc_loss = 0
    n_itc_ex = 0
    n_itc_t2v_corrects = 0
    n_itc_v2t_corrects = 0

    st = time.time()
    val_log = {'valid/mlm_loss': 0, 'valid/mlm_acc': 0,
               'valid/itm_loss': 0, 'valid/itm_acc': 0,
               'valid/itc_loss': 0, 'valid/itc_t2v_acc': 0, 'valid/itc_v2t_acc': 0
               }
    debug_step = 5
    val_loaders = val_loader if isinstance(val_loader, dict) else {
        "unnamed_val_loader": val_loader}
    LOGGER.info(f"In total {len(val_loaders)} val loaders")
    for loader_name, val_loader in val_loaders.items():
        if loader_name in ["msrvtt"]:
            continue
        LOGGER.info(f"Loop val_loader {loader_name}.")
        for val_step, batch in enumerate(val_loader):
            # use iter to reset MetaLoader
            # forward pass
            if not cfg.use_itm:
                batch['itm_labels'] = None
            outputs = model(**batch)

            # mlm
            if cfg.use_mlm and outputs["mlm_acc"] is not None:
                mlm_loss += outputs["mlm_loss"].mean().item()
                n_mlm_tokens += 1
                n_mlm_corrects += outputs["mlm_acc"].mean().item()

            # itm
            if cfg.use_itm:
                itm_loss += outputs["itm_loss"].mean().item()
                n_itm_ex += 1
                n_itm_corrects += outputs["itm_acc"].mean().item()

            # itc
            # if cfg.use_itc:
            #     itc_loss += outputs["itc_loss"].mean().item()
            #     n_itc_ex += 1
            #     n_itc_t2v_corrects += outputs["t2v_acc"].mean().item()
            #     n_itc_v2t_corrects += outputs["v2t_acc"].mean().item()

            if cfg.debug and val_step >= debug_step:
                break
    # Gather across all processes
    mlm_loss = sum(all_gather_list(mlm_loss))
    n_mlm_corrects = sum(all_gather_list(n_mlm_corrects))
    n_mlm_tokens = sum(all_gather_list(n_mlm_tokens))
    itm_loss = sum(all_gather_list(itm_loss))
    n_itm_corrects = sum(all_gather_list(n_itm_corrects))
    n_itm_ex = sum(all_gather_list(n_itm_ex))
    itc_loss = sum(all_gather_list(itc_loss))
    n_itc_ex = sum(all_gather_list(n_itc_ex))
    n_itc_t2v_corrects = sum(all_gather_list(n_itc_t2v_corrects))
    n_itc_v2t_corrects = sum(all_gather_list(n_itc_v2t_corrects))

    if n_mlm_tokens != 0:
        val_log.update({
            'valid/mlm_loss': float(mlm_loss / n_mlm_tokens),
            'valid/mlm_acc': float(n_mlm_corrects / n_mlm_tokens)
        })
    if n_itm_ex != 0:
        val_log.update({
            'valid/itm_loss': float(itm_loss / n_itm_ex),
            'valid/itm_acc': float(n_itm_corrects / n_itm_ex)
        })
    if n_itc_ex != 0:
        val_log.update({
            'valid/itc_loss': float(itc_loss / n_itc_ex),
            'valid/itc_t2v_acc': float(n_itc_t2v_corrects / n_itc_ex),
            'valid/itc_v2t_acc': float(n_itc_v2t_corrects / n_itc_ex)
        })

    TB_LOGGER.log_scalar_dict(val_log)
    LOGGER.info(f"validation finished in {int(time.time() - st)} seconds, "
                f"[mlm_acc (per token)]: {val_log['valid/mlm_acc'] * 100:.2f} "
                f"[itm_acc (per example)]: {val_log['valid/itm_acc'] * 100:.2f} "
                f"[itc_t2v_acc (per example)]: {val_log['valid/itc_t2v_acc'] * 100:.2f} "
                f"[itc_v2t_acc (per example)]: {val_log['valid/itc_v2t_acc'] * 100:.2f} "
                )
    model.train()
    return val_log


"""
To release the burden of cpu, we preprocess the training meta data and divide it 
into partial files. reload_train_loader will reload the training dataset according to 
the reload_path.
"""
def reload_train_loader(cfg, tokenizer, n_gpu, reload_path):
    LOGGER.info(f"Reload Train Loader from {reload_path}...")
    train_loaders = {}
    for db in cfg.train_datasets:
        train_loaders[db.name] = mk_pretrain_dataloader(
            dataset_name=db.name, vis_format=db.vis_format,
            anno_path=db.txt, vis_dir=db.vis,
            cfg=cfg, tokenizer=tokenizer, is_train=True
        )
        if "ratio" in db:
            train_loaders[db.name] = (train_loaders[db.name], db.ratio)

    train_loader = MetaLoader(train_loaders,
                              accum_steps=cfg.gradient_accumulation_steps,
                              distributed=n_gpu > 1)
    train_loader = PrefetchLoader(train_loader, None)
    return train_loader

def start_training():
    cfg = shared_configs.get_pretraining_args()
    data_mount(cfg)
    set_random_seed(cfg.seed)

    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())

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
    train_loaders, val_loaders = setup_dataloaders(cfg, tokenizer)
    train_loader = MetaLoader(train_loaders,
                              accum_steps=cfg.gradient_accumulation_steps,
                              distributed=n_gpu > 1)
    img_norm = None
    train_loader = PrefetchLoader(train_loader, img_norm)
    val_loaders = {k: PrefetchLoader(v, img_norm)
                   for k, v in val_loaders.items()}

    """
    To release the burden of cpu, we preprocess the training meta data and divide it 
    into 12 partial files for one epoch, defaultly. The following code calculate the 
    reload_steps and recalculate the batchsize and n_batches_in_epoch. 
    """
    NUM_SHARDS = 12
    # RELOAD_STEPS = 6540
    total_train_batch_size = int(
        n_gpu * cfg.train_batch_size *
        cfg.gradient_accumulation_steps * cfg.max_n_example_per_group)
    RELOAD_STEPS = int(train_loader.n_batches_in_epoch * cfg.train_batch_size * 0.98 / total_train_batch_size)
    JSON_SHARDS = [f"part{i+1}.jsonl" for i in range(NUM_SHARDS*cfg.num_train_epochs)]
    LOGGER.info(f"Will Reload Train Loader Every {RELOAD_STEPS} Steps.")
    n_batches_in_epoch = train_loader.n_batches_in_epoch * NUM_SHARDS

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
        TB_LOGGER.create(join(cfg.output_dir, 'log'))
        # pbar = tqdm(total=cfg.num_train_steps)
        model_saver = ModelSaver(join(cfg.output_dir, "ckpt"))
        add_log_to_file(join(cfg.output_dir, "log", "log.txt"))
    else:
        LOGGER.disabled = True
        # pbar = NoOp()
        model_saver = NoOp()
        restorer = NoOp()

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
    LOGGER.info(f"  Total #steps = {cfg.num_train_steps}")
    LOGGER.info(f"  Total #epochs = {total_n_epochs}.")
    LOGGER.info(f"  Save every {cfg.valid_steps} steps, in total {actual_num_valid} times")
    LOGGER.info(f"  Only Validate every {cfg.only_valid_steps} steps")

    # quick hack for amp delay_unscale bug
    with optimizer.skip_synchronize():
        optimizer.zero_grad()
        if global_step == 0:
            optimizer.step()
    debug_step = 5

    tasks = []
    for name, flag in zip(["mlm", "itm", "itc"], [cfg.use_mlm, cfg.use_itm, cfg.use_itc]):
        if flag:
            tasks.append(name)
    task2loss = {t: RunningMeter(f'train_loss/{t}', smooth=0)
                 for t in tasks}
    task2loss["loss"] = RunningMeter('train_loss/loss', smooth=0)

    if global_step // RELOAD_STEPS > 0:
        for db in cfg.train_datasets:
            db.txt = os.path.join(os.path.dirname(db.txt), JSON_SHARDS[global_step // RELOAD_STEPS])
        train_loader = reload_train_loader(cfg, tokenizer, n_gpu, db.txt)

    train_loader_iter = iter(train_loader)
    step = 0
    while global_step < cfg.num_train_steps:
        try:
            task, batch = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            task, batch = next(train_loader_iter)

        # forward pass
        if not cfg.use_itm:
            batch['itm_labels'] = None
            
        outputs = model(**batch)

        mlm_loss, itm_loss, itc_loss = 0, 0, 0
        if cfg.use_mlm:
            mlm_loss = outputs["mlm_loss"]
            task2loss["mlm"](mlm_loss.item())
        if cfg.use_itm:
            itm_loss = outputs["itm_loss"]
            task2loss["itm"](itm_loss.item())
        if cfg.use_itc:
            vis_feat = hvd.allgather(outputs['vis_features'])
            text_feat = hvd.allgather(outputs['text_features'])
            t2v = torch.matmul(text_feat, vis_feat.permute(1, 0)) / cfg.temp  # temperature
            v2t = t2v.permute(1, 0)
            t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
            v2t_label = t2v_label
            simple_loss = 0.5 * (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()
            if hasattr(cfg, "itc_weight"):
                simple_loss = simple_loss * cfg.itc_weight
            itc_loss = simple_loss
            task2loss["itc"](itc_loss.item())

        loss = mlm_loss + itm_loss + itc_loss
        task2loss["loss"](loss.item())
        
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
            TB_LOGGER.log_scalar_dict({l.name: l.val
                                       for l in task2loss.values()
                                       if l.val is not None})
            n_epoch = int(1. * n_gpu * cfg.gradient_accumulation_steps *
                          global_step / n_batches_in_epoch)
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

            # learning rate scheduling align
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
            # pbar.update(1)
            
            # checkpoint
            if global_step % cfg.valid_steps == 0:
                LOGGER.info(f'Step {global_step}: start Save')
                # validate(model, val_loaders, cfg)
                # validate_retrieval(model, val_loaders, cfg)
                model_saver.save(step=global_step, model=model)

            if global_step % cfg.only_valid_steps == 0:
                LOGGER.info(f'Step {global_step} Training Loss: {loss.item()}')
                LOGGER.info(f'Step {global_step}: start validation')
                validate(model, val_loaders, cfg)
                validate_retrieval(model, val_loaders, cfg)

            # reload training dataset
            if global_step > 0 and global_step % RELOAD_STEPS == 0:
                torch.cuda.empty_cache()
                for db in cfg.train_datasets:
                    db.txt = os.path.join(os.path.dirname(db.txt), JSON_SHARDS[global_step // RELOAD_STEPS])
                train_loader = reload_train_loader(cfg, tokenizer, n_gpu, db.txt)
                train_loader_iter = iter(train_loader)

        step += 1

    if global_step % cfg.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        validate(model, val_loaders, cfg)
        validate_retrieval(model, val_loaders, cfg)
        model_saver.save(step=global_step, model=model)

# change data path based on mount path
def data_mount(cfg):
    keys = ["e2e_weights_path",
            "mmdetection_weights_path",
            "bert_weights_path",
            "tokenizer_dir",
            "output_dir"]
    for key in keys:
        if cfg[key] is not None:
            cfg[key] = os.path.join(cfg.data_mount_dir, cfg[key])

    for db in cfg.train_datasets:
        db.txt = os.path.join(cfg.data_mount_dir, db.txt)
        db.vis = os.path.join(cfg.data_mount_dir, db.vis)

    for db in cfg.val_datasets:
        db.txt = os.path.join(cfg.data_mount_dir, db.txt)
        db.vis = os.path.join(cfg.data_mount_dir, db.vis)

if __name__ == '__main__':
    # Initialize Horovod
    hvd.init()
    start_training()
