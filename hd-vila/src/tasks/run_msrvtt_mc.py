import torch
import os
import time
import random
from transformers import BertConfig, BertTokenizerFast
from src.modeling.modeling_stage import HDVILAForPreTraining
from src.modeling.e2e_model import HDVILA

from src.datasets.dataset_video_mc import (
    MSRVTTMCCollator, HDVILAMSRVTTMCEvalDataset)
from src.datasets.dataloader import PrefetchLoader
from src.datasets.data_utils import ImageNorm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.configs.config import shared_configs
from src.utils.misc import set_random_seed
from src.utils.logger import LOGGER
from src.utils.basic_utils import (
    load_jsonl, load_json, save_json, merge_dicts)
from src.utils.load_save import load_state_dict_with_mismatch

from tqdm import tqdm
from os.path import join
from easydict import EasyDict as edict
from apex import amp
from torch.utils.data.distributed import DistributedSampler
import horovod.torch as hvd
from src.utils.distributed import all_gather_list


def mk_msrvtt_mc_datalist(raw_datalist, cfg):
    """
    Args:
        raw_datalist: list(dict)
        cfg:

    Returns:

    """
    LOGGER.info(f"Loaded data size {len(raw_datalist)}")
    if cfg.data_ratio != 1.0:
        random.shuffle(raw_datalist)
        raw_datalist = raw_datalist[:int(len(raw_datalist) * cfg.data_ratio)]
        LOGGER.info(f"Use {100 * cfg.data_ratio}% of the loaded data: {len(raw_datalist)}")

    datalist = []
    for raw_d in raw_datalist:
        d = dict(
            id=raw_d["qid"],
            vid_id=raw_d["clip_name"],
            answer=raw_d["answer"],
            options=raw_d["options"],
        )
        datalist.append(d)
    LOGGER.info(f"datalist {len(datalist)}")
    return datalist


def mk_msrvtt_mc_eval_dataloader(anno_path, lmdb_dir, cfg, tokenizer):
    """
    eval_retrieval: bool, will sample one video per batch paired with multiple text.
    Returns:

    """
    raw_datalist = load_jsonl(anno_path)
    datalist = mk_msrvtt_mc_datalist(raw_datalist, cfg)
    frm_sampling_strategy = cfg.frm_sampling_strategy
    if frm_sampling_strategy == "rand":
        frm_sampling_strategy = "middle"
    dataset = HDVILAMSRVTTMCEvalDataset(
        tokenizer=tokenizer,
        vis_dir=lmdb_dir,
        anno_path=anno_path,
        cfg=cfg,
        is_train=False,
        is_inference=True,
        vis_format = "video"
    )
    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank(),
        shuffle=False)
    collator = MSRVTTMCCollator(
        tokenizer=tokenizer, max_length=cfg.max_txt_len)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.inference_batch_size,
                            shuffle=False,
                            sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=collator.collate_batch)
    img_norm = None
    dataloader = PrefetchLoader(dataloader, img_norm)
    return dataloader


def setup_model(cfg, device=None):
    LOGGER.info("Setup model...")
    # has to be a BertConfig instance
    model_cfg = load_json(cfg.model_config)
    model_cfg = BertConfig(**model_cfg)
    # add downstream model config
    add_attr_list = [
        "num_labels", "classifier", "cls_hidden_scale",
        "loss_type", "margin",
    ]
    addition_cfgs={
    "backbone_channels": [256, 512, 1024, 2048],
    "resnet_depth": 50,
    "resnet_frozen_stage": -1,
    "timesformer_depth": 4,
    "timesformer_heads": 16,
    "bert_mean": 1
    }
    for k in addition_cfgs:
        setattr(model_cfg, k, addition_cfgs[k])

    for k in add_attr_list:
        setattr(model_cfg, k, cfg[k])
    transformer_model_cls = HDVILAForPreTraining

    # we separate the CNN and the transformer in order to use different optimizer for each
    # transformer still has a CNN layer inside, used to down sample grid.
    LOGGER.info("setup e2e model")
    model = HDVILA(
        model_cfg,
        transformer_cls=transformer_model_cls,
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


def forward_step(model, batch, cfg, n_options=5):
    """shared for training and validation"""
    repeat_counts = [e * n_options for e in batch["n_examples_list"]]
    batch["n_examples_list"] = repeat_counts

    outputs = model(batch)  # dict
    return outputs


@torch.no_grad()
def inference_retrieval_mc(model, val_loader, eval_file_path, cfg, n_options=5):
    model.eval()
    pred_id2ans = dict()
    st = time.time()
    LOGGER.info(f"Evaluate retrieval MC: {len(val_loader)}")
    if hvd.rank() == 0:
        pbar = tqdm(total=len(val_loader), desc="eval")

    for batch in val_loader:
        # compile shared text part
        question_ids = batch["question_ids"]
        bsz = len(question_ids)
        del batch["question_ids"]

        # multi-frame test, scores across frames of the same video will be pooled together
        # batch["visual_inputs"]  (B, T, C, H, W)
        pool_method = cfg.score_agg_func
        # could be 1, where only a single clip is evaluated
        num_clips = cfg.inference_n_clips
        num_frm = cfg.num_frm
        # (B, T=num_clips*num_frm, C, H, W) --> (B, num_clips, num_frm, C, H, W)
        outputs = model(**batch)
        vis_feat = outputs['vis_features'].cpu()  # (B, dim)
        text_feat = outputs['text_features'].cpu()   # (B*5, dim)
        B, dim = vis_feat.shape[0], vis_feat.shape[1]
        text_feat = text_feat.view(B, n_options, dim).permute(0, 2, 1)


        probs = torch.bmm(vis_feat.unsqueeze(1), text_feat).squeeze(1)

        probs = probs.view(-1, n_options)  # (B, 5)
        pred_answers = probs.max(1)[1].tolist()  # (B, )
        for qid, pred_ans in zip(question_ids, pred_answers):
            pred_id2ans[qid] = int(pred_ans)

        if hvd.rank() == 0:
            pbar.update(1)

    # ###### Saving with Horovod ####################
    # dummy sync
    _ = None
    all_gather_list(_)
    n_gpu = hvd.size()
    eval_dir = join(cfg.output_dir, f"results_mc_{os.path.splitext(os.path.basename(eval_file_path))[0]}")
    os.makedirs(eval_dir, exist_ok=True)
    if n_gpu > 1:
        # with retrial, as azure blob fails occasionally.
        max_save_load_trial = 10
        save_trial = 0
        while save_trial < max_save_load_trial:
            try:
                LOGGER.info(f"Save results trial NO. {save_trial}")
                save_json(
                    pred_id2ans,
                    join(eval_dir, f"tmp_results_mc_rank{hvd.rank()}.json"))
                break
            except Exception as e:
                print(f"Saving exception: {e}")
                save_trial += 1

    # dummy sync
    _ = None
    all_gather_list(_)
    # join results
    if n_gpu > 1 and hvd.rank() == 0:
        pred_id2ans = []
        for rk in range(n_gpu):
            pred_id2ans.append(load_json(
                join(eval_dir, f"tmp_results_mc_rank{rk}.json")))
        pred_id2ans = merge_dicts(pred_id2ans)
        LOGGER.info('results joined')

    if hvd.rank() == 0:
        retrieval_qa_metrics = val_loader.dataset.evaluate_qa_accuracy(pred_id2ans, force_same=True)
        LOGGER.info(f"validation finished in {int(time.time() - st)} seconds. scores: {retrieval_qa_metrics}")
    else:
        retrieval_qa_metrics = None

    model.train()
    return pred_id2ans, retrieval_qa_metrics


def start_inference(cfg):
    set_random_seed(cfg.seed)
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    if hvd.rank() != 0:
        LOGGER.disabled = True

    inference_res_dir = join(
        cfg.output_dir,
        f"mc_results_{os.path.splitext(os.path.basename(cfg.inference_txt_db))[0]}/"
        f"step_{cfg.inference_model_step}_{cfg.inference_n_clips}_{cfg.score_agg_func}"
    )

    if hvd.rank() == 0:
        os.makedirs(inference_res_dir, exist_ok=True)
        save_json(cfg, join(inference_res_dir, "raw_args.json"),
                  save_pretty=True)

    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), bool(cfg.fp16)))

    # overwrite cfg with stored_cfg,
    # but skip keys containing the keyword 'inference'
    stored_cfg_path = join(cfg.output_dir, "log/args.json")
    stored_cfg = edict(load_json(stored_cfg_path))
    for k, v in cfg.items():
        if k in stored_cfg and "inference" not in k and "output_dir" not in k and "path" not in k and "dir" not in k:
            setattr(cfg, k, stored_cfg[k])

    # addition_cfgs = {
    #     "num_frm": 11,
    #     "sample_rate": 4,
    #     "crop_size": [640, 512],
    # }
    # for k in addition_cfgs:
    #     setattr(input_cfg, k, addition_cfgs[k])

    tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer_dir)

    # setup models
    cfg.model_config = join(cfg.output_dir, "log/model_config.json")
    e2e_weights_path = join(
        cfg.output_dir, f"ckpt/model_{cfg.inference_model_step}.pt")
    cfg.e2e_weights_path = e2e_weights_path
    model = setup_model(cfg, device=device)
    model.eval()

    # FIXME separate scaling for each loss
    model = amp.initialize(
        model, enabled=cfg.fp16, opt_level='O2')

    global_step = 0
    # prepare data
    cfg.data_ratio = 1.

    val_loader = mk_msrvtt_mc_eval_dataloader(
        anno_path=cfg.inference_txt_db,
        lmdb_dir=cfg.inference_img_db,
        cfg=cfg, tokenizer=tokenizer,
    )

    LOGGER.info(cfg)
    LOGGER.info("Starting inference...")
    LOGGER.info(f"***** Running inference with {n_gpu} GPUs *****")
    LOGGER.info(f"  Batch size = {cfg.inference_batch_size}")

    LOGGER.info(f'Step {global_step}: start validation')
    ret_results, ret_scores = inference_retrieval_mc(
        model, val_loader, cfg.inference_txt_db, cfg)

    if hvd.rank() == 0:
        save_json(cfg, join(inference_res_dir, "merged_args.json"),
                  save_pretty=True)
        save_json(ret_results, join(inference_res_dir, "mc_test_results.json"),
                  save_pretty=True)
        save_json(ret_scores, join(inference_res_dir, "mc_test_scores.json"),
                  save_pretty=True)


def data_mount(cfg):
    keys = ["e2e_weights_path",
            "mmdetection_weights_path",
            "bert_weights_path",
            "tokenizer_dir",
            "output_dir",
            "inference_txt_db",
            "inference_img_db"]

    for key in keys:
        if cfg[key] is not None:
            cfg[key] = os.path.join(cfg.data_mount_dir, cfg[key])


if __name__ == '__main__':
    # Initialize Horovod
    hvd.init()
    input_cfg = shared_configs.get_video_retrieval_args()
    data_mount(input_cfg)
    assert input_cfg.do_inference

    start_inference(input_cfg)
