import torch
import torch.distributed as dist
import deepspeed
import argparse
import os
from mmcv import Config
from src.models import LFVILA_Retrieval

from src.tools import Trainer_Retrieval
from src.datasets.dataloader import build_dataloader
from src.optimization.lr_scheduler import build_scheduler
from src.optimization.optimizer import build_optimizer_parameters
from src.utils.logger import LOGGER, add_log_to_file
from src.utils.dist import master_process
from src.utils.misc import mkdirp, set_random_seed
from src.utils.load import load_model_weights_with_mismatch


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./src/configs/pretrain_test_stage2.yaml')
    parser.add_argument('--blob_mount_dir', default="/blob_mount")
    parser.add_argument('--deepspeed_sparse_attention',action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument('--fp16', action='store_true', help='enable fp16')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--distributed',action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--only_val', action='store_true')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    set_random_seed(args.seed)

    config = Config.fromfile(args.config)

    LOGGER.info(config)
    LOGGER.info(args)

    if not master_process(args):
        LOGGER.disabled = True
    if master_process(args):
        mkdirp(os.path.join(args.blob_mount_dir, config.TRAINING.save_dir,"log"))
        add_log_to_file(os.path.join(args.blob_mount_dir, config.TRAINING.save_dir,"log/log.txt"))

    model =  LFVILA_Retrieval(args, config)

    if config.WEIGHTS.model_weight != '':
        LOGGER.info(f"Loading model weights from {config.WEIGHTS.model_weight}")
        load_model_weights_with_mismatch(model, os.path.join(args.blob_mount_dir, config.WEIGHTS.model_weight))
    else:
        if config.WEIGHTS.swin_weight != '':
            LOGGER.info(f"Loading video encoder weights from {config.WEIGHTS.swin_weight}")
            
            load_model_weights_with_mismatch(model.video_encoder, 
                                            os.path.join(args.blob_mount_dir, config.WEIGHTS.swin_weight),
                                            load_swin=True,
                                            pretrained2d=config.WEIGHTS.pretrained_2d)
        if config.WEIGHTS.bert_weight != '':
            LOGGER.info(f"Loading bert weights from {config.WEIGHTS.bert_weight}")
            load_model_weights_with_mismatch(model.text_encoder, os.path.join(args.blob_mount_dir, config.WEIGHTS.bert_weight),load_bert=True)
            model._init_sent_embedding()

    parameter_group = build_optimizer_parameters(config, model)

    # init deepspeed
    if args.distributed:

        model_engine, optimizer, _, _ = deepspeed.initialize(args = args,
                                                            model=model,
                                                            model_parameters=parameter_group,
                                                            config=config.deepspeed_config
                                                        )
        print(dist.get_rank())
    

    LOGGER.info(f'Training with {dist.get_world_size()} gpus')
    
    dataset_trains, dataset_vals, dataloader_trains, dataloader_vals = build_dataloader(args, config)

    dataloader_train = dataloader_trains['RetrievalDataset-train']
    steps_per_epoch = len(dataloader_train)
    scheduler = build_scheduler(config, optimizer, steps_per_epoch)

    args.fp16 = model_engine.fp16_enabled()
    if args.fp16:
        LOGGER.info('Enable fp16 Training')

    trainer = Trainer_Retrieval(args, config, model_engine, optimizer, scheduler, dataloader_train, dataloader_vals['RetrievalDataset-val'])

    LOGGER.info('start first evaluate')

    trainer.evaluate(dataloader_vals['RetrievalDataset-val'])
    
    if not args.only_val:
        trainer.train(args.resume)

if __name__ == '__main__':
    deepspeed.init_distributed()
    main()

