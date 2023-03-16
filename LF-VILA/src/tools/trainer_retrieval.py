import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from tensorboardX import SummaryWriter

from src.utils.logger import LOGGER
from src.utils.dist import concat_all_gather
from src.utils.dist import master_process
from src.utils.metrics import compute_rt_metrics
from src.utils.misc import AverageMeter


class Trainer_Retrieval():
    def __init__(self, args, config, model, optimizer, scheduler,
                    dataloader_train, rt_dataloader_val=None):
        self.config = config
        self.local_rank = model.local_rank
        self.global_step = 0
        self.start_epoch = 0
        self.total_epochs = config.TRAINING.EPOCHS
        self.dataloader_train = dataloader_train
        self.rt_dataloader_val = rt_dataloader_val
        
        self.args = args

        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.best_performance = 0

        if master_process(self.args):
            self.summary_writer = SummaryWriter(log_dir=os.path.join(args.blob_mount_dir,config.TRAINING.save_dir,'tb_log'))

    def _checkpoint(self,PATH, ckpt_id, epoch, global_step):

        """Utility function for checkpointing model + optimizer dictionaries
        The main purpose for this is to be able to resume training from that instant again
        """
        checkpoint_state_dict = {
            'epoch': epoch,
            'global_step': global_step,
        }
        save_dir = os.path.join(self.args.blob_mount_dir,PATH, 'checkpoint')
        save_trial = 0
        while save_trial < 10:
            try:
                LOGGER.info(f"checkpointing trial NO. {save_trial}")
                success = self.model.save_checkpoint(save_dir, ckpt_id, checkpoint_state_dict)
                status_msg = 'checkpointing: PATH={}, ckpt_id={}'.format(save_dir, ckpt_id)
                if success:
                    LOGGER.info(f"Success {status_msg}")
                    break
            except Exception as e:
                save_trial += 1
                LOGGER.warning(f"Failure {status_msg}")


    def _save_model(self,PATH, epoch, step):
        
        save_dir = os.path.join(self.args.blob_mount_dir,PATH, 'saved_model', 'epoch_{0:03d}_step_{1:05d}'.format(epoch, step))
        save_trial = 0
        while save_trial < 10:
            try:
                sucess = self.model.save_fp16_model(save_dir)
                break
            except Exception as e:
                save_trial += 1
                LOGGER.warning(f"Failure save model")

    def _resume(self,PATH, tag=None):
        save_dir = os.path.join(self.args.blob_mount_dir,PATH, 'checkpoint')
        LOGGER.info(f"resume from {save_dir}")
        _, checkpoint_state_dict = self.model.load_checkpoint(save_dir)
        self.start_epoch = checkpoint_state_dict['epoch']
        self.global_step = checkpoint_state_dict['global_step']
        del checkpoint_state_dict

    def report_step_metrics(self, lr=0, loss=None, is_train=True):
        ##### Record the LR against global_step on tensorboard #####
        if master_process(self.args):

            if is_train:
                prefix = 'Train'
                self.summary_writer.add_scalar(f'Train/lr', lr, self.global_step)
            else:
                prefix = 'Val'

            self.summary_writer.add_scalar(f'{prefix}/total_loss', loss['total_loss'],self.global_step)
            self.summary_writer.add_scalar(f'{prefix}/ct_global_loss', loss['ct_global_loss'],self.global_step)

            ##### Recording  done. #####

            if self.global_step % self.config.TRAINING.print_step == 0 or not is_train:

                LOGGER.info(f"training_progress: {prefix} step={self.global_step}, \n"
                    f"[total_loss]: {loss['total_loss']}, \n"
                    f"[ct_global_loss]: {loss['ct_global_loss']}, \n"
                    )

    @torch.no_grad()
    def evaluate(self, dataloader_val):

        LOGGER.info(f"start evaluate.")

        self.model.eval()
        st = time.time()

        text_global_feats = []
        video_global_feats = []

        indexes = []

        valid_len = len(dataloader_val.dataset)

        if self.config.TRAINING.save_feats:
            save_dir = os.path.join(self.args.blob_mount_dir, self.config.TRAINING.save_dir, 'save_feats')
        
        for step, batch in enumerate(dataloader_val):

            video_frames = batch['video_frames'].to(self.local_rank)
            text_ids = batch['text_ids'].to(self.local_rank)
            attention_mask = batch['attention_mask'].to(self.local_rank)

            if self.config.TRAINING.save_feats:
                indexes.append(concat_all_gather(batch['index'].to(self.local_rank)))

                del batch['index']

            if self.args.fp16:
                video_frames = video_frames.half()
                attention_mask = attention_mask.half()
     
            output = self.model(video_frames, 
                                text_ids, 
                                attention_mask)

            video_global_feat = output['video_global_feat']
            text_global_feat = output['text_global_feat']
            video_global_feats.append(video_global_feat)
            text_global_feats.append(text_global_feat)


        text_global_feats = torch.cat(text_global_feats)[:valid_len]
        video_global_feats = torch.cat(video_global_feats)[:valid_len]
        global_sim_matrix = torch.matmul(text_global_feats, video_global_feats.permute(1,0)).cpu().numpy()

        if master_process(self.args) and self.config.TRAINING.save_feats:
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir,'text_global_feats.npy'), text_global_feats.cpu().numpy())
            np.save(os.path.join(save_dir,'video_global_feats.npy'), video_global_feats.cpu().numpy())

        if master_process(self.args) and self.config.TRAINING.save_feats:
            indexes = torch.cat(indexes)[:valid_len]
            np.save(os.path.join(save_dir,'indexes.npy'), indexes.cpu().numpy())


        v2tr1,v2tr5,v2tr10,_,_,_ = compute_rt_metrics(global_sim_matrix.T)
        t2vr1,t2vr5,t2vr10,t2vr50,t2vmedr, t2vmeanr = compute_rt_metrics(global_sim_matrix)

        LOGGER.info(f"validation global finished in {int(time.time() - st)} seconds, "
                    f"validated on {video_global_feats.shape[0]} videos \n"
                    f"t2v recall@1: {t2vr1 * 100:.4f} "
                    f"t2v recall@5: {t2vr5* 100:.4f} "
                    f"t2v recall@10: {t2vr10 * 100:.4f} "
                    f"t2v recall@50: {t2vr50 * 100:.4f} "
                    f"t2vmedr: {t2vmedr:.4f} ")
        
        if t2vr1 > self.best_performance and self.global_step>0:
            LOGGER.info(f"save best model")
            self._save_model(self.config.TRAINING.save_dir, 0000, 0000)
            self.best_performance = t2vr1

        self.model.train()

  
    def train(self, resume):
        self.model.train()
        if resume:
            self._resume(self.config.TRAINING.save_dir)
            LOGGER.info(f'resume from {self.start_epoch}, global step {self.global_step}')

        LOGGER.info(f'begin training from {self.start_epoch}')
        for epoch in range(self.start_epoch, self.total_epochs):

            if self.args.distributed:
                self.dataloader_train.sampler.set_epoch(epoch)

            for step, batch in enumerate(self.dataloader_train):
 
                video_frames = batch['video_frames'].to(self.local_rank)
                text_ids = batch['text_ids'].to(self.local_rank)
                attention_mask = batch['attention_mask'].to(self.local_rank)

                if self.args.fp16:
                    video_frames = video_frames.half()
                    attention_mask = attention_mask.half()

                output = self.model(video_frames, 
                                    text_ids, 
                                    attention_mask)

                tasks = ['total_loss', 'ct_global_loss']

                total_loss = output['ct_global_loss']

                output['total_loss'] = total_loss

                task2loss = {t: 0 for t in tasks}
                for t in tasks:
                    if output[t] != 0:
                        task2loss[t] = output[t].item()
                

                self.model.backward(total_loss)
                self.model.step()

                self.global_step += 1
                self.scheduler.step_update(self.global_step)
                lr = self.scheduler._get_lr(self.global_step)[0]
                self.report_step_metrics(lr, task2loss)

                if self.global_step % self.config.TRAINING.eval_step == 0:
                    self.evaluate(self.rt_dataloader_val)

                if self.global_step % self.config.TRAINING.checkpoint_step == 0:
                    self._checkpoint(self.config.TRAINING.save_dir, self.global_step, epoch, self.global_step)


                if self.global_step % self.config.TRAINING.save_step == 0:
                    self._save_model(self.config.TRAINING.save_dir, epoch, step)
                    
            self.start_epoch = epoch
            LOGGER.info(epoch)
