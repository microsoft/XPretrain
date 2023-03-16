import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


from tensorboardX import SummaryWriter

from src.utils.logger import LOGGER
from src.utils.dist import concat_all_gather
from src.utils.dist import master_process
from src.utils.metrics import compute_rt_metrics

from src.utils.misc import AverageMeter


class Trainer_Pretrain():
    def __init__(self, args, config, model, optimizer, scheduler,
                    dataloader_train, task_dataloader_val=None, pre_dataloader_val=None):
        self.config = config
        self.local_rank = model.local_rank
        self.global_step = 0
        self.start_epoch = 0
        self.total_epochs = config.TRAINING.EPOCHS
        self.dataloader_train = dataloader_train
        self.task_dataloader_val = task_dataloader_val
        self.pre_dataloader_val = pre_dataloader_val
        
        self.args = args

        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer

        if master_process(self.args) and config.log_tb:
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
        dist.barrier()


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
                if self.config.log_tb:
                    self.summary_writer.add_scalar(f'Train/lr', lr, self.global_step)
            else:
                prefix = 'Pretrain_Val'

            if self.config.log_tb:

                self.summary_writer.add_scalar(f'{prefix}/total_loss', loss['total_loss'],self.global_step)
                self.summary_writer.add_scalar(f'{prefix}/ct_global_loss', loss['ct_global_loss'],self.global_step)
                self.summary_writer.add_scalar(f'{prefix}/ct_time_loss', loss['ct_time_loss'],self.global_step)
                self.summary_writer.add_scalar(f'{prefix}/vtm_loss', loss['vtm_loss'],self.global_step)
                self.summary_writer.add_scalar(f'{prefix}/mlm_loss', loss['mlm_loss'],self.global_step)

            ##### Recording  done. #####

            if self.global_step % self.config.TRAINING.print_step == 0 or not is_train:

                LOGGER.info(f"training_progress: {prefix} step={self.global_step}, \n"
                    f"[total_loss]: {loss['total_loss']}, \n"
                    f"[ct_global_loss]: {loss['ct_global_loss']}, \n"
                    f"[ct_time_loss]: {loss['ct_time_loss']}, \n"
                    f"[vtm_loss]: {loss['vtm_loss']}, \n"
                    f"[mlm_loss]: {loss['mlm_loss']}, \n"
                    )

    @torch.no_grad()
    def evaluate(self, dataloader_val, stage=2, pretrain_val=False):

        if pretrain_val:
            LOGGER.info(f"start evaluate on pretrain-val dataset.")
        else:
            LOGGER.info(f"start evaluate on task dataset.")

        self.model.eval()
        st = time.time()

        text_global_feats = []
        video_global_feats = []
        mlm_accs = []
        vtm_accs = []

        valid_len = len(dataloader_val.dataset)

        for step, batch in enumerate(dataloader_val):

            video_frames = batch['video_frames'].to(self.local_rank)
            text_ids = batch['text_ids'].to(self.local_rank)
            attention_mask = batch['attention_mask'].to(self.local_rank)

            if stage==2:
                mlm_labels = batch['mlm_labels'].to(self.local_rank)
            else:
                mlm_labels=None

            if self.args.fp16:
                video_frames = video_frames.half()
                attention_mask = attention_mask.half()
     
            output = self.model(video_frames, 
                                text_ids, 
                                attention_mask,
                                mlm_labels = mlm_labels,
                                stage=stage,
                                is_train=False,
                                is_pretrain_val=pretrain_val)
            if pretrain_val:
                tasks = ['total_loss', 'ct_global_loss','ct_time_loss','mlm_loss','vtm_loss']


                total_loss = output['ct_global_loss'] + output['ct_time_loss'] + output['mlm_loss'] + output['vtm_loss']

                output['total_loss'] = total_loss
                
                task2loss = {t: AverageMeter() for t in tasks}
                for t in tasks:
                    if output[t] != 0:
                        all_loss = concat_all_gather(torch.tensor([output[t].item()],device=output[t].device))
                        task2loss[t].update(all_loss.mean())

            if stage == 1:  

                video_global_feat = output['video_global_feat']
                text_global_feat = output['text_global_feat']
                video_global_feats.append(video_global_feat)
                text_global_feats.append(text_global_feat)


            if stage == 2:
       
                mlm_acc = concat_all_gather(output['mlm_acc'])
                mlm_accs.append(mlm_acc.cpu().numpy())
   
                vtm_acc = concat_all_gather(output['vtm_acc'])
                vtm_accs.append(vtm_acc.cpu().numpy())

        if stage == 1:

            text_global_feats = torch.cat(text_global_feats)[:valid_len]
            video_global_feats = torch.cat(video_global_feats)[:valid_len]
            global_sim_matrix = torch.matmul(text_global_feats, video_global_feats.permute(1,0)).cpu().numpy()

            v2tr1,v2tr5,v2tr10,_,_,_ = compute_rt_metrics(global_sim_matrix.T)
            t2vr1,t2vr5,t2vr10,_,_,_ = compute_rt_metrics(global_sim_matrix)

            LOGGER.info(f"validation global finished in {int(time.time() - st)} seconds, "
                        f"validated on {video_global_feats.shape[0]} videos"
                        f"t2v recall@1: {t2vr1 * 100:.4f} "
                        f"t2v recall@5: {t2vr5* 100:.4f} "
                        f"t2v recall@10: {t2vr10 * 100:.4f} ")

            global_t2vr1 = t2vr1

        if stage == 2:

            mlm_accs = np.vstack(mlm_accs)
            mlm_acc = np.mean(mlm_accs)

            vtm_accs = np.vstack(vtm_accs)
            vtm_acc = np.mean(vtm_accs)

            LOGGER.info(f"validation finished in {int(time.time() - st)} seconds, "
                    f"mlm acc: {mlm_acc * 100:.4f} "
                    f"vtm acc: {vtm_acc * 100:.4f} ")

            if master_process(self.args) and self.config.log_tb:
                self.summary_writer.add_scalar(f'Valid/mlm_acc', mlm_acc*100, self.global_step)
                self.summary_writer.add_scalar(f'Valid/vtm_acc', vtm_acc*100, self.global_step)

        if pretrain_val:
            for t in tasks:
                task2loss[t] = task2loss[t].avg
            self.report_step_metrics(loss=task2loss, is_train=False)

            if stage == 1 and master_process(self.args) and self.config.log_tb:
                self.summary_writer.add_scalar(f'Pretrain_Val/t2vr1', global_t2vr1*100, self.global_step)
        else:
            if stage == 1 and master_process(self.args) and self.config.log_tb:
                self.summary_writer.add_scalar(f'Task_Val/t2vr1', global_t2vr1*100, self.global_step)

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
  
                if self.config.stage == 2 and self.config.TRAINING.use_mlm:
                    mlm_labels = batch['mlm_labels'].to(self.local_rank)
                else:
                    mlm_labels=None


                if self.args.fp16:
                    video_frames = video_frames.half()
                    attention_mask = attention_mask.half()

                output = self.model(video_frames,  
                                    text_ids, 
                                    attention_mask,
                                    mlm_labels = mlm_labels,
                                    stage=self.config.stage)

                tasks = ['total_loss', 'ct_global_loss', 'ct_time_loss','mlm_loss','vtm_loss']

                total_loss =  output['ct_global_loss'] + output['ct_time_loss'] + output['mlm_loss'] + output['vtm_loss']

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
                    if self.config.stage==1:
                        self.evaluate(self.task_dataloader_val, stage=1)
                        self.evaluate(self.pre_dataloader_val, stage=1, pretrain_val=True)

                    if self.config.stage==2:
                        self.evaluate(self.pre_dataloader_val, stage=2, pretrain_val=True)

                if self.global_step % self.config.TRAINING.checkpoint_step == 0:
                    self._checkpoint(self.config.TRAINING.save_dir, self.global_step, epoch, self.global_step)

                if self.global_step % self.config.TRAINING.save_step == 0:
                    self._save_model(self.config.TRAINING.save_dir, epoch, step)
                
                if self.global_step > self.config.TRAINING.BREAK_STEP:
                    LOGGER.info(f"Job finished")
                    break
                    
            self.start_epoch = epoch
            LOGGER.info(epoch)
            if self.global_step > self.config.TRAINING.BREAK_STEP:
                LOGGER.info(f"Job finished")
                break

