import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import time
from pathlib import Path
import torch

from tensorboardX import SummaryWriter

from src.utils.logger import LOGGER
from src.utils.dist import concat_all_gather
from src.utils.dist import master_process



class Trainer_Video_Classification():
    def __init__(self, args, config, model, optimizer, scheduler, 
                    dataloader_train=None, dataloader_val=None):
        self.config = config
        self.local_rank = model.local_rank
        self.global_step = 0
        self.start_epoch = 0
        self.total_epochs = config.TRAINING.EPOCHS
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

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

    def report_step_metrics(self, lr, loss, acc):
        ##### Record the LR against global_step on tensorboard #####
        if master_process(self.args):

            self.summary_writer.add_scalar(f'Train/lr', lr, self.global_step)
            self.summary_writer.add_scalar(f'Train/loss', loss, self.global_step)

        ##### Recording  done. #####

        if self.global_step % self.config.TRAINING.print_step == 0:
            LOGGER.info(f"training_progress: step={self.global_step}, \n"
                f"[lr]: {lr}, \n"
                f"[loss]: {loss}, \n"
                f"[acc]: {acc}, \n"
                )

    @torch.no_grad()
    def evaluate(self, dataloader_val, stage=2):
        LOGGER.info(f"start evaluate.")

        self.model.eval()
        st = time.time()

        predictions = []
        accs = []

        for step, batch in enumerate(dataloader_val):

            video_frames = batch['video_frames'].to(self.local_rank)
            labels = batch['labels'].to(self.local_rank)

            if self.args.fp16:
                video_frames = video_frames.half()
     
            output = self.model(video_frames, labels = labels, is_train=False)

            acc = concat_all_gather(output['acc'])
            prediction = concat_all_gather(output['prediction'])

            accs.append(acc.cpu().numpy())
            predictions.append(prediction.cpu().numpy())

        predictions = np.vstack(predictions)
     
        accs = np.vstack(accs)
        acc = np.mean(accs)

        LOGGER.info(f"validation finished in {int(time.time() - st)} seconds, "
                f"validated on {predictions.shape[0]} videos"
                f"acc: {acc * 100:.4f} ")

        if master_process(self.args):
            self.summary_writer.add_scalar(f'Valid/acc', acc*100, self.global_step)

        if acc > self.best_performance and self.global_step>0:
            LOGGER.info(f"save best model")
            self._save_model(self.config.TRAINING.save_dir, 0000, 0000)
            self.best_performance = acc

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
                labels = batch['labels'].to(self.local_rank)


                if self.args.fp16:
                    video_frames = video_frames.half()


                output = self.model(video_frames, labels = labels)

                loss = output['loss']
                total_loss = loss

                acc = output['acc'].mean().item()

                self.model.backward(total_loss)
                self.model.step()

                total_loss_value = total_loss.item()

                self.global_step += 1
                self.scheduler.step_update(self.global_step)
                lr = self.scheduler._get_lr(self.global_step)[0]
                self.report_step_metrics(lr, total_loss_value, acc)

                if self.global_step % self.config.TRAINING.eval_step == 0:
                    self.evaluate(self.dataloader_val, stage=2)

                if self.global_step % self.config.TRAINING.checkpoint_step == 0:
                    self._checkpoint(self.config.TRAINING.save_dir, self.global_step, epoch, self.global_step)


                if self.global_step % self.config.TRAINING.save_step == 0:
                    self._save_model(self.config.TRAINING.save_dir, epoch, step)
                    
            self.start_epoch = epoch
            LOGGER.info(epoch)

