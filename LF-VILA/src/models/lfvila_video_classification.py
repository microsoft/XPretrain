from locale import LC_NUMERIC
from src.models.bert import (
    BertConfig, BertModel, BertOnlyMLMHead, BertOnlyNSPHead, BertForMaskedLM)
from src.models.video_encoder import SwinTransformer3D
from src.models.text_encoder import TextEncoderForPretraining
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import random
import einops
from src.utils.logger import LOGGER
from timm.models.vision_transformer import Block


class LFVILA_Video_Classification(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        
        self.cfg = config
        self.video_encoder = SwinTransformer3D(**config.VideoEncoder)
        bert_config = BertConfig.from_json_file(config.bert_config)

        self.video_downsample = nn.MaxPool2d((2,3), stride=(1,1))

        self.video_global_proj = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.video_frame_proj = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)

        self.classifier = nn.Linear(bert_config.hidden_size, self.cfg.DATA.classification_labels)


    def downsample_video_embd(self, video_embd):
        B, N, H, W, C = video_embd.size() # B, N, H, W, C
        video_embd = video_embd.permute(0,1,4,2,3)
        video_embd = self.video_downsample(video_embd.view(B*N, C, H, W))
        video_embd = video_embd.permute(0,2,3,1) # B*N, H, W, C
        video_embd = video_embd.view(B, N, video_embd.size(-3), video_embd.size(-2),video_embd.size(-1))
        video_embd = video_embd.flatten(2,3) # B, N, X, C

        video_feat = video_embd.mean(dim=[1, 2])
        video_frame_feat = video_embd.mean(dim=2)

        return video_feat, video_frame_feat


    def forward(self, video_frames, labels = None):
        B, C, N, H, W = video_frames.size()
        video_global_embd, _ = self.video_encoder(video_frames) # B, N, H, W, C
        video_global_feat, video_frame_feat = self.downsample_video_embd(video_global_embd)

        video_global_feat = F.normalize(self.video_global_proj(video_global_feat),dim=-1)

        video_frame_feat = F.normalize(self.video_frame_proj(video_frame_feat),dim=-1)

 
        logits = self.classifier(video_global_feat)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        acc = logits.max(dim=-1)[1] == labels
        acc = acc.float().mean(dim=0, keepdim=True)

        return dict(video_global_feat = video_global_feat,
                    video_frame_feat = video_frame_feat,
                    prediction = logits,
                    loss = loss,
                    acc = acc)




    
    

