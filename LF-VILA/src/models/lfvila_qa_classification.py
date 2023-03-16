from locale import LC_NUMERIC
from src.models.bert import (
    BertConfig, BertModel, BertOnlyMLMHead, BertOnlyNSPHead, BertForMaskedLM)
from src.models.video_encoder import SwinTransformer3D
from src.models.text_encoder import TextEncoderForClassification
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import random
import einops
from src.utils.logger import LOGGER
from timm.models.vision_transformer import Block
from src.models.lfvila_pretrain import VideoTokenPos, SentEmbedding


class LFVILA_QA_Classification(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.cfg = config
        self.video_encoder = SwinTransformer3D(**config.VideoEncoder)
        bert_config = BertConfig.from_json_file(config.bert_config)
        setattr(bert_config,'stage',config.stage)
        setattr(bert_config,'num_local_layers',config.num_local_layers)
        setattr(bert_config,'stage1_layers',config.stage1_layers)
        setattr(bert_config,'bert_frozen_stage',config.bert_frozen_stage)
        if hasattr(config, "label_smoothing"):
            label_smoothing = config.label_smoothing
        else:
            label_smoothing = 0
        self.text_encoder = TextEncoderForClassification(args, config=bert_config, 
                                                classification_labels=config.DATA.classification_labels,
                                                label_smoothing=label_smoothing)


        self.video_downsample = nn.MaxPool2d((2,3), stride=(1,1))

        self.video_token_pos = VideoTokenPos(num_patches=config.final_num_patches,
                                                num_frames=config.DATA.sample_frame,
                                                hidden_size=bert_config.hidden_size)

        setattr(bert_config,'type_vocab_size',config.type_vocab_size)
        self.sent_embedding = SentEmbedding(bert_config)

    def _init_sent_embedding(self):
        self.sent_embedding.position_embeddings.weight.data.copy_(self.text_encoder.bert.embeddings.position_embeddings.weight.data)

    def downsample_video_embd(self, video_embd):
        sample_clip = self.cfg.DATA.sample_clip
        B, N, H, W, C = video_embd.size() # B, N, H, W, C
        video_embd = video_embd.permute(0,1,4,2,3)
        video_embd = self.video_downsample(video_embd.view(B*N, C, H, W))
        video_embd = video_embd.permute(0,2,3,1) # B*N, H, W, C
        video_embd = video_embd.view(B, N, video_embd.size(-3), video_embd.size(-2),video_embd.size(-1))
        video_embd = video_embd.flatten(2,3) # B, N, X, C

        video_feat = video_embd.view(B, sample_clip, int(N/sample_clip), -1, C)
        video_feat = video_feat.mean(dim=[2,3])

        return video_feat, video_embd


    def forward(self, video_frames, text_ids, attention_mask=None, labels = None):

        B, C, N, H, W = video_frames.size()
        video_global_embd, video_local_embd = self.video_encoder(video_frames)
        video_local_feat2, video_stage1_embd = self.downsample_video_embd(video_global_embd)

        B,M,L = text_ids.shape
        text_local_embd = self.text_encoder(text_ids.view(B*M,L), attention_mask=attention_mask.view(B*M,L), return_dict=True, stage=0).view(B, M, L, -1)

        B,M,L,C = text_local_embd.shape
        text_segment_id = torch.arange(M, device=text_local_embd.device).repeat(B,1).repeat_interleave(L,dim=1)

        text_local_embd = self.sent_embedding(text_local_embd.view(B,M*L,-1), text_segment_id)
 
        text_local_cls = text_local_embd.view(B,M,L,-1)[:,:,0,:].mean(dim=1)
        text_global_embd = torch.cat([text_local_cls.unsqueeze(1),text_local_embd], dim=-2) 
        

        attention_mask = torch.cat([torch.tensor([1.],dtype=attention_mask.dtype, device=attention_mask.device).repeat(B,1),attention_mask.view(B,M*L)], dim=-1)

        text_global_embd = self.text_encoder(text_global_embd, attention_mask=attention_mask, return_dict=True, stage=1)

        video_stage1_embd = self.video_token_pos(video_stage1_embd)

        video_stage1_embd = video_stage1_embd.flatten(1,2)

        visual_attention_mask = attention_mask.new_ones(
            video_stage1_embd.shape[:2])
        attention_mask = torch.cat(
            [attention_mask, visual_attention_mask], dim=-1)

        stage1_embedding_output = torch.cat(
            [text_global_embd, video_stage1_embd],
            dim=1) 
        fusion_output = self.text_encoder(stage1_embedding_output, attention_mask=attention_mask, labels=labels,  return_dict=True, stage=2)

        loss = fusion_output['loss']
        acc = fusion_output['acc']
        prediction = fusion_output['logits']

        return dict(loss=loss,
                    acc = acc,
                    prediction=prediction)
