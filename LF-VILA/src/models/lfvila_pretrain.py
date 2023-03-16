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
from src.utils.dist import SyncFunction
from src.utils.misc import vector_gather
from timm.models.vision_transformer import Block


class VideoTokenPos(nn.Module):
    def __init__(self,num_patches=6, num_frames=32, hidden_size=768):
        super().__init__()
        self.s_pos_embed = nn.Parameter(0.02*torch.randn(1, 1, num_patches, hidden_size), requires_grad=True)
        self.t_pos_embed = nn.Parameter(0.02*torch.randn(1, num_frames, 1, hidden_size), requires_grad=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, video_embd):
        video_embd = video_embd + self.s_pos_embed + self.t_pos_embed
        video_embd = self.norm(video_embd)
        return video_embd

class SentEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.hidden_size
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.segment_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(cfg.max_position_embeddings).expand((1, -1)))

    def forward(self, inputs_embeds, token_type_ids):
        segment_embeddings = self.segment_embeddings(token_type_ids) # B, N, C
        seq_length = inputs_embeds.shape[1]
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings =  inputs_embeds + position_embeddings + segment_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LFVILA_Pretrain(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.cfg = config
        self.args = args
        self.video_encoder = SwinTransformer3D(**config.VideoEncoder)
        bert_config = BertConfig.from_json_file(config.bert_config)
        setattr(bert_config,'stage',config.stage)
        setattr(bert_config,'num_local_layers',config.num_local_layers)
        setattr(bert_config,'stage1_layers',config.stage1_layers)
        setattr(bert_config,'bert_frozen_stage',config.bert_frozen_stage)
        self.text_encoder = TextEncoderForPretraining(args, config=bert_config)
        self.video_downsample = nn.MaxPool2d((2,3), stride=(1,1))

        self.video_local_proj = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.text_local_proj = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)

        self.video_global_proj = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.text_global_proj = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)

        if config.stage == 2:
            self._freeze_stage_one_params()
            self.video_token_pos = VideoTokenPos(num_patches=config.final_num_patches,
                                                num_frames=config.DATA.sample_frame,
                                                hidden_size=bert_config.hidden_size)

        setattr(bert_config,'type_vocab_size',config.type_vocab_size)
        self.sent_embedding = SentEmbedding(bert_config)

    def _init_sent_embedding(self):
        self.sent_embedding.position_embeddings.weight.data.copy_(self.text_encoder.bert.embeddings.position_embeddings.weight.data)

    def _freeze_stage_one_params(self):
        freeze_modules = ["video_encoder", "video_local_proj", "text_local_proj", "video_global_proj", "text_global_proj", "sent_embedding"]
        for i in freeze_modules:
            m = getattr(self, i)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        for m in [self.text_encoder.bert.embeddings]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        for i in range(0, 12):
            m = self.text_encoder.bert.encoder.layer[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def ct_global_loss(self, video_feat, text_feat):
        temp = self.cfg.TRAINING.temp
        t2v = torch.matmul(video_feat, text_feat.permute(1, 0)) / temp  # temperature
        v2t = t2v.permute(1, 0)
        t2v_label = torch.arange(t2v.shape[0], device=t2v.device)
        v2t_label = t2v_label
        loss = (F.cross_entropy(t2v, t2v_label) + F.cross_entropy(v2t, v2t_label)).mean()
        return loss

    def ct_time_loss(self, video_local_feat,text_local_feat):
        b,m,c = video_local_feat.shape
        temp = self.cfg.TRAINING.time_temp
        num_key = self.cfg.TRAINING.num_key
        num_value = self.cfg.TRAINING.num_value
        num_other_neg=self.cfg.TRAINING.num_other_neg
        
        key_indices = torch.cat([torch.randperm(m).unsqueeze(0) for x in range(b)],dim=0)[:,:num_key].to(text_local_feat.device)
        value_indices = torch.cat([torch.randperm(m).unsqueeze(0) for x in range(b)],dim=0)[:,:num_value].to(text_local_feat.device)
        text_key_feat = vector_gather(text_local_feat, key_indices) 
        video_value_feat = vector_gather(video_local_feat, value_indices)

        if num_other_neg > 0:
            other_neg_indices = torch.cat([torch.randperm(m).unsqueeze(0) for x in range(b)],dim=0)[:,0].to(text_local_feat.device)
            video_other_neg = vector_gather(video_local_feat, other_neg_indices)
            video_other_neg = torch.cat([video_other_neg.roll(shifts=x, dims=0).unsqueeze(1) for x in range(num_other_neg)],dim=1)
            video_value_feat = torch.cat([video_value_feat, video_other_neg],dim=1)

        sim_t2v = torch.matmul(text_key_feat, video_value_feat.permute(0,2,1)).flatten(0,1) / temp

        t2v_label = ((value_indices.unsqueeze(1) - key_indices.unsqueeze(2))).abs().argmin(dim=-1).flatten(0,1)

        minus = ((value_indices.unsqueeze(1) - key_indices.unsqueeze(2))).abs()
        mask = ((minus[:,:,0] - minus[:,:,-1]) == 0 ).flatten(0,1)
        t2v_label = t2v_label.masked_fill_(mask, -100)
        
        video_key_feat = vector_gather(video_local_feat, key_indices) 
        text_value_feat = vector_gather(text_local_feat, value_indices)

        if num_other_neg > 0:
            text_other_neg = vector_gather(text_local_feat, other_neg_indices)
            text_other_neg = torch.cat([text_other_neg.roll(shifts=x, dims=0).unsqueeze(1) for x in range(num_other_neg)],dim=1)
            text_value_feat = torch.cat([text_value_feat, text_other_neg],dim=1)

        sim_v2t = torch.matmul(video_key_feat, text_value_feat.permute(0,2,1)).flatten(0,1) / temp

        v2t_label = t2v_label

        loss = (F.cross_entropy(sim_t2v, t2v_label) + F.cross_entropy(sim_v2t, v2t_label)).mean()
    
        return loss


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

    def shuffle_embd_for_vtm(self, video_embd):
        B, L, C = video_embd.shape
        video_embd_neg = torch.roll(video_embd[:(B//2)],1,0)
        video_embd = torch.cat([video_embd_neg, video_embd[(B//2):]], dim=0)
        vtm_label = torch.cat([torch.zeros((B//2),device=video_embd.device,dtype=torch.long), torch.ones((B-B//2),device=video_embd.device,dtype=torch.long)])
        return video_embd, vtm_label


    def forward(self, video_frames, text_ids, 
                    attention_mask,  mlm_labels = None, 
                    stage=2,is_train=True,is_pretrain_val=False):

        # extract video feature
        B, C, N, H, W = video_frames.size()
        video_global_embd, video_local_embd = self.video_encoder(video_frames) # B, N, H, W, C

        video_local_feat1, _ = self.downsample_video_embd(video_local_embd)
        video_local_feat2, video_stage1_embd = self.downsample_video_embd(video_global_embd)

        # extract text feature
        B,M,L = text_ids.shape
        text_local_embd = self.text_encoder(text_ids.view(B*M, L), attention_mask=attention_mask.view(B*M, L), return_dict=True, stage=0).view(B, M, L, -1) # B, M, L, C

        if stage == 1:

            text_local_feat = text_local_embd[:,:,0,:] # B, M, C
            video_local_feat = F.normalize(self.video_local_proj(video_local_feat1),dim=-1)
            text_local_feat = F.normalize(self.text_local_proj(text_local_feat),dim=-1)
        else:
            video_local_feat, text_local_feat = None, None

        B,M,L,C = text_local_embd.shape

        text_segment_id = torch.arange(M, device=text_local_embd.device).repeat(B,1).repeat_interleave(L,dim=1)# B, N
        text_local_embd = self.sent_embedding(text_local_embd.view(B,M*L,-1), text_segment_id)

        text_local_cls = text_local_embd.view(B,M,L,-1)[:,:,0,:].mean(dim=1) # B,C
        text_global_embd = torch.cat([text_local_cls.unsqueeze(1),text_local_embd], dim=-2) # b, 1+M*L, c
        attention_mask = torch.cat([torch.tensor([1.],dtype=attention_mask.dtype, device=attention_mask.device).repeat(B,1),attention_mask.view(B,M*L)], dim=-1) # b, 1+M*L
        text_global_embd = self.text_encoder(text_global_embd, attention_mask=attention_mask, return_dict=True, stage=1) # B, 1+M*L, C
   
        if stage == 1:
            text_global_feat = text_global_embd[:,0,:] # B, C
            video_global_feat = video_local_feat2.mean(dim=1)

            video_global_feat = F.normalize(self.video_global_proj(video_global_feat),dim=-1)
            text_global_feat = F.normalize(self.text_global_proj(text_global_feat),dim=-1)

        else:
            text_global_feat, video_global_feat = None, None

        if stage == 1:
            if self.args.distributed:
                text_global_feat = SyncFunction.apply(text_global_feat)
                video_global_feat = SyncFunction.apply(video_global_feat)

                if self.cfg.TRAINING.use_time_match:
                    text_local_feat = SyncFunction.apply(text_local_feat)
                    video_local_feat = SyncFunction.apply(video_local_feat)                    

        ct_global_loss, ct_time_loss = 0, 0
        if is_train or is_pretrain_val:
            if stage == 1:
                ct_global_loss = self.ct_global_loss(video_global_feat, text_global_feat)
                weight=self.cfg.TRAINING.ct_global_loss_weight
                ct_global_loss = weight*ct_global_loss

                if self.cfg.TRAINING.use_time_match:
                    ct_time_loss = self.ct_time_loss(text_local_feat,video_local_feat)
                    weight=self.cfg.TRAINING.ct_time_loss_weight
                    ct_time_loss = weight*ct_time_loss

        if stage == 1:
 
            return dict(text_global_feat = text_global_feat,
                        video_global_feat = video_global_feat,
                        ct_global_loss = ct_global_loss,
                        ct_time_loss = ct_time_loss,
                        mlm_loss=0,
                        vtm_loss=0,
                        mlm_prediction=0,
                        mlm_acc = 0,
                        vtm_acc = 0
                        )

        video_stage1_embd = self.video_token_pos(video_stage1_embd)

        video_stage1_embd = video_stage1_embd.flatten(1,2)

        visual_attention_mask = attention_mask.new_ones(
            video_stage1_embd.shape[:2])
        attention_mask = torch.cat(
            [attention_mask, visual_attention_mask], dim=-1)


        video_stage1_embd, vtm_labels = self.shuffle_embd_for_vtm(video_stage1_embd)

        stage1_embedding_output = torch.cat([text_global_embd, video_stage1_embd], dim=1)

        mlm_labels = torch.cat([-100*mlm_labels.new_ones(mlm_labels.shape[:1]).unsqueeze(1), mlm_labels, -100*mlm_labels.new_ones(video_stage1_embd.shape[:2])], dim=1)

        fusion_output = self.text_encoder(stage1_embedding_output, attention_mask=attention_mask, mlm_labels = mlm_labels, vtm_labels=vtm_labels,  return_dict=True, stage=2)


        mlm_loss = self.cfg.TRAINING.mlm_loss_weight * fusion_output['mlm_loss']
        mlm_acc = fusion_output['mlm_acc']
        mlm_prediction = fusion_output['mlm_logits']
        vtm_acc = fusion_output['vtm_acc']
        vtm_loss = self.cfg.TRAINING.vtm_loss_weight * fusion_output['vtm_loss']

        return dict(mlm_loss=mlm_loss,
                    vtm_loss=vtm_loss,
                    mlm_prediction=mlm_prediction,
                    mlm_acc = mlm_acc,
                    vtm_acc = vtm_acc,
                    ct_global_loss = 0.,
                    ct_time_loss = 0.,
                    )
    

