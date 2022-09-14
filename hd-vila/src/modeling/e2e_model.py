from src.modeling.modeling_stage import (
    HDVILAForPreTraining)
import torch
from torch import nn
import torch.nn.functional as F
import math
import json
from transformers import BertConfig
from mmcv.runner import load_checkpoint
from src.datasets.data_utils import repeat_tensor_rows
from src.utils.load_save import load_state_dict_with_mismatch
from src.modeling.resnet_mmdetection import ResNet
from src.modeling.timesformer import TimeSformer


class HDVILA(nn.Module):
    """
    if involves generation, set if_generate=True,
    transformer_cls: adapt to different tasks,
    config: processed in main file, this file gives a ExampleConfig
    """
    def __init__(self, config,
                 transformer_cls=HDVILAForPreTraining,
                 stage=2):
        super(HDVILA, self).__init__()
        self.mean_value = [123.675, 116.28, 103.53]
        self.std_value = [58.395, 57.12, 57.375]
        self.mean = None
        self.std = None
        assert stage in [1, 2]
        self.stage = stage
        setattr(config, "stage", stage)

        self.cnn = ResNet(depth=config.resnet_depth, frozen_stages=config.resnet_frozen_stage)
        self.cnn_low = ResNet(depth=config.resnet_depth, frozen_stages=config.resnet_frozen_stage)
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(config.backbone_channel_in_size, config.hidden_size, kernel_size=1, stride=1,
                      padding=0, groups=1, bias=False,
                      dilation=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.grid_encoder_low = nn.Sequential(
            nn.Conv2d(config.backbone_channels[-2], config.hidden_size, kernel_size=1, stride=1,
                      padding=0, groups=1, bias=False,
                      dilation=1),
            nn.GELU(),
        )

        if not hasattr(config, "timesformer_type"):
            setattr(config, "timesformer_type", "new")

        self.timesformer = TimeSformer(depth=config.timesformer_depth, num_frames=7, H=10, W=16,
                                       embed_dim=config.hidden_size, num_heads=config.timesformer_heads,
                                       timesformer_type=config.timesformer_type)
        self.grid_encoder_combine = nn.Sequential(
            nn.Conv2d(2 * config.hidden_size, config.hidden_size, kernel_size=1, stride=1,
                      padding=0, groups=1, bias=False,
                      dilation=1),
            nn.GELU(),
        )

        self.transformer = transformer_cls(config)


    def forward(self, img_middle, img_other, text_input_ids, text_input_mask, mlm_labels=None, itm_labels=None):
        """
        batch['img_middle']: [b, clip_num, 3, h, w] uint8, RGB
        batch['img_other']: [b, clip_num, frm_num-1, 3, h/4, w/4] uint8, 0-255, RGB
        batch['text_input_ids']: [b, length] LongTensor
        batch['text_input_mask']: [b, length] LongTensor
        batch['mlm_labels']: [b, length] LongTensor
        batch['itm_labels']: [b, length] LongTensor
        """
        stage_features_tuple, visual_features = self.extract_features(img_middle, img_other)  # visual_features ([B*clip_num, C_, H_, W_],...)

        if img_middle is not None:
            b, clip_num = img_middle.shape[:2]

        else:
            b, clip_num = img_other.shape[:2]
            

        frm_num = 1
        c, h, w = visual_features.shape[-3:]

        visual_features = visual_features.reshape(b, clip_num, frm_num, c, h, w)
        visual_features = visual_features.permute(1, 0, 2, 4, 5, 3)

        outputs = self.transformer(visual_inputs=visual_features, text_input_ids=text_input_ids, text_input_mask=text_input_mask,
                                   mlm_labels=mlm_labels, itm_labels=itm_labels)  # dict

        return outputs

    def denormalize(self, images):
        # sr_images [B, 3, H, W]
        if self.mean is None:
            self.mean = torch.tensor(self.mean_value).view(1, 3, 1, 1).type_as(images).to(images.device)
        if self.std is None:
            self.std = torch.tensor(self.std_value).view(1, 3, 1, 1).type_as(images).to(images.device)
        return (self.std*images+self.mean)/255.

    def normalize(self, images):
        # images [B, 3, h, w]
        if self.mean is None:
            self.mean = torch.tensor(self.mean_value).view(1, 3, 1, 1).type_as(images).to(images.device)
        if self.std is None:
            self.std = torch.tensor(self.std_value).view(1, 3, 1, 1).type_as(images).to(images.device)
        return (images-self.mean)/self.std

    def extract_features(self, img_middle, img_other):
        if img_middle is None:
            return self.extract_features_other(img_other)

        if img_other is None:
            return self.extract_features_middle(img_middle)

        b, clip_num, c, h, w = img_middle.shape
        img_middle = self.normalize(img_middle.reshape(-1, c, h, w))
        _, _, frm_num, _, h_, w_ = img_other.shape
        frm_num = frm_num + 1
        img_other = self.normalize(img_other.reshape(-1, c, h_, w_))

        stage_features = self.cnn(img_middle)
        img_middle = self.grid_encoder(stage_features[-1])
        middle_feature_3 = F.interpolate(stage_features[-2], scale_factor=1/4, recompute_scale_factor=False)
        middle_feature_3 = self.grid_encoder_low(middle_feature_3)

        img_other = self.cnn_low.forward_to_stage(img_other)
        img_other = self.grid_encoder_low(img_other)

        middle_feature_3 = middle_feature_3.reshape(-1, 1, middle_feature_3.shape[-3], middle_feature_3.shape[-2], middle_feature_3.shape[-1])
        img_other = img_other.reshape(-1, frm_num-1, img_other.shape[-3], img_other.shape[-2], img_other.shape[-1])

        temporal_feature = torch.cat((img_other[:, :int(frm_num/2)], middle_feature_3, img_other[:, int(frm_num/2):]), 1)
        temporal_feature = self.timesformer(temporal_feature)
        temporal_feature = temporal_feature[:, int(frm_num/2)]

        middle_feature = self.grid_encoder_combine(torch.cat((img_middle, temporal_feature), 1))

        return stage_features, middle_feature

    def extract_features_other(self, img_other):
        b, clip_num, frm_num, c, h_, w_ = img_other.shape
        img_other = self.normalize(img_other.reshape(-1, c, h_, w_))

        img_other = self.cnn_low.forward_to_stage(img_other)
        img_other = self.grid_encoder_low(img_other)

        img_other = img_other.reshape(-1, frm_num, img_other.shape[-3], img_other.shape[-2],
                                          img_other.shape[-1])

        temporal_feature = img_other
        temporal_feature = self.timesformer(temporal_feature)
        temporal_feature = temporal_feature[:, int(frm_num / 2)]

        middle_feature = temporal_feature

        return None, middle_feature

    def extract_features_middle(self, img_middle):

        b, clip_num, c, h, w = img_middle.shape
        img_middle = self.normalize(img_middle.reshape(-1, c, h, w))


        stage_features = self.cnn(img_middle)
        img_middle = self.grid_encoder(stage_features[-1])

        middle_feature = img_middle

        return stage_features, middle_feature

    def load_separate_ckpt(self, cnn_weights_path=None, bert_weights_path=None):
        if cnn_weights_path:
            load_checkpoint(self.cnn, cnn_weights_path, map_location='cpu')
            load_checkpoint(self.cnn_low, cnn_weights_path, map_location='cpu')

        if bert_weights_path:
            load_state_dict_with_mismatch(self.transformer, bert_weights_path)

    def freeze_cnn_backbone(self):
        for n, p in self.cnn.feature.named_parameters():
            p.requires_grad = False
        for n, p in self.cnn_low.feature.named_parameters():
            p.requires_grad = False

    def freeze_stage_one_params(self):
        freeze_modules = ["cnn", "cnn_low", "grid_encoder", "grid_encoder_low", "grid_encoder_combine",\
                          "timesformer"]
        for i in freeze_modules:
            m = getattr(self, i)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        for m in [self.transformer.t_proj, self.transformer.v_proj, self.transformer.bert.embeddings, self.transformer.bert.pooler1]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        for i in range(0, 12):
            m = self.transformer.bert.encoder.layer[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    

    def freeze_stage_one_vision_params(self):
        freeze_modules = ["cnn", "cnn_low", "grid_encoder", "grid_encoder_low", "grid_encoder_combine",\
                          "timesformer"]
        for i in freeze_modules:
            m = getattr(self, i)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


def convert_pth_file(path, out_path):
    from collections import OrderedDict
    a = torch.load(path)
    new_state_dict = {"state_dict": OrderedDict()}

    for key in a['state_dict']:
        if "backbone." in key:
            new_state_dict['state_dict'][key.replace("backbone.", "")] = a['state_dict'][key]

    torch.save(new_state_dict, out_path)


if __name__ == "__main__":
    from src.utils.basic_utils import load_json
    from thop import profile
    model_cfg = load_json("src/configs/base_model_large.json")
    model_cfg = BertConfig(**model_cfg)
    add_attrs = {
        "pixel_random_sampling_size": 160,
        "use_itc": 1,
        "score_agg_func": "lse",
        "backbone_channels": [256, 512, 1024, 2048],
        "resnet_depth": 50,
        "resnet_frozen_stage": -1,
        "timesformer_depth": 4,
        "timesformer_heads": 16,
        "bert_mean": 1
    }
    for k in add_attrs:
        setattr(model_cfg, k, add_attrs[k])
    
    model = HDVILA(config=model_cfg,
                transformer_cls=HDVILAForPreTraining,
                stage=2)

    batch = {'img_middle': torch.randn(1, 2, 3, 640, 1024),
            'img_other': torch.randn(1, 2, 6, 3, 160, 256),
            'text_input_ids': torch.ones(1, 50).long(),
            'text_input_mask': torch.ones(1, 50).long(),
            'mlm_labels': torch.ones(1, 50).long(),
            'itm_labels': None
    }

    macs, params = profile(model, inputs=(batch["img_middle"],
                                          batch["img_other"],
                                          batch["text_input_ids"],
                                          batch["text_input_mask"],
                                          batch["mlm_labels"],
                                          batch["itm_labels"]))

    print(macs, params)

    # output = model(**batch)
    # for k in output:
    #     print(k, output[k].shape)







