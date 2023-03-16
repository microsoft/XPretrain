import torch
import torch.nn as nn
from functools import partial
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from src.modeling.CLIP_ViP import CLIPModel, clip_loss
from src.modeling.CLIP import CLIPModel as CLIP

class VidCLIP(nn.Module):
    def __init__(self, args):
        super(VidCLIP, self).__init__()
        clipconfig = CLIPConfig.from_pretrained(args.clip_config)
        setattr(clipconfig, "vision_additional_config", args.clip_vision_additional_config)
        self.vision_additional_config = args.clip_vision_additional_config
        if args.clip_weights:
            if self.vision_additional_config.type == "ViP":
                self.clipmodel = CLIPModel.from_pretrained(args.clip_weights, config=clipconfig)
            else:
                self.clipmodel = CLIP.from_pretrained(args.clip_weights, config=clipconfig)
        else:
            if self.vision_additional_config.type == "ViP":
                self.clipmodel = CLIPModel(clipconfig)
            else:
                self.clipmodel = CLIP(clipconfig)
        
        # init logit scale from 
        logit_scale_value = self.vision_additional_config.logit_scale_init_value
        self.clipmodel.logit_scale.data.fill_(logit_scale_value)
    
    def overload_logit_scale(self, overload_logit_scale):
        self.clipmodel.logit_scale.data.fill_(overload_logit_scale)

    def forward(self, video, text_input_ids, text_input_mask, \
                image=None, caption_ids=None, caption_masks=None):
        """
        video [B, n_clips*num_frms, C, H, W]
        text_input_ids [B, L]
        text_input_mask [B, L]
        image [B, img_num, C, H, W]
        caption_ids [B, img_num, L]
        caption_masks [B, img_num, L]
        """
        B, N, C, H, W = video.shape

        if self.vision_additional_config.type == "ViP":
            inputs = {"input_ids": text_input_ids,
                    "attention_mask": text_input_mask,
                    "pixel_values": video,
                    "return_loss": False}
            outputs = self.clipmodel(**inputs)
            results = {}
            results["text_features"] = outputs["text_embeds"]
            results["vis_features"] = outputs["image_embeds"]
            # results["loss"] = outputs["loss"]
        else:
            video = video.reshape(-1, C, H, W)
            inputs = {"input_ids": text_input_ids,
                    "attention_mask": text_input_mask,
                    "pixel_values": video}
            outputs = self.clipmodel(**inputs)
            vis_features = outputs["vision_model_output"][1]

            vis_features = self.clipmodel.visual_projection(vis_features)
            vis_features = vis_features / vis_features.norm(dim=-1, keepdim=True)
            vis_features = vis_features.reshape(B, N, -1).mean(1)
            vis_features = vis_features / vis_features.norm(dim=-1, keepdim=True)
            
            results = {}
            results["text_features"] = outputs["text_embeds"]
            results["vis_features"] = vis_features
        if image is not None:
            B, img_num, C, H, W = image.shape
            L = caption_ids.shape[-1]
            inputs = {"input_ids": caption_ids.reshape(-1, L),
                    "attention_mask": caption_masks.reshape(-1, L),
                    "pixel_values": image.reshape(-1, 1, C, H, W),
                    "return_loss": False}
            outputs = self.clipmodel(**inputs)
            results["img_features"] = outputs["image_embeds"]
            results["cap_features"] = outputs["text_embeds"]
        
        return results
    
    def forward_video(self, video):
        inputs = {"pixel_values": video,
                "if_norm": True}
        video_features = self.clipmodel.get_image_features(**inputs)
        return video_features
    
    def forward_text(self, text_input_ids, text_input_mask):
        inputs = {"input_ids": text_input_ids,
                "attention_mask": text_input_mask,
                "if_norm": True}
        text_features = self.clipmodel.get_text_features(**inputs)
        return text_features

    def freeze_text_encoder(self, freeze_text_proj):
        freeze_list = [self.clipmodel.text_model]
        if freeze_text_proj:
            freeze_list.append(self.clipmodel.text_projection)
        for m in freeze_list:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

