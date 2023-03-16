import torch
from src.utils.logger import LOGGER
import re


def load_model_weights_with_mismatch(model, loaded_path, load_swin=False, load_bert=False, pretrained2d=False, change_window=False):
    """operated in-place, no need to return `model`"""

    if load_swin:
        loaded_state_dict = process_swin_weights(model, loaded_path, pretrained2d)
    elif load_bert:
        loaded_state_dict = torch.load(loaded_path, map_location="cpu")
    else:
        loaded_state_dict = torch.load(loaded_path, map_location="cpu")

    if change_window and not load_bert and not load_swin:
        LOGGER.info(f"Window Size mismatched, start processing")
        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in loaded_state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del loaded_state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in loaded_state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del loaded_state_dict[k]

        # delete relative_position_bias_table since we always re-init it
        relative_position_bias_table_keys = [k for k in loaded_state_dict.keys() if "relative_position_bias_table" in k]

        # bicubic interpolate relative_position_bias_table if not match
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = loaded_state_dict[k]
            relative_position_bias_table_current = model.state_dict()[k]

            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()

            i_layer = int(k.split('.blocks')[0].split('.')[-1])

            layer_window_size = model.video_encoder.window_size[i_layer]

            L2 = (2*layer_window_size[0]-1) * (2*layer_window_size[1]-1) * (2*layer_window_size[2]-1)

            if nH1 != nH2:
                LOGGER.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    W1 = (2*layer_window_size[1]-1) * (2*layer_window_size[2]-1)
 
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, -1, W1).float(), size=(2*layer_window_size[0]-1, W1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
                    loaded_state_dict[k] = relative_position_bias_table_pretrained.half()

    model_keys = set([k for k in list(model.state_dict().keys())])
    load_keys = set(loaded_state_dict.keys())


    # process mismatched 
    if 'sent_embedding.segment_embeddings.weight' in loaded_state_dict and 'sent_embedding.segment_embeddings.weight' in model.state_dict():
        l1,c = loaded_state_dict['sent_embedding.segment_embeddings.weight'].shape
        l2,c = model.state_dict()['sent_embedding.segment_embeddings.weight'].shape
        if l1 != l2:
            LOGGER.info(f"process segment embedding: source{l1} target{l2}")
            new_segment_embeddings_weight = torch.cat([loaded_state_dict['sent_embedding.segment_embeddings.weight'], model.state_dict()['sent_embedding.segment_embeddings.weight'][l1:,:]],dim=0)
            loaded_state_dict['sent_embedding.segment_embeddings.weight'] = new_segment_embeddings_weight


    toload = {}
    mismatched_shape_keys = []
    for k in model_keys:
        if k in load_keys:
            if model.state_dict()[k].shape != loaded_state_dict[k].shape:
                mismatched_shape_keys.append(k)
            else:
                toload[k] = loaded_state_dict[k]


    LOGGER.info("You can ignore the keys with `num_batches_tracked` or from task heads")
    LOGGER.info("Keys in loaded but not in model:")
    diff_keys = load_keys.difference(model_keys)
    LOGGER.info(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
    LOGGER.info("Keys in model but not in loaded:")
    diff_keys = model_keys.difference(load_keys)
    LOGGER.info(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
    LOGGER.info("Keys in model and loaded, but shape mismatched:")
    LOGGER.info(f"In total {len(mismatched_shape_keys)}, {sorted(mismatched_shape_keys)}")
    model.load_state_dict(toload, strict=False)

    del loaded_state_dict, toload, mismatched_shape_keys

def process_swin_weights(model, loaded_path=None, pretrained2d=False, load_pos=True):
    """Inflate the swin2d parameters to swin3d.

    The differences between swin3d and swin2d mainly lie in an extra
    axis. To utilize the pretrained parameters in 2d model,
    the weight of swin2d models should be inflated to fit in the shapes of
    the 3d counterpart.
    """

    checkpoint = torch.load(loaded_path, map_location='cpu')
    loaded_state_dict = checkpoint['model']


    # other weights
    loaded_state_dict['local_feat_proj.reduction.weight'] = loaded_state_dict['layers.2.downsample.reduction.weight']
    loaded_state_dict['local_feat_proj.norm.weight'] = loaded_state_dict['layers.2.downsample.norm.weight']
    loaded_state_dict['local_feat_proj.norm.bias'] = loaded_state_dict['layers.2.downsample.norm.bias']

    loaded_state_dict['norm_local.weight'] = loaded_state_dict['norm.weight']
    loaded_state_dict['norm_local.bias'] = loaded_state_dict['norm.bias']

    # rename checkpoint keys layer.block.

    origin_stage_layer = set()
    for k in loaded_state_dict:
        x = re.findall('(layers\..+blocks\..+?\.)', k)
        if len(x) > 0:
            origin_stage_layer.add(x[0])
    origin_stage_layer = list(origin_stage_layer)
    origin_stage_layer.sort(key=lambda x: int(x.split('.')[-2]))
    origin_stage_layer.sort(key=lambda x: int(x.split('.')[1]))
    

    current_stage_layer = set()
    for k in model.state_dict().keys():
        x = re.findall('(layers\..+blocks\..+?\.)', k)
        if len(x) > 0:
            current_stage_layer.add(x[0])
    current_stage_layer = list(current_stage_layer)
    current_stage_layer.sort(key=lambda x: int(x.split('.')[-2]))
    current_stage_layer.sort(key=lambda x: int(x.split('.')[1]))

    replace_rule_dict = {origin_stage_layer[i]:current_stage_layer[i] for i in range(len(current_stage_layer))}

    
    loaded_state_dict_new = {}
    for k in list(loaded_state_dict.keys()):
        x = re.findall('(layers\..+blocks\..+?\.)', k)
        if len(x) > 0:
            x = x[0]
            new_k = re.sub('(layers\..+blocks\..+?\.)',replace_rule_dict[x],k)
            loaded_state_dict_new[new_k] = loaded_state_dict[k]

    for k,v in loaded_state_dict_new.items():
        loaded_state_dict[k] = v

    origin_stage_layer = set()
    for k in loaded_state_dict:
        x = re.findall('(layers\..+downsample\.)', k)
        if len(x) > 0:
            origin_stage_layer.add(x[0])
    origin_stage_layer = list(origin_stage_layer)
    origin_stage_layer.sort()
    

    current_stage_layer = set()
    for k in model.state_dict().keys():
        x = re.findall('(layers\..+downsample\.)', k)
        if len(x) > 0:
            current_stage_layer.add(x[0])
    current_stage_layer = list(current_stage_layer)
    current_stage_layer.sort()

    replace_rule_dict = {origin_stage_layer[i]:current_stage_layer[i] for i in range(len(current_stage_layer))}

    loaded_state_dict_new = {}
    for k in list(loaded_state_dict.keys()):
        x = re.findall('(layers\..+downsample\.)', k)
        if len(x) > 0:
            x = x[0]
            new_k = re.sub('(layers\..+downsample\.)',replace_rule_dict[x],k)
            loaded_state_dict[new_k] = loaded_state_dict[k]
    
    for k,v in loaded_state_dict_new.items():
        loaded_state_dict[k] = v

    for k in list(loaded_state_dict.keys()):
        if not k in model.state_dict().keys():
            del loaded_state_dict[k]
    

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in loaded_state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del loaded_state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in loaded_state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del loaded_state_dict[k]

    # delete relative_position_bias_table since we always re-init it
    relative_position_bias_table_keys = [k for k in loaded_state_dict.keys() if "relative_position_bias_table" in k]


    if load_pos:
    # bicubic interpolate relative_position_bias_table if not match
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = loaded_state_dict[k]
            relative_position_bias_table_current = model.state_dict()[k]

            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()

            i_layer = int(k.split('.blocks')[0][7:])

            layer_window_size = model.window_size[i_layer]

            L2 = (2*layer_window_size[1]-1) * (2*layer_window_size[2]-1)
            wd = layer_window_size[0]
            if nH1 != nH2:
                LOGGER.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2*layer_window_size[1]-1, 2*layer_window_size[2]-1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
            loaded_state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)

    else:
        for k in relative_position_bias_table_keys:
            del loaded_state_dict[k]

    if pretrained2d:
        if model.patch_size[1] != 4:
            weight = loaded_state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,model.patch_size[0],2,2) / model.patch_size[0] / 4

        else:
            weight = loaded_state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,model.patch_size[0],1,1) / model.patch_size[0]

        loaded_state_dict['patch_embed.proj.weight'] = weight

    return loaded_state_dict





