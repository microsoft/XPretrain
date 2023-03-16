import json
import torch



def build_optimizer_parameters(config, model):

    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'pos_embed','relative_position_bias_table']

    if "weight_decay" in config.TRAINING.keys():
        weight_decay = config.TRAINING["weight_decay"]
    else:
        weight_decay = 0.01


    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in param_optimizer
            if not any(nd in n for nd in no_decay) and p.requires_grad
        ],
        'weight_decay':
        weight_decay
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
        'weight_decay':
        0.0
    }]
    
    return optimizer_grouped_parameters