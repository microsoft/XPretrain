"""
Transformer part of HDVILA
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from .Transformers import BertPreTrainedModel
from .Transformers import (
    BertPreTrainingHeads, BertEmbeddings, BertEncoder, BertPooler)
# from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
LayerNorm = torch.nn.LayerNorm
from src.utils.basic_utils import flat_list_of_lists

def get_random_sample_indices(
        seq_len, num_samples=100, device=torch.device("cpu")):
    """
    Args:
        seq_len: int, the sampled indices will be in the range [0, seq_len-1]
        num_samples: sample size
        device: torch.device

    Returns:
        1D torch.LongTensor consisting of sorted sample indices
        (sort should not affect the results as we use transformers)
    """
    if num_samples >= seq_len:
        # return all indices
        sample_indices = np.arange(seq_len)
    else:
        sample_indices = np.random.choice(
            seq_len, size=num_samples, replace=False)
        sample_indices = np.sort(sample_indices)
    return torch.from_numpy(sample_indices).long().to(device)


BertLayerNorm = LayerNorm


class VisualInputEmbedding(nn.Module):
    """
    Takes input of both image and video (multi-frame)
    """
    def __init__(self, config):
        super(VisualInputEmbedding, self).__init__()
        self.config = config

        # sequence embedding
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.row_position_embeddings = nn.Embedding(
            config.max_grid_row_position_embeddings,
            config.hidden_size)
        self.col_position_embeddings = nn.Embedding(
            config.max_grid_col_position_embeddings,
            config.hidden_size)
        self.token_type_embeddings = nn.Embedding(1, config.hidden_size)
        self.LayerNorm = BertLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, grid):
        """
        Args:
            grid: (B, n_frm, H, W, C), note that #frm can be 1

        Returns:

        """
        bsz, _, _, _, hsz = grid.shape

        # temporal mean pooling
        grid = grid.mean(1)  # (B, H, W, d)
        grid = self.add_2d_positional_embeddings(grid)  # (B, H, W, d)
        # image token sequence
        visual_tokens = grid.view(bsz, -1, hsz)  # (B, H*W, d)

        # perform random sampling. It is only used in training phase
        # of pre-training, but not used in inference or downstream tasks.
        if hasattr(self.config, "pixel_random_sampling_size") and \
                self.config.pixel_random_sampling_size > 0 and self.training:
            sampled_indices = get_random_sample_indices(
                seq_len=visual_tokens.shape[1],
                num_samples=self.config.pixel_random_sampling_size,
                device=visual_tokens.device
            )
            visual_tokens = visual_tokens.index_select(
                dim=1, index=sampled_indices)  # (B, #samples, d)
        visual_tokens_shape = visual_tokens.shape[:-1]  # (B, H*W)
        device = visual_tokens.device

        # image token type embeddings.
        token_type_ids = torch.zeros(
            visual_tokens_shape, dtype=torch.long, device=device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # embeddings = visual_tokens + position_embeddings + token_type_embeddings
        embeddings = visual_tokens + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # (B, H*W, d)

    def add_temporal_postion_embeddings(self, grid):
        """
        Args:
            grid: (B, n_frms, H, W, d)

        Returns:
            (B, n_frms, H, W, d)
        """
        n_frms, height, width, hsz = grid.shape[-4:]

        # add row-wise position embeddings
        temporal_position_ids = torch.arange(
            n_frms, dtype=torch.long, device=grid.device)  # (n_frms, )
        t_position_embeddings = self.temporal_position_embeddings(
            temporal_position_ids)  # (n_frms, d)
        new_shape = (1, n_frms, 1, 1, hsz)  # (1, n_frms, 1, 1, d)
        grid = grid + t_position_embeddings.view(
            *new_shape)  # broadcast automatically

        return grid

    def add_2d_positional_embeddings(self, grid):
        """
        Args:
            grid: (B, *, H, W, d)

        Returns:
            (B, *, H, W, d)
        """
        height, width, hsz = grid.shape[-3:]

        # add row-wise position embeddings
        row_position_ids = torch.arange(
            height, dtype=torch.long, device=grid.device)  # (H, )
        row_position_embeddings = self.row_position_embeddings(
            row_position_ids)  # (H, d)
        row_shape = (1, ) * (len(grid.shape) - 3) + (
            height, 1, hsz)  # (1, *1, H, 1, d)
        grid = grid + row_position_embeddings.view(
            *row_shape)  # broadcast automatically

        # add column-wise position embeddings
        col_position_ids = torch.arange(
            width, dtype=torch.long, device=grid.device)  # (W, )
        col_position_embeddings = self.col_position_embeddings(
            col_position_ids)  # (W, d)
        col_shape = (1, ) * (len(grid.shape) - 3) + (
            1, width, hsz)  # (1, *1, 1, W, d)
        grid = grid + col_position_embeddings.view(
            *col_shape)  # broadcast automatically
        return grid


class HDVILABaseModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    config keys:
        text_model: str, text model name, default "bert-based-uncased"
        pretrained: bool, use pre-trained vision_model, default True
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler1 = BertPooler(config)

        if config.stage == 2:
            self.visual_embeddings = VisualInputEmbedding(config)
            self.pooler2 = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, clip_num, text_input_ids, visual_inputs, attention_mask):
        r"""Modified from BertModel
        text_input_ids: (B, Lt)
        visual_inputs: (B, #frame, H, W, C)
        attention_mask: (B, Lt)  with 1 indicates valid, 0 indicates invalid position.
        """
        input_shape = text_input_ids.size()
        device = text_input_ids.device

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        text_input_ids = self.embeddings(
            input_ids=text_input_ids)  # (B, Lt, D)
        visual_inputs = self.visual_embeddings(
            visual_inputs)  # (B, Lv, d)
        visual_attention_mask = attention_mask.new_ones(
            visual_inputs.shape[:2])  # (B, Lv)
        attention_mask = torch.cat(
            [attention_mask, visual_attention_mask], dim=-1)  # (B, lt+Lv, d)
        embedding_output = torch.cat(
            [text_input_ids, visual_inputs],
            dim=1)  # (B, Lt+Lv, d)

        attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        embedding_output = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=self.get_head_mask(
                None, self.config.num_hidden_layers)  # required input
        )
        embedding_output = embedding_output[0]
        pooled_output = self.pooler(embedding_output)

        embedding_output = embedding_output.view(clip_num, -1, embedding_output.shape[-2], embedding_output.shape[-1])
        pooled_output = pooled_output.view(clip_num, -1, pooled_output.shape[-1])

        # outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
        return embedding_output, pooled_output

    def forward_stage1(self, text_input_ids, attention_mask):
        r"""Modified from BertModel
        text_input_ids: (B, Lt)
        visual_inputs: (B, #frame, H, W, C)
        attention_mask: (B, Lt)  with 1 indicates valid, 0 indicates invalid position.
        """
        input_shape = text_input_ids.size()
        device = text_input_ids.device

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        text_input_ids = self.embeddings(
            input_ids=text_input_ids)  # (B, Lt, D)

        attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        text_input_ids = self.encoder.forward_stage(
            text_input_ids,
            attention_mask=attention_mask,
            head_mask=self.get_head_mask(
                None, self.config.num_hidden_layers),  # required input
            stage=1
        )
        text_input_ids = text_input_ids[0]
        pooled_output = self.pooler1(text_input_ids)

        # outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
        return text_input_ids, pooled_output

    def forward_stage2(self, clip_num, text_input_ids, visual_inputs, attention_mask):
        r"""Modified from BertModel
        text_input_ids: (B, Lt)
        visual_inputs: (B, #frame, H, W, C)
        attention_mask: (B, Lt)  with 1 indicates valid, 0 indicates invalid position.
        """
        input_shape = text_input_ids.size()
        device = text_input_ids.device

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.

        visual_inputs = self.visual_embeddings(
            visual_inputs)  # (B, Lv, d)
        visual_attention_mask = attention_mask.new_ones(
            visual_inputs.shape[:2])  # (B, Lv)
        attention_mask = torch.cat(
            [attention_mask, visual_attention_mask], dim=-1)  # (B, lt+Lv, d)
        embedding_output = torch.cat(
            [text_input_ids, visual_inputs],
            dim=1)  # (B, Lt+Lv, d)

        attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        embedding_output = self.encoder.forward_stage(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=self.get_head_mask(
                None, self.config.num_hidden_layers),  # required input
            stage=2
        )
        embedding_output = embedding_output[0]
        pooled_output = self.pooler2(embedding_output)

        embedding_output = embedding_output.view(clip_num, -1, embedding_output.shape[-2], embedding_output.shape[-1])
        pooled_output = pooled_output.view(clip_num, -1, pooled_output.shape[-1])

        # outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
        return embedding_output, pooled_output


class HDVILAForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = HDVILABaseModel(config)
        if config.stage == 2:
            self.cls = BertPreTrainingHeads(config)
            self.pool_method = self.config.score_agg_func
        
        self.t_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

    def get_output_embeddings(self):
        if self.config.stage == 2:
            return self.cls.predictions.decoder
        else:
            return None

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None:
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())


    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """ Clone module weights instead of tie
        """
        output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        
        # output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = torch.nn.functional.pad(
                output_embeddings.bias.data,
                (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings 

    def forward(
        self,
        visual_inputs,
        text_input_ids,
        text_input_mask,
        mlm_labels=None,
        itm_labels=None,
        **kwargs
    ):
        r"""
        text_input_ids: (B, Lt)
        visual_inputs: (clips*B, #frame=1, H, W, C)
        text_input_mask: (B, Lt)  with 1 indicates valid, 0 indicates invalid position.
        mlm_labels: (B, Lt)
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        itm_label: (B, )  with 1 indicates positive pair, 0 indicates negative pair.
        """
        clip_num, b, frm_num, h, w, c = visual_inputs.shape
        text_input_ids, pooled_output1 = self.bert.forward_stage1(text_input_ids=text_input_ids, attention_mask=text_input_mask)

        if self.config.bert_mean:
            pooled_output1 = (text_input_ids * text_input_mask.unsqueeze(-1)).sum(1) / torch.sum(text_input_mask, dim=1, keepdim=True)
            pooled_output1 = self.bert.pooler1.activation(self.bert.pooler1.dense(pooled_output1))

        visual_outputs = visual_inputs.mean((0, 2, 3, 4))
        if self.config.stage == 1:
            return dict(
                text_features=F.normalize(self.t_proj(pooled_output1), dim=-1, p=2),
                vis_features=F.normalize(self.v_proj(visual_outputs), dim=-1, p=2),
            )
        text_input_ids = text_input_ids.repeat(clip_num, 1, 1)
        text_input_mask = text_input_mask.repeat(clip_num, 1)

        visual_inputs = visual_inputs.reshape(clip_num * b, frm_num, h, w, c)
        sequence_output, pooled_output = self.bert.forward_stage2(clip_num=clip_num,
                                           text_input_ids=text_input_ids,
                                           visual_inputs=visual_inputs,
                                           attention_mask=text_input_mask)

        txt_len = text_input_mask.shape[1]
        vtoken_output = sequence_output[:, :, txt_len:, :]

        if self.pool_method == "mean":
            sequence_output = sequence_output.mean(0)
            pooled_output = pooled_output.mean(0)
        elif self.pool_method == "max":
            sequence_output = sequence_output.max(0)[0]
            pooled_output = pooled_output.max(0)[0]
        elif self.pool_method == "lse":
            sequence_output = torch.logsumexp(sequence_output, dim=0)
            pooled_output = torch.logsumexp(pooled_output, dim=0)
        else:
            raise ValueError(f"Invalid value for pool_method, "
                             f"got {self.pool_method}, expect one of [`mean`, `max`, `lse`]")

        # Only use the text part (which is the first `Lt` tokens) to save computation,
        # this won't cause any issue as cls only has linear layers.
        sequence_output, pooled_output = self.cls(
            sequence_output[:, :txt_len], pooled_output)

        loss_fct = CrossEntropyLoss(reduction="mean")
        if mlm_labels is not None:
            if itm_labels is not None:
                mlm_labels[itm_labels == 0] = -100
            mlm_loss = loss_fct(
                sequence_output.view(-1, self.config.vocab_size),
                mlm_labels.view(-1))
            mlm_mask = mlm_labels != -100
            if mlm_mask.sum().item() > 0:
                mlm_acc = (sequence_output[mlm_mask].max(dim=-1)[1] == mlm_labels[mlm_mask]).float().mean()
            else:
                mlm_acc = None
        else:
            mlm_loss = 0
            mlm_acc = 0

        if itm_labels is not None:
            itm_loss = loss_fct(
                pooled_output.view(-1, 2), itm_labels.view(-1))
            itm_acc = (pooled_output.max(dim=-1)[1] == itm_labels).float().mean()
        else:
            itm_loss = 0
            itm_acc = 0


        
        text_features = F.normalize(self.t_proj(pooled_output1), dim=-1, p=2)
        vis_features = F.normalize(self.v_proj(visual_outputs), dim=-1, p=2)

        return dict(
            vtoken_output=vtoken_output,
            mlm_acc=mlm_acc,
            mlm_loss=mlm_loss,
            itm_acc=itm_acc,
            itm_loss=itm_loss,
            text_features=text_features,
            vis_features=vis_features
        )


def instance_bce_with_logits(logits, labels, reduction="mean"):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(
        logits, labels, reduction=reduction)
    if reduction == "mean":
        loss *= labels.size(1)
    return loss


HDVILAForSequenceClassificationConfig = dict(
    cls_hidden_scale=2,   # mlp intermediate layer hidden size scaler
    classifier="mlp",  # classfied type, [mlp, linear]
    num_labels=3129,  # number of labels for classifier output
    loss_type="bce"  # [BCE, CE, KLDivLoss] only used when num_labels > 1
)


class HDVILAForSequenceClassification(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self, config):
        super(HDVILAForSequenceClassification, self).__init__(config)
        self.config = config

        self.bert = HDVILABaseModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size,
                      config.hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(config.hidden_size * 2, config.num_labels)
        )

        self.init_weights()

    def forward(self, text_input_ids, visual_inputs,
                text_input_mask, labels=None, **kwargs):

        clip_num, b, frm_num, h, w, c = visual_inputs.shape
        text_input_ids, _ = self.bert.forward_stage1(text_input_ids=text_input_ids,
                                                                  attention_mask=text_input_mask)

        text_input_ids = text_input_ids.repeat(clip_num, 1, 1)
        text_input_mask = text_input_mask.repeat(clip_num, 1)

        visual_inputs = visual_inputs.reshape(clip_num * b, frm_num, h, w, c)
        sequence_output, pooled_output = self.bert.forward_stage2(clip_num=clip_num,
                                                                  text_input_ids=text_input_ids,
                                                                  visual_inputs=visual_inputs,
                                                                  attention_mask=text_input_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits, loss = self.calc_loss(logits, labels)
        return dict(
            logits=logits,
            loss=loss
        )

    def calc_loss(self, logits, labels):
        if labels is not None:
            if self.config.num_labels == 1:  # regression
                loss_fct = MSELoss(reduction="none")
                # labels = labels.to(torch.float)
                loss = loss_fct(
                    logits.view(-1), labels.view(-1))
            else:
                if self.config.loss_type == 'bce':  # [VQA]
                    loss = instance_bce_with_logits(
                        logits, labels, reduction="none")
                elif self.config.loss_type == "ce":  # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss(reduction="none")
                    loss = loss_fct(
                        logits.view(-1, self.config.num_labels),
                        labels.view(-1))
                else:
                    raise ValueError("Invalid option for config.loss_type")
        else:
            loss = 0
        return logits, loss


class HDVILAForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super(HDVILAForMultipleChoice, self).__init__(config)
        self.config = config

        self.bert = HDVILABaseModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size,
                      config.hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(config.hidden_size * 2, 1)
        )
        self.init_weights()

    def forward(self, text_input_ids, visual_inputs,
                text_input_mask, labels=None, **kwargs):
        """
        Args:
            text_input_ids: (B * num_labels, Lt)
            visual_inputs: (B, Lv, d)
            text_input_mask: (B * num_labels, Lt)
            labels: (B, ), in [0, num_labels-1]

        Returns:

        """
        clip_num, b, frm_num, h, w, c = visual_inputs.shape
        text_input_ids, _ = self.bert.forward_stage1(text_input_ids=text_input_ids,
                                                                  attention_mask=text_input_mask)

        text_input_ids = text_input_ids.repeat(clip_num, 1, 1)
        text_input_mask = text_input_mask.repeat(clip_num, 1)

        visual_inputs = visual_inputs.reshape(clip_num * b, frm_num, h, w, c)
        repeats = int(text_input_ids.shape[0]/visual_inputs.shape[0])

        visual_inputs = repeat_tensor_rows_num(visual_inputs, repeats)
        sequence_output, pooled_output = self.bert.forward_stage2(clip_num=clip_num,
                                                                  text_input_ids=text_input_ids,
                                                                  visual_inputs=visual_inputs,
                                                                  attention_mask=text_input_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = logits.view(clip_num, b, self.config.num_labels)
        logits, loss = self.calc_loss(logits, labels)
        return dict(
            logits=logits,
            loss=loss
        )

    def calc_loss(self, logits, labels):
        if labels is not None:
            if self.config.loss_type == "ce":  # cross_entropy [GQA, Retrieval, Captioning]
                logits = logits.view(-1, self.config.num_labels)

            if self.config.num_labels == 1:  # regression
                loss_fct = MSELoss(reduction="none")
                # labels = labels.to(torch.float)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.config.loss_type == 'bce':  # [VQA]
                    loss = instance_bce_with_logits(
                        logits, labels, reduction="none")
                elif self.config.loss_type == "ce":  # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss(reduction="none")
                    # logits = logits.view(-1, self.config.num_labels)
                    loss = loss_fct(logits, labels.view(-1))
                else:
                    raise ValueError("Invalid option for config.loss_type")
        else:
            loss = 0
        return logits, loss

class HDVILAForRegression(BertPreTrainedModel):
    def __init__(self, config):
        super(HDVILAForRegression, self).__init__(config)
        self.config = config

        self.bert = HDVILABaseModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ELU(),
            nn.BatchNorm1d(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1))

        self.init_weights()

    def forward(self, clip_num, text_input_ids, visual_inputs,
                text_input_mask, labels=None, **kwargs):
        """
        Args:
            text_input_ids: (B * num_labels, Lt)
            visual_inputs: (B, Lv, d)
            text_input_mask: (B * num_labels, Lt)
            labels: (B, ), in [0, num_labels-1]

        Returns:

        """
        outputs = self.bert(clip_num=clip_num,
            text_input_ids=text_input_ids,
            visual_inputs=visual_inputs,
            attention_mask=text_input_mask,  # (B, Lt) note this mask is text only!!!
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)
        logits, loss = self.calc_loss(logits, labels)
        return dict(
            logits=logits,
            loss=loss
        )

    def calc_loss(self, logits, labels):
        if labels is not None:
            if self.config.loss_type == "mse":  # regression
                loss_fct = MSELoss(reduction="none")
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                raise ValueError(f"Invalid option {self.config.loss_type} for config.loss_type")
        else:
            loss = 0
        return logits, loss


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, hidden_states):
        return self.classifier(hidden_states)


class HDVILAForVideoTextRetrieval(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self, config):
        super(HDVILAForVideoTextRetrieval, self).__init__(config)
        self.config = config

        self.bert = HDVILABaseModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size,
                      config.hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(config.hidden_size * 2, config.num_labels)
        )
        self.margin = config.margin
        self.init_weights()

    def forward(self, clip_num, text_input_ids, visual_inputs,
                text_input_mask, labels=None, sample_size=-1, **kwargs):
        outputs = self.bert(clip_num=clip_num,
            text_input_ids=text_input_ids,
            visual_inputs=visual_inputs,
            attention_mask=text_input_mask,  # (B, Lt) note this mask is text only!!!
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # rank (B, 1) or ce (B, 2)
        logits, loss = self.calc_loss(logits, labels, sample_size=sample_size)
        return dict(
            logits=logits,
            loss=loss
        )

    def calc_loss(self, logits, labels, sample_size=-1):
        if labels is not None:
            if self.config.loss_type == "ce":
                loss_fct = CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    logits.view(-1, self.config.num_labels),
                    labels.view(-1))
            elif self.config.loss_type == "rank":
                # triplet loss
                rank_scores_sigmoid = torch.sigmoid(logits).squeeze()  # (B * (#pos=1 + #neg), )
                assert sample_size > 0  # video batch size
                # wrong! scores = rank_scores_sigmoid.contiguous().view(-1, sample_size)
                scores = rank_scores_sigmoid.contiguous().view(sample_size, -1)
                pos = scores[:, :1]  # (B, #pos=1)
                neg = scores[:, 1:]  # (B, #neg)
                loss = torch.clamp(self.margin + neg - pos, min=0)
            else:
                raise ValueError("Invalid option for config.loss_type")
        else:
            loss = 0
        return logits, loss

def repeat_tensor_rows_num(raw_tensor, row_repeats):
    """ repeat raw_tensor[i] row_repeats[i] times.
    Args:
        raw_tensor: (B, *)
        row_repeats: list(int), len(row_repeats) == len(raw_tensor)
    """
    if row_repeats == 1:
        return raw_tensor
    else:
        rows = len(raw_tensor)
        indices = torch.LongTensor(
            flat_list_of_lists([[i] * row_repeats for i in range(rows)])
        ).to(raw_tensor.device)
        return raw_tensor.index_select(0, indices)
