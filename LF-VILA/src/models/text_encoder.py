from .bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead
from transformers.utils import logging
import torch
from transformers.modeling_outputs import MaskedLMOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch.nn.functional as F


logger = logging.get_logger(__name__)

class TextEncoderForPretraining(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, args, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(args, config, add_pooling_layer=True)
        self.cls = BertOnlyMLMHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        mlm_labels=None,
        vtm_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        stage = 2
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage = stage
        )

        sequence_output = outputs[0]

        if stage == 1 or stage == 0:
            return sequence_output

        pooled_output = outputs.pooler_output
        prediction_scores = self.cls(sequence_output)

        seq_relationship_score = self.seq_relationship(pooled_output)

        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = 0
        mlm_acc = 0
        if mlm_labels is not None:
            if vtm_labels is not None:
                # ignore wrong pairs
                B = prediction_scores.size(0)
            
                masked_lm_loss = loss_fct(prediction_scores[(B//2):].view(-1, self.config.vocab_size), mlm_labels[(B//2):].view(-1))
            else:
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))

            mlm_mask = mlm_labels != -100
            if mlm_mask.sum().item() > 0:
                mlm_acc = (prediction_scores[mlm_mask].max(dim=-1)[1] == mlm_labels[mlm_mask]).float().mean(dim=0, keepdim=True)
            else:
                mlm_acc = 0

        if vtm_labels is not None:
            vtm_loss = loss_fct(
                seq_relationship_score.view(-1, 2), vtm_labels.view(-1))
            
            vtm_acc = (seq_relationship_score.max(dim=-1)[1] == vtm_labels).float().mean(dim=0, keepdim=True)
        else:
            vtm_loss = 0
            vtm_acc = 0

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return dict(
            mlm_loss=masked_lm_loss,
            mlm_logits=prediction_scores,
            last_hidden_states=sequence_output,
            mlm_acc = mlm_acc,
            vtm_loss = vtm_loss,
            vtm_acc = vtm_acc,
            vtm_score = seq_relationship_score
        )

class TextEncoderForMultichoice(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, args, config):
        super().__init__(config)

        self.bert = BertModel(args, config, add_pooling_layer=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, 1)
        self.span_classifier = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        span_labels = None,
        span_label_weights = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        stage=2,
        subtitle_fuse=None,
        num_choices = 5,
        num_frame = 32
    ):

        """
        Args:
            input_ids: (B, num_labels, L) or (B, num_labels, L, C)
            attention_mask: (B, num_labels, L)
            labels: (B, ), in [0, num_labels-1]

        Returns:

        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage=stage
        )

        sequence_output = outputs[0]

        if stage == 1 or stage == 0:    
            return sequence_output 
        
        # video_token_embedding
        video_token_embedding = sequence_output[:,-num_frame*6:,:] 
        video_token_embedding = video_token_embedding.view(-1 , num_frame, 6, video_token_embedding.shape[-1]).mean(dim=2) 
        span_prediction = self.span_classifier(video_token_embedding) 
        span_prediction = span_prediction.view(-1, num_choices, num_frame, 2)
        span_prediction = span_prediction.max(dim=1).values

        pooled_output = outputs.pooler_output 

        if subtitle_fuse is not None:
            pooled_output = pooled_output + subtitle_fuse
        pooled_output = self.dropout(pooled_output) 
        logits = self.classifier(pooled_output)    
        
        reshaped_logits = logits.view(-1, num_choices) 

        loss = None
        acc = 0
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

            acc = (reshaped_logits.max(dim=-1)[1] == labels).float().mean(dim=0, keepdim=True)

        span_loss = None
        span_acc = 0
        if span_labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            span_loss = loss_fct(span_prediction.flatten(0,1), span_labels.flatten(0,1))
            span_loss = (span_loss * span_label_weights.flatten(0,1)).mean()

            span_acc = (span_prediction.flatten(0,1).max(dim=-1)[1] == span_labels.flatten(0,1)).float().mean(dim=0, keepdim=True)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return dict(
            loss=loss,
            span_loss = span_loss,
            logits=reshaped_logits,
            acc = acc,
            span_acc = span_acc
        )

class TextEncoderForClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, args, config, classification_labels, label_smoothing=0):
        super().__init__(config)

        self.bert = BertModel(args, config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, classification_labels)
        self.label_smoothing = label_smoothing

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        stage = 2
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage = stage
        )

        sequence_output = outputs[0]

        if stage == 1 or stage == 0:
            return sequence_output

        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output) 
        logits = self.classifier(pooled_output) 

        labels = labels.view(-1)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss = loss_fct(logits, labels)

            acc = (logits.max(dim=-1)[1] == labels).float().mean(dim=0, keepdim=True)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return dict(
            loss=loss,
            logits=logits,
            acc = acc
        )

