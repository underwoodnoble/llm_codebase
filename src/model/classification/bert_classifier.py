from typing import Optional, Tuple, Union

import torch
from torch import nn

from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import PretrainedConfig
from transformers import BertModel, BertPreTrainedModel


class BertRewardModel(BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig, num_of_classes=1):
        super().__init__()
        self.model = BertModel(config)
        self.class_head = nn.Linear(config.hidden_size, num_of_classes, bias=False)
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pooling_type: str = "average",
        padding_side: str = "right",
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        last_hidden_states = transformer_outputs.last_hidden_states # (batch_size, seq_len, hidden_size)

        
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]
        
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(dim=-1)).to(last_hidden_states.device)
            else:
                sequence_lengths = -1
        
        if attention_mask is None:
            attention_mask = torch.ne(input_ids, self.config.pad_token_id).float() # (batch_size, seq_len)
        
        attention_mask_ext = attention_mask.unsqueeze(-1) # (batch_size, seq_len, 1)
        if pooling_type == "last":
            if padding_side == 'right':
                pooled_hidden_state = last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths-1]
            else:
                pooled_hidden_state = last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), -1]
        elif pooling_type == "average":
            pooled_hidden_state = (last_hidden_states * attention_mask_ext).sum(dim=1) / attention_mask_ext.sum(1)
        elif pooling_type == "max":
            pooled_hidden_state = (last_hidden_states * attention_mask_ext).max(dim=1)
        else:
            raise ValueError("The pooling method {} is not implemented!!".format(pooling_type))
        

        logits = self.class_head(pooled_hidden_state)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                self.config.problem_type = 'regression'
            if self.config.problem_type == 'regression':
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            elif self.config.problem_type == "ranking":
                pass
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            
            
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions
        )

