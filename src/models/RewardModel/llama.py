from transformers import LlamaModel, LlamaPreTrainedModel, LlamaConfig
from torch import nn
import torch
from typing import Optional, List, Union, Tuple
from ..utils import RewardModelOutput


class LlamaRewardModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.rm_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pooling_type: str = "last",
        padding_side: str = "right",
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, RewardModelOutput]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        last_hidden_state = transformer_outputs.last_hidden_state

        lm_logits = self.lm_head(last_hidden_state)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]
        
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(dim=-1)).to(last_hidden_state.device)
            else:
                sequence_lengths = 1
        
        if attention_mask is None:
            attention_mask = torch.ne(input_ids, self.config.pad_token_id).float()
        
        attention_mask_ext = attention_mask.unsqueeze(-1)
        if pooling_type in ["last", "eos"]:
            offset = 1 if pooling_type == "eos" else 2
            if padding_side == 'right':
                pooled_hidden_state = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths - offset]
            else:
                pooled_hidden_state = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), -offset]
        elif pooling_type == "average":
            pooled_hidden_state = (last_hidden_state * attention_mask_ext).sum(dim=1) / attention_mask_ext.sum(dim=1)
        elif pooling_type == 'max':
            pooled_hidden_state = (last_hidden_state * attention_mask_ext).max(dim=1)[0]
        else:
            raise ValueError(f"The pooling method {pooling_type} is not implemented.")
        
        pooled_logits = self.rm_head(pooled_hidden_state)

        ret = {
            "last_hidden_state": last_hidden_state,
            "hidden_states": transformer_outputs.hidden_states if output_hidden_states else None,
            "attentions": transformer_outputs.attentions if output_attentions else None,
            "lm_logits": lm_logits,
            "rm_logits": pooled_logits,
            "rm_embeddings": pooled_hidden_state
        }

        if not return_dict:
            return tuple(v for v in ret.values() if v is not None)
        return RewardModelOutput(**ret)