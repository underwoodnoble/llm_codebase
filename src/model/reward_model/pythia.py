from transformers import GPTNeoXForCausalLM
from typing import Optional, Tuple
import torch
from torch import nn

class PythiaRewardModel(GPTNeoXForCausalLM):
    def __init__(self, config) -> None:
        super().__init__(config)
        # self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.reward_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        pooling_type: str = "average",
        padding_side: str = "right",
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_hidden_states = True

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        all_layer_embeddings = outputs[2]

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)



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
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(dim=-1)).to(hidden_states.device)
            else:
                sequence_lengths = -1

        if attention_mask is None:
            attention_mask = torch.ne(input_ids, self.config.pad_token_id).float()

        attention_mask_ext = attention_mask.unsqueeze(-1)
        if pooling_type in ["last", "eos"]:
            offset = 1 if pooling_type == "eos" else 2
            if padding_side == "right":
                pooled_hidden_state = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths - offset]
            else:
                pooled_hidden_state = hidden_states[torch.arange(batch_size, device=hidden_states.device), - offset]
        elif pooling_type == "average":
            pooled_hidden_state = (hidden_states * attention_mask_ext).sum(dim=1) / attention_mask_ext.sum(dim=1)
        elif pooling_type == "max":
            pooled_hidden_state = (hidden_states * attention_mask_ext).max(dim=1)[0]
        else:
            raise ValueError("The pooling method {} is not implemented!!".format(pooling_type))

        # if not return_dict:
        #     output = (lm_logits,) + outputs[1:]
        #     return ((lm_loss,) + output) if lm_loss is not None else output

        pooled_logits = self.reward_head(pooled_hidden_state)

        return {
            "lm_logits": lm_logits,
            "rm_logits": pooled_logits,
            "hidden_states": hidden_states,
            "rm_embeddings": pooled_hidden_state,
            "all_embeddings": all_layer_embeddings
        }
