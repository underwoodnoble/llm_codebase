from ...BaseModel.Baichuan2 import BaichuanConfig, BaichuanModel, BaichuanPreTrainedModel
import torch
from torch import nn
from typing import Optional, List, Dict


class BaichuanClassiferConfig(BaichuanConfig):
    def __init__(self, cls_info=None, **kwargs):
        super().__init__()
        self._update_cls_info(cls_info)
    
    def _update_cls_info(self, cls_info: Optional[dict]=None):
        self.cls_info: Optional[dict] = cls_info
        self.cls_label_nums: Optional[dict] = {key: len(item['labels']) for key, item in cls_info.items()} if cls_info is not None else None
        self.cls_labels: Optional[dict] = {key: item['labels'] for key, item in cls_info.items()} if cls_info is not None else None
        self.cls_coeffs: Optional[dict] = {key: item['coeff'] for key, item in cls_info.items()} if cls_info is not None else None

class BaichuanClassifer(BaichuanPreTrainedModel):
    config_class = BaichuanConfig
    def __init__(self, config: BaichuanConfig, baichuan_model_path=None):
        super().__init__(config)
        if baichuan_model_path is None:
            self.model = BaichuanModel(config)
        else:
            self.model = BaichuanModel.from_pretrained(baichuan_model_path)

        self.cls_heads = nn.ModuleDict(
            {key: nn.Linear(config.hidden_size, value) for key, value in self.config.cls_label_nums.items()}
        )
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def floating_point_ops(self, inputs):
        return 0

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels_dict: Optional[Dict[str, torch.LongTensor]] = None,
        pooling_type: str = 'last',
        padding_side: str = 'right',
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model.forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = transformer_outputs[0]

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
        if pooling_type in ['last', 'eos']:
            offset = 1 if pooling_type == 'eos' else 2
            if padding_side == 'right':
                pooled_hidden_state = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths - offset]
            else:
                pooled_hidden_state = hidden_states[torch.arange(batch_size, device=hidden_states.device), -offset]
        elif pooling_type == "average":
            pooled_hidden_state = (hidden_states * attention_mask_ext).sum(dim=1) / attention_mask_ext.sum(dim=1)
        elif pooling_type == "max":
            pooled_hidden_state = (hidden_states * attention_mask_ext).max(dim=1)[0]
        else:
            raise ValueError("The pooling method {} is not implemented!!".format(pooling_type))

        output_logits_dict = {key: head_layer(pooled_hidden_state) for key, head_layer in self.cls_heads.items()}

        if labels_dict is not None:
            loss_fct = nn.CrossEntropyLoss()
            cls_loss_list = [
                self.config.cls_coeffs[key] * loss_fct(
                    output_logits_dict[key].view(-1, self.config.cls_label_nums[key]),
                    labels_dict[key].view(-1)
                ).mean().unsqueeze(0)
                for key in labels_dict
            ]
            cls_loss = torch.cat(cls_loss_list).sum()
        else:
            cls_loss = None
        
        return {
            "hidden_states": transformer_outputs[0],
            "output_logits_dict": output_logits_dict,
            "cls_loss": cls_loss,
            "cls_embeddings": pooled_hidden_state
        }

    def predict(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pooling_type: str = 'last',
        padding_side: str = 'right',
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        with torch.no_grad():
            output_label_logits = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pooling_type=pooling_type,
                padding_side=padding_side,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )['output_logits_dict']

            output_labels = {key: [self.config.cls_labels[key][i] for i in value.argmax(1)] 
                            for key, value in output_label_logits.items()}
        return output_labels