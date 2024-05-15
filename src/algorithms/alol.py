from typing import Dict, Any

import torch
from torch import nn

from .base import BaseTrainer
from .utils import IGNORE_INDEX
from ..arguments import ALOLTrainingArguments, ALOLDataArguments


def alol_transform(data_args: ALOLDataArguments):
    def transform(example: Dict[str, Any]):
        return {
            "prompt": example[data_args.prompt_name],
            "answer": example[data_args.answer_name],
            "weight": example.get(data_args.weight_name, 1.0),
            "advantage": example.get(data_args.reward_name, 1.0)
        }
    
    return transform


class ALOLTrainer(BaseTrainer):
    """This class implement the algorithm in the https://arxiv.org/pdf/2305.14718
    """
    args: ALOLTrainingArguments

    @staticmethod
    def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor = None, gather: bool = True) -> torch.Tensor:
        """
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)
        """
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:] if labels is not None else None

        return BaseTrainer.logprobs_from_logits(shift_logits, shift_labels, gather)

        
    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False):
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['labels']
        )

        ref_model_outputs = self.compute_ref_model_outputs(inputs['input_ids'], inputs['attention_mask'])
        if self.args.kl_coef is None:
            logprobs = self.logprobs_from_logits(model_outputs['logits'], inputs['labels'])
            ref_logprobs = self.logprobs_from_logits(ref_model_outputs['logits'], inputs['labels'])
            importance_weight = logprobs - ref_logprobs # (batch_size, seq_len-1)
            if not self.args.token_level:
                importance_weight = importance_weight.sum(-1) # (batch_size)
            cliped_importance_weight = torch.min(importance_weight, torch.clip(torch.exp(importance_weight), min=-self.args.clip_range, max=self.args.clip_range))
            loss = -inputs['advantage'] * cliped_importance_weight
            loss = loss if not self.args.token_level else loss.sum(-1)
        else:
            # calculate logprob
            if self.args.kl_penalty_mode == 'full':
                logprob = self.logprobs_from_logits(model_outputs.logits, gather=False) # (batch_size, seq_len-1, vocab_size)
                ref_logprob = self.logprobs_from_logits(ref_model_outputs.logits, gather=False) # (batch_size, seq_len-1, vocab_size)
            else:
                logprob = self.logprobs_from_logits(model_outputs.logits, labels=inputs['labels']) # (batch_size, seq_len-1)
                ref_logprob = self.logprobs_from_logits(ref_model_outputs.logits, labels=inputs['labels']) # (batch_size, seq_len-1)
            
            kl_divergence = self.compute_kl_divergence(logprob, ref_logprob, kl_penalty=self.args.kl_penalty_mode)
            # mask
            shift_labels = inputs['labels'][:, 1:]
            mask = torch.not_equal(shift_labels, IGNORE_INDEX)
            kl_divergence = (kl_divergence * mask).sum() / mask.sum()

            # logprobs for labels
            logprob = self.logprobs_from_logits(model_outputs.logits, labels=inputs['labels']) # (batch_size, sqe_len-1)
            loss = -inputs['advantage'] * logprob.sum(dim=-1) + kl_divergence

        return (loss, model_outputs.logits) if return_outputs else loss
