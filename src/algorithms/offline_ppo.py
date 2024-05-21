from typing import Dict, Any

import torch
from torch import nn

from .base import BaseTrainer
from .utils import IGNORE_INDEX
from ..arguments import OfflinePPOTrainingArguments, OfflinePPODataArguments


def offline_ppo_transform(data_args: OfflinePPODataArguments):
    def transform(example: Dict[str, Any]):
        return {
            "prompt": example[data_args.prompt_name],
            "answer": example[data_args.answer_name],
            "weight": example.get(data_args.weight_name, 1.0),
            "advantage": example.get(data_args.reward_name, 1.0)
        }
    
    return transform


class OfflinePPOTrainer(BaseTrainer):
    """This class implement the algorithm in the https://arxiv.org/pdf/2305.14718
    """
    args: OfflinePPOTrainingArguments

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

        shift_labels = inputs['labels'][:, 1:]
        mask = torch.not_equal(shift_labels, IGNORE_INDEX)
        if self.args.kl_penalty_mode == 'full':
            logprobs = self.logprobs_from_logits(model_outputs.logits, gather=False) # (batch_size, seq_len-1, vocab_size)
            ref_logprobs = self.logprobs_from_logits(ref_model_outputs.logits, gather=False)
            importance_weight = (torch.gather(logprobs, 2, (shift_labels * mask).unsqueeze(2)).squeeze(-1) - \
                torch.gather(ref_logprobs, 2, (shift_labels * mask).unsqueeze(2)).squeeze(-1)).exp()
        else:
            logprobs = self.logprobs_from_logits(model_outputs.logits, labels=inputs['labels']) # (batch_size, seq_len-1)
            ref_logprobs = self.logprobs_from_logits(ref_model_outputs.logits, labels=inputs['labels']) # (batch_size, seq_len-1)           
            importance_ratio = (logprobs - ref_logprobs).exp()

        kl_divergence = self.compute_kl_divergence(logprobs, ref_logprobs, kl_penalty=self.args.kl_penalty_mode) # (batch_size, seq_len-1)
        cliped_importance_weight = torch.clip(importance_ratio, min=-self.args.clip_range, max=self.args.clip_range)
        loss = -(inputs['advantage'] - kl_divergence) * mask * cliped_importance_weight
        loss = loss.sum(-1)

        return (loss, model_outputs.logits) if return_outputs else loss
