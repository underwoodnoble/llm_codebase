import pyarrow as pa
from typing import Dict, Any
from pydantic import BaseModel

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
                "reward": example.get(data_args.reward_name, 1.0),
                "data_type": 'rl' if 'reward' in example else 'lm'
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
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']
        rewards = inputs['rewards']
        weights = inputs['weights']
        lm_mask = inputs['lm_mask']
        
        # Calculate model outputs
        model_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # Calculate ref model outputs
        ref_model_outputs = self.compute_ref_model_outputs(input_ids, attention_mask=attention_mask)

        shift_labels = labels[:, 1:]
        label_mask = torch.not_equal(shift_labels, IGNORE_INDEX)
        
        if self.args.kl_penalty_mode == 'full':
            logprobs = self.logprobs_from_logits(model_outputs.logits, gather=False) # (batch_size, seq_len-1, vocab_size)
            ref_logprobs = self.logprobs_from_logits(ref_model_outputs.logits, gather=False)
            target_logprobs = torch.gather(logprobs, 2, (shift_labels * label_mask).unsqueeze(2)).squeeze(-1)
            ref_target_logprobs = torch.gather(ref_logprobs, 2, (shift_labels * label_mask).unsqueeze(2)).squeeze(-1)
            importance_ratio = (target_logprobs - ref_target_logprobs).exp()
            lm_loss = -target_logprobs.mean(-1)
        else:
            logprobs = self.logprobs_from_logits(model_outputs.logits, labels=labels) # (batch_size, seq_len-1)
            ref_logprobs = self.logprobs_from_logits(ref_model_outputs.logits, labels=labels) # (batch_size, seq_len-1)           
            importance_ratio = (logprobs - ref_logprobs).exp()
            lm_loss = -logprobs.mean(-1)

        kl_divergence = self.compute_kl_divergence(logprobs, ref_logprobs, kl_penalty=self.args.kl_penalty_mode) # (batch_size, seq_len-1)
        cliped_importance_weight = torch.clip(importance_ratio, min=-self.args.clip_range, max=self.args.clip_range)
        rl_loss = -(rewards - self.args.kl_coef * kl_divergence) * label_mask * cliped_importance_weight
        rl_loss = (rl_loss.mean(-1) * (1 - lm_mask) * weights).sum() / (1 - lm_mask).sum()
        lm_loss = (lm_loss * lm_mask * weights).sum() / (1 - lm_mask).sum()
        loss = rl_loss + self.args.lm_coef * lm_loss

        return (loss, model_outputs.logits) if return_outputs else loss
