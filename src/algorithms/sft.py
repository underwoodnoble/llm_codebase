from typing import Dict, Any

import torch

from ..arguments import SFTDataArguments, SFTTrainingArguments
from .base import BaseTrainer
from .utils import IGNORE_INDEX
from ..utils.general_utils import print_object_on_main_process


def sft_transform(data_args: SFTDataArguments):
    def transform(example: Dict[str, Any]):
        return {
            "prompt": example[data_args.prompt_name],
            "answer": example[data_args.answer_name],
            "weight": example.get(data_args.weight_name, 1.0)
        }
    return transform


class SFTTrainer(BaseTrainer):
    args: SFTTrainingArguments

    @staticmethod
    def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor = None, gather: bool = True) -> torch.Tensor:
        """
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)
        """
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:] if labels is not None else None

        return BaseTrainer.logprobs_from_logits(shift_logits, shift_labels, gather)


    def compute_loss(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False):
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )
        logprob = self.logprobs_from_logits(model_outputs.logits, labels=inputs['labels'])
        shift_labels = inputs['labels'][:, 1:]
        mask = torch.not_equal(shift_labels, IGNORE_INDEX)
        loss = -(logprob * mask).sum(dim=-1) / mask.sum(dim=-1)

        if self.args.kl_coef is not None:
            with torch.no_grad():
                ref_model_outputs = self.ref_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
            # calculate logprob
            if self.args.kl_penalty_mode == 'full':
                logprob = self.logprobs_from_logits(model_outputs.logits, gather=False) # (batch_size, seq_len-1, vocab_size)
                ref_logprob = self.logprobs_from_logits(ref_model_outputs.logits, gather=False) # (batch_size, seq_len-1, vocab_size)
            else:
                ref_logprob = self.logprobs_from_logits(ref_model_outputs.logits, labels=inputs['labels']) # (batch_size, seq_len-1)
            kl_divergence = self.compute_kl_divergence(logprob, ref_logprob, kl_penalty=self.args.kl_penalty_mode)
            kl_divergence = (kl_divergence * mask).sum(-1) / mask.sum(-1)

            loss += self.kl_contorller.value * kl_divergence

            # log kl and kl coef
            train_eval = 'train' if model.training else 'eval'
            self.store_metrics({"kl": kl_divergence.sum()}, train_eval)
            self.store_metrics({"kl_coef": self.kl_contorller.value}, train_eval)
            self.kl_step_buffer.append(kl_divergence)

        loss = (inputs['weights'] * loss).mean()
        return (loss, model_outputs.logits) if return_outputs else loss
