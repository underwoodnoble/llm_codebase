from typing import Dict, Any

import torch

from ..arguments import SFTDataArguments, SFTTrainingArguments
from .base import BaseLLMTrainer
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


class SFTTrainer(BaseLLMTrainer):
    args: SFTTrainingArguments

    def compute_loss(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False):
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )
        
        # Calculate lm loss
        shift_logits = model_outputs.logits[:, :-1, :]
        shift_labels = inputs['labels'][:, 1:]
        logprob = self.logprobs_from_logits(shift_logits, labels=shift_labels)
        mask = torch.not_equal(shift_labels, IGNORE_INDEX)
        loss = -(logprob * mask).sum(dim=-1) / mask.sum(dim=-1) # (batch_size)
        loss = (loss * inputs['weights']).mean()

        if self.args.kl_coef is not None:
            ref_model_outputs = self.compute_ref_model_outputs(inputs['input_ids'], inputs['attention_mask'])
            kl_divergence = self.compute_kl_divergence(model_outputs.logits, ref_model_outputs.logits, inputs['labels'])
            if self.args.token_level:
                kl_divergence = (kl_divergence * mask).sum(-1) / mask.sum(-1)
            kl_divergence = kl_divergence.mean()
            loss += self.kl_contorller.value * kl_divergence

            # log kl and kl coef
            train_eval = 'train' if model.training else 'eval'
            self.store_metrics({"kl": kl_divergence.mean()}, train_eval)
            self.store_metrics({"kl_coef": self.kl_contorller.value}, train_eval)
            self.kl_step_buffer.append(kl_divergence.mean().item())

        return (loss, model_outputs.logits) if return_outputs else loss
