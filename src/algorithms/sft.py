from ..arguments import SFTDataArguments
import torch
from .base import BaseTrainer
from typing import Dict
from transformers.trainer_pt_utils import LabelSmoother


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def sft_transform(data_args: SFTDataArguments):
    def transform(example: Dict):
        if data_args.prompt_name != 'prompt' or data_args.answer_name != 'prompt' or data_args.weight_name != 'weight':
            return {
                "prompt": example[data_args.prompt_name],
                "answer": example[data_args.answer_name],
                "weight": example.get(data_args.weight_name, 1.0)
            }

        return example
    return transform


class SFTTrainer(BaseTrainer):
    def _is_create_ref_model(self) -> bool:
        return self.args.kl_coeff is not None


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
            label=inputs['labels']
        )

        if self.args.kl_coeff is not None:
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
                logprob = self.logprobs_from_logits(model_outputs.logits, labels=inputs['labels']) # (batch_size, seq_len-1)
                ref_logprob = self.logprobs_from_logits(ref_model_outputs.logits, labels=inputs['labels']) # (batch_size, seq_len-1)
            
            kl_divergence = self.compute_kl_divergence(logprob, ref_logprob, kl_penalty=self.args.kl_penalty_mode)
            # mask
            shift_labels = inputs['labels'][:, 1:]
            mask = torch.not_equal(shift_labels, IGNORE_TOKEN_ID)
            kl_divergence = (kl_divergence * mask).sum() / mask.sum()
            loss = model_outputs.loss + self.args.kl_coeff * kl_divergence
        
        else:
            loss = model_outputs.loss

        return (loss, model_outputs['logits']) if return_outputs else loss