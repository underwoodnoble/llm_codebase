from typing import Dict, Any, List

import torch
from torch import nn
from transformers import PreTrainedTokenizer

from .base import BaseLLMTrainer
from .utils import IGNORE_INDEX, llm_tokenize
from ..arguments import OfflinePPOTrainingArguments, OfflinePPODataArguments
from ..utils.general_utils import print_object_on_main_process


def offline_ppo_transform(data_args: OfflinePPODataArguments):
    def transform(example: Dict[str, Any]):
        return {
                "prompt": example[data_args.prompt_name],
                "answer": example[data_args.answer_name],
                "weight": example.get(data_args.weight_name, 1.0),
                "reward": example.get(data_args.reward_name, 0.0),
                "data_type": 'rl' if data_args.reward_name in example else 'lm'
            }
    
    return transform


def offline_ppo_data_collator(
    tokenizer: PreTrainedTokenizer,
    args: OfflinePPOTrainingArguments):
    def collator(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = []
        texts = []
        rewards = []
        weights = []
        lm_mask = []
        for example in examples:
            prompts.append(example['prompt'])
            texts.append(example['prompt'] + example['answer'])
            rewards.append(example['reward'])
            weights.append(example['weight'])
            lm_mask.append(int(example['data_type'] == 'lm'))

        ret = llm_tokenize(prompts, texts, tokenizer, args)
        ret['rewards'] = torch.tensor(rewards)
        if args.reward_normalize:
            mean = ret['rewards'].mean()
            std = ret['rewards'].std()
            ret['rewards'] = (ret['rewards'] - mean) / max(std, 1e-6)
        ret['weights'] = torch.tensor(weights)
        ret['lm_mask'] = torch.tensor(lm_mask)
        return ret

    
    return collator


class OfflinePPOTrainer(BaseLLMTrainer):
    """This class implement the algorithm in the https://arxiv.org/pdf/2305.14718
    """
    args: OfflinePPOTrainingArguments


    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False):
        rewards = inputs['rewards']
        weights = inputs['weights']
        lm_mask = inputs['lm_mask']
        
        # Calculate model outputs
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )

        # Calculate ref model outputs
        ref_model_outputs = self.compute_ref_model_outputs(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

        shift_labels = inputs['labels'][:, 1:]
        shift_logits = model_outputs.logits[:, :-1, :]
        shift_ref_logits = ref_model_outputs.logits[:, :-1, :]
        mask = torch.ne(shift_labels, IGNORE_INDEX)
        
        # kl_divergence: (batch_size, seq_len-1) if token_level else (batch_size)
        kl_divergence = self.compute_kl_divergence(model_outputs.logits.detach(), ref_model_outputs.logits, inputs['labels'])

        
        advantage = rewards - self.args.kl_coef * kl_divergence # (batch_size, seq_len-1) if token_level else (batch_size)
        logprobs = self.logprobs_from_logits(shift_logits, shift_labels) # (batch_size ,seq_len-1)
        ref_logprobs = self.logprobs_from_logits(shift_ref_logits, shift_labels) # (batch_size ,seq_len-1)
        
        # Calculate rl loss
        if self.args.token_level:
            importance_ratio = (logprobs - ref_logprobs).exp() # (batch_size, seq_len-1)
            cliped_importance_ratio = torch.clip(importance_ratio, 1 - self.args.clip_range, 1 + self.args.clip_range) # (batch_size)
            rl_loss = - torch.minimum(advantage * cliped_importance_ratio, advantage * importance_ratio) # (batch_size, seq_len-1)
            rl_loss = (rl_loss * mask).sum(-1) / max(mask.sum(-1), 1) # (batch_size)
        else:
            importance_ratio = ((logprobs  - ref_logprobs)*mask).sum(-1) / max(1, mask.sum(-1)) # (batch_size)
            importance_ratio = importance_ratio.exp()
            cliped_importance_ratio = torch.clip(importance_ratio, 1 - self.args.clip_range, 1 + self.args.clip_range) # (batch_size)
            rl_loss = -torch.minimum(advantage * cliped_importance_ratio, advantage * importance_ratio) # (batch_size)
        rl_loss = (rl_loss * (1 - lm_mask) * weights).sum() / max((1 - lm_mask).sum(), 1)

        # Calculate lm loss
        lm_loss = -(logprobs * mask).sum(-1) / max(mask.sum(-1), 1)
        lm_loss = (lm_loss * lm_mask * weights).sum() / max(lm_mask.sum(), 1)

        loss = rl_loss + self.args.lm_coef * lm_loss
        # log
        if self.args.debug_mode:
            print_object_on_main_process('model_output', model_outputs.logits)
            print_object_on_main_process('model_output_max_logtis', model_outputs.logits.max())
            print_object_on_main_process('kl_divergence', kl_divergence)
            print_object_on_main_process('importance_ratio', importance_ratio)
            print_object_on_main_process('cliped_importance_ratio', cliped_importance_ratio)
            print_object_on_main_process('rl_loss', rl_loss)
            print_object_on_main_process('lm_loss', lm_loss)
        train_eval = 'train' if model.training else 'eval'
        self.store_metrics({"kl": kl_divergence.mean().item()}, train_eval)
        self.store_metrics({"kl_coef": self.kl_contorller.value}, train_eval)
        positive_mask = (rewards > 0) * (1 - lm_mask)
        negative_mask = (rewards < 0) * (1 - lm_mask)
        
        if self.args.debug_mode:
            print(f"rewards: {rewards}  ", f"lm_mask: {lm_mask}  ", f"postive_mask: {positive_mask}  ", f"negative_mask: {negative_mask}")

        if positive_mask.any():
            self.store_metrics({"kl_positive": (kl_divergence * positive_mask).sum() / positive_mask.sum()})
        if negative_mask.any():
            self.store_metrics({"kl_negative": (kl_divergence * negative_mask).sum() / negative_mask.sum()})
        if (1 - lm_mask).any():
            self.store_metrics({"rl_loss": rl_loss})
        if lm_mask.any():
            self.store_metrics({"kl_lm": (kl_divergence * lm_mask).sum() / lm_mask.sum()})
            self.store_metrics({"lm_loss": lm_loss})
        
        # store current kl
        self.kl_step_buffer.append(kl_divergence.mean().item())

        return (loss, model_outputs.logits) if return_outputs else loss
