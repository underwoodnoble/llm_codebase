from ..utils import print_rank_0
from transformers import Trainer, PreTrainedModel
from typing import Union, Optional, Dict, Callable, List, Tuple, Any
from torch import nn
import torch

class KTOTrainer(Trainer):
    @staticmethod
    def compute_lm_loglikelihood(logits: torch.Tensor, labels: torch.Tensor) -> torch.FloatTensor:
        batch_size, _, vocab_size = logits.shape

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        loglikelihood = loss_fct(shift_logits, shift_labels.to(shift_logits.device))
        return loglikelihood

    @staticmethod
    def compute_ref_kl(model: Union[PreTrainedModel, nn.Module],
                       kl_inputs: Dict[str, torch.Tensor],
                       batch_size: int) -> torch.FloatTensor:
        total_kl = torch.tensor(0.0, device=model.device)
        for i in range(0, len(kl_inputs), batch_size):
            batch_input_ids = kl_inputs['input_ids'][i:i+batch_size]
            batch_attention_mask = kl_inputs['attention_mask'][i:i+batch_size]
            batch_labels = kl_inputs['labels'][i:i+batch_size]
            with torch.no_grad():
                policy_outputs = model(
                    input_ids=batch_input_ids.to(model.device),
                    attention_mask=batch_attention_mask.to(model.device)
                )
                ref_outputs = model.ref_model(
                    input_ids=batch_input_ids.to(model.ref_model.device),
                    attention_mask=batch_attention_mask.to(model.ref_model.device)
                )
                policy_loglikelihood = KTOTrainer.compute_lm_loglikelihood(policy_outputs['logits'], batch_labels)
                ref_loglikelihood = KTOTrainer.compute_lm_loglikelihood(ref_outputs['logits'], batch_labels)
                total_kl += (policy_loglikelihood.exp()*(policy_loglikelihood.to(total_kl.device) - ref_loglikelihood.to(total_kl.device))).sum()
        return total_kl / len(kl_inputs)

    @staticmethod
    def compute_reward(model: Union[PreTrainedModel, nn.Module],
                       input_ids: torch.Tensor,
                       attention_mask: torch.Tensor,
                       labels: torch.Tensor) -> torch.FloatTensor:
        policy_outputs = model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device)
        )
        policy_loglikelihood = KTOTrainer.compute_lm_loglikelihood(policy_outputs['logits'], labels)
        with torch.no_grad():
            ref_outputs = model.ref_model(
                input_ids=input_ids.to(model.ref_model.device),
                attention_mask=attention_mask.to(model.ref_model.device)
            )
            ref_loglikelihood = KTOTrainer.compute_lm_loglikelihood(ref_outputs['logits'], labels)
        return policy_loglikelihood.sum(dim=-1) - ref_loglikelihood.sum(dim=-1)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Dict[str, torch.Tensor]],
        return_outputs=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if self.args.debug_mode:
            print_rank_0(f"check inputs: {inputs}")
        
        device = model.device
        target = inputs['target']
        kl_inputs = inputs['kl']
        input_ids = target['input_ids'].to(device)
        attention_mask = target['attention_mask'].to(device)
        labels = target['labels'].to(device)
        scores = target['scores'].to(device)

        rewards = self.args.kto_beta * KTOTrainer.compute_reward(model, input_ids, attention_mask, labels)
        if self.args.debug_mode:
            print_rank_0(rewards)
        ref_kl = KTOTrainer.compute_ref_kl(model, kl_inputs, self.args.eval_batch_size)
        ref_kl = self.accelerator.gather(ref_kl).mean().clamp(min=0)
        
        weights: torch.FloatTensor = (scores > 0).float() * self.args.kto_desirable_weight + (scores < 0).float() * self.args.kto_undesirable_weight
        v = nn.functional.sigmoid((rewards.to(device) - ref_kl.to(device)) * scores)
        loss = (weights * (1 - v)).mean()
        if self.args.debug_mode:
            print_rank_0(loss)

        return (loss, rewards) if return_outputs else loss