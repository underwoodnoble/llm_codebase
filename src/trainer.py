from transformers import Trainer
from torch import nn
import torch
from typing import Dict, List, Optional


def ranking_loss(logits: torch.Tensor, scores: torch.Tensor):
    """Compute ranking loss according to logits and scores
    logits: torch.Tensor  shape: (batch_size, pairs)
    scores: (batch_size, pairs)
    """
    logits_diff = logits.unsqueeze(1) - logits.unsqueeze(2) # shape: (batch_size, pairs, pairs)
    score_mask_larger = (scores.unsqueeze(1) > scores.unsqueeze(2)) * 1.
    score_mask_smaller = (scores.unsqueeze(1) < scores.unsqueeze(2)) * 1.
    score_mask = score_mask_larger - score_mask_smaller
    pad_mask = (scores >= 0).unsqueeze(1) * 1. * (scores >= 0).unsqueeze(2)

    total_mask = (score_mask_larger + score_mask_smaller) * pad_mask

    log_porb = nn.functional.logsigmoid(logits_diff * score_mask * pad_mask)
    total_loss = - (log_porb * total_mask).sum()
    total_pairs = total_mask.sum()
    return total_loss / total_pairs if total_pairs > 0 else total_loss

class RewardModelTrainer(Trainer):
    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False):
        device = model.device
        input_ids = inputs['input_ids'].to(device) # shape: (batch_size, num_sample, seq_len)
        attention_mask = inputs['attention_mask'].to(device)
        scores = inputs['scores'].to(device)
        batch_size, num_sample = scores.shape

        outputs = model(input_ids=input_ids.reshape(batch_size*num_sample, -1), attention_mask=attention_mask.reshape(batch_size*num_sample, -1))
        logits: torch.Tensor = outputs.logits # shape: (batch_size*num_sample, 1)
        logits = logits.view(batch_size, num_sample)

        rl = ranking_loss(logits, scores)
        if model.config.model_type == 'bert':
            total_loss = rl
        else:
            pass

        return (total_loss, logits) if return_outputs else total_loss

    def prediction_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor], prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None):
        device = model.device
        labels = inputs['scores'].to(device)

        with torch.no_grad():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
        
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, logits, labels)