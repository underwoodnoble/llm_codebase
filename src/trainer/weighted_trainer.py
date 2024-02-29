from transformers import Trainer
import torch
from torch import nn
from typing import Dict

class WeightedTrainer(Trainer):
    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False):
        device = model.device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = inputs['labels'].to(device)
        scores = inputs['scores'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits: torch.Tensor = outputs['logits']
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        all_probs = nn.functional.softmax(shift_logits, dim=-1)
        ix, iy = torch.meshgrid(torch.arange(shift_labels.shape[0]), torch.arange(shift_labels.shape[1]), indexing='ij')
        probs = all_probs[ix, iy, shift_labels] # (batch_size, seq_len - 1)
        positive_mask = scores.view(shift_labels.shape[0], -1) > 0
        negative_mask = scores.view(shift_labels.shape[0], -1) < 0
        positive_probs = probs * positive_mask
        negative_probs = (1 - probs) * negative_mask
        probs = positive_probs + negative_probs
        loss = -(torch.log(probs)*scores.abs().view(shift_labels.shape[0], -1)).mean()
        return (loss, logits) if return_outputs else loss