from transformers import Trainer
import torch
from torch import nn
from typing import Dict
import torch.nn.functional as F

class RRHFTrainer(Trainer):
    def gather_logits_labels(self, logits: torch.Tensor, labels: torch.Tensor):
        mask = (labels != self.args.ignore_token_id)
        new_logits = logits.clone()
        labels[labels == self.args.ignore_token_id] = 0
        output = torch.gather(new_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        output = output * mask
        return output


    def get_score(self, label_logit: torch.Tensor, labels: torch.Tensor):
        """
        label_logit: (batch_size*num_of_sample, seq_len)
        labels: (batch_size*num_of_sample, seq_len)
        """
        mask: torch.Tensor = (labels != self.args.ignore_token_id).float()
        length = mask.sum(-1)
        scores = label_logit.sum(-1) / (length ** self.args.length_penalty)
        return scores
        
    def rrhf_loss(self, scores: torch.Tensor, rw_scores: torch.Tensor):
        diff = scores.unsqueeze(-1) - scores.unsqueeze(1) # (batch_size, num_of_example, num_of_example). the item in the poisiton (i, j) means how much the ith score greater than jth reward.
        rw_diff = scores.unsqueeze(-1) - rw_scores.unsqueeze(1)
        aval = torch.bitwise_and(diff < 0, rw_diff > 0)
        return -diff[aval].sum()

    def sft_loss(self, label_logit: torch.Tensor, rw_scores: torch.Tensor):
        max_idx = torch.argmax(rw_scores, dim=-1)
        return -label_logit[:, max_idx].mean()

    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False):
        """
        inputs:
            input_ids: batch_size, num_of_sample, seq_length
            
        """
        device = model.device
        input_ids = inputs['input_ids'].to(device) # (batch_size, num_of_example, seq_len)
        attention_mask = inputs['attention_mask'].to(device) # (batch_size, num_of_example, seq_len)
        labels = inputs['labels'].to(device) # (batch_size, num_of_example, seq_len)
        rewards = inputs['scores'].to(device) # (batch_size, num_of_example)

        batch_size, num_of_example, seq_len = input_ids.shape

        logits = model(input_ids=input_ids.view(batch_size*num_of_example, seq_len), attention_mask=attention_mask.view(batch_size*num_of_example, seq_len))['logits'] # (batch_size*num_of_example, seq_len, vocab_size)
        logits = logits[:, :seq_len-1, :]
        labels = labels[:, :, 1:]
        logtis = F.log_softmax(logits, dim=-1) # (batch_size*num_of_example, seq_length-1, vocab_size)
        label_logit = self.gather_logits_labels(logtis, labels.view(batch_size*num_of_example, seq_len-1)) # (batch_size*num_of_example, seq_length-1)
        scores: torch.Tensor = self.get_score(label_logit, labels.view(batch_size*num_of_example, seq_len-1)) # (batch_size*num_of_example)
        rrhf_loss = self.rrhf_loss(scores.view(batch_size, num_of_example), rewards)
        sft_loss = self.sft_loss(label_logit, rewards)
        loss = self.args.rrhf_weight * rrhf_loss + sft_loss
        return (loss, scores) if return_outputs else loss