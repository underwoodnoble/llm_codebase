from torch._tensor import Tensor
from torch.nn.modules import Module
from torch.utils.data import Dataset
from transformers import Trainer
from torch import nn
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import print_rank_0


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

def lm_loss(lm_logits: torch.Tensor, input_ids: torch.Tensor, scores: torch.Tensor, loss_mask: torch.Tensor, score_thresh=0.9, eps=1e-7):
    """Compute language model loss
    lm_logits: torch.Tensor shape: (batch_size*num_sample, seq_len, vocab_size)
    input_ids: torch.Tensor shape: (batch_size*num_sample, seq_len)
    loss_mask: torch.Tensor shape: (batch_size*num_sample, seq_len)
    scores: torch.Tensor shape: (batch_size, num_sample)
    """
    b_n, _, vocab_size = lm_logits.shape

    token_ce_loss = nn.functional.cross_entropy(
        input=lm_logits[:, :-1, :].reshape(-1, vocab_size),
        target=input_ids[:, 1:].reshape(-1),
        reduction="none"
    ).view(b_n, -1) # (b_n, seq_len-1)

    sentence_ce_loss = (token_ce_loss * loss_mask[:, 1:].float()).sum(dim=1) / loss_mask[:, 1:].float().sum(dim=1) # (b_n)
    score_mask = (scores.reshape(-1) > score_thresh).float()
    return (sentence_ce_loss * score_mask).sum() / (score_mask.sum() + eps)

class RewardModelTrainer(Trainer):
    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False):
        device = model.device
        input_ids = inputs['input_ids'].to(device) # shape: (batch_size, num_sample, seq_len)
        attention_mask = inputs['attention_mask'].to(device)
        scores = inputs['scores'].to(device)
        batch_size, num_sample = scores.shape

        outputs = model(input_ids=input_ids.reshape(batch_size*num_sample, -1), attention_mask=attention_mask.reshape(batch_size*num_sample, -1))
        if self.args.model_type == 'bert':
            logits: torch.Tensor = outputs.logits # shape: (batch_size*num_sample, 1)
            logits = logits.view(batch_size, num_sample)
            rl = ranking_loss(logits, scores)
            total_loss = rl
        else:
            logits: torch.Tensor = outputs['rm_logits']
            logits = logits.view(batch_size, num_sample)
            rl = ranking_loss(logits, scores)
            total_loss = rl

            if self.args.add_lm_loss:
                lm_logits: torch.Tensor = outputs['lm_logits'] # shape: (batch_size*num_sample, seq_length, vocab_size)
                lml = lm_loss(lm_logits, input_ids.view(batch_size*num_sample, -1), 
                              scores, attention_mask.view(batch_size * num_sample, -1), self.args.lm_score_thresh)

                total_loss += self.args.lm_loss_coeff * lml

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

        
class ContrastiveTrainer(Trainer):
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
        loss = -torch.log(probs).mean()
        return (loss, logits) if return_outputs else loss

        
class MultiObjectClassificationTrainer(Trainer):
    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False):
        device = model.device



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

        logits = model(input_ids=input_ids.view(batch_size*num_of_example, seq_len), attention_mask=attention_mask.view(batch_size*num_of_example, seq_len))['logits']
        logtis = F.log_softmax(logits, dim=-1) # (batch_size*num_of_example, seq_length, vocab_size)
        label_logit = self.gather_logits_labels(logtis, labels.view(batch_size*num_of_example, seq_len)) # (batch_size*num_of_example, seq_length)
        scores: torch.Tensor = self.get_score(label_logit, labels) # (batch_size*num_of_example)
        rrhf_loss = self.rrhf_loss(scores.view(batch_size, num_of_example), rewards)
        sft_loss = self.sft_loss(label_logit, rewards)
        loss = rrhf_loss + sft_loss
        return (loss, scores) if return_outputs else loss