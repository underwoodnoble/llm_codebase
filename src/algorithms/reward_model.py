from typing import Dict, Any, Union, Tuple, List, Callable

import torch
from torch import nn
from transformers import PreTrainedTokenizer
from .base import BaseTrainer
from .utils import llm_tokenize
from ..arguments.data_arguments import RMDataArguments
from ..arguments.training_arguments import RMTrainingArguments
from ..models.utils import RewardModelOutput
from ..utils.general_utils import print_object_on_main_process


def rm_transform(data_args: RMDataArguments):
    def transform(example: Dict[str, Any]):
        if data_args.text_name in example:
            texts = example[data_args.text_name]
            scores = example[data_args.score_name]
        else:
            texts = [
                example[data_args.prompt_name] + example[data_args.chosen_name],
                example[data_args.prompt_name] + example[data_args.rejected_name]
            ]
            scores = [1, 0]
        return {
            "texts": texts,
            "scores": scores,
            "weight": example.get(data_args.weight_name, 1.0)
        }
    
    return transform


def rm_data_collator(tokenizer: PreTrainedTokenizer, args: RMTrainingArguments) -> Callable[[Dict[str, any]], Dict[str, torch.Tensor]]:
    def collator(examples: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        if args.debug_mode:
            print_object_on_main_process('data_example', examples[0])
        batch_size = len(examples)
        num_sample = max([len(example['texts']) for example in examples])
        all_texts = []
        all_scores = []
        weights = []
        for example in examples:
            # normalize scores to make sure that the minimum score is zero
            min_score = min(example['scores'])
            example['scores'] = [score - min_score for score in example['scores']]
            if len(example['texts']) < num_sample:
                example['texts'].extend([' ']*(num_sample - len(example['texts'])))
                # pading scores are -1
                example['scores'].extend([-1]*(num_sample - len(example['scores'])))

            all_texts.extend(example['texts'])
            all_scores.extend(example['scores'])
            weights.append(example['weight'])

        ret = llm_tokenize([""] * len(all_texts), all_texts, tokenizer, args)
        ret['scores'] = torch.tensor(all_scores).reshape(batch_size, -1)
        ret['weights'] = torch.tensor(weights)
        if args.debug_mode:
            print_object_on_main_process('scores', ret['scores'])
        return ret

    return collator


class RMTrainer(BaseTrainer):
    args: RMTrainingArguments
    
    @staticmethod
    def compute_ranking_loss(logits: torch.Tensor, scores: torch.Tensor, weights: torch.Tensor):
        """Compute ranking loss according to logits and scores
        logits: (batch_size, num_sample)
        scores: (batch_size, num_sample)
        """
        logits_diff = logits.unsqueeze(1) - logits.unsqueeze(2) # shape: (batch_size, num_sample, num_sample)
        score_mask_larger = (scores.unsqueeze(1) > scores.unsqueeze(2)) * 1.
        score_mask_smaller = (scores.unsqueeze(1) < scores.unsqueeze(2)) * 1.
        score_mask = score_mask_larger - score_mask_smaller
        
        # padding score is -1, other scores are greater or equal than 0.0. We use -0.5 to compare to avoid float precision issue.
        pad_mask = (scores > -0.5).unsqueeze(1) * 1. * (scores > -0.5).unsqueeze(2)

        total_mask = (score_mask_larger + score_mask_smaller) * pad_mask
        
        log_prob = nn.functional.logsigmoid(logits_diff * score_mask)
        loss = - ((log_prob * total_mask).sum(dim=(-1, -2)) * weights).sum()
        pairs = total_mask.sum()
        return loss / pairs if pairs > 0 else loss
    
    @staticmethod
    def compute_lm_loss(
        lm_logits: torch.FloatTensor,
        labels: torch.Tensor,
        scores: torch.FloatTensor,
        attention_mask: torch.Tensor,
        weights: torch.Tensor,
        score_thresh: float = 0.9,
        eps: float = 1e-7):
        """
        lm_logits: (batch_size*num_sample, seq_len, vocab_size)
        labels: (batch_size*num_sample, seq_len)
        attention_mask: (batch_size*num_sample, seq_len)
        scores: (batch_size, num_sample)
        """
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        b_n, _, vocab_size = lm_logits.shape
        
        token_ce_loss = nn.functional.cross_entropy(
            input=shift_logits.reshape(-1, vocab_size),
            target=shift_labels.reshape(-1),
            reduction='none'
        ).view(b_n, -1) # (b_n, seq_len-1)
        
        sentence_ce_loss = (token_ce_loss * attention_mask[:, 1:].float()).sum(dim=1)  / attention_mask[:, 1:].float().sum(dim=1)
        score_mask = (scores.reshape(-1) > score_thresh).float()
        return (sentence_ce_loss * score_mask * weights).sum() / (score_mask.sum() + eps)

    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor],
                    return_outputs: bool=False) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        device = model.device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        scores = inputs['scores'].to(device)
        weights = inputs['weights'].to(device)
        batch_size, num_sample = scores.shape
        
        outputs: RewardModelOutput = model(input_ids=input_ids.reshape(batch_size*num_sample, -1), attention_mask=attention_mask.reshape(batch_size*num_sample, -1))

        rm_logits = outputs.rm_logits # shape: (batch_size*num_sample, 1)
        rm_logits = rm_logits.view(batch_size, num_sample)
        total_loss = RMTrainer.compute_ranking_loss(rm_logits, scores, weights)
        if self.args.model_type != 'bert' and self.args.add_lm_loss:
            lm_logits = outputs.lm_logits # shape: (batch_size*num_sample, seq_length, vocab_size)
            lm_loss = RMTrainer.compute_lm_loss(lm_logits, inputs['labels'].reshape(batch_size*num_sample, -1),
                                                scores, attention_mask.reshape(batch_size*num_sample, -1), self.args.lm_score_thresh)
            total_loss += self.args.lm_loss_coeff * lm_loss
        
        return (total_loss, rm_logits) if return_outputs else total_loss

    def prediction_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor | Any], prediction_loss_only: bool, ignore_keys: List[str] | None = None) -> Tuple[torch.Tensor | None]:
        device = model.device
        labels = inputs['scores'].to(device)
        
        with torch.no_grad():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
        
        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)