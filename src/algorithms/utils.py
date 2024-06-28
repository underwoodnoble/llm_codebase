from typing import List, Dict
from copy import deepcopy

import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother

from ..arguments.training_arguments import BaseTrainingArguments


IGNORE_INDEX = LabelSmoother.ignore_index


class FixedKLController:
    """Fixed KL controller"""


    def __init__(self, kl_coef):
        self.value = kl_coef


    def update(self, current):
        pass

    
class AdaptiveKLController:
    """
    Adaptive KL controller
    """

    
    def __init__(self, init_kl_coef, target) -> None:
        """
        init_kl_coef: base kl coefficient
        """
        self.value = init_kl_coef
        self.target = target
    
    def update(self, current):
        proprotional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        mult = 1 + proprotional_error * 0.1
        self.value *= mult
        
        
def llm_tokenize(prompts: List[str], texts: List[str], tokenizer: PreTrainedTokenizer, args: BaseTrainingArguments) -> Dict[str, torch.Tensor]:
    input_ids = []
    labels = []
    for prompt, text in zip(prompts, texts):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        response_start_idx = len(prompt_ids)
        if prompt_ids != text_ids[:response_start_idx]:
            response_start_idx -= 1
        if args.add_special_tokens:
            response_start_idx += 1
            text_ids = [tokenizer.bos_token_id] + text_ids + [tokenizer.eos_token_id]
        label = deepcopy(text_ids)
        if args.only_predict_answer:
            label[:response_start_idx] = [IGNORE_INDEX] * response_start_idx
        if len(text_ids) > args.model_max_length:
            text_ids = text_ids[-tokenizer.model_max_length:]
            label = label[-tokenizer.model_max_length:]

        input_ids.append(torch.tensor(text_ids))
        labels.append(torch.tensor(label))
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    if args.pad_labels_with_ignore:
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)

    if args.debug_mode: # If debug_model is True, then pad the sequence into the max token length.
        input_ids = F.pad(input_ids, (0, tokenizer.model_max_length - input_ids.shape[0]), mode='constant', value=tokenizer.pad_token_id)
        labels = F.pad(labels, (0, tokenizer.model_max_length - input_ids.shape[0]), mode='constant', value=IGNORE_INDEX
                    if args.pad_labels_with_ignore else tokenizer.pad_token_id)

    attention_mask = torch.ne(input_ids, tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }