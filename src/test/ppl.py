from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Dict
from torch.nn.utils.rnn import pad_sequence
import torch
from copy import deepcopy


def compute_ppl(data: Dict[str, Dict[str, List[str]]], tokenizer: PreTrainedTokenizer, model: PreTrainedModel, ignore_token_id=-100):
    '''calculate ppl
    '''
    input_ids = []
    labels = []
    device = model.device
    for prompt, answer in zip(data['prompt'], data['answer']):
        prompt_ids = tokenizer.encode(prompt)
        text_ids = tokenizer.encode(prompt + answer)
        label = deepcopy(text_ids)
        label[:len(prompt_ids)] = [ignore_token_id] * len(prompt_ids)
        input_ids.append(torch.tensor(text_ids[-tokenizer.model_max_length:]))
        labels.append(torch.tensor(label[-tokenizer.model_max_length:]))

    ppls = []
    with torch.no_grad():
        for i, l in zip(input_ids, labels):
            outputs = model(input_ids=i.unsqueeze(0).to(device), labels=l.unsqueeze(0).to(device))
        neg_log_likelihood = outputs.loss
        ppls.append(torch.exp(neg_log_likelihood).item())
    return [{"prompt": prompt, "answer": answer, "ppl": ppl} for prompt, answer, ppl in zip(data['prompt'], data['answer'], ppls)]