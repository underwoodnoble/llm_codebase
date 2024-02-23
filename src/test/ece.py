from typing import Dict, List, Union
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
from torch.nn.functional import sigmoid
import numpy as np

def compute_preference_confidence(inputs: Dict[str, torch.Tensor], tokenizer: PreTrainedTokenizer, model: PreTrainedModel):
    input_ids = inputs['input_ids'].to(model.device) # (batch_size, num_of_sample, seq_len)
    attention_mask = inputs['attention_mask'].to(model.device) # (batch_size, num_of_sample, seq_len)
    scores = inputs['scores'].to(model.device)
    batch_size, num_of_sample, seq_len = input_ids.shape
    with torch.no_grad():
        logits: torch.Tensor = model.forward(input_ids=input_ids.view(batch_size*num_of_sample, -1),
                                             attention_mask=attention_mask.view(batch_size*num_of_sample, -1))['rm_logits'] #(batch_size*num_of_sample, 1)
    logits = logits.view(batch_size, -1) # (batch_size, num_of_sample)
    
    logit_diff = logits.unsqueeze(dim=1) - logits.unsqueeze(dim=-1) # (batch_size, num_of_sample, num_of_sample)
    probs = sigmoid(logit_diff).flatten().tolist()
    
    scores_diff = (scores.unsqueeze(dim=1) - scores.unsqueeze(dim=-1)).flatten().tolist()
    preds = []
    ground_truth = []
    for i in range(len(scores_diff)):
        score = scores_diff[i]
        if score == 0.0:
            continue
        elif score > 0.0:
            preds.append(probs[i])
            ground_truth.append(1)
        else:
            preds.append(probs[i])
            ground_truth.append(0)
    return preds, ground_truth


def compute_ece(y_true, y_prob, n_bins=5, strategy="uniform"):
    if len(y_true) == 0:
        return 0., 0., 0.
    
    if strategy == "quantile":
        quantiles = np.linspace(0., 1., n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0., 1., n_bins+1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy must be either 'quantile' or 'uniform'."
        )
    
    # the ith element in binids indicate which bin the ith element in y_prob belong to.
    binids = np.searchsorted(bins[1:-1], y_prob)

    # the ith element in bin_sums is the average probability of positive examples that model predict in the ith bin
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    # the ith element in bin_true is the real probablility of positive examples in the ith bin
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    # the ith element in bin_total is the total num of examples belong to the ith bin
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0

    try:
        expected_error = np.abs(bin_sums - bin_true).sum() / len(y_prob)
        average_error = (np.abs(bin_sums[nonzero] - bin_true[nonzero]) / bin_total[nonzero]).mean()
        max_error = (np.abs(bin_sums[nonzero] - bin_true[nonzero]) / bin_total[nonzero]).max()
    except:
        expected_error, average_error, max_error = 0., 0., 0.
    return expected_error, average_error, max_error