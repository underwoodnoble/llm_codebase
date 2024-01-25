from transformers import EvalPrediction
import torch
import numpy as np


def compute_reward_metrics(predict: EvalPrediction):
    logits = torch.from_numpy(predict.predictions)
    scores = torch.from_numpy(predict.label_ids)

    logits_diff = logits.unsqueeze(1) - logits.unsqueeze(2) # shape: (batch_size, num_sample, num_sample)
    
    score_mask_larger = (scores.unsqueeze(1) > scores.unsqueeze(2)) * 1.
    score_mask_smaller = (scores.unsqueeze(1) < scores.unsqueeze(2)) * 1.
    score_mask = score_mask_larger - score_mask_smaller
    pad_mask = (scores >= 0).unsqueeze(1) * 1. * (scores >= 0).unsqueeze(2)

    # caculate accuracy
    pred_compare = ((logits_diff * score_mask).detach() > 0.) * 1.
    total_mask = (score_mask_larger + score_mask_smaller) * pad_mask
    correct_compare = pred_compare * total_mask

    all_acc = correct_compare.sum() / total_mask.sum()
    first_two_acc = (correct_compare[:, 0, 1]).sum() / (total_mask[:, 0, 1]).sum()

    return {"Preference total Acc": all_acc.item(), "First-two Acc": first_two_acc.item()}


def compute_classification_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = (predictions == labels).sum() / len(labels)
    return {
        "acc": acc
    }

def compute_contrastive_metrics(predict: EvalPrediction):
    logits = torch.from_numpy(predict.predictions)