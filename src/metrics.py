from transformers import EvalPrediction
import torch
import numpy as np
from .utils import print_rank_0
from .test import compute_ece
from torch.nn.functional import sigmoid


def compute_reward_metrics(args, predict: EvalPrediction):
    def rm_calibration_errors(args, labels: torch.Tensor, probs: torch.Tensor, masks: torch.Tensor, num_bins):
        label_list = labels.reshape(-1).tolist()
        prob_list = probs.reshape(-1).tolist()
        mask_list = masks.reshape(-1).tolist()

        y_true, y_prob = [], []
        for label, prob, mask in zip(label_list, prob_list, mask_list):
            if mask:
                y_true.append(label)
                y_prob.append(prob)
        
        if args.debug_mode:
            print_rank_0(f">>> Check calibration inputs mask filtered...")
            print_rank_0(f">>>>>>>>> y_true: {y_true[:10]}")
            print_rank_0(f">>>>>>>>> y_prob: {y_prob[:10]}")
        
        return compute_ece(np.array(y_true), np.array(y_prob), n_bins=num_bins)


    logits = torch.from_numpy(predict.predictions) # (batch_size, num_sample)
    scores = torch.from_numpy(predict.label_ids) # (batch_size, num_sample)

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

    # caculate ece
    calibration_errors = {}
    if args.rm_calibration:
        for num_bins in args.calibration_bins:
            expected_error, average_error, max_error = rm_calibration_errors(
                args=args,
                labels=score_mask_larger,
                probs=sigmoid(logits_diff),
                masks=total_mask,
                num_bins=num_bins
            )

            calibration_errors[f"calibration_ECE_bin{num_bins}"] = expected_error
            #calibration_errors[f"calibration_ACE_bin{num_bins}"] = average_error
            #calibration_errors[f"calibration_MCE_bin{num_bins}"] = max_error
    
    if args.debug_mode:
        print_rank_0(f">>> Check eval_prediction outputs...")
        print_rank_0(f">>> correct_compare: {correct_compare}")
        print_rank_0(f">>> total_mask: {total_mask}")
        print_rank_0(f">>> all_acc: {all_acc}")
        print_rank_0(f">>> calibration error: {calibration_errors}")

    return {"Preference total Acc": all_acc.item(), "First-two Acc": first_two_acc.item(), **calibration_errors}


def compute_classification_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc: np.ndarray = (predictions == labels).sum() / len(labels)
    return {
        "acc": acc.item()
    }