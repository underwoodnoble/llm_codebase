from typing import Union, Optional, List, Tuple, Callable, Dict, Literal, Any
from copy import deepcopy
from collections import defaultdict
from contextlib import contextmanager

import torch
import deepspeed
from datasets import Dataset
from torch import nn
from transformers import (Trainer, PreTrainedModel,
    PreTrainedTokenizerBase, TrainerCallback, EvalPrediction)
from peft import PeftModel

from .utils import IGNORE_INDEX, FixedKLController, AdaptiveKLController
from ..arguments.training_arguments import BaseTrainingArguments, BaseLLMTrainingArguments
from ..utils.general_utils import print_object_on_main_process


class BaseTrainer(Trainer):
    args: BaseTrainingArguments


    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module, PeftModel] = None,
        args: BaseTrainingArguments = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        if ref_model is not None:
            self.ref_model = ref_model
            for param in self.ref_model.parameters():
                param.requires_grad = False

            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
            
        # kl setting
        if self.args.adaptive_kl_ctrl:
            self.kl_contorller = AdaptiveKLController(self.args.kl_coef, self.args.kl_target)
        else:
            self.kl_contorller = FixedKLController(self.args.kl_coef)
        self.kl_step_buffer = []
    

    def _prepare_deepspeed(self, model: PreTrainedModel):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
        del config_kwargs['optimizer']
        del config_kwargs['scheduler']

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size" : 0.9 * hidden_size * hidden_size
                        }
                    )
        
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model = deepspeed.initialize(model=model, config=config_kwargs)[0].module
        model.eval()
        return model

    @staticmethod
    def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor = None, gather: bool = True) -> torch.Tensor:
        """
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)
        """
        logp = nn.functional.log_softmax(logits, dim=2)

        if not gather:
            return logp
        mask = torch.not_equal(labels, IGNORE_INDEX)
        logpy = torch.gather(logp, 2, (labels * mask).unsqueeze(2)).squeeze(-1)
        return logpy

        
    def kl_divergence_form_logprobs(self, logprobs: torch.FloatTensor, ref_logprobs: torch.FloatTensor) -> torch.FloatTensor:
        """
        if kl_penalty is not 'full'
        logprob: (batch_size, seq_len)
        ref_logprob: (batch_size, seq_len)
        if kl_penalty is 'full'
        logprob: (batch_size, seq_len, vocab_size)
        ref_logprob: (batch_size, seq_len, vocab_size)

        output: (batch_size, seq_len)
        """
        kl_penalty = self.args.kl_penalty_mode
        if kl_penalty == 'kl':
            return logprobs - ref_logprobs
        
        if kl_penalty == 'abs':
            return (logprobs - ref_logprobs).abs()
        
        if kl_penalty == 'mse':
            return 0.5 * (logprobs - ref_logprobs).square()
        
        if kl_penalty == 'full':
            return nn.functional.kl_div(ref_logprobs, logprobs, log_target=True, reduction='none').sum(-1)


    def compute_ref_model_outputs(self, input_ids, attention_mask):
        with torch.no_grad():
            if hasattr(self, "ref_model"):
                    return self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    return self.model(input_ids=input_ids, attention_mask=attention_mask)
            

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        ret = super().training_step(model, inputs)
        if len(self.kl_step_buffer) != 0:
            # calculate current kl
            current_kl = torch.tensor(self.kl_step_buffer).mean()
            # update kl coef
            self.kl_contorller.update(current_kl)
            # clear kl step buffer
            self.kl_step_buffer = []
        return ret

            
    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)


    def log(self, logs: Dict[str, float]) -> None:
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            if len(metrics) == 0:
                logs[key] = 0.0
            else:
                logs[key] = torch.tensor(metrics).mean().item()
                self._stored_metrics[train_eval][key] = []
        return super().log(logs)
    

class BaseLLMTrainer(BaseTrainer):
    args: BaseLLMTrainingArguments
    
    
    def compute_kl_divergence(self, logits: torch.FloatTensor,
                              ref_logits: torch.FloatTensor,
                              labels: torch.Tensor):
        """
        logits: (batch_size, seq_len, vocab_size)
        ref_logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)
        output: (batch_size, seq_len-1) if token_level else (batch_size)
        """
        shift_logits = logits[:, :-1, :]
        shift_ref_logits = ref_logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = torch.not_equal(shift_labels, IGNORE_INDEX)        
        if self.args.token_level:
        # calculate token level kl
            if self.args.kl_penalty_mode == 'full':
                logprobs = self.logprobs_from_logits(shift_logits, gather=False) # (batch_size, seq_len-1, vocab_size)
                ref_logprobs = self.logprobs_from_logits(shift_ref_logits, gather=False) # (batch_size, seq_len-1, vocab_size)
            else:
                logprobs = self.logprobs_from_logits(shift_logits, shift_labels)
                ref_logprobs = self.logprobs_from_logits(shift_ref_logits, shift_labels)
            return self.kl_divergence_form_logprobs(logprobs, ref_logprobs) * mask
        else:
        # calculate sentence level kl
            logprobs = self.logprobs_from_logits(shift_logits, shift_labels) # (batch_size, seq_len-1)
            ref_logprobs = self.logprobs_from_logits(shift_ref_logits, shift_labels) # (batch_size, seq_len-1)
            if self.args.debug_mode:
                print_object_on_main_process('shift_logits', shift_logits)
                print_object_on_main_process('shift_labels', shift_labels)
                print_object_on_main_process('logprobs', logprobs)
                print_object_on_main_process('ref_logprobs', ref_logprobs)
            return (logprobs * mask).sum(-1) / max(mask.sum(-1), 1) - (ref_logprobs * mask).sum(-1) / max(mask.sum(-1), 1)
        