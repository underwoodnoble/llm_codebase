from typing import Union, Optional, List, Tuple, Callable, Dict, Literal, Any
from copy import deepcopy
from collections import defaultdict
from contextlib import contextmanager

from datasets import Dataset
import torch
from torch import nn
from transformers import (Trainer, PreTrainedModel,
    PreTrainedTokenizerBase, TrainerCallback, EvalPrediction)
import deepspeed
from peft import PeftModel

from .utils import IGNORE_INDEX, FixedKLController, AdaptiveKLController
from ..arguments.training_arguments import BaseTrainingArguments


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

        
    @staticmethod
    def compute_kl_divergence(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty='kl') -> torch.FloatTensor:
        """
        if kl_penalty is not 'full'
        logprob: (batch_size, seq_len)
        ref_logprob: (batch_size, seq_len)
        if kl_penalty is 'full'
        logprob: (batch_size, seq_len, vocab_size)
        ref_logprob: (batch_size, seq_len, vocab_size)

        output: (batch_size, seq_len)
        """
        if kl_penalty == 'kl':
            return logprob - ref_logprob
        
        if kl_penalty == 'abs':
            return (logprob - ref_logprob).abs()
        
        if kl_penalty == 'mse':
            return 0.5 * (logprob - ref_logprob).square()
        
        if kl_penalty == 'full':
            return nn.functional.kl_div(ref_logprob, logprob, log_target=True, reduction='none').sum(-1)


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
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)