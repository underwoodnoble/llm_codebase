from transformers import (Trainer, PreTrainedModel,
PreTrainedTokenizerBase, EvalPrediction, TrainerCallback)
from ..arguments import SFTDataArguments, SFTTrainingArguments
from typing import Union, Optional, Callable, Dict, List, Tuple
from torch import nn
from datasets import Dataset
import torch
import deepspeed
from copy import deepcopy
from ..utils.general_utils import print_rank_0


def sft_transform(example, data_args: SFTDataArguments):
    if data_args.data_format == 'qa':
        if data_args.prompt_name != 'prompt' or data_args.answer_name != 'prompt':
            return {
                "prompt": example[data_args.prompt_name],
                "answer": example[data_args.answer_name]
            }
    else:
        pass

    return example

    
def sft_data_collator():
    pass



class SFTTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        args: SFTTrainingArguments = None,
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
        self.args = args
        if self.args.kl is not None:
            self.ref_model = ref_model
            for param in self.ref_model.parameters():
                param.requires_grad = False

            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
    
    @staticmethod
    def compute_lm_loglikeli(logits, labels):
        batch_size, seq_length, vocab_size = logits.shape
            
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels).reshape(batch_size, -1) # [bs * seq_len]
        ignore_mask = labels != -100
        
        mean_loss = loss.sum(dim=-1) / ignore_mask.sum(dim=-1)

        return - mean_loss #loss.reshape(batch_size, -1).mean(dim=-1)


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


    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.debug_mode:
            print_rank_0(f"check inputs :{inputs}")
            
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

        with torch.no_grad():
            # model.ref_model.eval()
            ref_model_outputs = self.ref_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            ref_logprob = self.compute_lm_loglikeli(ref_model_outputs.logits, inputs['labels']) #[batch_size]

        if self.args.debug_mode:
            print_rank_0(f"check ref_model output: {ref_logprob}")

        logprob = self.compute_lm_loglikeli(model_outputs.logits, inputs['labels'])

        # for importance sampling
        # loss = - logprob + self.args.llm_kl_coeff * (logprob - ref_logprob)

        # for MC kl
        kl_divergence = logprob.exp() * (logprob - ref_logprob)

        if self.args.use_kl_mask:
            lm_size, kl_size = (1-inputs['sft_mask']).sum(), (inputs['sft_mask']).sum()
            
            lm_loss = (-logprob * (1 - inputs['sft_mask']) * inputs['weights']).sum() / lm_size if lm_size > 0 else lm_size
            kl_loss = (inputs['sft_mask'] * kl_divergence).sum() / kl_size if kl_size > 0 else kl_size

            total_loss = lm_loss + self.args.lm_kl_coeff * kl_loss
        else:        
            loss = - logprob + self.args.lm_kl_coeff * kl_divergence

        #total_loss = - logprob * self.args.lm_loss_coeff * inputs['sft_mask'] + ppo_loss * (1 - inputs['sft_mask'])
        
        total_loss = (loss * inputs['weights']).mean() # [batch_size]

        # total_loss = weighted_loss + args.llm_kl_coeff * kl_divergence

        if self.args.debug_mode:          
            print_rank_0(f"check loss : {loss}")
            print_rank_0(f"check kl divergence : {kl_divergence}")

        return (total_loss, model_outputs['logits']) if return_outputs else total_loss