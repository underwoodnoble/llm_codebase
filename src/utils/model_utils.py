from typing import Dict, Callable, Tuple, Type, Optional

import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
    PreTrainedModel, PreTrainedTokenizer, PretrainedConfig,
    AutoModel, AutoConfig)
from trl import DPOTrainer

from .general_utils import print_rank_0
from ..arguments import GenericTrainingArguments
from ..algorithms.base import BaseTrainer
from ..algorithms import SFTTrainer, OfflinePPOTrainer, RMTrainer, sft_data_collator, offline_ppo_data_collator, rm_data_collator


def set_special_tokens(tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> None:
    DEFAULT_PAD_TOKEN = "<pad>"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # If not set add_eos_token to True, Llama tokenizer do not add eos token in encoding automatically.
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings: torch.Tensor = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        if model.get_output_embeddings() is not None:
            output_embeddings: torch.Tensor = model.get_output_embeddings().weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg


def load_causal_lm(
    training_args: GenericTrainingArguments,
    load_ref_model: bool
    ) -> Tuple[PreTrainedTokenizer, PreTrainedModel, Optional[PreTrainedModel]]:
    def _load(model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        truncation_side='left',
        padding_side='right',
        trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        set_special_tokens(tokenizer, model) 
        return tokenizer, model

    print_rank_0(f"Load model from {training_args.model_name_or_path}")
    tokenizer, model = _load(training_args.model_name_or_path)
    if load_ref_model:
        _, ref_model = _load(training_args.ref_model_name_or_path
                            if training_args.ref_model_name_or_path else training_args.model_name_or_path)
    else:
        ref_model = None

    return tokenizer, model, ref_model


def load_rm_model(
    training_args: GenericTrainingArguments,
    load_ref_model: bool)-> Tuple[PreTrainedTokenizer, PreTrainedModel, Optional[PreTrainedModel]]:
    from ..models.utils import get_reward_config, get_reward_model
    
    def _load(model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            truncation_size='left',
            padding_side='right',
            trust_remote_code=True
        )
        base_config = AutoConfig.from_pretrained(model_name_or_path)
        config = get_reward_config(type(base_config)).from_pretrained(model_name_or_path)
        base_model_class = AutoModel._model_mapping[type(base_config)]
        model = get_reward_model(base_model_class).from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            config=config
        )
        set_special_tokens(tokenizer, model)
        return tokenizer, model
    
    model_name_or_path = training_args.rm_model_name_or_path if hasattr(training_args, 'rm_model_name_or_path') else training_args.model_name_or_path
    print_rank_0(f"Loading model from {model_name_or_path}")
    tokenizer, model = _load(model_name_or_path)
    if load_ref_model:
        _, ref_model = _load(training_args.ref_model_name_or_path if training_args.ref_model_name_or_path else model_name_or_path)
    else:
        ref_model = None
    return tokenizer, model, ref_model


def load_tokenizer_and_model(training_args: GenericTrainingArguments, algorithm: str):
    reward_model, value_model = None, None
    if algorithm in ['sft', 'offline_ppo', 'dpo']:
        if algorithm == 'sft':
            load_ref_model = training_args.kl_coef is not None and training_args.peft_config_path is None
        elif algorithm in ['offline_ppo', 'dpo']:
            load_ref_model = training_args.peft_config_path is None
        tokenizer, model, ref_model = load_causal_lm(
            training_args,
            load_ref_model
        )
    elif algorithm == 'rm':
        tokenizer, model, ref_model = load_rm_model(
            training_args,
            load_ref_model = training_args.kl_coef is not None and training_args.peft_config_path is None
        )

    elif algorithm == 'ppo_v2':
        tokenizer, model, ref_model = load_causal_lm(
            training_args,
            load_ref_model = training_args.peft_config_path is None
        )
        _, reward_model, _ = load_rm_model(training_args, load_ref_model=False)
        if not training_args.share_backbone_between_policy_and_value:
            _, value_model, _ = load_rm_model(training_args, load_ref_model=False)
        else:
            class ValueModel(torch.nn.Module):
                base_model_prefix = 'model'
                def __init__(self, backbone: torch.nn.Module):
                    super().__init__()
                    self.model = backbone
                    self.value_head = torch.nn.Linear(self.model.config.hidden_size, 1, bias=False)
                @property
                def score(self):
                    return self.value_head
            
            value_model = ValueModel(getattr(model, model.base_model_prefix))

    tokenizer.model_max_length = training_args.model_max_length

    return tokenizer, model, ref_model, reward_model, value_model


def get_collator_and_trainer(algorithm) -> Tuple[Callable[[Dict[str, any]], Dict[str, torch.Tensor]], Type[BaseTrainer]]:
    MAP: Dict[str, Tuple[Callable[[Dict[str, any]], Dict[str, torch.Tensor]], Type[BaseTrainer]]] = {
        "sft": (sft_data_collator, SFTTrainer),
        "offline_ppo": (offline_ppo_data_collator, OfflinePPOTrainer),
        'rm': (rm_data_collator, RMTrainer),
        'dpo': (lambda x, y: None, DPOTrainer)
    }

    return MAP[algorithm]