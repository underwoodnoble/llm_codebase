from ..arguments import GenericTrainingArguments
from transformers import (AutoTokenizer, AutoModelForCausalLM,
    PreTrainedModel, PreTrainedTokenizer)
from typing import Tuple, Optional
import torch
from .general_utils import print_rank_0


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
        output_embeddings: torch.Tensor = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def load_causal_lm(
    training_args: GenericTrainingArguments,
    load_ref_model: bool
    ) -> Tuple[PreTrainedTokenizer, PreTrainedModel, Optional[PreTrainedModel]]:
    def _load():
        tokenizer = AutoTokenizer.from_pretrained(
        training_args.model_name_or_path,
        truncation_side='left',
        padding_side='right',
        trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            training_args.model_name_or_path,
            trust_remote_code=True
        )
        set_special_tokens(tokenizer, model) 
        return tokenizer, model

    print_rank_0(f"Load model from {training_args.model_name_or_path}")
    tokenizer,model = _load()
    if load_ref_model:
        _, ref_model = _load()
    else:
        ref_model = None
    
    return tokenizer, model, ref_model


def load_rm_model(training_args: GenericTrainingArguments):
    pass


def load_tokenizer_and_model(training_args: GenericTrainingArguments, algorithm: str):
    if algorithm in ['sft']:
        tokenizer, model, ref_model = load_causal_lm(
            training_args,
            training_args.kl_coeff is not None
        )
    
    tokenizer.model_max_length = training_args.model_max_length

    return tokenizer, model, ref_model