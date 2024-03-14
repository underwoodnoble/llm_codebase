from dataclasses import dataclass, field
import transformers
from typing import Optional, List


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    debug_mode: Optional[bool] = field(default=False)
    # task arguments
    task_type: Optional[str] = field(default='reward')

    #######################################################################################
    # data arguments
    data_dir: str = field(default=None, metadata={"help": "the directory to load data."})
    data_paths: List[str] = field(default=None, metadata={"help": "train dataset paths"})

    eval_data_dir: str = field(default=None, metadata={"help": "the directory to load evaluation datasets."})
    eval_data_paths: List[str] = field(default=None, metadata={"help": "evaluation dataset paths."})
    eval_dataset_merge_mode: Optional[str] = field(default='separate',
                                                 metadata={"help": "How to evaluate multiple evalution datasets. Must be one of ['separate', 'merge', 'both']"})

    sep_token: Optional[str] = field(default='<sep>', metadata={"help": "the token that can use to seperate the query and answer in text"})

    ## classification data
    cls_data_text_name: Optional[str] = field(default='text', metadata={"help": "text's names"})
    cls_data_label_name: Optional[str] = field(default='label', metadata={"help": "text label names"})
    cls_data_label_nums: Optional[int] = field(default=None, metadata={"help": "num of label types"})

    ## sft data
    sft_data_prompt_name: Optional[str] = field(default='prompt', metadata={"help": "prompt name."})
    sft_data_answer_name: Optional[str] = field(default='answer', metadata={"help": "answer name"})

    ## preference data
    preference_data_text_name: Optional[str] = field(default='texts', metadata={"help": "key in preference data that indicate texts"})
    preference_data_score_name: Optional[str] = field(default="scores", metadata={"help": "key in preference data that indicate scores"})

    ## Weighted data
    weighted_data_prompt_name: Optional[str] = field(default=None)
    weighted_data_answer_name: Optional[str] = field(default=None)
    weighted_data_score_name: Optional[str] = field(default=None)

    #########################################################################################
    # model arguments
    model_type: Optional[str] = field(default='bert', metadata={"help": "base model to use."})
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "pretrained model path"})
    model_max_length: Optional[int] = field(default=512, metadata={"help": "the max sentence sequence length."})
    ignore_token_id: Optional[int] = field(default=-100, metadata={"help": "token id used to inplace query ids."})
    set_special_tokens: Optional[bool] = field(default=True, metadata={"help": "whether or not add special token in tokenizer."})
    
    #########################################################################################
    # training arguments
    truncation_side: Optional[str] = field(default='left', metadata={"help": "which side to truncate when sequence is too long."})
    padding_side: Optional[str] = field(default='right', metadata={"help": "which side to padding."})
    only_predict_answer: Optional[bool] = field(default=True, metadata={"help": "Only predict the answer."})
    pad_labels_with_ignore: Optional[bool] = field(default=False, metadata={"help": "Whether use ignore token to pad labels."})
    evaluate_at_beginning: Optional[bool] = field(default=True, metadata={"help": "Whether evaluate at the beginning."})
    save_training_states: Optional[bool] = field(default=False, metadata={"help": "Whether or not save training states."})
    training_type: Optional[str] = field(default='full')
    ## Lora arguments
    lora_rank: Optional[int] = field(default=8)
    lora_alpha: Optional[int]= field(default=32)
    lora_dropout: Optional[float] = field(default=0.1)

    ## Reward model training arguments
    add_lm_loss: Optional[bool] = field(default=True, metadata={"help": "add language model loss when training reward model"})
    lm_loss_coeff: Optional[float] = field(default=0., metadata={"help": "the coefficient for language modeling loss."})
    lm_score_thresh: Optional[float] = field(default=0.85, metadata={"help": "the threshold to select response for language modeling."})
    pooling_type: Optional[str] = field(default='last', metadata={"help": "how to pooling the last hidden states"})
    rm_calibration: Optional[bool] = field(default=True)
    calibration_bins: List[int] = field(default_factory=lambda:[5])

    ## RRHF
    length_penalty: Optional[float] = field(default=1.0, metadata={"help": "length penalty in RRHF"})
    rrhf_weight: Optional[float] = field(default=1.0, metadata={"help": "weights of ranking loss in RRHF"})

    ## DPO
    construct_method: Optional[str] = field(default="best_to_rest", metadata={"help": "The method used to construct preference pairs. [best_over_rest, best_over_worst, one_over_rest]"})
    dpo_beta: Optional[float] = field(default=0.1, metadata={"help":"The beta factor in DPO loss. Higher beta means less divergence from the initial policy."})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "The maximum length of the prompt."})

    # KTO
    kto_pair_prompt_name: Optional[str] = field(default='prompt')
    kto_pair_answer_name: Optional[str] = field(default='completion')
    kto_pair_label_name: Optional[str] = field(default='label')
    kto_beta: Optional[float] = field(default=0.1)
    kto_desirable_weight: Optional[float] = field(default=1.0)
    kto_undesirable_weight: Optional[float] = field(default=1.0)
    kto_num_of_kl_completions_per_prompt: Optional[int] = field(default=1000)


    def __post_init__(self):
        super().__post_init__()
        valid_task_types = ["reward", "classification", "multi_objective_classification", 
                            "SFT", "offline_rejection_sampling", "offline_RRHF", "weighted_learning", "DPO",
                            "KTO"]
        if self.task_type not in valid_task_types:
            raise ValueError(f"Invalid task type. Expected one of {valid_task_types}, but got {self.task_type}")

        if self.data_dir is not None and self.data_paths is not None:
            raise ValueError(f"Only one of data_dir and data_paths should be set.")
        if self.eval_data_dir is not None and self.eval_data_paths is not None:
            raise ValueError(f"Only one of eval_dir and eval_data_paths should be set.")
        if self.eval_dataset_merge_mode not in ['separate', 'merge', 'both']:
            raise ValueError(f"Ivalid eval_dataset_merge_mode, Expected one of ['separate', 'merge', 'both'], bug got {self.eval_dataset_merge_mode}")

        valid_model_types = ['bert', 'llama', 'baichuan', 'other']
        if self.model_type not in valid_model_types:
            raise ValueError(f"Invalid model type. Expected one of {valid_model_types}, but got {self.model_type}.")
        if self.model_name_or_path is None:
            raise ValueError(f"model_name_or_path must be assigned.")

        valid_training_type = ['full', 'lora', 'p-tuning']
        if self.training_type not in valid_training_type:
            raise ValueError(f"Invalid training_type. Expected one of {valid_training_type}, but got {self.training_type}")