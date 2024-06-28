from typing import Optional, List
import warnings
from dataclasses import dataclass, field

import transformers


@dataclass
class BaseTrainingArguments(transformers.TrainingArguments):
    debug_mode: Optional[bool] = field(default=False, metadata={"help": "Enable the debug mode."})

    model_type: Optional[str] = field(default='bert', metadata={"help": "Model type."})
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Pretrained model path."})
    ref_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Reference model path."})
    model_max_length: Optional[int] = field(default=512, metadata={"help": "Model max length."})

    add_special_tokens: Optional[bool] = field(default=False, metadata={"help": "Whether to add the bos token and eos token automatically "})
    pad_labels_with_ignore: Optional[bool] = field(default=True, metadata={"help": "Pad labels with ignore token id."})

    # kl arguments
    kl_coef: Optional[float] = field(default=None, metadata={"help": "KL penalty weight."})
    kl_penalty_mode: Optional[str] = field(default='kl', metadata={"help": "KL penalty mode. One of ['kl', 'abs', 'mse', 'full']. In llm training, this only works if the 'token_level' is true"})
    adaptive_kl_ctrl: Optional[bool] = field(default=False, metadata={"help": "Using adaptive KL controller."})
    kl_target: Optional[float] = field(default=6., metadata={"help": "The expected KL divergence value. Effective when the KL controller is 'AdaptiveKLController'."})

    # peft
    peft_config_path: Optional[str] = field(default=None, metadata={"help": "PEFT configuration path."})


    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False


@dataclass
class BaseLLMTrainingArguments(BaseTrainingArguments):
    token_level: Optional[bool] = field(default=False, metadata={"help": "Consider each token to be an action."})
    # Todo: Discard this parameter.
    only_predict_answer: Optional[bool] = field(default=True, metadata={"help": "Only calculate the loss of answer."})


@dataclass
class SFTTrainingArguments(BaseLLMTrainingArguments):
    pass


@dataclass
class RMTrainingArguments(BaseLLMTrainingArguments):
    add_lm_loss: Optional[bool] = field(default=False, metadata={"help": "add language model loss when training reward model."})
    lm_loss_coeff: Optional[float] = field(default=0., metadata={"help": "the coefficient for language modeling loss."})
    lm_score_thresh: Optional[float] = field(default=0.85, metadata={"help": "the threshold to select response for language modeling."})
    pooling_type: Optional[str] = field(default='last', metadata={"help": "how to pooling the last hidden states"})
    rm_calibration: Optional[bool] = field(default=True)
    calibration_bins: List[int] = field(default_factory=lambda:[5])

    def __post_init__(self):
        super().__post_init__()
        if self.pooling_type not in ['last', 'eos', 'average', 'max']:
            raise ValueError(f"pooling_type must be one of ['last', 'eos', 'average', 'max'], getting '{self.pooling_type}'")

@dataclass
class OfflinePPOTrainingArguments(BaseLLMTrainingArguments):
    clip_range: float = field(default=0.2)
    lm_coef: float = field(default=0.5)


    def __post_init__(self):
        super().__post_init__()
