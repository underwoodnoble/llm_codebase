from dataclasses import dataclass, field
import transformers
from typing import Optional, List

@dataclass
class BaseTrainingArguments(transformers.TrainingArguments):
    debug_mode: Optional[bool] = field(default=False, metadata={"help": "Enable the debug mode."})

    model_type: Optional[str] = field(default='bert', metadata={"help": "Model type."})
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Pretrained model path."})
    ref_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Reference model path."})
    model_max_length: Optional[int] = field(default=512, metadata={"help": "Model max length."})

    add_special_tokens: Optional[bool] = field(default=False, metadata={"help": "Whether to add the bos token and eos token automatically "})
    only_predict_answer: Optional[bool] = field(default=True, metadata={"help": "Only calculate the loss of answer."})
    pad_labels_with_ignore: Optional[bool] = field(default=True, metadata={"help": "Pad labels with ignore token id."})

    # kl arguments
    kl_coef: Optional[float] = field(default=None, metadata={"help": "KL penalty weight."})
    kl_penalty_mode: Optional[str] = field(default='kl', metadata={"help": "KL penalty mode. One of ['kl', 'abs', 'mse', 'full']"})
    adaptive_kl_ctrl: Optional[bool] = field(default=False, metadata={"help": "Using adaptive KL controller."})
    kl_target: Optional[float] = field(default=6., metadata={"help": "The expected KL divergence value. Effective when the KL controller is 'AdaptiveKLController'"})


    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False


@dataclass
class SFTTrainingArguments(BaseTrainingArguments):
    pass


@dataclass
class RMTrainingArguments(BaseTrainingArguments):
    pass


@dataclass
class OfflinePPOTrainingArguments(BaseTrainingArguments):
    pass