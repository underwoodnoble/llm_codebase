from dataclasses import dataclass, field
import transformers
from typing import Optional, List

@dataclass
class BaseTrainingArguments(transformers.TrainingArguments):
    debug_mode: Optional[bool] = field(default=False, metadata={"help": "Enable the debug mode."})

    model_type: Optional[str] = field(default='bert', metadata={"help": "Model type."})
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Pretrained model path."})
    model_max_length: Optional[int] = field(default=512, metadata={"help": "Model max length."})


@dataclass
class SFTTrainingArguments(BaseTrainingArguments):
    kl: Optional[float] = field(default=None, metadata={"help": "KL penalty weight."})


@dataclass
class RMTrainingArguments(BaseTrainingArguments):
    pass