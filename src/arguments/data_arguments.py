from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class BaseDataArguments:
    data_paths: List[Path] = field(default=None, metadata={"help": "Training dataset files or dirs."})
    eval_data_paths: Optional[List[Path]] = field(default=None, metadata={"help": "evaluation dataset files or dirs."})
    eval_dataset_merge_mode: str = field(
        default='separate',
        metadata={"help": "How to evaluate multiple evalution datasets. Must be one of ['separate', 'merge']"})

    streaming: bool = field(
        default=False,
        metadata={"help": "Using IterableDataset"}
    )

    def __post_init__(self):
        if self.data_paths is None and self.eval_data_paths is None:
            raise ValueError("data_paths and eval_data_paths must have at least one that is not None")

@dataclass
class SFTDataArguments(BaseDataArguments):
    prompt_name: Optional[str] = field(default="prompt", metadata={"help": "The field corresponding to prompt."})
    answer_name: Optional[str] = field(default="answer", metadata={"help": "The field corresponding to answer."})
    weight_name: Optional[str] = field(default="weight", metadata={"help": "The field corresponding to data weight."})

    
    def __post_init__(self):
        super().__post_init__()


@dataclass
class RMDataArguments(BaseDataArguments):
    text_name: Optional[str] = field(default="texts", metadata={"help": "The field corresponding to texts."})

    prompt_name: Optional[str] = field(default='prompt', metadata={"help": "The field corresponding to prompt. This parameter is valid when there is no text field"})
    chosen_name: Optional[str] = field(default='chosen', metadata={"help": "The field corresponding to chosen response. This parameter is valid when there is no text field"})
    rejected_name: Optional[str] = field(default='rejected', metadata={"help": "The field corresponding to rejected response. This parameter is valid when there is no text field"})

    score_name: Optional[str] = field(default="scores", metadata={"help": "The field corresponding to scores."})
    weight_name: Optional[str] = field(default="weight", metadata={"help": "The field corresponding to weight."})
    
    def __post_init__(self):

        super().__post_init__()


@dataclass
class OfflinePPODataArguments(BaseDataArguments):
    prompt_name: Optional[str] = field(default="prompt", metadata={"help": "The field corresponding to prompt."})
    answer_name: Optional[str] = field(default="answer", metadata={"help": "The field corresponding to answer."})
    weight_name: Optional[str] = field(default="weight", metadata={"help": "The field corresponding to data weight."})
    reward_name: Optional[str] = field(default="reward", metadata={"help": "The field corresponding to reward."})
    data_type_name: Optional[str] = field(default="type", metadata={"help": "The field corresponding to data type. (rl or lm)"})
    
    def __post_init__(self):
        return super().__post_init__()

        
@dataclass
class DPODataArguments(BaseDataArguments):
    prompt_name: Optional[str] = field(default="prompt", metadata={"help": "The field corresponding to prompt."})
    chosen_name: Optional[str] = field(default="chosen", metadata={"help": "The field corresponding to chosen response."})
    rejected_name: Optional[str] = field(default="rejected", metadata={"help": "The field corresponding to rejected response."})

    def __post_init__(self):
        return super().__post_init__()

        
@dataclass
class PPOv2DataArguments(BaseDataArguments):
    prompt_name: Optional[str] = field(default='prompt', metadata={"help": "The field corresponding to prompt."})

    def __post_init__(self):
        return super().__post_init__()