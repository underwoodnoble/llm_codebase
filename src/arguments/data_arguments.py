from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class BaseDataArguments:
    data_paths: Optional[List[Path]] = field(default=None, metadata={"help": "Training dataset files or dirs."})
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
    data_format: str = field(default='qa', metadata={"help": "Data format. Should be 'qa' or 'dialogue'"})

    prompt_name: Optional[str] = field(default="prompt", metadata={"help": "The field corresponding to prompt."})
    answer_name: Optional[str] = field(default="answer", metadata={"help": "The field corresponding to answer."})

    user_name: Optional[str] = field(default="qa", metadata={"help": "The field corresponding to user."})
    assistant_name: Optional[str] = field(default='dialogue', metadata={"help": "The field corresponding to assistant."})

    weight_name: Optional[str] = field(default="weight", metadata={"help": ""})

    
    def __post_init__(self):
        super().__post_init__()
        if self.data_format not in ['qa', 'dialogue']:
            raise ValueError(f"Do not support {self.data_format} data format.")


@dataclass
class RMDataArguments(BaseDataArguments):
    pass