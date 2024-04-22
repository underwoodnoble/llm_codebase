from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class BaseDataArguments:
    data_dirs: Optional[List[str]] = field(default=None, metadata={"help": "The directories for loading training dataset."})
    data_paths: Optional[List[str]] = field(default=None, metadata={"help": "Training dataset paths."})

    eval_data_dirs: Optional[str] = field(default=None, metadata={"help": "The directories for loading evaluation dataset."})
    eval_data_paths: Optional[List[str]] = field(default=None, metadata={"help": "evaluation dataset paths."})
    eval_dataset_merge_mode: str = field(
        default='separate',
        metadata={"help": "How to evaluate multiple evalution datasets. Must be one of ['separate', 'merge', 'both']"})

    streaming: bool = field(
        default=False,
        metadata={"help": "Using IterableDataset"}
    )

    def __post_init__(self):
        if self.data_dirs is None and self.data_paths is None:
            raise ValueError("data_dirs and data_paths must have at least one that is not None")


@dataclass
class SFTDataArguments(BaseDataArguments):
    data_format: str = field(default='qa', metadata={"help": "Data format. Should be 'qa' or 'dialogue'"})

    prompt_name: Optional[str] = field(default="prompt", metadata={"help": "The field corresponding to prompt."})
    answer_name: Optional[str] = field(default="answer", metadata={"help": "The field corresponding to answer."})

    user_name: Optional[str] = field(default="qa", metadata={"help": "The field corresponding to user."})
    assistant_name: Optional[str] = field(default='dialogue', metadata={"help": "The field corresponding to assistant."})

    
    def __post_init__(self):
        super().__post_init__()
        if self.data_format not in ['qa', 'dialogue']:
            raise ValueError(f"Do not support {self.data_format} data format.")


@dataclass
class RMDataArguments(BaseDataArguments):
    pass