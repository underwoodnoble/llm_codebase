from dataclasses import dataclass, field
import transformers
from typing import Optional, List


@dataclass
class CustomArguments(transformers.TrainingArguments):
    # task arguments
    task_type: Optional[str] = field(default='reward')

    # data arguments
    data_dir: str = field(default=None, metadata={"help": "the directory to load data."})
    data_paths: List[str] = field(default=None, metadata={"help": "train dataset paths"})

    eval_data_dir: str = field(default=None, metadata={"help": "the directory to load evaluation datasets."})
    eval_data_paths: List[str] = field(default=None, metadata={"help": "evaluation dataset paths."})

    # model arguments
    model_type: Optional[str] = field(default='bert', metadata={"help": "base model to use."})
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "pretrained model path"})
    preference_data_texts_name: Optional[str] = field(default='texts', metadata={"help": "key in preference data that indicate texts"})
    preference_data_scores_name: Optional[str] = field(default="scores", metadata={"help": "key in preference data that indicate scores"})
    model_max_length: Optional[str] = field(default=512, metadata={"help": "the max sentence sequence length."})
    
    # training arguments
    truncation_side: Optional[str] = field(default='left', metadata={"help": "which side to truncate when sequence is too long."})
    padding_side: Optional[str] = field(default='right', metadata={"help": "which side to padding."})


    def __post_init__(self):
        super().__post_init__()
        valid_task_types = ["reward"]
        if self.task_type not in valid_task_types:
            raise ValueError(f"Invalid task type. Expected one of {valid_task_types}, but got {self.task_type}")

        if self.data_dir is None and self.data_paths is None:
            raise ValueError(f"One of data_dir and data_paths must be set.")
        if self.data_dir is not None and self.data_paths is not None:
            raise ValueError(f"Only one of data_dir and data_paths should be set.")
        if self.eval_data_dir is None and self.eval_data_paths is None:
            raise ValueError(f"One of eval_dir and eval_data_path must be set.")
        if self.eval_data_dir is not None and self.eval_data_paths is not None:
            raise ValueError(f"Only one of eval_dir and eval_data_paths should be set.")

        valid_model_types = ['bert', 'llama']
        if self.model_type not in valid_model_types:
            raise ValueError(f"Invalid model type. Expected one of {valid_model_types}, but got {self.model_type}.")
        if self.model_name_or_path is None:
            raise ValueError(f"model_name_or_path must be assigned.")