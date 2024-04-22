from .training_arguments import SFTTrainingArguments, RMTrainingArguments
from .data_arguments import SFTDataArguments, RMDataArguments
from .inference_arguments import InferenceArguments
from typing import TypeVar


GenericTrainingArguments = TypeVar(
    "GenericTrainingArguments",
    SFTTrainingArguments,
    RMTrainingArguments
)


GenericDataArguments = TypeVar(
    "GenericDataArguments",
    SFTDataArguments,
    RMDataArguments
)