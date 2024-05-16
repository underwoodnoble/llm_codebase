from typing import TypeVar
from .training_arguments import SFTTrainingArguments, RMTrainingArguments, OfflinePPOTrainingArguments
from .data_arguments import SFTDataArguments, RMDataArguments, OfflinePPODataArguments
from .inference_arguments import InferenceArguments


GenericTrainingArguments = TypeVar(
    "GenericTrainingArguments",
    SFTTrainingArguments,
    RMTrainingArguments,
    OfflinePPOTrainingArguments
)


GenericDataArguments = TypeVar(
    "GenericDataArguments",
    SFTDataArguments,
    RMDataArguments,
    OfflinePPODataArguments
)