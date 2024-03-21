from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from dataclasses import dataclass, field
from torch import FloatTensor

@dataclass
class RewardModelOutput(ModelOutput):
    lm_logits: FloatTensor = None
    rm_logits: FloatTensor = None
    last_hidden_states: FloatTensor = None
    rm_embeddings: FloatTensor = None

class RefModel(PreTrainedModel):
    def __init__(self, ip, port):
        pass