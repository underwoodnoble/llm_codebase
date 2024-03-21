from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from dataclasses import dataclass, field
from torch import FloatTensor

@dataclass
class RewardModelOutput(BaseModelOutput):
    lm_logits: FloatTensor = None
    rm_logits: FloatTensor = None
    rm_embeddings: FloatTensor = None

class RefModel(PreTrainedModel):
    def __init__(self, ip, port):
        pass