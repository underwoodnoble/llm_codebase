import numpy as np
from transformers.trainer_pt_utils import LabelSmoother



IGNORE_INDEX = LabelSmoother.ignore_index


class FixedKLController:
    """Fixed KL controller"""


    def __init__(self, kl_coef):
        self.value = kl_coef


    def update(self, current):
        pass

    
class AdaptiveKLController:
    """
    Adaptive KL controller
    """

    
    def __init__(self, init_kl_coef, target) -> None:
        """
        init_kl_coef: base kl coefficient
        """
        self.value = init_kl_coef
        self.target = target
    
    def update(self, current):
        proprotional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        mult = 1 + proprotional_error * 0.1
        self.value *= mult