"""Main class, holding information about models and training/testing routines."""

import torch
import numpy as np
from ..consts import BENCHMARK
from ..utils import cw_loss
torch.backends.cudnn.benchmark = BENCHMARK

from .witch_base import _Witch

class comput_pass_loss():
    def __int__(self,poison_grad, target_grad, target_gnorm):
        self.poison_gard = poison_grad
        self.target_grad = target_grad
        self.target_gnorm = target_gnorm
    def _passenger_loss(self, poison_grad, target_grad, target_gnorm):
        """Compute the blind passenger loss term."""
        passenger_loss = 0
        poison_norm = 0

        indices = torch.arange(len(target_grad))

        SIM_TYPE = ['similarity']
        for i in indices:
            if self.args.loss in ['scalar_product', *SIM_TYPE]: # default
                passenger_loss -= (target_grad[i] * poison_grad[i]).sum()
            elif self.args.loss == 'cosine1':
                passenger_loss -= torch.nn.functional.cosine_similarity(target_grad[i].flatten(), poison_grad[i].flatten(), dim=0)
            elif self.args.loss == 'SE':
                passenger_loss += 0.5 * (target_grad[i] - poison_grad[i]).pow(2).sum()
            elif self.args.loss == 'MSE':
                passenger_loss += torch.nn.functional.mse_loss(target_grad[i], poison_grad[i])

            if self.args.loss in SIM_TYPE or self.args.normreg != 0: # default is true
                poison_norm += poison_grad[i].pow(2).sum() # Adjust passenger_loss

        passenger_loss = passenger_loss / target_gnorm  # this is a constant

        if self.args.loss in SIM_TYPE: # default is true
            passenger_loss = 1 + passenger_loss / poison_norm.sqrt()
        return passenger_loss

