"""Main class, holding information about models and training/testing routines."""

import torch
import numpy as np
from ..consts import BENCHMARK
from ..utils import cw_loss
torch.backends.cudnn.benchmark = BENCHMARK

from .witch_base import _Witch

class WitchGradientMatching(_Witch):
    """Brew passenger poison with given arguments.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def _define_objective(self, poisoned_FT_data, FT_label, FT_index, PT_data, PT_label, PT_index):
        """Implement the closure here."""
        if hasattr(self.args, 'flag_average_or_max'):
            if self.args.flag_average_or_max=='max':
                def closure(model, model_name, criterion, weight_FT_PT, target_grad, target_gnorm):
                    """This function will be evaluated on all GPUs."""  # noqa: D401
                    if model_name == 'raw_gru' or model_name.startswith('raw_gru'):
                        # 判断model是否是含有rnn结构
                        with torch.backends.cudnn.flags(enabled=False):
                            poisoned_outputs_FT = model(poisoned_FT_data, FT_index)
                            outputs_PT = model(PT_data, PT_index)

                            poison_loss_FT = criterion(poisoned_outputs_FT, FT_label)
                            loss_PT = criterion(outputs_PT, PT_label)

                            poison_loss = poison_loss_FT * weight_FT_PT[0] + loss_PT * weight_FT_PT[1]

                            params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
                            poison_grad = torch.autograd.grad(poison_loss, params_to_optimize, retain_graph=True, create_graph=True)

                            passenger_loss = self._passenger_loss(poison_grad, target_grad, target_gnorm)

                            # if self.args.centreg != 0: # 默认为0
                            #     passenger_loss = passenger_loss + self.args.centreg * poison_loss
                            return passenger_loss #.backward(retain_graph=self.retain)
                            # return passenger_loss.detach().cpu()
                    else:
                        raise NotImplementedError(f'Unknown model {model_name}.')
                    return closure
        def closure(model, model_name, criterion, weight_FT_PT, target_grad, target_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            if model_name in ['raw_gru','cnn'] or model_name.startswith('raw_gru'):
                # 判断model是否是含有rnn结构
                with torch.backends.cudnn.flags(enabled=False):
                    poisoned_outputs_FT = model(poisoned_FT_data, FT_index)
                    outputs_PT = model(PT_data, PT_index)

                    poison_loss_FT = criterion(poisoned_outputs_FT, FT_label)
                    loss_PT = criterion(outputs_PT, PT_label)

                    # poison_loss = poison_loss_FT * weight_FT_PT[0] + loss_PT * weight_FT_PT[1]
                    poison_loss = poison_loss_FT # * weight_FT_PT[0] + loss_PT * weight_FT_PT[1]

                    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
                    poison_grad = torch.autograd.grad(poison_loss, params_to_optimize, retain_graph=True, create_graph=True)

                    passenger_loss = self._passenger_loss(poison_grad, target_grad, target_gnorm)

                    # if self.args.centreg != 0: # 0
                    #     passenger_loss = passenger_loss + self.args.centreg * poison_loss
                    passenger_loss.backward(retain_graph=self.retain) # witch_base.py::31
                    return passenger_loss.detach().cpu()
            else:
                raise NotImplementedError(f'Unknown model {model_name}.')
        return closure

    def _passenger_loss(self, poison_grad, target_grad, target_gnorm):
        """Compute the blind passenger loss term."""
        passenger_loss = 0
        poison_norm = 0

        # _, indices = torch.topk(torch.stack([p.norm() for p in poison_grad], dim=0), 5)
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
                poison_norm += poison_grad[i].pow(2).sum() # adjust passenger_loss

        passenger_loss = passenger_loss / target_gnorm  # this is a constant

        if self.args.loss in SIM_TYPE: # default is true
            passenger_loss = 1 + passenger_loss / poison_norm.sqrt()
        # print poison_norm.sqrt()
        # if self.args.normreg != 0:
        #     passenger_loss += self.args.normreg * poison_norm.sqrt()
        # print(passenger_loss)
        return passenger_loss

