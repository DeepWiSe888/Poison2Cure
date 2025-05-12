"""Main class, holding information about models and training/testing routines."""

import torch

from collections import OrderedDict
from ..utils import cw_loss

from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK
from .modules import MetaMonkey

from .witch_base import _Witch
import numpy as np

class WitchMetaPoison(_Witch):
    """Brew metapoison with given arguments.

    Note: This function does not work in single-model-multi-GPU mode, due to the weights being fixed to a single GPU.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """


    def _define_objective(self, poisoned_FT_data, FT_label, FT_index, PT_data, PT_label, PT_index, FT_auth_loader, FT_priv_loader):
        # def closure(model, criterion, optimizer, *args):
        def closure(model, model_name, criterion, weight_FT_PT, target_grad, target_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401

            model = MetaMonkey(model)

            for _ in range(2):
                poisoned_outputs_FT = model(poisoned_FT_data, FT_index, model.parameters)
                outputs_PT = model(PT_data, PT_index, model.parameters)

                poison_loss_FT = criterion(poisoned_outputs_FT, FT_label)
                loss_PT = criterion(outputs_PT, PT_label)

                poison_loss = poison_loss_FT * weight_FT_PT[0] + loss_PT * weight_FT_PT[1]

                poison_grad = torch.autograd.grad(poison_loss, model.parameters.values(), retain_graph=True, create_graph=True, only_inputs=True)


                current_lr = 0.01
                model.parameters = OrderedDict((name, param - current_lr * grad_part) for ((name, param), grad_part) in zip(model.parameters.items(), poison_grad))
            

            loss_priv = 0
            loss_auth = 0
            total_auth = 0
            total_priv = 0
            for data, label, indeces in FT_auth_loader:
                indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
                data = data.to(**self.setup)
                label = label.to(**self.setup)
                output = model(data,indeces,model.parameters)
                loss_auth += criterion(output, label)
                total_auth += label.size(0)
            for data, label, indeces in FT_priv_loader:
                indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
                data = data.to(**self.setup)
                label = label.to(**self.setup)
                output = model(data,indeces,model.parameters)
                loss_priv += criterion(output, label)
                total_priv += label.size(0)
            loss = loss_auth / total_auth - loss_priv / total_priv

            loss.backward()

            return loss.detach().cpu()
        return closure
