"""Implement victim behavior, for single-victim, ensemble and stuff."""
import torch

from .victim_ensemble import _VictimEnsemble
from .victim_single import _VictimSingle
from .victim_single_mulStep import _VictimMultiStep

def Victim(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.ensemble == 1: # Default
        # _VictimMultiStep or _VictimSingle
        if hasattr(args, 'multiStep_flag'):
            if args.multiStep_flag is True:
                return _VictimMultiStep(args, setup)
        elif hasattr(args, 'ensemble_flag'):
            if args.ensemble_flag is True:
                return _VictimEnsemble(args, setup)
        return _VictimSingle(args, setup)
    elif args.ensemble > 1:
        if hasattr(args, 'multiStep_flag'):
            if args.multiStep_flag is True:
                return _VictimMultiStep(args, setup)
        return _VictimEnsemble(args, setup)


from .optimization_strategy import training_strategy
__all__ = ['Victim', 'training_strategy']
