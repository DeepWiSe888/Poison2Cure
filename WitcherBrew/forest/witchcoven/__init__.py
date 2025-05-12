"""Interface for poison recipes."""
from .witch_matching import WitchGradientMatching
from .witch_metapoison import WitchMetaPoison
from .domain_adaption_tools import ForeverDataIterator
import torch


def Witch(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.recipe == 'gradient-matching':
        return WitchGradientMatching(args, setup)
    elif args.recipe == 'metapoison':
        print('Using MetaPoison recipe')
        return WitchMetaPoison(args, setup)
    else:
        raise NotImplementedError()

__all__ = ['Witch', 'ForeverDataIterator']
