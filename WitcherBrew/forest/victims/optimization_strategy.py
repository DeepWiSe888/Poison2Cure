"""Optimization setups."""

from dataclasses import dataclass
from torch.optim.lr_scheduler import StepLR

BRITTLE_NETS = ['convnet', 'mobilenet', 'vgg', 'alexnet']  # handled with lower learning rate

def training_strategy(model_name, args):
    """Parse training strategy."""
    if args.optimization == 'basic_p2p':
        defs = Raw_GRU_OptStgy(model_name, args)
    elif args.optimization == 'basic_p2p_joint_FT':
        defs = Raw_GRU_OptStgy_JointFT(model_name, args)
    elif args.optimization == 'basic_p2p_joint_en':
        defs = Raw_GRU_OptStgy_EN(model_name, args)
    else:
        raise NotImplementedError(f'Unknown training strategy {args.optimization}.')
    return defs


@dataclass
class Strategy:
    """Default usual parameters, not intended for parsing."""

    epochs : int
    batch_size : int
    optimizer : str
    lr : float
    scheduler : str
    weight_decay : float
    augmentations : bool
    privacy : dict
    validate : int
    case_name: str

    def __init__(self, model_name, args):
        """Defaulted parameters. Apply overwrites from args."""
        if args.epochs is not None:
            self.epochs = args.epochs
        if args.noaugment:
            self.augmentations = None
        else:
            self.augmentations = args.data_aug


@dataclass
class Raw_GRU_OptStgy(Strategy):
    """Default usual parameters, defines a config object."""

    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        self.case_name = 'raw_gru'
        self.lr_mlp = 0.00005
        self.lr_gru = 0.000001


        self.lr_cls = 0.
        self.epochs = 20  # training epoch after poison:60
        self.batch_size = 64
        self.optimizer = 'AdamW'
        self.scheduler = 'none'    # None/none/linear/step
        self.weight_decay = 5e-4
        self.augmentations = None 
        self.privacy = dict(clip=None, noise=None)
        self.adversarial_steps = 0
        self.validate = 10
        self.weight_FT_PT = [1.,1.] # Wights of FT loss and PT loss

        super().__init__(model_name, args)

class Raw_GRU_OptStgy_JointFT(Strategy): # Optimizer for const
    """Default usual parameters, defines a config object."""

    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        self.case_name = 'raw_gru'

        self.lr_mlp = 0.00005
        self.lr_gru = 0.000001
        self.lr_cls = None

        self.epochs = 20
        self.batch_size = 64
        self.optimizer = 'AdamW'
        self.scheduler = 'none'
        self.weight_decay = 5e-4
        self.augmentations = None 
        self.privacy = dict(clip=None, noise=None)
        self.adversarial_steps = 0
        self.validate = 10
        self.weight_FT_PT =  [1.,1.]

        super().__init__(model_name, args)

class Raw_GRU_OptStgy_EN(Strategy): # Optimizer for EN

    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        self.case_name = 'raw_gru'

        self.lr_mlp = 0.00003
        self.lr_gru = 0.000001
        self.lr_cls = None

        self.epochs = 20
        self.batch_size = 64
        self.optimizer = 'AdamW'
        self.scheduler = 'none'
        self.weight_decay = 5e-4
        self.augmentations = None
        self.privacy = dict(clip=None, noise=None)
        self.adversarial_steps = 0
        self.validate = 10
        self.weight_FT_PT =  [1.,1.]

        super().__init__(model_name, args)
