"""Definition for multiple victims that share a single GPU (sequentially)."""

import torch
import numpy as np
from collections import defaultdict

from ..utils import set_random_seed, average_dicts
from ..consts import BENCHMARK
from .context import GPUContext
torch.backends.cudnn.benchmark = BENCHMARK

from .victim_base import _VictimBase


class _VictimEnsemble(_VictimBase):
    """
    Implement model-specific code and behavior for multiple models on a single GPU.
    --> Running in sequential mode!
    """

    """ Methods to initialize a model."""

    def initialize(self, seed=None):
        if self.args.modelkey is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.modelkey
        set_random_seed(self.model_init_seed)
        print(f'Initializing ensemble from random key {self.model_init_seed}.')

        self.models, self.definitions, self.criterions, self.optimizers, self.schedulers, self.epochs = [], [], [], [], [], []
        self.model_name = []
        self.weight_FT_PT=None

        if self.args.ensemble_case == 'M1':
            for idx in range(self.args.ensemble):
                model_name = self.args.net[0]
                feat_len = self.args.feat_len_ensemble[idx]
                layer_num = self.args.layer_gru_ensemble[idx]
                if self.args.use_FT_model_flag:
                    str_model_name = model_name + '_F_' + str(feat_len) + '_L_' + str(layer_num) + '_FT'
                else:
                    str_model_name = model_name + '_F_' + str(feat_len) + '_L_' + str(layer_num)
                model_name = [model_name, feat_len, layer_num]
                model, defs, criterion, optimizer, scheduler, weight_FT_PT = self._initialize_model(model_name)
                model.to(**self.setup)
                self.model_name.append(str_model_name)
                self.models.append(model)
                self.definitions.append(defs)
                self.criterions.append(criterion)
                self.optimizers.append(optimizer)
                self.schedulers.append(scheduler)
                print(f'{model_name[0]} with feat len {model_name[1]} and layer num {model_name[2]} initialized as model {idx}')
            self.defs = self.definitions[0]
            self.weight_FT_PT = weight_FT_PT


            if self.args.use_FT_model_flag:
                self.valid_model_name = self.args.net[0] + '_True_FT'
            else:
                self.valid_model_name = self.args.net[0] + '_True'
            feat_len = self.args.true_feat_len
            layer_num = self.args.true_layer_gru
            model_name = self.args.net[0]
            model_name = [model_name, feat_len, layer_num]
            model, defs, criterion, optimizer, scheduler, weight_FT_PT = self._initialize_model(model_name)
            model.to(**self.setup)    
            self.valid_model = model
            self.valid_definition = defs
            self.valid_criterion = criterion
            self.valid_optimizer = optimizer
            self.valid_scheduler = scheduler
            print(f'VALID MODEL: true {model_name[0]} as model initialized')
            self.valid_defs = self.valid_definition
        elif self.args.ensemble_case == 'M':
            self.args.ensemble += 1
            self.args.feat_len_ensemble.append(64)
            self.args.layer_gru_ensemble.append(2)
            for idx in range(self.args.ensemble):
                model_name = self.args.net[0]
                feat_len = self.args.feat_len_ensemble[idx]
                layer_num = self.args.layer_gru_ensemble[idx]
                if self.args.use_FT_model_flag:
                    str_model_name = model_name + '_F_' + str(feat_len) + '_L_' + str(layer_num) + '_FT'
                else:
                    str_model_name = model_name + '_F_' + str(feat_len) + '_L_' + str(layer_num)
                model_name = [model_name, feat_len, layer_num]
                model, defs, criterion, optimizer, scheduler, weight_FT_PT = self._initialize_model(model_name)
                model.to(**self.setup)
                self.model_name.append(str_model_name)
                self.models.append(model)
                self.definitions.append(defs)
                self.criterions.append(criterion)
                self.optimizers.append(optimizer)
                self.schedulers.append(scheduler)
                print(f'{model_name[0]} with feat len {model_name[1]} and layer num {model_name[2]} initialized as model {idx}')
            self.defs = self.definitions[0]
            self.weight_FT_PT = weight_FT_PT

            # 以下加载一下攻击目标,即用来验证的模型
            if self.args.use_FT_model_flag:
                self.valid_model_name = self.args.net[0] + '_True_FT'
            else:
                self.valid_model_name = self.args.net[0] + '_True'
            feat_len = self.args.true_feat_len       #  64
            layer_num = self.args.true_layer_gru     #  2
            model_name = self.args.net[0]
            model_name = [model_name, feat_len, layer_num]
            model, defs, criterion, optimizer, scheduler, weight_FT_PT = self._initialize_model(model_name)
            model.to(**self.setup)
            self.valid_model = model
            self.valid_definition = defs
            self.valid_criterion = criterion
            self.valid_optimizer = optimizer
            self.valid_scheduler = scheduler
            print(f'VALID MODEL: true {model_name[0]} as model initialized')
            self.valid_defs = self.valid_definition

        elif self.args.ensemble_case == 'R':

            for idx in range(self.args.ensemble):
                model_name = self.args.net[0]
                feat_len = self.args.true_feat_len
                layer_num = self.args.true_layer_gru
                rand_init_idx = self.args.rand_init_list[idx]
                if self.args.use_FT_model_flag:
                    str_model_name = model_name + '_R_' + str(rand_init_idx) + '_FT'
                else:
                    str_model_name = model_name + '_R_' + str(rand_init_idx)
                model_name = [model_name, feat_len, layer_num]
                model, defs, criterion, optimizer, scheduler, weight_FT_PT = self._initialize_model(model_name)
                model.to(**self.setup)
                self.model_name.append(str_model_name)
                self.models.append(model)
                self.definitions.append(defs)
                self.criterions.append(criterion)
                self.optimizers.append(optimizer)
                self.schedulers.append(scheduler)
                print(f'{model_name[0]} with R{rand_init_idx}-th initial point initialized as model {idx}')
            self.defs = self.definitions[0]
            self.weight_FT_PT = weight_FT_PT

            model_name = self.args.net[0]
            if self.args.use_FT_model_flag:
                str_model_name = model_name + '_True_FT'
            else:
                str_model_name = model_name + '_True'
            self.valid_model_name = str_model_name
            feat_len = self.args.true_feat_len
            layer_num = self.args.true_layer_gru
            model_name = [model_name, feat_len, layer_num]
            model, defs, criterion, optimizer, scheduler, weight_FT_PT = self._initialize_model(model_name)
            model.to(**self.setup)    
            self.valid_model = model
            self.valid_definition = defs
            self.valid_criterion = criterion
            self.valid_optimizer = optimizer
            self.valid_scheduler = scheduler
            print(f'VALID MODEL: true {model_name[0]} as model initialized')
            self.valid_defs = self.valid_definition

        elif self.args.ensemble_case == 'S':
            for idx in range(self.args.ensemble):
                model_name = self.args.net[0]
                feat_len = self.args.true_feat_len
                layer_num = self.args.true_layer_gru
                stgr_epoh_idx = self.args.staggered_epoch_list[idx]
                if self.args.use_FT_model_flag:
                    str_model_name = model_name + '_S_' + str(stgr_epoh_idx) + '_FT'
                else:
                    str_model_name = model_name + '_S_' + str(stgr_epoh_idx)
                model_name = [model_name, feat_len, layer_num]
                model, defs, criterion, optimizer, scheduler, weight_FT_PT = self._initialize_model(model_name)
                model.to(**self.setup)
                self.model_name.append(str_model_name)
                self.models.append(model)
                self.definitions.append(defs)
                self.criterions.append(criterion)
                self.optimizers.append(optimizer)
                self.schedulers.append(scheduler)
                print(f'{model_name[0]} with S{stgr_epoh_idx}-th initial point initialized as model {idx}')
            self.defs = self.definitions[0]
            self.weight_FT_PT = weight_FT_PT

            model_name = self.args.net[0]
            if self.args.use_FT_model_flag:
                str_model_name = model_name + '_True_FT'
            else:
                str_model_name = model_name + '_True'
            self.valid_model_name = str_model_name
            feat_len = self.args.true_feat_len
            layer_num = self.args.true_layer_gru
            model_name = [model_name, feat_len, layer_num]
            model, defs, criterion, optimizer, scheduler, weight_FT_PT = self._initialize_model(model_name)
            model.to(**self.setup)
            self.valid_model = model
            self.valid_definition = defs
            self.valid_criterion = criterion
            self.valid_optimizer = optimizer
            self.valid_scheduler = scheduler
            print(f'VALID MODEL: true {model_name[0]} as model initialized')
            self.valid_defs = self.valid_definition

    def load_pretrained(self, model_dir):
        """Load a pretrained model."""
        import os
        # self.model.module.load_state_dict(torch.load(os.path.join(model_dir, self.model_name + '.pth')))   # Multiple GPUs
        for idx in range(len(self.model_name)):
            self.models[idx].load_state_dict(torch.load(os.path.join(model_dir, self.model_name[idx] + '.pth'),map_location='cuda:0'))
            print(f'Loaded pretrained model from {os.path.join(model_dir, self.model_name[idx] + ".pth")}')
        valid_model_dir = self.args.pretrained_dir_true
        self.valid_model.load_state_dict(torch.load(os.path.join(valid_model_dir, self.valid_model_name + '.pth'),map_location='cuda:0'))
        print(f'Loaded pretrained valid model from {os.path.join(valid_model_dir, self.valid_model_name + ".pth")}')
    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def _iterate_valid(self, kettle, poison_delta, max_epoch=None):

        stats = defaultdict(list)

        if max_epoch is None:
            max_epoch = self.defs.epochs

        def loss_fn(model, outputs, labels):
            return self.valid_criterion(outputs, labels)

        single_setup = (self.valid_model, self.valid_defs, self.valid_criterion, self.valid_optimizer, self.valid_scheduler)
        for self.epoch in range(max_epoch):
            print(f'Validating valid model at epoch {self.epoch} ...')
            self._step(kettle, poison_delta, loss_fn, self.epoch, stats, *single_setup)
            if self.args.dryrun:
                break
        return stats

    def _iterate(self, kettle, poison_delta, max_epoch=None):

        multi_model_setup = (self.models, self.definitions, self.criterions, self.optimizers, self.schedulers)

        # Only partially train ensemble for poisoning if no poison is present
        if max_epoch is None:
            max_epoch = self.defs.epochs
        if poison_delta is None and self.args.stagger:
            # stagger_list = [int(epoch) for epoch in np.linspace(0, max_epoch, self.args.ensemble)]
            # stagger_list = [int(epoch) for epoch in np.linspace(0, max_epoch, self.args.ensemble + 2)[1:-1]]
            stagger_list = [int(epoch) for epoch in range(self.args.ensemble)]
            print(f'Staggered pretraining to {stagger_list}.')
        else:
            stagger_list = [max_epoch] * self.args.ensemble

        run_stats = list()
        for idx, single_model in enumerate(zip(*multi_model_setup)):
            stats = defaultdict(list)
            model, defs, criterion, optimizer, scheduler = single_model

            # Move to GPUs
            model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

            def loss_fn(model, outputs, labels):
                return criterion(outputs, labels)
            for epoch in range(stagger_list[idx]):
                self._step(kettle, poison_delta, loss_fn, epoch, stats, *single_model)
                if self.args.dryrun:
                    break
            # Return to CPU
            if torch.cuda.device_count() > 1:
                model = model.module
            model.to(device=torch.device('cpu'))
            run_stats.append(stats)

        if poison_delta is None and self.args.stagger:
            average_stats = run_stats[-1]
        else:
            average_stats = average_dicts(run_stats)

        # Track epoch
        self.epochs = stagger_list

        return average_stats

    def step(self, kettle, poison_delta, poison_targets, true_classes):
        """Step through a model epoch. Optionally minimize target loss during this.

        This function is limited because it assumes that defs.batch_size, defs.max_epoch, defs.epochs
        are equal for all models.
        """
        multi_model_setup = (self.models, self.criterions, self.optimizers, self.schedulers)

        run_stats = list()
        for idx, single_model in enumerate(zip(*multi_model_setup)):
            model, criterion, optimizer, scheduler = single_model

            # Move to GPUs
            model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

            def loss_fn(model, outputs, labels):
                normal_loss = criterion(outputs, labels)
                model.eval()
                if self.args.adversarial != 0:
                    target_loss = 1 / self.defs.batch_size * criterion(model(poison_targets), true_classes)
                else:
                    target_loss = 0
                model.train()
                return normal_loss + self.args.adversarial * target_loss

            self._step(kettle, poison_delta, loss_fn, self.epochs[idx], defaultdict(list), *single_model)
            self.epochs[idx] += 1
            if self.epochs[idx] > self.defs.epochs:
                self.epochs[idx] = 0
                print('Model reset to epoch 0.')
                model, criterion, optimizer, scheduler = self._initialize_model()
            # Return to CPU
            if torch.cuda.device_count() > 1:
                model = model.module
            model.to(device=torch.device('cpu'))
            self.models[idx], self.criterions[idx], self.optimizers[idx], self.schedulers[idx] = model, criterion, optimizer, scheduler

    """ Various Utilities."""

    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""
        def apply_dropout(m):
            """https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/6."""
            if type(m) == torch.nn.Dropout:
                m.train()
        [model.eval() for model in self.models]
        if dropout:
            [model.apply(apply_dropout) for model in self.models]

    def reset_learning_rate(self):
        """Reset scheduler objects to initial state."""
        for idx in range(self.args.ensemble):
            _, _, _, optimizer, scheduler = self._initialize_model()
            self.optimizers[idx] = optimizer
            self.schedulers[idx] = scheduler

    def grad_UE_target_loss_valid(self, auth_weight, FT_auth_loader, FT_priv_loader):        
        return self.grad_UE_target_loss_FTver(auth_weight, FT_auth_loader, FT_priv_loader, self.valid_model, self.valid_criterion)
               

    def grad_UE_target_loss(self, auth_weight, FT_auth_loader, FT_priv_loader):
        grad_list, norm_list = [], []
        for idx in range(self.args.ensemble):
            with GPUContext(self.setup, self.models[idx]) as model:
                criterion = self.criterions[idx]
                grad, norm = self.grad_UE_target_loss_FTver(auth_weight, FT_auth_loader, FT_priv_loader, model, criterion)
                grad_list.append(grad)
                norm_list.append(norm)
        return grad_list, norm_list


    def grad_UE_target_loss_FTver(self, auth_weight, FT_auth_loader, FT_priv_loader, model, criterion):
        loss_auth = 0.
        loss_priv = 0.
        total_auth = 0
        total_priv = 0
        model.train()
        for data, label, indeces in FT_auth_loader:
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            data = data.to(**self.setup)
            label = label.to(**self.setup)
            output = model(data,indeces)
            loss_auth += criterion(output, label)
            total_auth += label.size(0)
        for data, label, indeces in FT_priv_loader:
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            data = data.to(**self.setup)
            label = label.to(**self.setup)
            output = model(data,indeces)
            loss_priv += criterion(output, label)
            total_priv += label.size(0)
        loss = loss_auth / total_auth - loss_priv / total_priv
        print(f'loss_auth: {loss_auth/total_auth}, loss_priv: {loss_priv/total_priv}, total_loss: {loss}')
        
        params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
        gradients = torch.autograd.grad(loss, params_to_optimize, only_inputs=True, allow_unused=True)
        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return gradients, grad_norm
    
    def gradient(self, images, labels, external_criterion=None):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        grad_list, norm_list = [], []
        for model, criterion in zip(self.models, self.criterions):
            with GPUContext(self.setup, model) as model:
                loss = criterion(model(images), labels)
                if external_criterion is None:
                    loss = criterion(model(images), labels)
                else:
                    loss = external_criterion(model(images), labels)
                grad_list.append(torch.autograd.grad(loss, model.parameters(), only_inputs=True, allow_unused=True))
                grad_norm = 0
                for grad in grad_list[-1]:
                    grad_norm += grad.detach().pow(2).sum()
                norm_list.append(grad_norm.sqrt())
        return grad_list, norm_list


    def compute(self, function, *args):
        """Compute function on all models.

        Function has arguments that are possibly sequences of length args.ensemble
        """
        outputs = []

        if hasattr(self.args, 'flag_average_or_max'):
            if self.args.flag_average_or_max=='max':
                output_values = []
                # tar_idx = self.current_sur_idx
                max_idx = 0
                max_value = -100
                max_output = None
                for idx, (model, criterion, optimizer) in enumerate(zip(self.models, self.criterions, self.optimizers)):

                    # with GPUContext(self.setup, model) as model:
                    model.to(**self.setup)
                    single_arg = [arg[idx] if hasattr(arg, '__iter__') else arg for arg in args] # 其实这句话对应在witch_base.py->_Witch()->_batched_step()->loss = victim.compute(...) 那里, 可以从那看到, 是在分别取出每一个 surrogate 对应地 target_grad和target_gnorm.
                    x = function(model, self.model_name[idx], criterion, self.weight_FT_PT, *single_arg)
                    output_values.append(x.detach().cpu())
                    if x.detach().cpu() > max_value:
                        max_value = x.detach().cpu()
                        if max_output is not None:
                            max_output.detach_()
                            torch.cuda.empty_cache()
                        max_output = x

                max_output.backward(retain_graph=True)
                max_output.detach().cpu()
                avg_output = torch.stack(output_values)
                return avg_output
        flag_average_or_max = 'stochastic'
        if flag_average_or_max == 'average':
            for idx, (model, criterion, optimizer) in enumerate(zip(self.models, self.criterions, self.optimizers)):
                with GPUContext(self.setup, model) as model:
                    single_arg = [arg[idx] if hasattr(arg, '__iter__') else arg for arg in args]
                    outputs.append(function(model, self.model_name[idx], criterion, self.weight_FT_PT, *single_arg))

            avg_output = torch.stack(outputs)
        elif flag_average_or_max == 'stochastic':
            tar_idx = np.random.choice(self.args.ensemble, 2, replace=False)
            for idx, (model, criterion, optimizer) in enumerate(zip(self.models, self.criterions, self.optimizers)):
                if idx not in tar_idx:
                    continue
                with GPUContext(self.setup, model) as model:
                    single_arg = [arg[idx] if hasattr(arg, '__iter__') else arg for arg in args]
                    outputs.append(function(model, self.model_name[idx], criterion, self.weight_FT_PT, *single_arg))
            avg_output = torch.stack(outputs)
        return avg_output
    
    def compute_valid(self, function, *args):
        # self.valid_model.eval()
        x = function(self.valid_model, self.valid_model_name, self.valid_criterion, self.weight_FT_PT, *args)
        # x.backward()
        return x.detach().cpu()