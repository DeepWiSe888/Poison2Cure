"""Main class, holding information about models and training/testing routines."""

import torch
import numpy as np
import warnings
from ..utils import cw_loss
from ..consts import NON_BLOCKING, BENCHMARK
from .domain_adaption_tools import ForeverDataIterator
from tqdm import tqdm
torch.backends.cudnn.benchmark = BENCHMARK

class _Witch():
    """Brew poison with given arguments.

    Base class.

    This class implements _brew(), which is the main loop for iterative poisoning.
    New iterative poisoning methods overwrite the _define_objective method.

    Noniterative poison methods overwrite the _brew() method itself.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize a model with given specs..."""
        self.args, self.setup = args, setup
        self.auth_weight = args.auth_weight
        self.retain = True if self.args.ensemble > 1 and self.args.local_rank is None else False # default args.local_rank=None
        self.stat_optimal_loss = None

    """ BREWING RECIPES """

    def brew(self, victim, kettle):
        """Recipe interface."""

        poison_delta,poison_loss_value = self._brew(victim, kettle)

        return poison_delta,poison_loss_value

    def _brew(self, victim, kettle):
        """Run generalized iterative routine."""
        print(f'Starting brewing procedure ...')
        self._initialize_brew(victim, kettle)

        poisons, scores = [], torch.ones(self.args.restarts) * 10_000

        for trial in range(self.args.restarts): # default args.restarts = 1
            poison_delta, target_losses, poison_loss_value = self._run_trial(victim, kettle)
            scores[trial] = target_losses
            poisons.append(poison_delta.detach())
            if self.args.dryrun:
                break

        optimal_score = torch.argmin(scores)
        self.stat_optimal_loss = scores[optimal_score].item()
        print(f'Poisons with minimal target loss {self.stat_optimal_loss:6.4e} selected.')
        poison_delta = poisons[optimal_score]

        # ==========================
        if self.args.attackoptim == 'noPGD':
            poison_flag = True
            AP_FT_loader = kettle.AP_FT_loader
            for FT_data, _, _ in AP_FT_loader:
                FT_data = FT_data.to(**self.setup)
            if poison_flag:
                poison_delta_temp = self._nopgd_step(poison_delta, FT_data, self.tau0)
            else:
                poison_delta_temp = self._nopgd_step(FT_data, FT_data, self.tau0)
            poison_delta.data = poison_delta_temp.detach().to('cpu')
        # ==========================

        return poison_delta, poison_loss_value


    def _initialize_brew(self, victim, kettle):
        """Implement common initialization operations for brewing."""
        # victim.eval(dropout=True)
        
        # Compute target gradients

        # Loss function
        grad_UE = self.args.craft_with_UE_eval
        # grad_UE = True
        if grad_UE:
            self.target_grad, self.target_gnorm = victim.grad_UE_target_loss(self.auth_weight, kettle.UE_valid_auth_loader, kettle.UE_valid_priv_loader)
        else:
            self.target_grad, self.target_gnorm = victim.grad_UE_target_loss(self.auth_weight, kettle.FT_auth_loader,kettle.FT_priv_loader)
            if self.args.ensemble > 1:
                self.valid_target_grad, self.valid_target_gnorm = victim.grad_UE_target_loss_valid(self.auth_weight, kettle.FT_auth_loader, kettle.FT_priv_loader)
        self.target_clean_grad = None
        self.tau0 = self.args.tau * (self.args.batch_size_UE / 512) / self.args.ensemble
        print(f'Using tau0 = {self.tau0} for brewing.')


    def _run_trial(self, victim, kettle):
        """Run a single trial."""
        poison_delta = kettle.initialize_poison()
        # print(f'A!!!: poison_delta.grad is {poison_delta.grad}')
        AP_FT_loader = kettle.AP_FT_loader
        AP_PT_loader = kettle.AP_PT_loader

        if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
            poison_delta.requires_grad_()
            if self.args.attackoptim in ['Adam', 'signAdam']: # default args.attackoptim is signAdam
                att_optimizer = torch.optim.SGD([poison_delta], lr=self.tau0, momentum=0.9, weight_decay=0)
            else:
                raise NotImplementedError('Unknown attack optimizer.')
            
            if self.args.scheduling:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[self.args.attackiter // 2.667, self.args.attackiter // 1.6,
                    self.args.attackiter // 1.142], gamma=0.1)
            poison_delta.grad = torch.zeros_like(poison_delta)
            # print(f'C!!!: poison_delta.grad is {poison_delta.grad}')
        elif self.args.attackoptim == 'PGD':
            poison_bounds = None
        elif self.args.attackoptim == 'noPGD':
            poison_bounds = None
            poison_delta.requires_grad_()
            att_optimizer = torch.optim.Adam([poison_delta], lr=self.tau0, weight_decay=0)  # tau0
            if self.args.scheduling:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[self.args.attackiter // 2.667, self.args.attackiter // 1.6,
                                                                                            self.args.attackiter // 1.142], gamma=0.1)
            poison_delta.grad = torch.zeros_like(poison_delta)
        else:
            raise NotImplementedError('Unknown attack optimizer.')

        poison_loss_value = []

        for step in tqdm(range(self.args.attackiter)):
            # poison_delta.requires_grad_()
            # print(f'J!!!: poison_delta.grad is {poison_delta.grad}')
            poison_losses = 0  # Misalignment loss between AP training and
            iter_AP_PT_loader = ForeverDataIterator(AP_PT_loader)
            
            # if step % 10 == 0:
                # victim.current_sur_idx = np.random.randint(0, self.args.ensemble)

            for FT_data, FT_label, FT_index in AP_FT_loader:
                FT_data = FT_data.to(**self.setup)
                FT_label = FT_label.to(**self.setup)
                PT_data, PT_label, PT_index = next(iter_AP_PT_loader)
                PT_data = PT_data.to(**self.setup)
                PT_label = PT_label.to(**self.setup)
                # Set FT_data, FT_label, FT_index, PT_data, PT_label to a batch
                batch = (FT_data, FT_label, FT_index, PT_data, PT_label, PT_index)
                # print(f'D!!!: poison_delta.grad is {poison_delta.grad}')
                # Passenger loss
                loss = self._batched_step(poison_delta, batch, victim, kettle)
                poison_losses += np.mean(loss)

                if self.args.dryrun:
                    break

            # Note that these steps are handled batch-wise for PGD in _batched_step
            # For the momentum optimizers, we only accumulate gradients for all poisons
            # and then use optimizer.step() for the update. This is math. equivalent
            # and makes it easier to let pytorch track momentum.
            if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']: # True
                # att_optimizer.zero_grad()
                if self.args.attackoptim in ['momPGD', 'signAdam']: # True
                    poison_delta.grad.sign_()
                    # print(f'E!!!: poison_delta.grad is {poison_delta.grad}')
                att_optimizer.step()
                # print(f'F!!!: poison_delta.grad is {poison_delta.grad}')
                if self.args.scheduling:
                    scheduler.step()
                # print(f'H!!!: poison_delta.grad is {poison_delta.grad}')
                att_optimizer.zero_grad()
                # print(f'I!!!: poison_delta.grad is {poison_delta.grad}')
                # if self.args.var_snr_constraint is not None:
                #     poison_delta = self._constraint_projection(poison_delta, FT_data)

            elif self.args.attackoptim == 'PGD':
                poison_delta.grad.sign_()
                att_optimizer.step()
                if self.args.scheduling:
                    scheduler.step()
                att_optimizer.zero_grad()
            elif self.args.attackoptim == 'noPGD':
                poison_delta.grad.sign_()
                att_optimizer.step()
                if self.args.scheduling:
                    scheduler.step()
                att_optimizer.zero_grad()
            else:
                raise NotImplementedError('Unknown attack optimizer.')

            poison_losses = poison_losses / len(AP_FT_loader)    # poison loss is the passenger loss
            poison_loss_value.append(poison_losses)
            if step % (self.args.attackiter // 10) == 0 or step == (self.args.attackiter - 1):
                print(f'\nIteration {step}: Target closure loss is {poison_losses:2.4f}')
                print(f'Detailed Loss {loss}')
                if self.args.dataset_name == 'data_for_validApprox':
                    continue
                if self.args.ensemble > 1:
                    temp_poison = poison_delta.detach().to(**self.setup)
                    # temp_poison does not require grad
                    for FT_data, FT_label, FT_index in AP_FT_loader:
                        FT_data = FT_data.to(**self.setup)
                        FT_label = FT_label.to(**self.setup)
                        PT_data, PT_label, PT_index = next(iter_AP_PT_loader)
                        PT_data = PT_data.to(**self.setup)
                        PT_label = PT_label.to(**self.setup)
                        # Set FT_data, FT_label, FT_index, PT_data, PT_label to a batch
                        batch = (FT_data, FT_label, FT_index, PT_data, PT_label, PT_index)
                        # print(f'D!!!: poison_delta.grad is {poison_delta.grad}')
                        # passenger loss
                        loss = self._batched_step(temp_poison, batch, victim, kettle, pure_test = True)
                    print(f'Alignment Loss on validate model {loss}')

            '''
            if self.args.step:  # default is False, 'Optimize the model for one epoch.'
                if self.args.clean_grad: # Compute the first-order poison gradient. default is false
                    victim.step(kettle, None, self.targets, self.true_classes) 
                else:
                    victim.step(kettle, poison_delta, self.targets, self.true_classes)
            '''

            if self.args.dryrun:
                break

        return poison_delta, poison_losses,poison_loss_value


    def _batched_step(self, poison_delta, batch, victim, kettle, pure_test=False):
        """Take a step toward minmizing the current target loss."""
        FT_data, FT_label, FT_index, PT_data, PT_label, PT_index = batch
        FT_index = FT_index.cpu().numpy().astype(np.int64).squeeze()
        PT_index = PT_index.cpu().numpy().astype(np.int64).squeeze()
        delta_slice = poison_delta.detach().to(**self.setup) # detach from previous computational graph.
        delta_slice.requires_grad_()

        poisoned_FT_data = FT_data + delta_slice


        # Perform differentiable data augmentation
        if self.args.paugment: # default is True
            # inputs = kettle.augment(inputs, randgen=randgen)
            raise NotImplementedError('Data augmentation not implemented yet.')

        # Define the loss objective and compute gradients
        if self.args.recipe == 'metapoison':
            closure = self._define_objective(poisoned_FT_data, FT_label, FT_index, PT_data, PT_label, PT_index,
                                             kettle.FT_auth_loader,
                                             kettle.FT_priv_loader)
        elif self.args.recipe == 'gradient-matching':
            closure = self._define_objective(poisoned_FT_data, FT_label, FT_index, PT_data, PT_label, PT_index)  # witch_matching.py::_define_objective(...)
        if pure_test:
            loss = victim.compute_valid(closure, self.valid_target_grad, self.valid_target_gnorm)  
        else:
            loss = victim.compute(closure, self.target_grad, self.target_gnorm)      # passenger loss
        # Update Step
        if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
            # print(f'poison_delta.grad.size() is {poison_delta.grad.size()}')
            # print(f'delta_slice.grad.size() is {delta_slice.grad.size()}')
            # print(f'FT_index size is {len(FT_index)}')
            # print(f'FT_index is {FT_index}')
            # print(f'poison_delta.grad.type() is {poison_delta.grad.type()}')
            # print(f'delta_slice.grad.type() is {delta_slice.grad.type()}')
            # print(f'poison_delta.grad is {poison_delta.grad}')
            if pure_test is False:
                poison_delta.grad = torch.zeros_like(delta_slice.grad).to(**self.setup)    # shape
                poison_delta.grad = delta_slice.grad.detach().squeeze() #.to(device=torch.device('cpu'))
            else:
                poison_delta.grad = torch.zeros_like(delta_slice).to(**self.setup)
        elif self.args.attackoptim == 'PGD':
            delta_slice = self._pgd_step(delta_slice, FT_data, self.tau0)
            poison_delta.data = delta_slice.detach().to('cpu')
        elif self.args.attackoptim == 'noPGD':
            if pure_test is False:
                poison_delta.grad = torch.zeros_like(delta_slice.grad).to(**self.setup)    # shape
                poison_delta.grad = delta_slice.grad.detach().squeeze() #.to(device=torch.device('cpu'))
            else:
                poison_delta.grad = torch.zeros_like(delta_slice).to(**self.setup)

        else: 
            raise NotImplementedError('Unknown attack optimizer.')

        return loss.numpy()

    def _define_objective(self):
        """Implement the closure here."""
        def closure(model, criterion, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            raise NotImplementedError()
            # return target_loss.item(), prediction.item()

    def _constraint_projection(self, poison_slice, FT_data):
        with torch.no_grad():
            if self.args.var_snr_constraint is not None:
                norm_poison_slice = torch.norm(poison_slice.data, dim=[1,2], keepdim=True)
                norm_FT_data = torch.norm(FT_data, dim=[1,2], keepdim=True)
                ratio_coefficient = 10.0**(0.1*self.args.var_snr_constraint/2.0)
                poison_slice.data = poison_slice.data / norm_poison_slice * torch.min(norm_poison_slice, norm_FT_data/ratio_coefficient)
        return poison_slice

    def _pgd_step(self, delta_slice, FT_data, tau):
        """PGD step."""
        with torch.no_grad():
            delta_slice.data -= delta_slice.grad.sign() * tau


        return delta_slice
    
    
    def _nopgd_step(self, delta_slice, FT_data, tau):
        """PGD step."""
        with torch.no_grad():
            # Gradient Step

            # delta_slice.data -= delta_slice.grad * tau

            # delta_slice.data -= delta_slice.grad.sign() * tau

            # Projection Step
            # Compute the delta_slice norm
            if self.args.dfs_ub_constraint is not None:
                signal_length = delta_slice.size(1)
                rfft_delta_slice = torch.fft.rfft(delta_slice,n=signal_length, dim = 1)
                rfft_delta_slice[:,-self.args.dfs_ub_constraint:,:] = 0
                delta_slice.data = torch.fft.irfft(rfft_delta_slice,n=signal_length, dim = 1)

            if self.args.var_snr_constraint is not None:
                norm_delta_slice = torch.norm(delta_slice.data, dim=[1,2], keepdim=True)
                norm_FT_data = torch.norm(FT_data, dim=[1,2], keepdim=True)
                ratio_coefficient = 10.0**(0.1*self.args.var_snr_constraint/2.0)
                delta_slice.data = delta_slice.data / norm_delta_slice * torch.min(norm_delta_slice, norm_FT_data/ratio_coefficient)

        return delta_slice
