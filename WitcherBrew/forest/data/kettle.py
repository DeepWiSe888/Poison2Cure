"""Data class, holding information about dataloaders and poison ids."""

import torch
from torch.utils.data import DataLoader
import numpy as np

import pickle

import datetime
import os
import warnings
import random
import PIL

from .datasets import construct_datasets, Subset
from data_split.seq_dataset_all import split_UE_dataset


from ..consts import PIN_MEMORY, BENCHMARK, DISTRIBUTED_BACKEND, SHARING_STRATEGY, MAX_THREADING
from ..utils import set_random_seed
torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


class Kettle():
    """Brew poison with given arguments.

    Data class.
    Attributes:
    - trainloader
    - validloader
    - poisonloader
    - poison_ids
    - trainset/poisonset/targetset

    Most notably .poison_lookup is a dictionary that maps image ids to their slice in the poison_delta tensor.

    Initializing this class will set up all necessary attributes.

    Other data-related methods of this class:
    - initialize_poison
    - export_poison

    """

    def __init__(self, args, batch_size, augmentations, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize with given specs..."""
        self.args, self.setup = args, setup
        self.batch_size = batch_size
        self.augmentations = augmentations
        # self.trainset, self.validset = self.prepare_data(normalize=False)
        self.num_classes = args.num_classes
        self.AP_PT_dataset, self.AP_FT_dataset, self.UE_valid_auth_dataset, self.UE_valid_priv_dataset, self.FT_auth_dataset, self.FT_priv_dataset = self.prepare_data()
        self.batch_size_AP = min(batch_size, len(self.AP_FT_dataset))
        self.batch_size_AP_PT = int(len(self.AP_PT_dataset) / (len(self.AP_FT_dataset) / self.batch_size_AP))
        if self.batch_size_AP < batch_size:
            print(f'!! Batch size for AP training reduced from {batch_size} to {self.batch_size_AP} due to dataset size limit of AP_FT being {len(self.AP_FT_dataset)}')
        self.batch_size_UE = args.batch_size_UE

        # num_workers = self.get_num_workers()
        num_workers = 0

        # Generate loaders:
        # PT dataset
        self.batch_size_AP_PT = 600
        self.AP_PT_loader = DataLoader(self.AP_PT_dataset, self.batch_size_AP_PT, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        print(f'batch_size_AP_PT is {self.batch_size_AP_PT}')
        print(f'AP_PT_loader has {len(self.AP_PT_loader)} batches. And the lost data is {len(self.AP_PT_dataset) - len(self.AP_PT_loader) * self.batch_size_AP_PT}/{len(self.AP_PT_dataset)}')
        self.batch_size_AP = 300
        self.AP_FT_loader = DataLoader(self.AP_FT_dataset, self.batch_size_AP, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        print(f'batch_size_AP_FT is {self.batch_size_AP}')
        print(f'AP_FT_loader has {len(self.AP_FT_loader)} batches. And the lost data is {len(self.AP_FT_dataset) - len(self.AP_FT_loader) * self.batch_size_AP}/{len(self.AP_FT_dataset)}')
        self.UE_valid_auth_loader = DataLoader(self.UE_valid_auth_dataset, self.batch_size_UE, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        self.UE_valid_priv_loader = DataLoader(self.UE_valid_priv_dataset, self.batch_size_UE, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

        self.FT_auth_loader = DataLoader(self.FT_auth_dataset, self.batch_size_UE, shuffle=False,
                                               drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        self.FT_priv_loader = DataLoader(self.FT_priv_dataset, self.batch_size_UE, shuffle=False,
                                               drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

    def get_num_workers(self):
        """Check devices and set an appropriate number of workers."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            max_num_workers = 4 * num_gpus
        else:
            max_num_workers = 4
        if torch.get_num_threads() > 1 and MAX_THREADING > 0:
            worker_count = min(min(2 * torch.get_num_threads(), max_num_workers), MAX_THREADING)
        else:
            worker_count = 0
        # worker_count = 200
        print(f'Data is loaded with {worker_count} workers.')
        return worker_count

    """ CONSTRUCTION METHODS """
    def prepare_data(self):
        
        AP_PT_dataset, AP_FT_dataset, UE_valid_auth_dataset, UE_valid_priv_dataset = \
                construct_datasets(self.args.dataset_name, self.args.dataset_dir, self.setup, normalize=False)
        FT_auth_dataset, FT_priv_dataset = split_UE_dataset(AP_FT_dataset, 10, self.args.auth_act_list, self.args.priv_act_list)

        # Train augmentations are handled separately as they possibly have to be backpropagated
        if self.augmentations != 'none' or self.args.paugment:
            raise NotImplementedError('Augmentations not implemented yet.')
        return AP_PT_dataset, AP_FT_dataset, UE_valid_auth_dataset, UE_valid_priv_dataset, FT_auth_dataset, FT_priv_dataset


    def initialize_poison(self, initializer=None):
        """Initialize according to args.init.

        Propagate initialization in distributed settings.
        """
        
        initializer = self.args.init # default is randn

        if initializer == 'all-zero':
            num_poison = len(self.AP_FT_dataset)
            sample_data, _, _ = self.AP_FT_dataset[0]
            init_poison_delta = torch.zeros(num_poison, *sample_data.shape).to(**self.setup)
        else:
            raise NotImplementedError()

        return init_poison_delta

    """ EXPORT METHODS """

    def export_poison(self, poison_delta, path=None, mode='automl'):
        """Export poisons in either packed mode (just ids and raw data) or in full export mode, exporting all images.

        In full export mode, export data into folder structure that can be read by a torchvision.datasets.ImageFolder

        In automl export mode, export data into a single folder and produce a csv file that can be uploaded to
        google storage.
        """
        if path is None:
            path = self.args.poison_path

        dm = torch.tensor(self.trainset.data_mean)[:, None, None]
        ds = torch.tensor(self.trainset.data_std)[:, None, None]

        def _torch_to_PIL(image_tensor):
            """Torch->PIL pipeline as in torchvision.utils.save_image."""
            image_denormalized = torch.clamp(image_tensor * ds + dm, 0, 1)
            image_torch_uint8 = image_denormalized.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8)
            image_PIL = PIL.Image.fromarray(image_torch_uint8.numpy())
            return image_PIL

        def _save_image(input, label, idx, location, train=True):
            """Save input image to given location, add poison_delta if necessary."""
            filename = os.path.join(location, str(idx) + '.png')

            lookup = self.poison_lookup.get(idx)
            if (lookup is not None) and train:
                input += poison_delta[lookup, :, :, :]
            _torch_to_PIL(input).save(filename)

        # Save either into packed mode, ImageDataSet Mode or google storage mode
        if mode == 'packed':
            data = dict()
            data['poison_setup'] = self.poison_setup
            data['poison_delta'] = poison_delta
            data['poison_ids'] = self.poison_ids
            data['target_images'] = [data for data in self.targetset]
            name = f'{path}poisons_packed_{datetime.date.today()}.pth'
            torch.save([poison_delta, self.poison_ids], os.path.join(path, name))

        elif mode == 'limited':
            # Save training set
            names = self.trainset.classes
            for name in names:
                os.makedirs(os.path.join(path, 'train', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'targets', name), exist_ok=True)
            for input, label, idx in self.trainset:
                lookup = self.poison_lookup.get(idx)
                if lookup is not None:
                    _save_image(input, label, idx, location=os.path.join(path, 'train', names[label]), train=True)
            print('Poisoned training images exported ...')

            # Save secret targets
            for enum, (target, _, idx) in enumerate(self.targetset):
                intended_class = self.poison_setup['intended_class'][enum]
                _save_image(target, intended_class, idx, location=os.path.join(path, 'targets', names[intended_class]), train=False)
            print('Target images exported with intended class labels ...')

        elif mode == 'full':
            # Save training set
            names = self.trainset.classes
            for name in names:
                os.makedirs(os.path.join(path, 'train', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'test', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'targets', name), exist_ok=True)
            for input, label, idx in self.trainset:
                _save_image(input, label, idx, location=os.path.join(path, 'train', names[label]), train=True)
            print('Poisoned training images exported ...')

            for input, label, idx in self.validset:
                _save_image(input, label, idx, location=os.path.join(path, 'test', names[label]), train=False)
            print('Unaffected validation images exported ...')

            # Save secret targets
            for enum, (target, _, idx) in enumerate(self.targetset):
                intended_class = self.poison_setup['intended_class'][enum]
                _save_image(target, intended_class, idx, location=os.path.join(path, 'targets', names[intended_class]), train=False)
            print('Target images exported with intended class labels ...')

        elif mode in ['automl-upload', 'automl-all', 'automl-baseline']:
            from ..utils import automl_bridge
            targetclass = self.targetset[0][1]
            poisonclass = self.poison_setup["poison_class"]

            name_candidate = f'{self.args.name}_{self.args.dataset}T{targetclass}P{poisonclass}'
            name = ''.join(e for e in name_candidate if e.isalnum())

            if mode == 'automl-upload':
                automl_phase = 'poison-upload'
            elif mode == 'automl-all':
                automl_phase = 'all'
            elif mode == 'automl-baseline':
                automl_phase = 'upload'
            automl_bridge(self, poison_delta, name, mode=automl_phase, dryrun=self.args.dryrun)

        elif mode == 'numpy':
            _, h, w = self.trainset[0][0].shape
            training_data = np.zeros([len(self.trainset), h, w, 3])
            labels = np.zeros(len(self.trainset))
            for input, label, idx in self.trainset:
                lookup = self.poison_lookup.get(idx)
                if lookup is not None:
                    input += poison_delta[lookup, :, :, :]
                training_data[idx] = np.asarray(_torch_to_PIL(input))
                labels[idx] = label

            np.save(os.path.join(path, 'poisoned_training_data.npy'), training_data)
            np.save(os.path.join(path, 'poisoned_training_labels.npy'), labels)

        elif mode == 'kettle-export':
            with open(f'kette_{self.args.dataset}{self.args.model}.pkl', 'wb') as file:
                pickle.dump([self, poison_delta], file, protocol=pickle.HIGHEST_PROTOCOL)

        elif mode == 'benchmark':
            foldername = f'{self.args.name}_{"_".join(self.args.net)}'
            sub_path = os.path.join(path, 'benchmark_results', foldername, str(self.args.benchmark_idx))
            os.makedirs(sub_path, exist_ok=True)

            # Poisons
            benchmark_poisons = []
            for lookup, key in enumerate(self.poison_lookup.keys()):  # This is a different order than we usually do for compatibility with the benchmark
                input, label, _ = self.trainset[key]
                input += poison_delta[lookup, :, :, :]
                benchmark_poisons.append((_torch_to_PIL(input), int(label)))

            with open(os.path.join(sub_path, 'poisons.pickle'), 'wb+') as file:
                pickle.dump(benchmark_poisons, file, protocol=pickle.HIGHEST_PROTOCOL)

            # Target
            target, target_label, _ = self.targetset[0]
            with open(os.path.join(sub_path, 'target.pickle'), 'wb+') as file:
                pickle.dump((_torch_to_PIL(target), target_label), file, protocol=pickle.HIGHEST_PROTOCOL)

            # Indices
            with open(os.path.join(sub_path, 'base_indices.pickle'), 'wb+') as file:
                pickle.dump(self.poison_ids, file, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            raise NotImplementedError()

        print('Dataset fully exported.')
