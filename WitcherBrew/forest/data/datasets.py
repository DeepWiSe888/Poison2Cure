"""Super-classes of common datasets to extract id information per image."""
import torch
import torchvision
from ..consts import *   # import all mean/std constants
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import glob
import random

from torchvision.datasets.imagenet import load_meta_file
from torchvision.datasets.utils import verify_str_arg

# Block ImageNet corrupt EXIF warnings
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def construct_datasets(dataset_root, data_dir, setup, normalize=False):
    """Construct datasets with appropriate transforms."""
    # Compute mean, std:
    if dataset_root in ['data_for_poison', 'data_for_validApprox']:
        # Load data
        AP_PT_data, AP_PT_label = load_npz_data(data_dir,'AP_PT')
        AP_PT_dataset = SequenceDataset(AP_PT_data, AP_PT_label, setup)
        AP_FT_data, AP_FT_label = load_npz_data(data_dir,'AP_FT')
        AP_FT_dataset = SequenceDataset(AP_FT_data, AP_FT_label, setup)

        UE_valid_auth_data, UE_valid_auth_label = load_npz_data(data_dir,'UE_valid_auth')
        UE_valid_auth_dataset = SequenceDataset(UE_valid_auth_data, UE_valid_auth_label, setup)

        UE_valid_priv_data, UE_valid_priv_label, UE_valid_priv_intendLabel = load_npz_data(data_dir,'UE_valid_priv', with_intend_label = True)
        UE_valid_priv_dataset = SequenceDataset(UE_valid_priv_data, UE_valid_priv_label, setup,intend_labels=UE_valid_priv_intendLabel)
    else:
        raise NotImplementedError('Unknown dataset_root: {}'.format(dataset_root))

    return AP_PT_dataset, AP_FT_dataset, UE_valid_auth_dataset, UE_valid_priv_dataset


def split_dataset(dataset, num_classes, x_per_class, hold_class=[]):
    class_indices = {i: [] for i in range(num_classes)}

    # Iterate over the dataset to distribute indices into class buckets
    for idx, (_, label) in enumerate(dataset):
        label_idx = torch.argmax(label).item()  # Assuming one-hot encoded labels
        class_indices[label_idx].append(idx)

    # Introduce randomness: Shuffle indices within each class
    FLAG_shuffle = True

    if FLAG_shuffle:
        for indices in class_indices.values():
            random.shuffle(indices)

    # Select X samples per class and separate the rest
    selected_indices = []
    remaining_indices = []
    for label_idx, indices in class_indices.items():
        if len(hold_class) > 0:
            if label_idx in hold_class:
                this_x_per_class = 0
            else:
                this_x_per_class = x_per_class
        else:
            this_x_per_class = x_per_class
        selected_indices.extend(indices[:this_x_per_class])
        remaining_indices.extend(indices[this_x_per_class:])

    # Create two datasets
    selected_dataset = Subset(dataset, selected_indices)
    remaining_dataset = Subset(dataset, remaining_indices)

    return selected_dataset, remaining_dataset   


def load_npz_data(file_dir, data_name, with_intend_label = False):
    # file_path = os.path.join(file_dir, file_name+'_dataset.npz')
    # np.savez(file_path, data=self.sequences, label=self.labels, intend_label=self.intend_labels)

    file_path = os.path.join(file_dir, data_name+'_dataset.npz')
    data = np.load(file_path, allow_pickle=True)
    sequences = data['data']
    labels = data['label']
    if with_intend_label:
        intend_labels = data['intend_label']
        return sequences, labels, intend_labels
    else:
        return sequences, labels


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, setup, intend_labels = None):
        self.sequences = sequences
        self.labels = labels
        self.intend_labels = intend_labels
        self.seq_len_list = None
        self.setup = setup
        self.data_mean = None
        self.data_std = None
        self.determine_seq_len()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # print(self.sequences.shape)
        sequence = self.sequences[idx]
        label = self.labels[idx][0]
        if self.intend_labels is not None:
            intend_label = self.intend_labels[idx][0]
            return sequence, label, self.seq_len_list[idx], intend_label
        if self.seq_len_list is not None:
            return sequence, label, self.seq_len_list[idx]
        else:
            return sequence, label

    def determine_seq_len(self):
        self.seq_len_list = []
        for i in range(self.sequences.shape[0]):
            cur_seq = self.sequences[i]
            # cur_seq is a 2D array, with shape (seq_len, feature_dim)
            # the len of cur_seq is the index of the first all -1 feature
            # determine the len of the cur_seq based on above rule
            seq_len = cur_seq.shape[0] - 1
            for j in range(cur_seq.shape[0]):
                if cur_seq[j][0] == -1 and cur_seq[j][1] == -1:
                    seq_len = j
                    break
            self.seq_len_list.append(seq_len)
        self.seq_len_list = np.array(self.seq_len_list).reshape(-1, 1)

class Subset(torch.utils.data.Subset):
    """Overwrite subset class to provide class methods of main class."""

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)

