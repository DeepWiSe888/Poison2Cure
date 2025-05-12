import os
import random
import scipy.io
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import copy
def split_UE_dataset(dataset, num_classes, auth_act_list, priv_act_list):
    class_indices = {i: [] for i in range(num_classes)}

    # Iterate over the dataset to distribute indices into class buckets
    for idx, (_, label, _) in enumerate(dataset):
        label_idx = np.argmax(label)  # Assuming one-hot encoded labels
        class_indices[label_idx].append(idx)

    # Introduce randomness: Shuffle indices within each class
    FLAG_shuffle = True

    if FLAG_shuffle:
        for indices in class_indices.values():
            random.shuffle(indices)

    # Select X samples per class and separate the rest
    auth_act_indices = []
    priv_act_indices = []
    for label_idx, indices in class_indices.items():
        if label_idx in auth_act_list:
            auth_act_indices.extend(indices)
        elif label_idx in priv_act_list:
            priv_act_indices.extend(indices)

    # Create two datasets
    UE_auth_dataset = Subset_SequenceDataset(dataset, auth_act_indices)
    UE_priv_dataset = Subset_SequenceDataset(dataset, priv_act_indices)

    return UE_auth_dataset, UE_priv_dataset



def split_dataset(dataset, num_classes, x_per_class, hold_class=[]):
    class_indices = {i: [] for i in range(num_classes)}

    # Iterate over the dataset to distribute indices into class buckets
    for idx, (_, label, _) in enumerate(dataset):
        label_idx = np.argmax(label)  # Assuming one-hot encoded labels
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
    selected_dataset = Subset_SequenceDataset(dataset, selected_indices)
    remaining_dataset = Subset_SequenceDataset(dataset, remaining_indices)

    return selected_dataset, remaining_dataset

def calculate_variance(cur_seq):
    copy_seq = copy.deepcopy(cur_seq)
    seq_len = cur_seq.shape[0]
    for j in range(cur_seq.shape[0]):
        if cur_seq[j][0] == -1 and cur_seq[j][1] == -1:
            seq_len = j
            break
    x = copy_seq[0:seq_len]
    sum_x2 = np.sum(x**2, axis=0)
    mean_x2 = sum_x2 / seq_len
    return mean_x2.sum()

def split_dataset_each_withVarRemove(dataset, num_classes, x_per_class, remove_ratio, hold_class=[]):
    class_indices = {i: [] for i in range(num_classes)}
    power_vec = np.zeros(len(dataset))
    
    # Iterate over the dataset to distribute indices into class buckets
    for idx, (data, label, _) in enumerate(dataset):
        label_idx = np.argmax(label)  # Assuming one-hot encoded labels
        class_indices[label_idx].append(idx)
        power_vec[idx] = calculate_variance(data)


    # Introduce randomness: Shuffle indices within each class
    FLAG_shuffle = True

    if FLAG_shuffle:
        for indices in class_indices.values():
            np.random.shuffle(indices)

    # Select X samples per class and separate the rest
    selected_indices_filtered = []
    selected_indices_full = []
    remaining_indices = []
    indicator_of_using = []
    cutoff_bound_list = []
    for label_idx, indices in class_indices.items():
        this_x_per_class = x_per_class[label_idx]
        selected_indices_this_class = indices[:this_x_per_class]
        power_vec_this_class = power_vec[indices[:this_x_per_class]]
        half_remove_number = int(np.round(remove_ratio / 2 * x_per_class[label_idx]))
        # remove half_remove_number of the samples with the smallest power and half_remove_number of the samples with the largest power
        selected_indices_full.extend(copy.copy(selected_indices_this_class))
        sorted_indices = np.argsort(power_vec_this_class)
        selected_indices_this_class = [selected_indices_this_class[i] for i in sorted_indices[half_remove_number:-half_remove_number].tolist()]
        indicator_vec = np.zeros(this_x_per_class)
        indicator_vec[sorted_indices[half_remove_number:-half_remove_number]] = 1
        lower_bound = power_vec_this_class[sorted_indices[half_remove_number]] - 1e-4
        upper_bound = power_vec_this_class[sorted_indices[-half_remove_number-1]] + 1e-4
        tiled_bound = np.tile(np.array([lower_bound, upper_bound]), (this_x_per_class,1))
        if label_idx not in hold_class:
            indicator_of_using.extend(indicator_vec)
            cutoff_bound_list.extend(tiled_bound)
        selected_indices_filtered.extend(selected_indices_this_class)
        remaining_indices.extend(indices[this_x_per_class:])
    indicator_of_using = np.array(indicator_of_using)
    cutoff_bound_list = np.array(cutoff_bound_list)
    # Create two datasets
    selected_dataset_filtered = Subset_SequenceDataset(dataset, selected_indices_filtered)
    selected_dataset_full = Subset_SequenceDataset(dataset, selected_indices_full)
    remaining_dataset = Subset_SequenceDataset(dataset, remaining_indices)

    return selected_dataset_filtered, selected_dataset_full, indicator_of_using, cutoff_bound_list, remaining_dataset


def split_dataset_each(dataset, num_classes, x_per_class, hold_class=[]):
    class_indices = {i: [] for i in range(num_classes)}
    # Iterate over the dataset to distribute indices into class buckets
    for idx, (data, label, _) in enumerate(dataset):
        label_idx = np.argmax(label)  # Assuming one-hot encoded labels
        class_indices[label_idx].append(idx)

    # Introduce randomness: Shuffle indices within each class
    FLAG_shuffle = True

    if FLAG_shuffle:
        for indices in class_indices.values():
            np.random.shuffle(indices)

    # Select X samples per class and separate the rest
    selected_indices = []
    remaining_indices = []
    for label_idx, indices in class_indices.items():
        if len(hold_class) > 0:
            if label_idx in hold_class:
                this_x_per_class = 0
            else:
                this_x_per_class = x_per_class[label_idx]
        else:
            this_x_per_class = x_per_class[label_idx]
        selected_indices.extend(indices[:this_x_per_class])
        remaining_indices.extend(indices[this_x_per_class:])

    # Create two datasets
    selected_dataset = Subset_SequenceDataset(dataset, selected_indices)
    remaining_dataset = Subset_SequenceDataset(dataset, remaining_indices)

    return selected_dataset, remaining_dataset

# Load data from .mat file
def load_data(file_dir, data_name):
    file_list = os.listdir(file_dir)
    file_list = [file for file in file_list if file.startswith(data_name+'_for_py_dataset_p_')]
    file_list.sort()
    print(file_list)

    all_sequences = []
    all_labels = []
    for file in file_list:
        file_path = os.path.join(file_dir, file)
        print(file_path)
        mat = scipy.io.loadmat(file_path)
        sequences = mat['data']
        labels = mat['label']
        # print(sequences.shape)
        # print(labels.shape)
        all_sequences.append(sequences[0])
        all_labels.append(labels[0])

    all_sequences = np.concatenate(all_sequences, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(all_sequences.shape)
    print(all_labels.shape)
    return all_sequences, all_labels


def load_data_all(file_dir):
    file_list = os.listdir(file_dir)
    file_list.sort()
    print(file_list)

    all_sequences = []
    all_labels = []
    all_domains = []
    for file in file_list:
        file_path = os.path.join(file_dir, file)
        print(file_path)
        mat = scipy.io.loadmat(file_path)
        sequences = mat['data']
        labels = mat['label']
        domains = mat['domain']

        all_sequences.append(sequences[0])
        all_labels.append(labels[0])
        all_domains.append(domains[0])

    all_sequences = np.concatenate(all_sequences, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_domains = np.concatenate(all_domains, axis=0)

    print(all_sequences.shape)
    print(all_labels.shape)
    print(all_domains.shape)
    return all_sequences, all_labels, all_domains

def select_domain(all_sequences, all_labels, all_domains, sel_domain):
    cache_ind_extract = np.where(all_domains == sel_domain)[0]
    target_x = all_sequences[cache_ind_extract]
    target_y = all_labels[cache_ind_extract]
    target_d = all_domains[cache_ind_extract]

    cache_ind_rest = np.where(all_domains != sel_domain)[0]
    source_x = all_sequences[cache_ind_rest]
    source_y = all_labels[cache_ind_rest]
    source_d = all_domains[cache_ind_rest]

    return target_x, target_y, target_d, source_x, source_y, source_d


def split_data_for_all(all_sequences, all_labels, all_domains, ratio):
    # action labels
    action = np.array([i for i in range(len(ratio))])
    d_in_data = np.unique(all_domains).astype(int)
    train_x = []
    train_y = []
    train_d = []
    test_x = []
    test_y = []
    test_d = []
    for d in d_in_data:
        cache_ind = np.where(all_domains == d)[0]
        data_cache = all_sequences[cache_ind]
        y_cache = all_labels[cache_ind]
        y_cache_num = [np.argmax(x[0]) for x in y_cache]

        for i in range(len(action)):
            ac = action[i]
            cache_ac_ind = np.where(y_cache_num == ac)[0]
            len_extract = int(len(cache_ac_ind) * ratio[i])
            cache_ac_ind_extract = np.copy(cache_ac_ind[0:len_extract])
            cache_ac_ind_rest = np.copy(cache_ac_ind[len_extract:len(cache_ac_ind)])

            train_x = np.append(train_x, data_cache[cache_ac_ind_extract])
            train_y = np.concatenate((train_y, np.copy(y_cache[cache_ac_ind_extract])), axis=0)
            train_d = np.concatenate((train_d, np.full((len(cache_ac_ind_extract)), d)), axis=0)
            test_x = np.append(test_x, data_cache[cache_ac_ind_rest])
            test_y = np.concatenate((test_y, np.copy(y_cache[cache_ac_ind_rest])), axis=0)
            test_d = np.concatenate((test_d, np.full((len(cache_ac_ind_rest)), d)), axis=0)
    train_x = np.concatenate(train_x, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    return train_x, train_y, train_d, test_x, test_y, test_d


# Load data from .mat file with h5py, used for the matlab data v7.3
def load_data_h5py(file_dir, data_name):
    file_list = os.listdir(file_dir)
    file_list = [file for file in file_list if file.startswith(data_name+'_for_py_dataset_p_')]
    file_list.sort()
    print(file_list)

    all_sequences = []
    all_labels = []
    for file in file_list:
        file_path = os.path.join(file_dir, file)
        print(file_path)
        mat = h5py.File(file_path, 'r')
        labels = np.array([i for i in range(len(mat['label']))], dtype=object).reshape(1,-1)
        sequences = np.array([i for i in range(len(mat['label']))], dtype=object).reshape(1,-1)
        sequences_cache = [mat[element[0]][:].transpose(1,0).astype(np.float64) for element in mat['data']]
        labels_cache = [mat[element[0]][:].transpose(1,0).astype(np.uint8) for element in mat['label']]

        for i in range(len(mat['label'])):
            labels[0,i] = labels_cache[i]
            sequences[0,i] = sequences_cache[i]
        # print(sequences.shape)
        # print(labels.shape)
        all_sequences.append(sequences[0])
        all_labels.append(labels[0])

    all_sequences = np.concatenate(all_sequences, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(all_sequences.shape)
    print(all_labels.shape)
    return all_sequences, all_labels


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        self.intend_labels = None
        self.seq_len_list = None
        self.determine_seq_len()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # print(self.sequences.shape)
        sequence = self.sequences[idx]
        label = self.labels[idx][0]
        if self.intend_labels is not None:
            intend_label = self.intend_labels[idx][0]
            return sequence, label, intend_label
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
            seq_len = cur_seq.shape[0]
            for j in range(cur_seq.shape[0]):
                if cur_seq[j][0] == -1 and cur_seq[j][1] == -1:
                    seq_len = j
                    break
            self.seq_len_list.append(seq_len-1)
        self.seq_len_list = np.array(self.seq_len_list).reshape(-1,1)

class Subset_SequenceDataset(SequenceDataset):
    def __init__(self, dataset, indices):
        super(Subset_SequenceDataset, self).__init__(dataset.sequences[indices], dataset.labels[indices])
        if dataset.intend_labels is not None:
            self.intend_labels = dataset.intend_labels[indices]
        else:
            self.intend_labels = None
        if dataset.seq_len_list is not None:
            self.seq_len_list = dataset.seq_len_list[indices]
        else:
            self.seq_len_list = None
    def append(self, dataset):
        self.sequences = np.concatenate((self.sequences, dataset.sequences), axis=0)
        self.labels = np.concatenate((self.labels, dataset.labels), axis=0)
        self.seq_len_list = np.concatenate((self.seq_len_list, dataset.seq_len_list), axis=0)

def add_intend_label(dataset:SequenceDataset, intend_label_list:list):
    dataset.intend_labels = copy.deepcopy(dataset.labels)
    for i in range(dataset.labels.shape[0]):
        label_idx = np.argmax(dataset.labels[i][0])
        if len(intend_label_list[label_idx]) > 0:
            intend_label_idx = intend_label_list[label_idx]
        else:
            intend_label_idx = label_idx
        dataset.intend_labels[i] = np.zeros_like(dataset.labels[i])
        dataset.intend_labels[i][0][intend_label_idx] = 1

def save_to_file(dataset:SequenceDataset, file_dir:str, file_name:str):
    file_path = os.path.join(file_dir, file_name + '_dataset.npz')
    if dataset.intend_labels is None:
        np.savez(file_path, data=dataset.sequences, label=dataset.labels)
    else:
        np.savez(file_path, data=dataset.sequences, label=dataset.labels, intend_label=dataset.intend_labels)