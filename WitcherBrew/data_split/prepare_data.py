'''
prepare your data
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import scipy.io
import numpy as np
import os
from data_split.seq_dataset_all import (SequenceDataset, split_dataset, load_data, load_data_all, select_domain, split_data_for_all,
                                        split_UE_dataset, add_intend_label, save_to_file)
from data_split.seq_dataset_spl import split_dataset_each
from data_split.domain_adaption_tools import ForeverDataIterator

DEVICE_ID = 0
SEED = 10
torch.cuda.set_device(DEVICE_ID)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# DEBUG_MODE = False
# try:
#     if DEBUG_MODE:
#         pydevd_pycharm.settrace('10.22.64.228', port=12340, stdoutToServer=True, stderrToServer=True)
# except:
#     pass


# Please set your path !!!!
FILE_PATH = '/home/lixin/PythonCode/PoisonWitch/Data/Py_Data/v7G'
FT_size_for_FT_list = [0,1,2,3]
sel_domain = 0
for FT_size_for_FT in FT_size_for_FT_list:
    # Please set your path !!!!
    FILE_SAVE_PATH = '/home/lixin/PythonCode/RemotePath/data_for_poison15D/data_for_poison' + str(sel_domain) + '_FTsize' + str(FT_size_for_FT)
    if not os.path.exists(FILE_SAVE_PATH):
        os.makedirs(FILE_SAVE_PATH)

    MAX_SEQ_LEN = None  # Maximum sequence length
    DATA_VEC_LEN = 370  # Length of the data vector
    NUM_CLASSES = 10    # Number of classes
    FLAG_Retrain = True
    FLAG_CrossDomain_FT = True
    FLAG_FT = True
    DEVICE = torch.device('cuda')
    max_epoch = 200
    FT_epoch = 50
    FT_ratio = 0.2
    PT_FT_ratio = 2
    BATCH_SIZE_train = 64

    auth_act_list = [0, 1, 2, 3, 6, 7, 8, 9]
    priv_act_list = [4, 5]
    priv_act_intend_label = [[],[],[],[], [8], [3], [],[],[],[]]
    # Custom Dataset

    all_sequences, all_labels, all_domains = load_data_all(FILE_PATH)
    Domain = [i for i in range(16)]
    tr_r = [0.9]*10
    ft_r = [0.0]*10
    per_class_FT_size_list= [[10] * NUM_CLASSES,[20] * NUM_CLASSES,[30] * NUM_CLASSES,[40] * NUM_CLASSES,[50] * NUM_CLASSES,[70] * NUM_CLASSES]
    # for i in range(6):    # set priv action as 0 or not
    #     per_class_FT_size_list[i][4] = 0
    #     per_class_FT_size_list[i][5] = 0
    # per_class_FT_size = per_class_FT_size_list[2]
    per_class_FT_size = per_class_FT_size_list[FT_size_for_FT]
    (target_x, target_y, target_d, source_x,
         source_y, source_d) = select_domain(all_sequences, all_labels, all_domains, sel_domain)
    (train_sequences, train_labels, train_domains,
         test_sequences_source, test_labels_source, test_domains_source) = split_data_for_all(source_x, source_y, source_d, tr_r)

    # adjust batch size
    if train_sequences.shape[0] % BATCH_SIZE_train != 1:
        BATCH_SIZE = BATCH_SIZE_train
    else:
        BATCH_SIZE = BATCH_SIZE_train - 2

    pre_train_dataset = SequenceDataset(train_sequences, train_labels)
    test_dataset = SequenceDataset(target_x, target_y)
    test_dataset_source = SequenceDataset(test_sequences_source, test_labels_source)
    train_size = len(pre_train_dataset)
    test_size = len(test_dataset)
    test_size_source = len(test_dataset_source)

    FT_size = sum(per_class_FT_size)          # FT number
    pure_test_size = test_size - FT_size      # test number
    # FT dataset from target domain
    FT_dataset, pure_test_dataset = split_dataset_each(test_dataset, NUM_CLASSES, per_class_FT_size, hold_class = [])

    # AP_PT_dataset
    per_class_PT_size = int(sum(per_class_FT_size) / NUM_CLASSES * PT_FT_ratio)
    PT_dataset, _ = split_dataset(pre_train_dataset, NUM_CLASSES, per_class_PT_size)

    FT_loader = DataLoader(FT_dataset, batch_size=BATCH_SIZE, shuffle=True)
    PT_loader = DataLoader(PT_dataset, batch_size=BATCH_SIZE*PT_FT_ratio, shuffle=True)
    PT_loader_iter = ForeverDataIterator(PT_loader, device=DEVICE)
    test_loader_source = DataLoader(test_dataset_source, batch_size=BATCH_SIZE, shuffle=False)
    FT_all_loader = DataLoader(ConcatDataset([FT_dataset, PT_dataset]),batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(pure_test_dataset, batch_size=BATCH_SIZE, shuffle=False)                         #

    # split data into auth data and priv data
    UE_auth_dataset, UE_priv_dataset = split_UE_dataset(pure_test_dataset, NUM_CLASSES, auth_act_list, priv_act_list)
    # save data
    save_to_file(FT_dataset, file_dir=FILE_SAVE_PATH, file_name='AP_FT')
    save_to_file(PT_dataset, file_dir=FILE_SAVE_PATH, file_name='AP_PT')
    save_to_file(UE_auth_dataset, file_dir=FILE_SAVE_PATH, file_name='UE_valid_auth')
    save_to_file(UE_priv_dataset, file_dir=FILE_SAVE_PATH, file_name='UE_valid_priv_original')
    # change the label
    add_intend_label(UE_priv_dataset, priv_act_intend_label)
    # save data
    save_to_file(UE_priv_dataset, file_dir=FILE_SAVE_PATH, file_name='UE_valid_priv')


