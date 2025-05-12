'''
Train a neural network for human activities recognition
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import scipy.io
import numpy as np
import os
import logging
from GRU_model import GRUClassifier
from seq_dataset_all import (SequenceDataset, split_dataset, load_data, load_data_all, select_domain, split_data_for_all,
                            split_UE_dataset, add_intend_label, save_to_file)
from seq_dataset_spl import split_dataset_each
from domain_adaption_tools import ForeverDataIterator

DEVICE_ID = 0
SEED = 10
torch.cuda.set_device(DEVICE_ID)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# PATH (Please adjust your path !!!!!)
FILE_PATH = '.../data_for_train'         # Data path (folder path)



# Creat save path
parent_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))
Save_root_dir = parent_directory + '/Res_GRU'   # Result and model path (folder path)
SAVE_PATH = Save_root_dir + "/Results_tr" + "_ft" + "_D0"
Load_PATH = SAVE_PATH
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Running parameters
max_epoch = 50              # Training epoch
FT_epoch = 200               # FT epoch
MAX_SEQ_LEN = None           # Maximum sequence length
DATA_VEC_LEN = 370           # Length of the data vector
NUM_CLASSES = 10             # Number of classes
BATCH_SIZE_train = 64
FLAG_Retrain = True         # Train a new model or load. True for the first time
FLAG_PreData = True          # Default: True. Prepare data or not.
FLAG_CrossDomain_FT = True
DEVICE = torch.device('cuda')


# Custom Dataset
auth_act_list = [0, 1, 2, 3, 6, 7, 8, 9]    # Without 4 and 5 because they are privacy action
priv_act_list = []                          # Default
priv_act_intend_label = [[],[],[],[], [], [], [],[],[],[]]  # Default
# Load data
all_sequences, all_labels, all_domains = load_data_all(FILE_PATH)
Domain = [i for i in range(16)]
tr_r = [0.9]*10
ft_r = [0.0]*10
FT_ratio = 0.2
PT_FT_ratio = 2
per_class_FT_size = [30] * NUM_CLASSES
per_class_FT_size[4] = 1
per_class_FT_size[5] = 1


sel_domain = 0
(target_x, target_y, target_d, source_x,
 source_y, source_d) = select_domain(all_sequences, all_labels, all_domains, sel_domain)

(train_sequences, train_labels, train_domains,
 test_sequences_source, test_labels_source, test_domains_source) = split_data_for_all(source_x, source_y, source_d, tr_r)

if train_sequences.shape[0] % BATCH_SIZE_train != 1:
    BATCH_SIZE = BATCH_SIZE_train
else:
    BATCH_SIZE = BATCH_SIZE_train - 2


# write output into txt file
logging_txt_train = os.path.join(SAVE_PATH, 'GRU_Output_Train.txt')
logging_txt_FT = os.path.join(SAVE_PATH, 'GRU_Output_FT.txt')

logger_Train = logging.getLogger("Train")
logger_FT = logging.getLogger("FT")

logger_Train.setLevel(logging.INFO)
logger_FT.setLevel(logging.INFO)

file_handler1 = logging.FileHandler(logging_txt_train, mode="w")
file_handler1.setFormatter(logging.Formatter("%(message)s"))
logger_Train.addHandler(file_handler1)

file_handler2 = logging.FileHandler(logging_txt_FT, mode="w")
file_handler2.setFormatter(logging.Formatter("%(message)s"))
logger_FT.addHandler(file_handler2)

# Data preparation
FLAG_NEW_GRU = True

train_dataset = SequenceDataset(train_sequences, train_labels)
test_dataset = SequenceDataset(target_x, target_y)
test_dataset_source = SequenceDataset(test_sequences_source, test_labels_source)
train_size = len(train_dataset)
test_size = len(test_dataset)
test_size_source = len(test_dataset_source)


def test(model, test_loader, add_name):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        class_total = [0] * NUM_CLASSES
        class_correct = [0] * NUM_CLASSES
        confusion_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES)
        tran_conf_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES)
        for sequences, labels, indeces in test_loader:
            sequences, labels = sequences.to(DEVICE).type(torch.float32), labels.to(DEVICE).type(torch.float32)
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            outputs = model(sequences, indeces)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
            # Acuracy
            c = (predicted == torch.argmax(labels, dim=1)).squeeze()
            for i in range(len(labels)):
                label = torch.argmax(labels[i])
                class_correct[label] += c[i].item()
                class_total[label] += 1
            # Confusion matrix
            for t, p in zip(torch.argmax(labels, dim=1).view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                tran_conf_matrix[p.long(), t.long()] += 1

        print(f'{add_name} Test accuracy: {100 * correct / total}%')
        for i in range(NUM_CLASSES):
            print(
                f'{add_name} -> Accuracy of Class {i} (Total {class_total[i]}): {100 * class_correct[i] / class_total[i]}%')

        temp_1 = confusion_matrix / confusion_matrix.sum(1, keepdim=True)
        temp_2 = confusion_matrix / confusion_matrix.sum(0, keepdim=True)
        accuracy_vec = temp_1.diag()
        precision_vec = temp_2.diag()
        f1_score_vec = 2 * accuracy_vec * precision_vec / (accuracy_vec + precision_vec)
        for i in range(NUM_CLASSES):
            print(f'{add_name} -> F1 score of Class {i}: {f1_score_vec[i]}')

        import matplotlib.pyplot as plt
        import seaborn as sns
        # Normalization
        normalized_conf_mat = confusion_matrix / confusion_matrix.sum(1, keepdim=True)
        normalized_tran_conf_matrix = tran_conf_matrix / tran_conf_matrix.sum(1, keepdim=True)

        # Confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(normalized_conf_mat.numpy(), annot=True, fmt='.2f', cmap='Blues')
        plt.title(add_name + ' confusion matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        # plt.show()
        # save fig
        plt.savefig(os.path.join(SAVE_PATH, add_name + 'confusion_matrix.png'))
        plt.close()

        # plot soft_conf_matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(normalized_tran_conf_matrix.numpy(), annot=True, fmt='.2f', cmap='crest')
        plt.title(add_name + ' precision matrix')
        plt.xlabel('True Label')
        plt.ylabel('Predicted Label')
        # plt.show()
        plt.savefig(os.path.join(SAVE_PATH, add_name + '_soft_confusion_matrix.png'))
        plt.close()



if FLAG_PreData: # prepare dataset
    FT_size = sum(per_class_FT_size)
    pure_test_size = test_size - FT_size
    print(f'FT_size: {FT_size}, per_class_FT_size: {per_class_FT_size}, pure_test_size: {pure_test_size}')
    FT_dataset, pure_test_dataset = split_dataset_each(test_dataset, NUM_CLASSES, per_class_FT_size, hold_class = priv_act_list)

    # AP_PT_dataset: Extract part of pretrain dataset for the FT.
    per_class_PT_size = int(sum(per_class_FT_size) / NUM_CLASSES * PT_FT_ratio)
    PT_dataset, _ = split_dataset(train_dataset, NUM_CLASSES, per_class_PT_size)

    FT_loader = DataLoader(FT_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(pure_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    PT_loader = DataLoader(PT_dataset, batch_size=BATCH_SIZE*PT_FT_ratio, shuffle=True)
    PT_loader_iter = ForeverDataIterator(PT_loader, device=DEVICE)

    test_loader_source = DataLoader(test_dataset_source, batch_size=BATCH_SIZE, shuffle=False)

    UE_auth_dataset, UE_priv_dataset = split_UE_dataset(pure_test_dataset, NUM_CLASSES, auth_act_list, priv_act_list)
    add_intend_label(UE_priv_dataset, priv_act_intend_label)
else:
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_source = DataLoader(test_dataset_source, batch_size=BATCH_SIZE, shuffle=False)

if FLAG_Retrain:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Model, Loss, Optimizer
model = GRUClassifier(DATA_VEC_LEN, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()

if FLAG_Retrain:   # train a new model or load trained model
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # Training Loop
    for epoch in range(max_epoch):  # number of epochs
        model.train()
        total = 0
        correct = 0
        for sequences, labels, indeces in train_loader:
            sequences, labels = sequences.to(DEVICE).type(torch.float32), labels.to(DEVICE).type(torch.float32)
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            # Forward pass
            outputs = model(sequences, indeces)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

        train_info = f'Epoch [{epoch + 1}/{max_epoch}], Loss: {loss.item():.4f}, Training accuracy: {100 * correct / total:.4f}%'
        print(train_info)
        logger_Train.info(train_info)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            pre_result = []
            real_result = []
            for sequences, labels, indeces in test_loader:
                sequences, labels = sequences.to(DEVICE).type(torch.float32), labels.to(DEVICE).type(torch.float32)
                indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
                outputs = model(sequences, indeces)
                _, predicted = torch.max(outputs.data, 1)
                _, labels_real = torch.max(labels.data, 1)
                a = predicted.data.cpu().numpy()
                total += labels.size(0)
                correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
                pre_result.extend(predicted.data.cpu().numpy())
                real_result.extend(labels_real.data.cpu().numpy())
            scipy.io.savemat(SAVE_PATH + '/target_pre_result.mat', {'pre_result': pre_result})
            scipy.io.savemat(SAVE_PATH + '/target_real_result.mat', {'real_result': real_result})
            target_info = f'Epoch [{epoch + 1}/{max_epoch}] Test accuracy of target domain: {100 * correct / total:.4f}%'
            print(target_info)
            logger_Train.info(target_info)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            pre_result = []
            real_result = []
            for sequences, labels, indeces in test_loader_source:
                sequences, labels = sequences.to(DEVICE).type(torch.float32), labels.to(DEVICE).type(torch.float32)
                indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
                outputs = model(sequences, indeces)
                _, predicted = torch.max(outputs.data, 1)
                _, labels_real = torch.max(labels.data, 1)
                total += labels.size(0)
                correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
                pre_result.extend(predicted.data.cpu().numpy())
                real_result.extend(labels_real.data.cpu().numpy())
            scipy.io.savemat(SAVE_PATH + '/source_pre_result.mat', {'pre_result': pre_result})
            scipy.io.savemat(SAVE_PATH + '/source_real_result.mat', {'real_result': real_result})
            source_info = f'Epoch [{epoch + 1}/{max_epoch}] Test accuracy of source domain: {100 * correct / total:.4f}%'
            print(source_info)
            logger_Train.info(source_info)

    # save model
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'raw_gru.pth'))
else:
    model.load_state_dict(torch.load(os.path.join(Load_PATH, 'raw_gru.pth')))
    # model.load_state_dict(torch.load(os.path.join(Load_PATH, 'raw_gru.pth')),strict=False)


# continue Train
# test(model, test_loader, 'Test 1')
# Differentiated learning rate for the GRU
for param in model.fc.parameters():
    param.requires_grad = False

FT_optimizer = torch.optim.AdamW([
    {'params': model.mlp.parameters(), 'lr': 0.001},
    {'params': model.gru.parameters(), 'lr': 0.0005}])

for epoch in range(FT_epoch):  # number of epochs
    model.train()

    for sequences, labels, indeces in FT_loader:
        sequences, labels = sequences.to(DEVICE).type(torch.float32), labels.to(DEVICE).type(torch.float32)
        indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
        # print(sequences.size())
        # print(labels.size())
        # Forward pass
        outputs = model(sequences, indeces)
        loss = criterion(outputs, labels)

        if FLAG_CrossDomain_FT:
            batch_src_seq, batch_src_label,batch_src_indeces = next(PT_loader_iter)
            batch_src_seq, batch_src_label = torch.tensor(batch_src_seq,dtype=torch.float32).to(DEVICE), \
                                             torch.tensor(batch_src_label,dtype=torch.float32).to(DEVICE)
            batch_src_indeces = batch_src_indeces.cpu().numpy().astype(np.int64).squeeze()
            outputs_src = model(batch_src_seq,batch_src_indeces)
            loss_src = criterion(outputs_src, batch_src_label)
            loss = loss + loss_src

        # Backward and optimize
        FT_optimizer.zero_grad()
        loss.backward()
        FT_optimizer.step()

    FT_loss_info = f'FT Epoch [{epoch + 1}/{FT_epoch}], Loss: {loss.item():.4f}'
    print(FT_loss_info)
    logger_FT.info(FT_loss_info)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        pre_result = []
        real_result = []
        for sequences, labels, indeces in test_loader:
            sequences, labels = sequences.to(DEVICE).type(torch.float32), labels.to(DEVICE).type(torch.float32)
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            outputs = model(sequences, indeces)
            _, predicted = torch.max(outputs.data, 1)
            _, labels_real = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
            pre_result.extend(predicted.data.cpu().numpy())
            real_result.extend(labels_real.data.cpu().numpy())
        scipy.io.savemat(SAVE_PATH + '/FT_pre_result.mat', {'pre_result': pre_result})
        scipy.io.savemat(SAVE_PATH + '/FT_real_result.mat', {'real_result': real_result})
        FT_target_info = f'FT Epoch [{epoch + 1}/{FT_epoch}] Test accuracy of target domain: {100 * correct / total}%'
        print(FT_target_info)
        logger_FT.info(FT_target_info)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        pre_result = []
        real_result = []
        for sequences, labels, indeces in test_loader_source:
            sequences, labels = sequences.to(DEVICE).type(torch.float32), labels.to(DEVICE).type(torch.float32)
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            outputs = model(sequences, indeces)
            _, predicted = torch.max(outputs.data, 1)
            _, labels_real = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
            pre_result.extend(predicted.data.cpu().numpy())
            real_result.extend(labels_real.data.cpu().numpy())
        scipy.io.savemat(SAVE_PATH + '/FT_source_pre_result.mat', {'pre_result': pre_result})
        scipy.io.savemat(SAVE_PATH + '/FT_real_source_result.mat', {'real_result': real_result})
        FT_source_info = f'FT Epoch [{epoch + 1}/{FT_epoch}] Test accuracy of source domain: {100 * correct / total:.4f}%'
        print(FT_source_info)
        logger_FT.info(FT_source_info)


torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'raw_gru_FT_0.pth'))
# test(model, test_loader, 'Test 2')
del model, target_x, target_y, target_d, source_x, source_y,\
    source_d, train_sequences, train_labels, train_domains,\
    test_sequences_source, test_labels_source, test_domains_source
torch.cuda.empty_cache()
