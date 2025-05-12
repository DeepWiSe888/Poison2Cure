"""Repeatable code parts concerning optimization and training schedules."""


import torch
import numpy as np
from sklearn.metrics import f1_score
import datetime
from .utils import print_and_save_stats, pgd_step
from ..consts import NON_BLOCKING, BENCHMARK, DEBUG_TRAINING
from ..witchcoven import ForeverDataIterator
torch.backends.cudnn.benchmark = BENCHMARK


def get_optimizers(model, args, defs):
    """Construct optimizer as given in defs."""
    if defs.case_name == 'raw_gru':
        if defs.optimizer == 'AdamW':
            if defs.lr_cls is None:
                for param in model.fc.parameters():
                    param.requires_grad = False

                optimizer = torch.optim.AdamW([
                    {'params': model.mlp.parameters(), 'lr': defs.lr_mlp},
                    {'params': model.gru.parameters(), 'lr': defs.lr_gru}], weight_decay=defs.weight_decay)
            else:
                optimizer = torch.optim.AdamW([
                    {'params': model.mlp.parameters(), 'lr': defs.lr_mlp},
                    {'params': model.gru.parameters(), 'lr': defs.lr_gru},
                    {'params': model.fc.parameters(), 'lr': defs.lr_cls}], weight_decay=defs.weight_decay)
        else:
            raise NotImplementedError(f'Unknown optimizer {defs.optimizer}.')
    elif defs.case_name == 'cnn_test':
        optimizer = torch.optim.AdamW([
                    {'params': model.mlp.parameters(), 'lr': defs.lr_mlp},
                    {'params': model.cnn.parameters(), 'lr': defs.lr_cnn},
                    {'params': model.fc.parameters(), 'lr': defs.lr_fc}], weight_decay=defs.weight_decay)
    else:
        raise NotImplementedError(f'Unknown case name {defs.case_name}.')
    if defs.scheduler is None:
        scheduler = None
    elif defs.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[defs.epochs // 2.667, defs.epochs // 1.6,
                                                                     defs.epochs // 1.142], gamma=0.1)
    elif defs.scheduler == 'none':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[10_000, 15_000, 25_000], gamma=1)
    elif defs.scheduler == 'step':
        lambda_func = lambda epoch: 1.05 ** epoch if epoch < 20 else (1.05 ** 20) * (0.94 ** (epoch - 19))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)
        # Example: epochs=160 leads to drops at 60, 100, 140.
    return optimizer, scheduler

def calculate_correlation(model, UE_valid_auth_loader,UE_valid_priv_loader, kettle, AP_PT_loader, AP_FT_loader, criterion, poison_delta, defs, loss_fn):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
    loss_auth = 0.
    loss_priv = 0.
    total_auth = 0
    total_priv = 0
    passenger_loss = 0
    for data, label, indeces in UE_valid_auth_loader:
        indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
        data = data.to(**kettle.setup)
        label = label.to(**kettle.setup)
        output = model(data, indeces)
        loss_auth += criterion(output, label)
        total_auth += label.size(0)
    for data, label, indeces,_ in UE_valid_priv_loader:
        indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
        data = data.to(**kettle.setup)
        label = label.to(**kettle.setup)
        output = model(data, indeces)
        loss_priv += criterion(output, label)
        total_priv += label.size(0)
    loss_UE = loss_auth  - loss_priv    # auth_weight = 1

    # with torch.backends.cudnn.flags(enabled=False):
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    gradients_UE = torch.autograd.grad(loss_UE, params_to_optimize, only_inputs=True, allow_unused=True)
    

    gradients_UE_detached = [g.detach() for g in gradients_UE if g is not None]
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
    
    # ============================================================
    iter_AP_PT_loader = ForeverDataIterator(AP_PT_loader)
    loss_AP = 0.
    for FT_data, FT_label, FT_idx in AP_FT_loader:
        PT_data, PT_label, PT_idx = next(iter_AP_PT_loader)
        FT_data = FT_data.to(**kettle.setup)
        FT_label = FT_label.to(**kettle.setup)
        FT_idx = FT_idx.cpu().numpy().astype(np.int64).squeeze()
        PT_data = PT_data.to(**kettle.setup)
        PT_label = PT_label.to(**kettle.setup)
        PT_idx = PT_idx.cpu().numpy().astype(np.int64).squeeze()
        # Prep Mini-Batch
        # optimizer.zero_grad()
        # Add adversarial pattern
        if poison_delta is not None:
            poisoned_FT_data = FT_data + poison_delta.to(**kettle.setup)
        else:
            poisoned_FT_data = FT_data

        # Add data augmentation
        if defs.augmentations != 'none':  # defs.augmentations is actually a string, but it is False if --noaugment
            raise NotImplementedError('Augmentations not implemented for this model.')
            # inputs = kettle.augment(inputs)

        # Get loss
        outputs_FT = model(poisoned_FT_data, FT_idx)
        loss_FT = criterion(outputs_FT, FT_label)
        outputs_PT = model(PT_data, PT_idx)
        loss_PT = loss_fn(model, outputs_PT, PT_label)
        loss_AP +=  loss_FT
    # passenger_gard = torch.autograd.grad(epoch_loss_p, model.parameters(), retain_graph=True, create_graph=True)
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    gradients_AP = torch.autograd.grad(loss_AP, params_to_optimize, only_inputs=True, allow_unused=True)

    gradients_AP_detached = [g.detach() for g in gradients_AP if g is not None]
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
    # =============================================
    gradients_UE_detached_norm = 0
    gradients_AP_detached_norm = 0
    passenger_loss = 0.
    for grad in gradients_UE_detached:
        gradients_UE_detached_norm += grad.detach().pow(2).sum()
    gradients_UE_detached_norm = gradients_UE_detached_norm.sqrt()

    indices_grad_UE = torch.arange(len(gradients_UE_detached))
    for i_grad in indices_grad_UE:
        passenger_loss -= (gradients_UE_detached[i_grad] * gradients_AP_detached[i_grad]).sum()
        gradients_AP_detached_norm += gradients_AP_detached[i_grad].pow(2).sum()
    passenger_loss = passenger_loss / gradients_UE_detached_norm
    passenger_loss = 1 + passenger_loss / gradients_AP_detached_norm.sqrt()
    passenger_loss = passenger_loss.cpu().numpy()
    return passenger_loss



def run_step(kettle, poison_delta, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler, weight_FT_PT, ablation=False):

    AP_PT_loader = kettle.AP_PT_loader
    AP_FT_loader = kettle.AP_FT_loader
    # ===============================================================================
    model.train()
    passenger_flag = True
    if passenger_flag:
        passenger_loss = calculate_correlation(model, kettle.UE_valid_auth_loader, kettle.UE_valid_priv_loader, kettle, AP_PT_loader, AP_FT_loader, criterion, poison_delta, defs, loss_fn)
    else:
        passenger_loss = 0.
#===================================================================

    epoch_loss, total_preds, correct_preds = 0, 0, 0
    epoch_loss_p = 0
    if DEBUG_TRAINING:
        # debug time
        data_timer_start = torch.cuda.Event(enable_timing=True)
        data_timer_end = torch.cuda.Event(enable_timing=True)
        forward_timer_start = torch.cuda.Event(enable_timing=True)
        forward_timer_end = torch.cuda.Event(enable_timing=True)
        backward_timer_start = torch.cuda.Event(enable_timing=True)
        backward_timer_end = torch.cuda.Event(enable_timing=True)

        stats['data_time'] = 0
        stats['forward_time'] = 0
        stats['backward_time'] = 0

        data_timer_start.record()

    iter_AP_PT_loader = ForeverDataIterator(AP_PT_loader)    # device=DEVICE
    FT_label_np_list = []
    PT_label_np_list = []
    predictions_FT_np_list = []
    predictions_PT_np_list = []
    for FT_data, FT_label, FT_idx in AP_FT_loader:
        PT_data, PT_label, PT_idx = next(iter_AP_PT_loader)
        # print(FT_idx)
        # print(PT_idx)
        FT_data = FT_data.to(**kettle.setup)
        FT_label = FT_label.to(**kettle.setup)
        FT_idx = FT_idx.cpu().numpy().astype(np.int64).squeeze()
        PT_data = PT_data.to(**kettle.setup)
        PT_label = PT_label.to(**kettle.setup)
        PT_idx = PT_idx.cpu().numpy().astype(np.int64).squeeze()
        # Prep Mini-Batch
        model.train()
        optimizer.zero_grad()

        if DEBUG_TRAINING:
            data_timer_end.record()
            forward_timer_start.record()

        # Add adversarial pattern
        if poison_delta is not None:
            poisoned_FT_data = FT_data + poison_delta.to(**kettle.setup)
        else:
            poisoned_FT_data = FT_data

        # Add data augmentation
        if defs.augmentations != 'none':  # defs.augmentations is actually a string, but it is False if --noaugment
            raise NotImplementedError('Augmentations not implemented for this model.')
            # inputs = kettle.augment(inputs)


        # Get loss
        outputs_FT = model(poisoned_FT_data, FT_idx)
        loss_FT = loss_fn(model, outputs_FT, FT_label)
        outputs_PT = model(PT_data, PT_idx)
        loss_PT = loss_fn(model, outputs_PT, PT_label)
        loss =loss_FT # + weight_FT_PT[1] * loss_PT

        if DEBUG_TRAINING:
            forward_timer_end.record()
            backward_timer_start.record()
        loss.backward()
        optimizer.step()

        predictions_FT = torch.argmax(outputs_FT.data, dim=1) 
        predictions_PT = torch.argmax(outputs_PT.data, dim=1)
        total_preds += FT_label.size(0) + PT_label.size(0)
        correct_preds += ( (predictions_FT == torch.argmax(FT_label, dim=1)).sum().item() 
                            + (predictions_PT == torch.argmax(PT_label, dim=1)).sum().item() )

        # epoch_loss_p += loss
        epoch_loss += loss.item()

        FT_label_np_list.append(torch.argmax(FT_label, dim=1).cpu().numpy())
        PT_label_np_list.append(torch.argmax(PT_label, dim=1).cpu().numpy())
        predictions_FT_np_list.append(predictions_FT.cpu().numpy())
        predictions_PT_np_list.append(predictions_PT.cpu().numpy())

        if DEBUG_TRAINING:
            backward_timer_end.record()
            torch.cuda.synchronize()
            stats['data_time'] += data_timer_start.elapsed_time(data_timer_end)
            stats['forward_time'] += forward_timer_start.elapsed_time(forward_timer_end)
            stats['backward_time'] += backward_timer_start.elapsed_time(backward_timer_end)

            data_timer_start.record()

        if defs.scheduler == 'cyclic':
            scheduler.step()
        if kettle.args.dryrun:
            break
    label_np_list = FT_label_np_list + PT_label_np_list
    predictions_np_list = predictions_FT_np_list + predictions_PT_np_list
    label_np_list = np.concatenate(label_np_list, axis=0)
    predictions_np_list = np.concatenate(predictions_np_list, axis=0)
    f1 = f1_score(label_np_list, predictions_np_list,average='micro')   # weight/micro

    FT_label_np_list = np.concatenate(FT_label_np_list, axis=0)
    PT_label_np_list = np.concatenate(PT_label_np_list, axis=0)
    predictions_FT_np_list = np.concatenate(predictions_FT_np_list, axis=0)
    predictions_PT_np_list = np.concatenate(predictions_PT_np_list, axis=0)


    if defs.scheduler is not None:
        scheduler.step()


    # if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
    target_test = kettle.args.valid_with_UE_eval
    # target_test = True
    if target_test:
        (auth_accuracy, priv_intend_accuracy, priv_risk_accuracy, auth_loss, priv_poisoned_loss, f1_auth, f1_inte_priv, f1_risk_priv,
         auth_label_np_list, predictions_auth_np_list,priv_risk_label_np_list, predictions_priv_np_list)\
            = run_validation(model, criterion, kettle.UE_valid_auth_loader, kettle.UE_valid_priv_loader, kettle.setup, kettle.args.dryrun)
    else:
        (auth_accuracy, priv_intend_accuracy, priv_risk_accuracy, auth_loss, priv_poisoned_loss, f1_auth, f1_inte_priv,
         f1_risk_priv,auth_label_np_list, predictions_auth_np_list,priv_risk_label_np_list, predictions_priv_np_list) \
            = run_validation_FT(model, criterion, kettle.FT_auth_loader, kettle.FT_priv_loader, kettle.setup,
                             kettle.args.dryrun)
    # else:
    #     auth_accuracy, priv_intend_accuracy, priv_risk_accuracy, auth_loss, priv_intend_loss = None, None, None, None, None

    if poison_delta is None:
        str_with_poison = 'clean'
        log_file_dir = kettle.args.log_file_dir
    else:
        str_with_poison = 'poisoned'
        log_file_dir = kettle.args.log_file_dir
    
    current_lr = optimizer.param_groups[0]['lr']
    # print_and_save_stats(log_file_dir, str_with_poison, epoch, stats, current_lr, epoch_loss / (len(AP_FT_loader) + 1), correct_preds / total_preds, auth_accuracy, priv_intend_accuracy, priv_risk_accuracy, auth_loss, priv_intend_loss)
    print_and_save_stats(log_file_dir, str_with_poison, epoch, stats, current_lr, epoch_loss / (len(AP_FT_loader) + 1),
                         correct_preds / total_preds, auth_accuracy, priv_intend_accuracy, priv_risk_accuracy,
                         auth_loss, priv_poisoned_loss, f1, f1_auth, f1_inte_priv, f1_risk_priv, passenger_loss,
                         auth_label_np_list, predictions_auth_np_list,priv_risk_label_np_list, predictions_priv_np_list, save_mat=False)

    if DEBUG_TRAINING:
        print(f"Data processing: {datetime.timedelta(milliseconds=stats['data_time'])}, "
              f"Forward pass: {datetime.timedelta(milliseconds=stats['forward_time'])}, "
              f"Backward Pass and Gradient Step: {datetime.timedelta(milliseconds=stats['backward_time'])}")
        stats['data_time'] = 0
        stats['forward_time'] = 0
        stats['backward_time'] = 0

def run_validation(model, criterion, auth_loader, priv_loader, setup, dryrun=False):
    """Get accuracy of model relative to dataloader."""
    model.eval()
    auth_correct = 0
    auth_total = 0
    auth_loss = 0
    priv_total = 0
    priv_intend_correct = 0
    priv_intend_loss = 0
    priv_poisoned_loss = 0.
    priv_risk_correct = 0 #

    auth_label_np_list = []
    priv_risk_label_np_list = []
    priv_inte_label_np_list = []
    predictions_auth_np_list = []
    predictions_priv_np_list = []

    with torch.no_grad():
        for data, label, indeces in auth_loader:
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            data = data.to(**setup)           
            label = label.to(**setup)
            outputs = model(data,indeces)
            _, predicted = torch.max(outputs.data, 1)
            auth_loss += criterion(outputs, label).item()
            auth_total += label.size(0)
            auth_correct += (predicted == torch.argmax(label, dim=1)).sum().item()

            auth_label_np_list.append(torch.argmax(label, dim=1).cpu().numpy())
            predictions_auth_np_list.append(predicted.cpu().numpy())

            if dryrun:
                break

        auth_label_np_list = np.concatenate(auth_label_np_list,axis=0)
        predictions_auth_np_list = np.concatenate(predictions_auth_np_list,axis=0)
        f1_auth = f1_score(auth_label_np_list, predictions_auth_np_list, average='weighted')

        for data, label, indeces, intend_label in priv_loader:
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            data = data.to(**setup)
            label = label.to(**setup)
            intend_label = intend_label.to(**setup)
            outputs = model(data,indeces)
            _, predicted = torch.max(outputs.data, 1)
            priv_intend_loss += criterion(outputs, intend_label).item()
            priv_poisoned_loss += criterion(outputs, label).item()
            priv_total += label.size(0)
            priv_intend_correct += (predicted == torch.argmax(intend_label, dim=1)).sum().item()
            priv_risk_correct += (predicted == torch.argmax(label, dim=1)).sum().item()

            priv_inte_label_np_list.append(torch.argmax(intend_label, dim=1).cpu().numpy())
            priv_risk_label_np_list.append(torch.argmax(label, dim=1).cpu().numpy())
            predictions_priv_np_list.append(predicted.cpu().numpy())

            if dryrun:
                break

        priv_risk_label_np_list = np.concatenate(priv_risk_label_np_list,axis=0)
        priv_inte_label_np_list = np.concatenate(priv_inte_label_np_list, axis=0)
        predictions_priv_np_list = np.concatenate(predictions_priv_np_list,axis=0)
        f1_inte_priv = f1_score(priv_inte_label_np_list, predictions_priv_np_list, average='weighted')
        f1_risk_priv = f1_score(priv_risk_label_np_list, predictions_priv_np_list, average='weighted')

    auth_accuracy = auth_correct / auth_total
    priv_intend_accuracy = priv_intend_correct / priv_total
    priv_risk_accuracy = priv_risk_correct / priv_total

    user_obj_loss = (auth_loss-priv_poisoned_loss)/(auth_total+priv_total)
    return (auth_accuracy, priv_intend_accuracy, priv_risk_accuracy, user_obj_loss, priv_poisoned_loss, f1_auth, f1_inte_priv, f1_risk_priv,
            auth_label_np_list, predictions_auth_np_list,priv_risk_label_np_list, predictions_priv_np_list)


def run_validation_FT(model, criterion, auth_loader, priv_loader, setup, dryrun=False):
    """Get accuracy of model relative to dataloader."""
    model.eval()
    auth_correct = 0
    auth_total = 0
    auth_loss = 0
    priv_total = 0
    priv_poisoned_loss = 0.
    priv_risk_correct = 0

    auth_label_np_list = []
    priv_risk_label_np_list = []
    predictions_auth_np_list = []
    predictions_priv_np_list = []

    with torch.no_grad():
        for data, label, indeces in auth_loader:
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            data = data.to(**setup)
            label = label.to(**setup)
            outputs = model(data,indeces)
            _, predicted = torch.max(outputs.data, 1)
            auth_loss += criterion(outputs, label).item()
            auth_total += label.size(0)
            auth_correct += (predicted == torch.argmax(label, dim=1)).sum().item()

            auth_label_np_list.append(torch.argmax(label, dim=1).cpu().numpy())
            predictions_auth_np_list.append(predicted.cpu().numpy())

            if dryrun:
                break

        auth_label_np_list = np.concatenate(auth_label_np_list,axis=0)
        predictions_auth_np_list = np.concatenate(predictions_auth_np_list,axis=0)
        f1_auth = f1_score(auth_label_np_list, predictions_auth_np_list, average='weighted')

        for data, label, indeces in priv_loader:
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            data = data.to(**setup)
            label = label.to(**setup)
            outputs = model(data,indeces)
            priv_poisoned_loss += criterion(outputs, label).item()
            _, predicted = torch.max(outputs.data, 1)
            priv_total += label.size(0)
            priv_risk_correct += (predicted == torch.argmax(label, dim=1)).sum().item()

            priv_risk_label_np_list.append(torch.argmax(label, dim=1).cpu().numpy())
            predictions_priv_np_list.append(predicted.cpu().numpy())

            if dryrun:
                break

        priv_risk_label_np_list = np.concatenate(priv_risk_label_np_list,axis=0)
        predictions_priv_np_list = np.concatenate(predictions_priv_np_list,axis=0)
        f1_risk_priv = f1_score(priv_risk_label_np_list, predictions_priv_np_list, average='weighted')

    auth_accuracy = auth_correct / auth_total
    priv_intend_accuracy = 0
    priv_risk_accuracy = priv_risk_correct / priv_total
    priv_intend_loss = 0
    f1_inte_priv  = 0

    return (auth_accuracy, priv_intend_accuracy, priv_risk_accuracy, auth_loss, priv_poisoned_loss, f1_auth, f1_inte_priv,
            f1_risk_priv,auth_label_np_list, predictions_auth_np_list,priv_risk_label_np_list, predictions_priv_np_list)

