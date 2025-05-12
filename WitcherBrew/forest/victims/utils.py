"""Utilites related to training models."""

import torch
import os
import scipy.io


def print_and_save_stats(file_dir, str_with_poison, epoch, stats, current_lr, train_loss, train_acc, 
                        auth_accuracy, priv_intend_accuracy, priv_risk_accuracy, auth_loss, priv_poisoned_loss,
                         f1, f1_auth, f1_inte_priv, f1_risk_priv,passenger_loss,
                         auth_label_np_list, predictions_auth_np_list,priv_risk_label_np_list, predictions_priv_np_list, save_mat=False):
    """Print info into console and into the stats object."""
    stats['train_losses'].append(train_loss)
    stats['train_accs'].append(train_acc)
    stats['train_f1'].append(f1)
    # open file_path to store the printed info
    file_path = os.path.join(file_dir, str_with_poison+'valid_log.txt')
    with open(file_path, 'a') as f:

        if auth_accuracy is not None:
            stats['auth_accuracy'].append(auth_accuracy)
            stats['auth_f1'].append(f1_auth)
            stats['priv_intend_accuracy'].append(priv_intend_accuracy)
            stats['priv_intend_f1'].append(f1_inte_priv)
            stats['priv_risk_accuracy'].append(priv_risk_accuracy)
            stats['priv_risk_f1'].append(f1_risk_priv)
            stats['auth_loss'].append(auth_loss)
            stats['priv_poisoned_loss'].append(priv_poisoned_loss)
            stats['passenger_loss'].append(passenger_loss)

            f.write(f'{str_with_poison} Epoch: {epoch:<3}|' 
                f'priv_risk_accuracy: {stats["priv_risk_accuracy"][-1]:7.2%} |'
                f'auth_accuracy: {stats["auth_accuracy"][-1]:7.2%} |' 
                f'auth_loss: {stats["auth_loss"][-1]:7.4f} |'
                f'priv_poisoned_loss: {stats["priv_poisoned_loss"][-1]:7.4f} |\n')
    '''
    else:
        if 'valid_accs' in stats:
            # Repeat previous answers if validation is not recomputed
            stats['auth_accuracy'].append(auth_accuracy)
            stats['priv_intend_accuracy'].append(priv_intend_accuracy)
            stats['priv_risk_accuracy'].append(priv_risk_accuracy)
            stats['auth_loss'].append(auth_loss)
            stats['priv_intend_loss'].append(priv_intend_loss)

        print(f'Epoch: {epoch:<3}| lr: {current_lr:.4f} | '
              f'Training    loss is {stats["train_losses"][-1]:7.4f}, train acc: {stats["train_accs"][-1]:7.2%} | ')
    '''
    if save_mat and str_with_poison == 'poisoned':
        SAVE_PATH = os.path.join(file_dir, str_with_poison + str(epoch) + 'np_pre.mat')
        pre = {'auth_label_np_list':auth_label_np_list, 'predictions_auth_np_list':predictions_auth_np_list,
                          'priv_risk_label_np_list':priv_risk_label_np_list,'predictions_priv_np_list':predictions_priv_np_list}
        scipy.io.savemat(SAVE_PATH, {'pre': pre})


def pgd_step(inputs, labels, model, loss_fn, dm, ds, eps=16, tau=0.01):
    """Perform a single projected signed gradient descent step, maximizing the loss on the given labels."""
    inputs.requires_grad = True
    tau = eps / 255 / ds * tau

    loss = loss_fn(model(inputs), labels)
    grads = torch.grad.autograd(loss, inputs, retain_graph=True, create_graph=True, only_inputs=True)
    inputs.requires_grad = False
    with torch.no_grad():
        # Gradient Step
        outputs = inputs + tau * grads

        # Projection Step
        outputs = torch.max(torch.min(outputs, eps / ds / 255), -eps / ds / 255)
        outputs = torch.max(torch.min(outputs, (1 + dm) / ds - inputs), -dm / ds - inputs)
    return outputs
