'''
A function to draw the histogram of the poison delta distribution.
'''

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy import stats
import scipy.io as sio


def save_draw_poison_delta_distribution(poison_delta,poison_loss_value, log_save_dir):

    save_path_data = os.path.join(log_save_dir, 'poison_delta.mat')
    save_path_loss = os.path.join(log_save_dir, 'poison_loss.mat')
    numpy_poison_delta = poison_delta.cpu().numpy()
    sio.savemat(save_path_data, {'poison_delta': numpy_poison_delta})
    sio.savemat(save_path_loss, {'poison_loss_value': poison_loss_value})

    vector_poison_delta = numpy_poison_delta.reshape(-1)
    sns.set_style('whitegrid')
    sns.set_context('paper')
    plt.figure(figsize=(8, 6))
    plt.title('Poison Delta Distribution')
    plt.xlabel('Poison Delta')
    plt.ylabel('Frequency')
    plt.hist(vector_poison_delta, bins=200, density=True)
    plt.savefig(os.path.join(log_save_dir, 'poison_delta_distribution.png'))
    plt.show()
    plt.close()

    # save the histogram of the poison delta distribution