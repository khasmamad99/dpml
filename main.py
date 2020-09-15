import os

import numpy as np
from matplotlib import pyplot as plt
import torch

from models.convnet import ConvNet
from sangexp import SangExp


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def plot_exp():
    kwargs = {
        'model_class' : ConvNet,
        'model_path' : './save/fashion.pt',
        'device': device,
        'data' : 'fashion',
        'batch_size' : 512,
        'test_batch_size' : 512,
        'download_path' : './data'
    }

    SangExp.full_run(**kwargs)


def plot_dp():
    path = 'save/mnist_dp.npy'
    stats = np.load(path)

    epsilons = [stat[0] for stat in stats]
    accuracies = [stat[2] for stat in stats]

    plt.figure(figsize=(6,4))
    plt.title('mnist')
    plt.plot(epsilons, accuracies, 'r-x')
    # plt.legend(['epsilon', 'accuracy'])
    
    plt.show()


plot_dp()

