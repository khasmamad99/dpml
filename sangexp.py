import os
import itertools
import random

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

from canexp import CanExp



class SangExp:

    @staticmethod
    def leakage(train_per_sample_loss, test_per_sample_loss, train_avg_loss):
        tp, tn, fp, fn = CanExp.perform_yeom_mi(
                train_per_sample_loss, test_per_sample_loss, train_avg_loss)
            
        count_tp, count_tn, count_fp, count_fn = [a.float().sum().item() for a in [tp, tn, fp, fn]]
        print("TP: {}, TN: {}, FP: {}, FN: {}".format(count_tp, count_tn, count_fp, count_fn))
        tpr = count_tp / (count_tp + count_fn)
        fpr = count_fp / (count_fp + count_tn)
        print("TPR: {}, FPR: {}".format(tpr, fpr))

        return (tpr, fpr)


    @staticmethod
    def perform(model, criterion, trainloader, testloader, bins=10):
        device = next(model.parameters()).device

        # calculcate importance of the features and sort them
        imp_mtx = SangExp.calc_feature_importance(model, trainloader)
        sorted_imp_mtx, indices = torch.sort(imp_mtx)

        min_imp, max_imp = sorted_imp_mtx[0], sorted_imp_mtx[-1]
        step = (max_imp - min_imp) / bins
        train_accs = []
        test_accs = []
        leakages = []
        thresholds = [-1.]
        thresholds.extend([min(min_imp + (i + 1) * step, max_imp) for i in range(bins)])

        feature_dist = []

        for i, thresh in enumerate(thresholds):
            # remove the features with importance value less than the threshold
            print("{} - removing features with importance <= {}".format(i, thresh))
            indices_to_remove = (indices[sorted_imp_mtx <= thresh]).long()
            count_removed = indices_to_remove.size(0)
            print("{} features removed...".format(count_removed))
            feature_dist.append(count_removed - sum(feature_dist))

            print("Forward passes to calculate loss and accuracy...")
            train_avg_acc, train_avg_loss, train_per_sample_loss = SangExp.forward_pass(model, criterion, trainloader, indices_to_remove)
            test_avg_acc, test_avg_loss, test_per_sample_loss = SangExp.forward_pass(model, criterion, testloader, indices_to_remove)

            train_accs.append(train_avg_acc)
            test_accs.append(test_avg_acc)

            print("Performing membership inference")
            leakages.append(SangExp.leakage(train_per_sample_loss, test_per_sample_loss, train_avg_loss))

        thresholds[0] = 0.
        return train_accs, test_accs, leakages, thresholds, feature_dist
        

    @staticmethod
    def normalize(data: torch.Tensor) -> torch.Tensor:
        return (data - data.mean(0)) / data.std(0)


    @staticmethod
    def normalize2(data: torch.Tensor) -> torch.Tensor:
        data -= data.min(1, keepdim=True)[0]
        data /= data.max(1, keepdim=True)[0]
        return data

    
    @staticmethod
    def forward_pass(model, criterion, dataloader, indices_to_remove):
        device = next(model.parameters()).device
        running_acc = 0.
        running_loss = 0.
        per_sample_loss = []
        len_data = 0
        criterion.reduction = 'none'
        with torch.no_grad():
            for xb, yb in dataloader:
                len_data += xb.size(0)
                xb, yb = xb.to(device), yb.to(device)
                out = model.controlled_forward(xb, indices_to_remove)
                preds = torch.argmax(out, dim=1)
                running_acc += (preds == yb).float().sum().item()
                loss = criterion(out, yb)
                per_sample_loss.append(loss)
                running_loss += loss.sum().item()

        per_sample_loss = torch.cat(per_sample_loss, 0)
        avg_loss = running_loss / len_data
        avg_acc = running_acc / len_data

        return avg_acc, avg_loss, per_sample_loss


    @staticmethod
    def calc_feature_importance(model, trainloader):
        device = next(model.parameters()).device
        importance_mtx = []
        num_0s = []
        with torch.no_grad():
            for xb, _ in trainloader:
                xb = xb.to(device)
                features = model.partial_forward(xb)
                num_0s.append(features)
                features = SangExp.normalize2(features)
                importance_mtx.append(features)
        
        print((torch.cat(num_0s) == 0.).float().sum(1).mean())
        importance_mtx = torch.cat(importance_mtx, 0).mean(0)

        
        return importance_mtx


    @staticmethod
    def get_model(model_class, model_path, device):
        model = model_class()
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss()

        return model, criterion


    @staticmethod
    def get_data(data='mnist', batch_size=512, test_batch_size=512, download_path='./data'):
        if not os.path.isdir(download_path):
            os.mkdir(download_path)

        kwargs = {'batch_size':batch_size, 'num_workers':1, 'pin_memory':True, 'shuffle':False}

        if data == 'mnist':
            transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])
            
            trainset = torchvision.datasets.MNIST(download_path, train=True, download=True,
                                transform=transform)
            testset = torchvision.datasets.MNIST(download_path, train=False, download=False,
                                transform=transform)

        elif data == 'fashion':
            trainset = torchvision.datasets.FashionMNIST(download_path, download=True, train=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
            testset = torchvision.datasets.FashionMNIST(download_path, download=False, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))


        elif data == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4914, 0.4822, 0.4465],
                    [0.2023, 0.1994, 0.2010])                             
                ])

            trainset = CIFAR10(download_path, train=True, transform=transform, download=True)
            testset = CIFAR10(download_path, train=False, transform=transform, download=False)


        elif data == 'cifar100':
            # if train:
            #     kwargs['batch_size'] = 512
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])

            trainset = CIFAR100(download_path, train=True, transform=transform, download=True)
            testset = CIFAR100(download_path, train=False, transform=transform, download=False)

        trainloader = torch.utils.data.DataLoader(trainset, **kwargs)
        kwargs['batch_size'] = test_batch_size
        testloader = torch.utils.data.DataLoader(testset, **kwargs)

        return trainset, testset, trainloader, testloader


    @staticmethod
    def shrink_dataset(dataset, samples_per_class):
        targets = np.array(dataset.targets, dtype=np.uint8)
        data = list(itertools.chain(*[dataset.data[targets == i][:samples_per_class] for i in range(len(dataset.classes))]))
        targets = list(itertools.chain(*[targets[targets == i][:samples_per_class] for i in range(len(dataset.classes))]))
        pairs = [(d, t) for d, t in zip(data, targets)]
        random.shuffle(pairs)

        dataset.data = np.array([p[0] for p in pairs])
        dataset.targets = torch.LongTensor([p[1] for p in pairs])

    
    @staticmethod
    def full_run(model_class, model_path, device, data, batch_size, test_batch_size, download_path):
        model, criterion = SangExp.get_model(model_class, model_path, device)
        trainset, testset, trainloader, testloader = SangExp.get_data(
            data, batch_size, test_batch_size, download_path
        )

        train_accs, test_accs, leakages, thresholds, feature_dist = SangExp.perform(model, criterion, trainloader, testloader)

        tpr = [t[0] for t in leakages]
        fpr = [t[1] for t in leakages]
        leakages = [t - f for t, f in zip(tpr, fpr)]

        relative_leakages = [(l - leakages[0]) / leakages[0] for l in leakages]
        relative_leakages = [ l - min(relative_leakages) for l in relative_leakages]
        relative_leakages = [ l / max(relative_leakages) for l in relative_leakages]

        
        plt.figure(figsize=(8,12))
        plt.subplot(3, 1, 1)
        plt.title(data)
        plt.plot(thresholds[1:], test_accs[1:],'r-x')
        plt.plot(thresholds[1:], leakages[1:], 'b-x')
        plt.plot(thresholds[1:], relative_leakages[1:], 'g-x')
        plt.legend(["test_acc", "leakage", 'relative_leakage'])

        plt.subplot(3, 1, 2)
        plt.plot(thresholds[1:], tpr[1:], 'g-x')
        plt.plot(thresholds[1:], fpr[1:], 'r-x')
        plt.legend(['TPR', 'FPR'])

        plt.subplot(3, 1, 3)
        plt.plot(thresholds[1:], feature_dist[1:], 'x-')
        plt.xlabel('importance')
        plt.ylabel('# of features')

        plt.show()
