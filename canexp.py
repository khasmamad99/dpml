import math
import random
import os
import glob

from matplotlib import pyplot as plt
import torch


class CanExp:

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def set_device(device):
        CanExp._device = device
    
    @staticmethod
    def get_device():
        return CanExp._device


    @staticmethod
    def shuffle_datasets(trainset, testset, ratio=0.5):
        train_pairs = [(x, y) for x, y in zip(trainset.data, trainset.targets)]
        test_pairs = [(x, y) for x, y in zip(testset.data, testset.targets)]
        pairs = train_pairs + test_pairs
        random.shuffle(pairs)
        num_train = int(len(pairs) * ratio)
        trainset.data = [p[0] for p in pairs[:num_train]]
        trainset.targets = [p[1] for p in pairs[:num_train]]
        testset.data = [p[0] for p in pairs[num_train:]]
        testset.targets = [p[1] for p in pairs[num_train:]]
        
        return trainset, testset


    @staticmethod
    def train_target_models(epochs, num_models, model_class, criterion, 
                            opt_class, opt_params, trainset, testset, batch_size,
                            epoch_shuffle=True, train_shuffle=False, random_init=True, 
                            save=False, save_path=None, silent=False):

        assert epochs > 0, "Number of epochs should be > 0"
        factor = epochs / 5.
        if epochs <= 10:
            factor = math.ceil(factor)
        else:
            factor = math.floor(factor)

        device = CanExp.get_device()

        # create loaders for calculating loss
        trainloader_no_shuffle = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        testloader_no_shuffle = torch.utils.data.DataLoader(
            testset, batch_size=batch_size*2, shuffle=False, num_workers=2
        )

        if save:
            if save_path is None:
                save_path = 'checkpoints'

            if not os.path.isdir(save_path):
                os.mkdir(save_path)

        criterion.reduction = "none"
        aux = []
        for i in range(1, num_models + 1):
            model = model_class()
            if not random_init:
                model.use_fixed_params()
            model.to(device)
            opt = opt_class(model.parameters(), **opt_params)

            if train_shuffle:
                trainset, testset = CanExp.shuffle_datasets(trainset, testset)

            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=epoch_shuffle, num_workers=2
            )

            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size*2, shuffle=False, num_workers=2
            )

            print("Starting training model {}...".format(i))
            for epoch in range(epochs):
                running_loss = 0.0
                running_acc = 0.0
                for xb, yb in trainloader:
                    opt.zero_grad()
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)

                    preds = torch.argmax(out, dim=1)
                    running_acc += (preds == yb).float().sum().item()

                    loss = criterion(out, yb)
                    with torch.no_grad():
                        running_loss += loss.sum().item()
                    loss = loss.mean()
                    loss.backward()
                    opt.step()

                if (not silent and epoch % factor == factor - 1) or epoch == epochs - 1:
                    running_loss /= len(trainset.targets)
                    running_acc /= len(trainset.targets)
                    test_acc = CanExp.get_accuracy(model, testloader)
                    if not silent:
                        print("Model {}, Epoch {}, Train Loss: {}, Train Acc: {}, Test Acc: {}".format(
                            i, epoch+1, running_loss, running_acc, test_acc
                        ))

            print("Training of model {} finished...".format(i))
            print("Train Loss: {}, Train Acc: {}, Test Acc: {}".format(
                running_loss, running_acc, test_acc
            ))

            train_per_sample_loss, train_avg_loss = CanExp.get_loss_stuff(model, criterion, trainloader_no_shuffle)
            test_per_sample_loss, _ = CanExp.get_loss_stuff(model, criterion, testloader_no_shuffle)
            aux.append((train_per_sample_loss, test_per_sample_loss, train_avg_loss))

            if save:
                print("Saving model {}...".format(i))
                full_path = os.path.join(save_path, "model{}_{}_{}.tar".format(i, int(epoch_shuffle), int(random_init)))
                torch.save({
                    "state_dict" : model.state_dict(),
                    "train_loss" : running_loss,
                    "train_acc" : running_acc,
                    "test_acc" :  test_acc,
                    "train_per_sample_loss" : train_per_sample_loss,
                    "test_per_sample_loss" : test_per_sample_loss
                }, full_path)

        print("Finished training all the models...")
        model_class.reset_params()
        return aux
    

    @staticmethod
    def get_accuracy(model, dataloader):
        device = next(model.parameters()).device
        len_data = 0
        with torch.no_grad():
            running_acc = 0.0
            for xb, yb in dataloader:
                len_data += xb.size(0)
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds = torch.argmax(out, dim=1)
                running_acc += (preds == yb).float().sum().item()
            
        avg_acc = running_acc / len_data
        return avg_acc


    @staticmethod
    def get_loss_stuff(model, criterion, dataloader):
        device = next(model.parameters()).device
        
        criterion.reduction = "none"
        with torch.no_grad():
            avg_loss = 0.0
            per_sample_loss = []
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                per_sample_loss.append(loss)
                avg_loss += loss.sum().item()

        per_sample_loss = torch.cat(per_sample_loss, 0)
        avg_loss /= (len(dataloader) * xb.size(0))

        return per_sample_loss, avg_loss


    @staticmethod
    def perform_yeom_mi(train_per_sample_loss, test_per_sample_loss, train_avg_loss):
        true_positives = train_per_sample_loss <= train_avg_loss
        true_negatives = test_per_sample_loss > train_avg_loss
        false_positives = test_per_sample_loss <= train_avg_loss
        false_negatives = train_per_sample_loss > train_avg_loss

        return (true_positives, true_negatives, false_positives, false_negatives)

    @staticmethod
    def load_models_aux(models_path, model_class=None, criterion=None, trainloader=None, testloader=None):
        aux = []
        model_files = glob.glob(os.path.join(models_path, "*.tar"))
        assert len(model_files) > 0, "No model files found in {}".format(models_path)
        for file in model_files:
            checkpoint = torch.load(file)

            try:
                train_per_sample_loss = checkpoint['train_per_sample_loss']
                test_per_sample_loss = checkpoint['test_per_sample_loss']
                train_avg_loss = checkpoint['train_loss']
            except:
                assert model_class is not None and criterion is not None and trainloader is not None and testloader is not None, \
                        "checkpoint does not include information on the loss of the model; model_class, criterion, trainloader, \
                        and testloader parameters must be given"
                
                model = model_class()
                model.load_state_dict(checkpoint['state_dict'])
                model.to(CanExp.get_device())
                train_per_sample_loss, train_avg_loss = CanExp.get_loss_stuff(model, criterion, trainloader)
                test_per_sample_loss, test_avg_loss = CanExp.get_loss_stuff(model, criterion, testloader)

            aux.append((train_per_sample_loss, test_per_sample_loss, train_avg_loss))

        return aux

    
    @staticmethod
    def yeom_mi(aux):
        cum_true_positives = []
        cum_true_negatives = []
        cum_false_positives = []

        for true_positives, true_negatives, false_positives in [CanExp.perform_yeom_mi(*ax) for ax in aux]:
            cum_true_positives.append(true_positives.unsqueeze(1))
            cum_true_negatives.append(true_negatives.unsqueeze(1))
            cum_false_positives.append(false_positives.unsqueeze(1))

        num_true_positives = torch.cat(cum_true_positives, 1).sum(1)
        num_true_negatives = torch.cat(cum_true_negatives, 1).sum(1)
        num_false_positives = torch.cat(cum_false_positives, 1).sum(1)

        counts_true_positives = [
            (num_true_positives == i).float().sum().item() for i in range(1, len(aux)+1)
            ]
        counts_true_negatives = [
            (num_true_negatives == i).float().sum().item() for i in range(1, len(aux)+1)
            ]

        counts_false_positives = [
            (num_false_positives == i).float().sum().item() for i in range(1, len(aux)+1)
            ]

        return counts_true_positives, counts_true_negatives, counts_false_positives

    
    @staticmethod
    def plot_attack_results(result_dict, title):
        line_types = ['rx-', 'bx-', 'gx-']
        legend = []
        for i, item in enumerate(result_dict.items()):
            name, values = item
            legend.append(name)
            plt.plot(range(1, len(values)+1), values, line_types[i])

        plt.title(title)
        plt.xlabel('# of models')
        plt.ylabel('# of samples')
        plt.legend(legend)

    
    @staticmethod
    def private_inference(models, sample, criterion):
        device = next(models[0].parameters()).device
        criterion.reduction = 'mean'
        losses = []
        with torch.no_grad():
            outs = [model(sample.to(device)) for model in models]
            preds = [torch.argmax(out) for out in outs]
            losses = [criterion(out, pred) for out, pred in zip(outs, preds)]