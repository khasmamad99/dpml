import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules
from tqdm import tqdm

from models.convnet import ConvNet
from models.resnet import resnet18

from sangexp import SangExp


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if not args.disable_dp:
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"   
        )
        return epsilon, best_alpha
    
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def test(args, model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(
                dim = 1, keepdim=True
            )   
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )        
    )
    return correct / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "-na",
        "--n_accumulation_steps",
        default=1,
        type=int,
        metavar="N",
        help="number of mini-batches to accumulate into an effective batch",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=200,
        metavar="TB",
        help="input batch size for testing (default: 1024)",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="mnist",
        metavar="D",
        help="dataset to train on (mnist, fashion, cifar10, cifar100"
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: .1)",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0.,
        type=float,
        metavar="W",
        help="SGD weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--momentum", default=0., type=float, metavar="M", help="SGD momentum"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process (default: 'cuda')",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model (default: false)",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Where MNIST is/will be stored",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    kwargs = {"num_workers": 1, "pin_memory": True}

    train_set, test_set, train_loader, test_loader = SangExp.get_data(
        data=args.data,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        download_path=args.data_root
    )

    
    if args.data == "mnist" or args.data == "fashion":
        model = ConvNet()
    elif args.data == "cifar10":
        model = resnet18(num_classes=10)
    elif args.data == "cifar100":
        model = resnet18(num_classes=100)

    model = convert_batchnorm_modules(model).to(device)


    optimizer = optim.SGD(
        model.parameters(),
        lr = args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    if not args.disable_dp:
        privacy_engine = PrivacyEngine(
            model,
            batch_size=args.batch_size * args.n_accumulation_steps,
            sample_size=len(train_set),
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
            clip_per_layer=False,
            enable_stat=False
        )
        privacy_engine.attach(optimizer)

    stats = []
    for epoch in range(args.epochs):
        stat = []
        if not args.disable_dp:
            epsilon, best_alpha = train(args, model, device, train_loader, optimizer, epoch)
            stat.append(epsilon)
            stat.append(best_alpha)
        else:
            train(args, model, device, train_loader, optimizer, epoch)
        acc = test(args, model, device, test_loader)
        stat.append(acc)
        stats.append(tuple(stat))

    name = "save/{}".format(args.data)
    if not args.disable_dp:
        name += "_dp"
    np.save(name, stats)
    torch.save(model.state_dict(), name+".pt")



if __name__ == "__main__":
    main()



