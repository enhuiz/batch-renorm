import torch
import argparse
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR100

from .bns import BN, BNRS, BRN


class Model(nn.Sequential):
    def __init__(self, in_channels, num_classes, Norm):
        super().__init__(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            Norm(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, dilation=2, padding=2),
            Norm(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, dilation=4, padding=4),
            Norm(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, dilation=8, padding=8),
            Norm(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(32, num_classes),
        )


class Ensemble(nn.ModuleList):
    def forward(self, x):
        ys = []
        for model in self:
            ys.append(model(x))
        return ys


def _loop_forever(dl):
    while True:
        yield from dl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="MNIST")
    args = parser.parse_args()

    if args.dataset == "MNIST":
        Dataset = MNIST
        in_channels = 1
        num_classes = 10
        lr = 1e-3
    elif args.dataset == "CIFAR100":
        Dataset = CIFAR100
        in_channels = 3
        num_classes = 100
        lr = 1e-3
    else:
        raise NotImplementedError(args.dataset)

    train_ds = Dataset(
        "data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    test_ds = Dataset(
        "data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
    )

    del train_ds, test_ds

    Norms = [BN, BRN, BNRS]
    models = [Model(in_channels, num_classes, Norm).to(args.device) for Norm in Norms]

    # Eliminate the effect of normalization
    for model in models[1:]:
        model.load_state_dict(models[0].state_dict())

    model = Ensemble(models)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_loss_curves = defaultdict(list)
    test_acc_curves = defaultdict(list)

    for i, (x, y) in enumerate(_loop_forever(train_dl)):
        if i > 10000:
            break

        x = x.to(args.device)
        y = y.to(args.device)

        hs = model(x)

        losses = []

        for j, h in enumerate(hs):
            loss = F.cross_entropy(h, y)
            losses.append(loss)

        if i % args.log_every == 0:
            for j, loss in enumerate(losses):
                print(f"Iteration {i}: Loss for Norm {Norms[j]}: {loss.item():.5g}.")
                training_loss_curves[j].append(loss.item())

        optimizer.zero_grad()
        sum(losses).backward()
        optimizer.step()

        if i % args.test_every == 0:
            model.eval()
            with torch.inference_mode():
                accs = defaultdict(list)
                for x, y in tqdm(test_dl):
                    x = x.to(args.device)
                    y = y.to(args.device)
                    hs = model(x)
                    for j, h in enumerate(hs):
                        accs[j].extend((h.argmax(dim=-1) == y).cpu().numpy())
                for j, v in accs.items():
                    test_acc_curves[j].append(np.mean(v))
                    print(f"Accuracy of Norm {Norms[j]}: {test_acc_curves[j][-1]:.4g}.")
            model.train()

            plt.subplot(121)
            plt.title("Training Loss")
            for j, m in enumerate(Norms):
                plt.plot(training_loss_curves[j], label=m.__name__)
            plt.yscale("log")
            plt.xlabel(f"x{args.log_every} iterations")
            plt.legend()
            plt.tight_layout()

            plt.subplot(122)
            plt.title("Test Accuracy")
            for j, m in enumerate(Norms):
                plt.plot(test_acc_curves[j], label=m.__name__)
            plt.xlabel(f"x{args.test_every} iterations")
            if args.dataset == "MNIST":
                plt.ylim(0.75, 1.0)
            plt.legend()
            plt.tight_layout()

            path = Path(
                "results",
                f"bs-{args.batch_size}-dataset-{args.dataset}.png",
            )
            plt.savefig(path)
            plt.clf()


if __name__ == "__main__":
    main()
