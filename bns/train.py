import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from functorch import combine_state_for_ensemble, vmap

from bns.bns import BN, BNRS, BRN


class Model(nn.Sequential):
    def __init__(self, Norm):
        super().__init__(
            nn.Conv2d(1, 32, 3, padding=1),
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
            nn.Linear(32, 10),
        )


def _loop_forever(dl):
    while True:
        yield from dl


def run(args):
    train_ds = MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    test_ds = MNIST(
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
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    del train_ds, test_ds

    Norms = [BN, BRN, BNRS]
    models = [Model(Norm).to(args.device) for Norm in Norms]
    fmodel, params, buffers = combine_state_for_ensemble(models)
    [p.requires_grad_() for p in params]

    optimizer = torch.optim.Adam(params, lr=args.lr)

    def _repeat_to_device(x: Tensor):
        return x[None].repeat_interleave(len(models), dim=0).to(args.device)

    for i, (x, y) in enumerate(_loop_forever(train_dl)):
        x = _repeat_to_device(x)
        y = y.to(args.device)

        h = vmap(fmodel)(params, buffers, x)

        losses = []

        for j, hj in enumerate(h):
            loss = F.cross_entropy(hj, y)
            losses.append(loss)
            if (i * 10) % args.val_every == 0:
                print(f"Iteration {i} Loss for Norm {Norms[j]}: {loss.item():.5g}")

        optimizer.zero_grad()
        sum(losses).backward()
        optimizer.step()

        if i % args.val_every == 0:
            with torch.inference_mode():
                fmodel.eval()

                accs = defaultdict(list)
                for x, y in (tqdm if args.verbose else lambda x: x)(test_dl):
                    x = _repeat_to_device(x)
                    y = y.to(args.device)
                    h = vmap(fmodel)(params, buffers, x)
                    model_batch_accs = (h.argmax(dim=-1) == y).float().tolist()

                    for j, batch_accs in enumerate(model_batch_accs):
                        accs[j].extend(batch_accs)

                for j, v in accs.items():
                    print(f"Accuracy of Norm {Norms[j]}: {np.mean(v):.4g}.")

                fmodel.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
