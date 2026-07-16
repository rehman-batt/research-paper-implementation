import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from engine.metrics import top_k_accuracy
from typing import Callable

def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer: Optimizer, loss_fn: Callable, device: str | torch.device ='cpu') -> tuple[float, float, float]:
    
    epoch_loss = 0.0
    epoch_top1_acc = 0.0
    epoch_top5_acc = 0.0

    for batch_id, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        pred = model(inputs)

        optimizer.zero_grad(set_to_none=True)

        loss = loss_fn(pred, labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_top1_acc += top_k_accuracy(pred, labels, k=1)
        epoch_top5_acc += top_k_accuracy(pred, labels, k=5)

    n = len(train_loader)
    return epoch_loss / n, epoch_top1_acc / n, epoch_top5_acc / n

def eval_one_epoch(model: nn.Module, val_loader: DataLoader, loss_fn: Callable, device: str | torch.device   ='cpu') -> tuple[float, float, float]:

    epoch_loss = 0.0
    epoch_top1_acc = 0.0
    epoch_top5_acc = 0.0

    with torch.no_grad():
        for batch_id, (inputs, labels) in enumerate(val_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            pred = model(inputs)

            loss = loss_fn(pred, labels)

            epoch_loss += loss.item()
            epoch_top1_acc += top_k_accuracy(pred, labels, k=1)
            epoch_top5_acc += top_k_accuracy(pred, labels, k=5)

    n = len(val_loader)
    return epoch_loss / n, epoch_top1_acc / n, epoch_top5_acc / n





