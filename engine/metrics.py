import torch
from torch import Tensor

def accuracy(pred: Tensor, labels: Tensor) -> float:
    # pred_class = torch.argmax(pred, dim=1)
    # correct_pred = (pred_class == labels).sum()
    # return (correct_pred / len(labels)).item()

    return top_k_accuracy(pred, labels, 1)
    

def top_k_accuracy(pred: Tensor, labels: Tensor, k: int) -> float:
    if k > pred.shape[1]:
        raise ValueError("K Value Cannot be Greater Than Number of Classes")
    
    topk = torch.topk(pred, k, dim=1).indices
    correct = torch.eq(topk, labels.unsqueeze(1))

    return (correct.sum() / len(labels)).item()