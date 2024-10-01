import torch
import torch.nn as nn
from config import Config

WEIGHTS = torch.tensor([1., 2., 4.])
c_train = nn.CrossEntropyLoss(weight=WEIGHTS.to(Config.device))
c_valid = nn.CrossEntropyLoss(weight=WEIGHTS)

def criterion(logits, targets, mode="train"):
    l = logits.view(logits.shape[0], -1)
    t = targets
    loss = 0
    n = 1
    for i in range(n):
        pred = l[:,i*3:i*3+3]
        gt = t[:,i]
        if mode == "train":
            loss+= c_train(pred, gt)/n
        elif mode == "valid":
            loss+= c_valid(pred, gt)/n
    return loss
