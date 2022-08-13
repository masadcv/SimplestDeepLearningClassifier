import torch.nn as nn


def get_loss(name, reduction="mean", weight=None):
    lossname_to_func = {"XENT": nn.CrossEntropyLoss(weight=weight, reduction=reduction)}
    return lossname_to_func[name]
