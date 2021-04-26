import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaGANLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(VanillaGANLoss, self).__init__()
        self.crit = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, status):
        """
            :param status: boolean, True/False
        """
        target = torch.empty_like(input).fill_(int(status))
        loss = self.crit(input, target)
        return loss


class LSGANLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(LSGANLoss, self).__init__()
        self.crit = nn.MSELoss(reduction=reduction)

    def forward(self, input, status):
        """
            :param status: boolean, True/False
        """
        target = torch.empty_like(input).fill_(int(status))
        loss = self.crit(input, target)
        return loss


class CharbonnierLoss(nn.Module):
    """ Charbonnier Loss (robust L1)
    """

    def __init__(self, eps=1e-6, reduction='sum'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)

        if self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'mean':
            loss = torch.mean(loss)
        else:
            raise NotImplementedError
        return loss


class CosineSimilarityLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(CosineSimilarityLoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        diff = F.cosine_similarity(input, target, dim=1, eps=self.eps)
        loss = 1.0 - diff.mean()

        return loss
