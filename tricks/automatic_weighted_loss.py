# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True).cuda()
        self.params = torch.nn.Parameter(params).cuda()
        self.cnt = 0

    def forward(self, *x, rank=0):
        loss_sum = 0
        self.cnt += 1
        if self.cnt % 100 == 0 and rank == 0:
            print((self.params[0]), (self.params[1]))
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2).cuda()
    print(awl.parameters())
    print(awl.state_dict())