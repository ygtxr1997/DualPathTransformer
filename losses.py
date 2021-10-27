import torch
from torch import nn


class CosFace(nn.Module):
    def __init__(self, s=64.0, m=0.4):  # default:m=0.4
        super(CosFace, self).__init__()
        self.s = s
        self.m = m
        self.a = 1.3  # 1.2
        self.r = 1.5  # 1.25
        self.k = 0.1  # 0.1

        self.avg_min = 0
        self.avg_max = 0
        self.avg_avg = 0
        self.cnt = 0

        print('CosFace:m=%f, a=%f, r=%f, k=%f' % (self.m, self.a, self.r, self.k))

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)

        cosine.acos_()
        import os
        rank = -1 if __name__ == "__main__" else int(os.environ['RANK'])
        self.cnt += 1
        self.avg_min += cosine[index, label[index]].min().data.cpu()
        self.avg_max += cosine[index, label[index]].max().data.cpu()
        self.avg_avg += cosine[index, label[index]].mean().data.cpu()
        if self.cnt % 100 == 0 and rank == 0:
            import logging
            logging.info('before:[%d],mean:[%.4f],range:[%.4f~%.4f]', self.cnt,
                         self.avg_avg / self.cnt,
                         self.avg_min / self.cnt,
                         self.avg_max / self.cnt)
        cosine.cos_()

        """ 1. vanilla """
        # cosine[index] -= m_hot

        """ 2. adaptive - piecewise """
        # big_index = torch.where(cosine[index].acos_() >= self.a)  # large angle -> m=0.3
        # sma_index = torch.where(cosine[index].acos_() < self.a)  # small angle -> m=0.5
        # m_hot[big_index] /= self.r
        # m_hot[sma_index] *= self.r
        # cosine[index] -= m_hot

        """ 3. adaptive - subtract linear """
        a = self.a
        k = self.k
        m_hot[range(0, index.size()[0]), label[index]] -= k * (cosine[index, label[index]].acos_() - a)
        cosine[index] -= m_hot

        ret = cosine * self.s
        return ret


class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

        self.avg_min = 0
        self.avg_max = 0
        self.avg_avg = 0
        self.cnt = 0

    def forward(self, cosine: torch.Tensor, label):
        # cosine: (batch, depart_id)
        # label: (batch, )
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        # index: (batch_valid, )
        # m_hot: (batch_valid, depart_id)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()

        import os
        rank = int(os.environ['RANK'])
        self.cnt += 1
        self.avg_min += cosine[index, label[index]].min().data.cpu()
        self.avg_max += cosine[index, label[index]].max().data.cpu()
        self.avg_avg += cosine[index, label[index]].mean().data.cpu()
        if self.cnt % 100 == 0 and rank == 0:
            import logging
            logging.info('before:[%d],mean:[%.4f],range:[%.4f~%.4f]', self.cnt,
                         self.avg_avg / self.cnt,
                         self.avg_min / self.cnt,
                         self.avg_max / self.cnt)

        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        # print('after:', cosine.min(), cosine.max())
        return cosine


class Linear(nn.Module):
    def __init__(self, a=0.88, b=0.88, s=64.0):
        super(Linear, self).__init__()
        self.a = a
        self.b = b
        self.s = s

        self.avg_min = 0
        self.avg_max = 0
        self.avg_avg = 0
        self.cnt = 0

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        # m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        # m_hot.scatter_(1, label[index, None], 1)

        a_hot = torch.ones(index.size()[0], cosine.size()[1], device=cosine.device)
        b_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        a_hot.scatter_(1, label[index, None], -self.a)
        b_hot.scatter_(1, label[index, None], self.b)

        # print('before-full:', cosine[index].min(), cosine[index].max())
        # print('before-pos:', cosine[index, label[index]].min(), cosine[index, label[index]].max())
        cosine[index, label[index]] = cosine[index, label[index]].acos_()

        import os
        rank = int(os.environ['RANK'])
        self.cnt += 1
        self.avg_min += cosine[index, label[index]].min().data.cpu()
        self.avg_max += cosine[index, label[index]].max().data.cpu()
        self.avg_avg += cosine[index, label[index]].mean().data.cpu()
        if self.cnt % 100 == 0 and rank == 0:
            import logging
            logging.info('before:[%d],mean:[%.4f],range:[%.4f~%.4f]', self.cnt,
                         self.avg_avg / self.cnt,
                         self.avg_min / self.cnt,
                         self.avg_max / self.cnt)

        # print('acos-full:', cosine[index].min(), cosine[index].max())
        # print('acos-pos:', cosine[index, label[index]].min(), cosine[index, label[index]].max())
        cosine[index] = cosine[index] * a_hot + b_hot
        # print('-a+b-full:', cosine[index].min(), cosine[index].max())
        # cosine.cos_().mul_(self.s)
        cosine.mul_(self.s)

        # print('after-full:', cosine[index].min(), cosine[index].max())
        return cosine


class Quadratic(nn.Module):
    def __init__(self, a=0.12, b=2.6, c=1.6, s=64.0):
        super(Quadratic, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.s = s
        print('[Quadratic]: a=%f, b=%f, c=%f' % (a, b, c))

        self.avg_min = 0
        self.avg_max = 0
        self.avg_avg = 0
        self.cnt = 0

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]

        a_hot = torch.ones(index.size()[0], cosine.size()[1], device=cosine.device)
        b_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        a_hot.scatter_(1, label[index, None], -self.a)
        b_hot.scatter_(1, label[index, None], self.b)

        # print('before-full:', cosine[index].min(), cosine[index].max())
        # print('before-pos:', cosine[index, label[index]].min(), cosine[index, label[index]].max())
        cosine[index, label[index]] = cosine[index, label[index]].acos_()

        import os
        rank = int(os.environ['RANK'])
        self.cnt += 1
        self.avg_min += cosine[index, label[index]].min().data.cpu()
        self.avg_max += cosine[index, label[index]].max().data.cpu()
        self.avg_avg += cosine[index, label[index]].mean().data.cpu()
        if self.cnt % 100 == 0 and rank == 0:
            import logging
            logging.info('before:[%d],mean:[%.4f],range:[%.4f~%.4f]', self.cnt,
                         self.avg_avg / self.cnt,
                         self.avg_min / self.cnt,
                         self.avg_max / self.cnt)

        cosine[index, label[index]] = cosine[index, label[index]] + self.b
        cosine[index, label[index]] = cosine[index, label[index]] * cosine[index, label[index]]
        cosine[index, label[index]] = -self.a * cosine[index, label[index]]
        cosine[index, label[index]] = cosine[index, label[index]] + self.c
        # print('acos-full:', cosine[index].min(), cosine[index].max())
        # print('acos-pos:', cosine[index, label[index]].min(), cosine[index, label[index]].max())
        # cosine[index] = cosine[index] * a_hot + b_hot
        # print('-a+b-full:', cosine[index].min(), cosine[index].max())
        # cosine.cos_().mul_(self.s)
        cosine.mul_(self.s)

        # print('after-full:', cosine[index].min(), cosine[index].max())
        return cosine


if __name__ == '__main__':
    cosine = torch.randn(6, 8) / 100
    cosine[0][2] = 0.3
    cosine[1][4] = 0.4
    cosine[2][6] = 0.5
    cosine[3][5] = 0.6
    cosine[4][3] = 0.7
    cosine[5][0] = 0.8
    label = torch.tensor([-1, 4, -1, 5, 3, -1])

    print('cosine:', cosine)
    print('label:', label)

    layer = CosFace(s=1)
    print('after forward:', layer(cosine, label))

    index = torch.where(label != -1)[0]
    m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
    m_hot.scatter_(1, label[index, None], 1)

    print('index:', index)
    print('label[index]:', label[index])
    print('cosine[index]:', cosine[index])
    print('m_hot:', m_hot)

    a = b = 0.88
    a_hot = torch.ones(index.size()[0], cosine.size()[1], device=cosine.device)
    b_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
    a_hot.scatter_(1, label[index, None], -a)
    b_hot.scatter_(1, label[index, None], b)
    print('a_hot:', a_hot)
    print('b_hot:', b_hot)

    s = 64.0
    print('before margin', cosine)
    cosine[index, label[index]] = cosine[index, label[index]].acos_()
    print('acos', cosine)
    cosine[index] = (cosine[index]) * a_hot + b_hot
    print('after margin', cosine)
    cosine.mul_(s)

