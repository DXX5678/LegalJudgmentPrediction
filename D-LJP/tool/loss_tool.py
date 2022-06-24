from torch import nn


class MultiLabelSoftmaxLoss(nn.Module):
    def __init__(self, task_num=0):
        super().__init__()
        self.task_num = task_num
        self.criterion = []
        for a in range(0, self.task_num):
            self.criterion.append(nn.CrossEntropyLoss())

    def forward(self, outputs, labels):
        if labels is None:
            return 0
        loss = 0
        for a in range(0, len(outputs[0])):
            o = outputs[:, a, :].view(outputs.size()[0], -1)
            loss += self.criterion[a](o, labels[:, a])

        return loss
