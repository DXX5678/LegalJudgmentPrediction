import torch
import torch.nn as nn
from tool.loss_tool import MultiLabelSoftmaxLoss
from tool.accuracy_tool import multi_label_accuracy

from model.Predicator import LJPredictor


class LJPLSTM(nn.Module):
    def __init__(self, batch_size, device, hidden_size=768, num_layers=2, bidirectional=True):
        super(LJPLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bi = bidirectional
        self.output_size = self.hidden_size
        self.num_layers = num_layers
        self.device = device
        if self.bi:
            self.output_size = self.output_size // 2

        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.output_size,
                            num_layers=self.num_layers, batch_first=True, bidirectional=self.bi)
        self.predictor = LJPredictor(hidden_size=self.hidden_size)

        self.embedding = nn.Embedding(21128, 768)

        self.criterion = {
            "zm": MultiLabelSoftmaxLoss(200),
            "ft": MultiLabelSoftmaxLoss(183)
        }
        self.accuracy_function = {
            "zm": multi_label_accuracy,
            "ft": multi_label_accuracy
        }

    def forward(self, data):
        x = data[0]
        x = self.embedding(x)
        batch_size = x.size()[0]
        hidden = (
            torch.autograd.Variable(
                torch.zeros(self.num_layers + int(self.bi) * self.num_layers, batch_size, self.output_size)).to(
                self.device),
            torch.autograd.Variable(
                torch.zeros(self.num_layers + int(self.bi) * self.num_layers, batch_size, self.output_size)).to(
                self.device))

        h, c = self.lstm(x, hidden)

        h_ = torch.max(h, dim=1)[0]
        result = self.predictor(h_)
        loss = 0
        loss += self.criterion["ft"](result["ft"], data[1])
        loss += self.criterion["zm"](result["zm"], data[2])
        acc_result = {"zm": None, "ft": None}
        acc_result["ft"] = self.accuracy_function["ft"](result["ft"], data[1], acc_result["ft"])
        acc_result["zm"] = self.accuracy_function["zm"](result["zm"], data[2], acc_result["zm"])
        return {"output": result, "loss": loss, "acc_result": acc_result}
