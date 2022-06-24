import torch
import torch.nn as nn
from model.Predicator import LJPredictor
from tool.loss_tool import MultiLabelSoftmaxLoss
from tool.accuracy_tool import multi_label_accuracy


class CNNEncoder(nn.Module):
    def __init__(self, hidden_size=768):
        super(CNNEncoder, self).__init__()

        self.emb_dim = hidden_size
        self.output_dim = self.emb_dim // 4

        self.min_gram = 2
        self.max_gram = 5
        self.convs = []
        for a in range(self.min_gram, self.max_gram + 1):
            self.convs.append(nn.Conv2d(1, self.output_dim, (a, self.emb_dim)))

        self.convs = nn.ModuleList(self.convs)
        self.feature_len = self.emb_dim
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size()[0]

        x = x.view(batch_size, 1, -1, self.emb_dim)

        conv_out = []
        gram = self.min_gram
        for conv in self.convs:
            y = self.relu(conv(x))
            y = torch.max(y, dim=2)[0].view(batch_size, -1)

            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1)

        return conv_out


class LSTMDecoder(nn.Module):
    def __init__(self, device, hidden_size=768):
        super(LSTMDecoder, self).__init__()
        self.feature_len = hidden_size
        self.device = device
        features = self.feature_len
        self.hidden_dim = features

        self.task_name = ["ft", "zm"]

        self.midfc = []
        for x in self.task_name:
            self.midfc.append(nn.Linear(features, features))

        self.cell_list = []
        for x in self.task_name:
            self.cell_list.append(nn.LSTMCell(self.feature_len, self.feature_len))
        """
        self.hidden_state_fc_list = []
        for a in range(0, len(self.task_name) + 1):
            arr = []
            for b in range(0, len(self.task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.hidden_state_fc_list.append(arr)

        self.cell_state_fc_list = []
        for a in range(0, len(self.task_name) + 1):
            arr = []
            for b in range(0, len(self.task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.cell_state_fc_list.append(arr)
        """
        self.midfc = nn.ModuleList(self.midfc)
        self.cell_list = nn.ModuleList(self.cell_list)
        # self.hidden_state_fc_list = nn.ModuleList(self.hidden_state_fc_list)
        # self.cell_state_fc_list = nn.ModuleList(self.cell_state_fc_list)

    def init_hidden(self, bs):
        self.hidden_list = []
        for a in range(1, len(self.task_name) + 1):
            self.hidden_list.append((torch.autograd.Variable(torch.zeros(bs, self.hidden_dim)).to(self.device),
                                     torch.autograd.Variable(torch.zeros(bs, self.hidden_dim)).to(self.device)))

    def forward(self, x):
        fc_input = x
        outputs = {}
        batch_size = x.size()[0]
        self.init_hidden(batch_size)

        first = []
        for a in range(0, len(self.task_name) + 1):
            first.append(True)
        for a in range(1, len(self.task_name) + 1):
            h, c = self.cell_list[a - 1](fc_input, self.hidden_list[a - 1])
            for b in range(1, len(self.task_name) + 1):
                hp, cp = self.hidden_list[b - 1]
                if first[b]:
                    first[b] = False
                    hp, cp = h, c
                """
                else:
                    hp = hp + self.hidden_state_fc_list[a][b](h)
                    cp = cp + self.cell_state_fc_list[a][b](c)
                """
                self.hidden_list[b - 1] = (hp, cp)
            outputs[self.task_name[a - 1]] = self.midfc[a - 1](h).view(batch_size, -1)

        return outputs


class TopJudge(nn.Module):
    def __init__(self, batch_size, device, hidden_size=768):
        super(TopJudge, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = CNNEncoder(hidden_size)
        self.decoder = LSTMDecoder(device, hidden_size)

        self.predictor = LJPredictor(hidden_size=self.hidden_size)
        self.dropout = nn.Dropout(0.4)

        self.criterion = {
            "zm": MultiLabelSoftmaxLoss(200),
            "ft": MultiLabelSoftmaxLoss(183)
        }
        self.accuracy_function = {
            "zm": multi_label_accuracy,
            "ft": multi_label_accuracy
        }

        self.embedding = nn.Embedding(21128, 768)

    def forward(self, data):
        x = data[0]
        x = self.embedding(x)
        hidden = self.encoder(x)
        hidden = self.dropout(hidden)
        result = self.decoder(hidden)
        for name in result:
            result[name] = self.predictor(self.dropout(result[name]))[name]

        loss = 0
        loss += self.criterion["ft"](result["ft"], data[1])
        loss += self.criterion["zm"](result["zm"], data[2])
        acc_result = {"zm": None, "ft": None}
        acc_result["ft"] = self.accuracy_function["ft"](result["ft"], data[1], acc_result["ft"])
        acc_result["zm"] = self.accuracy_function["zm"](result["zm"], data[2], acc_result["zm"])
        return {"output": result, "loss": loss, "acc_result": acc_result}
