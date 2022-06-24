import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from LJPapp.model.Attention import Attention
from LJPapp.model.Predicator import LJPredictor
from LJPapp.tool.loss_tool import MultiLabelSoftmaxLoss
from LJPapp.tool.accuracy_tool import multi_label_accuracy


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

    def forward(self, x, attn):
        fc_input = x
        outputs = {}
        batch_size = x.size()[0]
        self.init_hidden(batch_size)
        h0 = torch.autograd.Variable(torch.zeros(batch_size, self.hidden_dim)).to(self.device)
        first = []
        for a in range(0, len(self.task_name) + 1):
            first.append(True)
        for a in range(1, len(self.task_name) + 1):
            if a == 1:
                h, c = self.cell_list[a - 1](fc_input, (h0, attn))
            else:
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


class TopMergeModel(nn.Module):
    def __init__(self, bert_path, batch_size, device, hidden_size=768,):
        super(TopMergeModel, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.hidden_size = hidden_size
        self.bert = BertModel.from_pretrained(bert_path)
        unfreeze_layers = ['bert.embeddings', 'layer.10', 'layer.11', 'bert.pooler', 'out.']
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        self.attn = Attention(query_size=self.hidden_size, key_size=self.hidden_size, value_size1=self.batch_size,
                              value_size2=self.hidden_size, output_size=self.hidden_size)
        self.decoder = LSTMDecoder(self.device, self.hidden_size)

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

    def forward(self, data):
        x = data[0]
        y, v = self.bert(x, output_all_encoded_layers=False)
        v_temp = v
        v = v.unsqueeze(0)
        attn_out = self.attn(v, v, v)
        attn_out = attn_out.squeeze(0)
        hidden = self.dropout(v_temp)
        result = self.decoder(hidden, attn_out)
        for name in result:
            result[name] = self.predictor(self.dropout(result[name]))[name]

        loss = 0
        loss += self.criterion["ft"](result["ft"], data[1])
        loss += self.criterion["zm"](result["zm"], data[2])
        acc_result = {"zm": None, "ft": None}
        acc_result["ft"] = self.accuracy_function["ft"](result["ft"], data[1], acc_result["ft"])
        acc_result["zm"] = self.accuracy_function["zm"](result["zm"], data[2], acc_result["zm"])
        return {"output": result, "loss": loss, "acc_result": acc_result}
