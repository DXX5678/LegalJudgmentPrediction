import torch
from torch import nn
from pytorch_pretrained_bert import BertModel
from model.Attention import Attention
from model.Predicator import LJPredictor
from tool.loss_tool import MultiLabelSoftmaxLoss
from tool.accuracy_tool import multi_label_accuracy


class MergeModelzm(nn.Module):
    def __init__(self, bert_path, batch_size, device, hidden_size=768, num_layers=2, bidirectional=True):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.bidirectional = bidirectional
        if bidirectional:
            self.output_size = self.output_size // 2
        self.num_layers = num_layers
        self.bert = BertModel.from_pretrained(bert_path)
        unfreeze_layers = ['bert.embeddings', 'layer.10', 'layer.11', 'bert.pooler', 'out.']
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        self.attn = Attention(query_size=self.hidden_size, key_size=self.hidden_size, value_size1=self.batch_size,
                              value_size2=self.hidden_size, output_size=self.output_size)
        self.bi_lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.output_size, num_layers=self.num_layers,
                               batch_first=True, bidirectional=self.bidirectional)
        self.predictor = LJPredictor(hidden_size=self.hidden_size)
        self.dropout = nn.Dropout(0.2)
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
        v = v.unsqueeze(0)
        attn_out = self.attn(v, v, v)
        c_0 = attn_out
        h_0 = torch.autograd.Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.output_size)).to(
            self.device)
        for i in range(self.num_layers * 2):
            if i == 0:
                c_0 = attn_out
            else:
                c_0 = torch.cat((c_0, attn_out), dim=0)
        c_0 = torch.autograd.Variable(c_0)
        output, (h_n, c_n) = self.bi_lstm(y, (h_0, c_0))
        output = torch.max(output, dim=1)[0]
        output = self.dropout(output)
        result = self.predictor(output)
        loss = 0

        loss += self.criterion["zm"](result["zm"], data[2])
        acc_result = {"zm": None, "ft": None}

        acc_result["zm"] = self.accuracy_function["zm"](result["zm"], data[2], acc_result["zm"])
        return {"output": result, "loss": loss, "acc_result": acc_result}
