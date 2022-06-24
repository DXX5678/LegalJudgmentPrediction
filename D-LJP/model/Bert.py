import torch.nn as nn
from model.Predicator import LJPredictor
from tool.loss_tool import MultiLabelSoftmaxLoss
from tool.accuracy_tool import multi_label_accuracy
from pytorch_pretrained_bert import BertModel


class LJPBert(nn.Module):
    def __init__(self, bert_path, batch_size, device, hidden_size=768):
        super(LJPBert, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.hidden_size = hidden_size
        self.bert = BertModel.from_pretrained(bert_path)
        unfreeze_layers = ['bert.embeddings', 'layer.9', 'layer.10', 'layer.11', 'bert.pooler', 'out.']
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        self.predictor = LJPredictor(hidden_size=self.hidden_size)

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
        _, y = self.bert(x, output_all_encoded_layers=False)
        result = self.predictor(y)
        loss = 0
        loss += self.criterion["ft"](result["ft"], data[1])
        loss += self.criterion["zm"](result["zm"], data[2])
        acc_result = {"zm": None, "ft": None}
        acc_result["ft"] = self.accuracy_function["ft"](result["ft"], data[1], acc_result["ft"])
        acc_result["zm"] = self.accuracy_function["zm"](result["zm"], data[2], acc_result["zm"])
        return {"output": result, "loss": loss, "acc_result": acc_result}
