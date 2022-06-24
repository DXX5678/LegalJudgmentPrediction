import torch
from torch import optim
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import DataLoader
from dataset.CAILDataset import CAILDataset
from model.MergeModel import MergeModel
from preprocess.preprocess_data import preprocess_data
from tool.eval_tool import valid

if __name__ == "__main__":
    eval_data_path = "data/data_valid.json"
    log_path = "log/"
    bert_tokenizer_path = "bert-chinese/vocab.txt"
    bert_path = "bert-chinese/"
    batch_size = 32
    learning_rate = 1e-4
    model_path = "model/result/17.pkl"
    state = torch.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MergeModel(bert_path, batch_size, device).to(device)
    model.load_state_dict(state['net'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    optimizer.load_state_dict(state['optimizer'])
    tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
    eval_data = preprocess_data(eval_data_path, 32)
    eval_dataset = CAILDataset(eval_data, tokenizer, device)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=0)
    valid(eval_dataloader, model, 0, log_path)

