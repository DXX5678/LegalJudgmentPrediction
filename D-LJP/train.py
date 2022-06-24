import torch
from torch import optim
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import DataLoader
from dataset.CAILDataset import CAILDataset
from model.MergeModel import MergeModel
from preprocess.preprocess_data import preprocess_data
from tool.train_tool import train

if __name__ == "__main__":
    train_data_path = "data/data_train.json"
    eval_data_path = "data/data_valid.json"
    test_data_path = "data/data_test.json"
    epochs = 5
    eval_per_epoch = 0
    output_path = "model/result/"
    log_path = "log/"
    bert_tokenizer_path = "bert-chinese/vocab.txt"
    bert_path = "bert-chinese/"
    batch_size = 32
    learning_rate = 1e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MergeModel(bert_path, batch_size, device).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
    train_data = preprocess_data(train_data_path, 198400)
    train_dataset = CAILDataset(train_data, tokenizer, device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    eval_data = preprocess_data(eval_data_path)
    eval_dataset = CAILDataset(eval_data, tokenizer, device)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=0)
    test_data = preprocess_data(test_data_path)
    test_dataset = CAILDataset(test_data, tokenizer, device)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    train(train_dataloader, eval_dataloader, test_dataloader, model, optimizer, epochs, eval_per_epoch, output_path,
          log_path)
