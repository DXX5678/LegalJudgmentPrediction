import torch
from torch.utils.data import Dataset


class CAILDataset(Dataset):
    def __init__(self, data, tokenizer, device, encoding="utf8"):
        self.input = data[0]
        self.device = device
        self.target_articles = data[1]
        self.target_accusations = data[2]
        self.tokenizer = tokenizer
        self.encoding = encoding
        self.len = len(self.input)

    def __getitem__(self, index):
        input_data = self.input[index]
        target_articles = torch.tensor(self.target_articles[index]).to(self.device)
        target_accusations = torch.tensor(self.target_accusations[index]).to(self.device)

        input_data = self.tokenizer.tokenize(input_data)
        if len(input_data) <= 510:
            input_data = ["[CLS]"] + input_data + ["[SEP]"]
            while len(input_data) < 512:
                input_data.append("[PAD]")
        else:
            input_data = input_data[0:510]
            input_data = ["[CLS]"] + input_data + ["[SEP]"]
        input_data = torch.tensor(self.tokenizer.convert_tokens_to_ids(input_data)).to(self.device)
        return input_data, target_articles, target_accusations

    def __len__(self):
        return self.len
