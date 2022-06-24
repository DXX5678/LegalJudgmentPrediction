import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        super().__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size

        self.attn = nn.Linear(self.query_size + self.key_size, self.value_size1)
        self.attn_combine = nn.Linear(self.query_size + self.value_size2, self.output_size)

    def forward(self, query, key, value):
        attn_weights = F.softmax(self.attn(torch.cat((query[0], key[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), value)
        output = torch.cat((query[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        return output
