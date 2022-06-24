from torch import nn


class LJPredictor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.article_fc = nn.Linear(self.hidden_size, 183 * 2)
        self.accusation_fc = nn.Linear(self.hidden_size, 200 * 2)

    def forward(self, h):
        article = self.article_fc(h)
        accusation = self.accusation_fc(h)
        batch_size = h.size()[0]
        article = article.view(batch_size, -1, 2)
        accusation = accusation.view(batch_size, -1, 2)
        return {"zm": accusation, "ft": article}
