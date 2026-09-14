import torch.nn as nn


class CTCHead(nn.Module):
    def __init__(self, input_size, vocab_size, dropout=0.1):
        super().__init__()

        hidden = vocab_size // 2

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_size, hidden)

        self.relu = nn.ReLU(inplace=True)

        self.linear2 = nn.Linear(hidden, vocab_size)

    def forward(self, inputs):
        hidden = self.dropout(inputs)
        hidden = self.relu(self.linear1(hidden))

        return self.linear2(hidden)
