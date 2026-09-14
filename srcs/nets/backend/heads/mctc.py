import torch
import torch.nn as nn


class MCTCHead(nn.Module):
    def __init__(
        self,
        input_size,
        initial_size,
        rhyme_size,
        tone_size,
        dropout=0.1,
    ):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_size, 1)
        self.linear2 = nn.Linear(input_size, rhyme_size)
        self.linear3 = nn.Linear(input_size, initial_size)
        self.linear4 = nn.Linear(input_size, tone_size)

        self.param1 = nn.Parameter(torch.zeros(rhyme_size, initial_size))
        self.param2 = nn.Parameter(torch.zeros(rhyme_size, tone_size))

    def forward(self, inputs):
        inputs = self.dropout1(inputs)

        blank = self.linear1(inputs).squeeze(-1)
        rhyme = self.linear2(inputs)

        initial = self.linear3(inputs).unsqueeze(-2)
        tone = self.linear4(inputs).unsqueeze(-2)

        return {
            "blank": blank,
            "rhyme": rhyme,
            "initial": initial + self.param1,
            "tone": tone + self.param2,
        }
