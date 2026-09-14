import torch.nn as nn

from srcs.nets.backend.nets_utils import make_activation


class TemporalBlock(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        kernel_size,
        dilation,
        dropout,
        relu_type,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            input_size,
            output_size,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.batchnorm1 = nn.BatchNorm1d(output_size)
        self.relu1 = make_activation(relu_type, output_size)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            output_size,
            output_size,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.batchnorm2 = nn.BatchNorm1d(output_size)
        self.relu2 = make_activation(relu_type, output_size)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = None
        if input_size != output_size:
            self.downsample = nn.Conv1d(input_size, output_size, kernel_size=1)

        self.relu = make_activation(relu_type, output_size)

    def forward(self, inputs):
        hidden = self.dropout1(self.relu1(self.batchnorm1(self.conv1(inputs))))
        hidden = self.dropout2(self.relu2(self.batchnorm2(self.conv2(hidden))))

        residual = inputs if self.downsample is None else self.downsample(inputs)
        return self.relu(hidden + residual)


class TCN(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_channels,
        kernel_size=3,
        dropout=0.2,
        relu_type="prelu",
    ):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number.")

        self.output_size = num_channels[-1]
        blocks = []

        for index, output_size in enumerate(num_channels):
            input_size = num_inputs if index == 0 else num_channels[index - 1]

            blocks.append(
                TemporalBlock(
                    input_size=input_size,
                    output_size=output_size,
                    kernel_size=kernel_size,
                    dilation=2**index,
                    dropout=dropout,
                    relu_type=relu_type,
                )
            )

        self.network = nn.ModuleList(blocks)

    def forward(self, inputs):
        hidden = inputs.transpose(1, 2)

        for block in self.network:
            hidden = block(hidden)

        return hidden.transpose(1, 2)
