import torch
import torch.nn as nn

from srcs.nets.backend.nets_utils import make_activation

STAGE_OUT_CHANNELS = {
    0.5: [-1, 24, 48, 96, 192, 1024],
    1.0: [-1, 24, 116, 232, 464, 1024],
    1.5: [-1, 24, 176, 352, 704, 1024],
    2.0: [-1, 24, 244, 488, 976, 2048],
}
STAGE_REPEATS = (4, 8, 4)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    return x.view(batchsize, -1, height, width)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super().__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            self.banch2 = nn.Sequential(
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
            self.banch2 = nn.Sequential(
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, x):
        if self.benchmodel == 1:
            x1 = x[:, : (x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2) :, :, :]
            out = self._concat(x1, self.banch2(x2))
        else:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, width_mult=1.0):
        super().__init__()

        if width_mult not in STAGE_OUT_CHANNELS:
            raise ValueError(
                "Width multiplier should be in [0.5, 1.0, 1.5, 2.0]. "
                f"Current value: {width_mult}"
            )

        self.stage_repeats = STAGE_REPEATS
        self.stage_out_channels = STAGE_OUT_CHANNELS[width_mult]
        self.output_size = self.stage_out_channels[-1]

        input_channel = self.stage_out_channels[1]
        features = []

        for stage_index, repeat_count in enumerate(self.stage_repeats):
            output_channel = self.stage_out_channels[stage_index + 2]

            for block_index in range(repeat_count):
                stride = 2 if block_index == 0 else 1
                benchmodel = 2 if block_index == 0 else 1

                features.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        stride,
                        benchmodel,
                    )
                )
                input_channel = output_channel

        self.features = nn.Sequential(*features)
        self.conv_last = conv_1x1_bn(input_channel, self.output_size)
        self.globalpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, inputs):
        hidden = self.features(inputs)
        hidden = self.conv_last(hidden)
        hidden = self.globalpool(hidden)

        return hidden.flatten(1)


class VideoShuffleNet(nn.Module):
    def __init__(self, width_mult=1.0, relu_type="prelu"):
        super().__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                24,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(24),
            make_activation(relu_type, 24),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.trunk = ShuffleNetV2(width_mult)
        self.output_size = self.trunk.output_size

    def forward(self, videos):
        batch_size = videos.size(0)
        hidden = self.frontend3D(videos.transpose(1, 2))
        time = hidden.size(2)
        hidden = hidden.transpose(1, 2).flatten(0, 1)
        hidden = self.trunk(hidden)

        return hidden.view(batch_size, time, -1)


def video_shufflenet(width_mult=1.0, relu_type="prelu"):
    return VideoShuffleNet(width_mult, relu_type)
