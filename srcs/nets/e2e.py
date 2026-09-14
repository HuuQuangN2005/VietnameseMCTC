import torch.nn as nn

from srcs.nets.backend.frontend.shufflenet import video_shufflenet
from srcs.nets.backend.heads.ctc import CTCHead
from srcs.nets.backend.heads.mctc import MCTCHead
from srcs.nets.backend.nets_utils import make_non_pad_mask
from srcs.nets.backend.TCN import TCN
from srcs.nets.loss.ctc import CTCLoss
from srcs.nets.loss.mctc import MCTCWELoss
from srcs.nets.utils import load_visual_pretrained, load_weights


class MCTCVSR(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=256,
        tcn_channels=512,
        num_layers=6,
        kernel_size=3,
        dropout=0.1,
        relu_type="prelu",
        visual_pretrained=None,
    ):
        super().__init__()
        self.frontend = video_shufflenet(relu_type=relu_type)
        self.tcn = TCN(
            num_inputs=self.frontend.output_size,
            num_channels=[tcn_channels] * num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            relu_type=relu_type,
        )
        self.projection = nn.Linear(self.tcn.output_size, hidden_dim)

        if visual_pretrained:
            load_visual_pretrained(self, visual_pretrained)

        self.head = MCTCHead(
            input_size=hidden_dim,
            initial_size=vocab_size["initial"],
            rhyme_size=vocab_size["rhyme"],
            tone_size=vocab_size["tone"],
            dropout=dropout,
        )
        self.loss_fn = MCTCWELoss()

    def forward(self, videos, video_lengths, labels=None, label_lengths=None):
        features = self.frontend(videos)
        features = self.tcn(features)
        features = self.projection(features)

        valid_mask = make_non_pad_mask(
            video_lengths.to(features.device),
            features.size(1),
        ).unsqueeze(-1)

        features = features * valid_mask.to(features.dtype)

        logits = self.head(features)
        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels, video_lengths, label_lengths)

        return {
            "loss": loss,
            "logits": logits,
            "input_lengths": video_lengths,
        }


class CTCVSR(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=256,
        tcn_channels=512,
        num_layers=6,
        kernel_size=3,
        dropout=0.1,
        relu_type="prelu",
        visual_pretrained=None,
    ):
        super().__init__()
        self.frontend = video_shufflenet(relu_type=relu_type)
        self.tcn = TCN(
            num_inputs=self.frontend.output_size,
            num_channels=[tcn_channels] * num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            relu_type=relu_type,
        )
        self.projection = nn.Linear(self.tcn.output_size, hidden_dim)

        if visual_pretrained:
            load_visual_pretrained(self, visual_pretrained)

        self.head = CTCHead(
            input_size=hidden_dim,
            vocab_size=vocab_size,
            dropout=dropout,
        )
        self.loss_fn = CTCLoss()

    def forward(self, videos, video_lengths, labels=None, label_lengths=None):
        features = self.frontend(videos)
        features = self.tcn(features)
        features = self.projection(features)

        valid_mask = make_non_pad_mask(
            video_lengths.to(features.device),
            features.size(1),
        ).unsqueeze(-1)

        features = features * valid_mask.to(features.dtype)

        logits = self.head(features)
        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels, video_lengths, label_lengths)

        return {
            "loss": loss,
            "logits": logits,
            "input_lengths": video_lengths,
        }


def get_model(model_cls, vocab_size, ckpt=None, **model_config):
    network = model_cls(vocab_size=vocab_size, **model_config)

    if ckpt:
        load_weights(network, ckpt)

    return network
