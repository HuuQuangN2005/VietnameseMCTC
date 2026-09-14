import torch
import torch.nn as nn


def make_activation(relu_type, num_channels):
    if relu_type == "prelu":
        return nn.PReLU(num_parameters=num_channels)

    if relu_type == "relu":
        return nn.ReLU(inplace=True)

    raise ValueError("relu_type must be 'prelu' or 'relu'.")


def make_non_pad_mask(lengths, max_length=None):
    lengths = torch.as_tensor(lengths)
    if lengths.ndim != 1:
        raise ValueError("lengths must be one-dimensional.")
    if lengths.numel() == 0:
        raise ValueError("lengths must not be empty.")

    if max_length is None:
        max_length = int(lengths.max().item())

    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def ctc_alignment_mask(labels, input_lengths, label_lengths):
    input_lengths = torch.as_tensor(input_lengths, device=labels.device)
    label_lengths = torch.as_tensor(label_lengths, device=labels.device)
    repeat_counts = torch.zeros_like(label_lengths)

    if labels.size(1) > 1:
        repeated = labels[:, 1:] == labels[:, :-1]

        if repeated.ndim > 2:
            repeated = repeated.all(dim=tuple(range(2, repeated.ndim)))

        pair_lengths = (label_lengths - 1).clamp_min(0)
        pair_mask = make_non_pad_mask(pair_lengths, labels.size(1) - 1)
        repeat_counts = (repeated & pair_mask).sum(dim=1)

    return input_lengths >= label_lengths + repeat_counts
