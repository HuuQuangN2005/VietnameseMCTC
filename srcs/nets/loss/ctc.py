import torch
import torch.nn as nn
import torch.nn.functional as F

from srcs.nets.backend.nets_utils import ctc_alignment_mask, make_non_pad_mask

BLANK_ID = 0


def collapse(ids, blank_id):
    sequence = []
    previous = None

    for current in ids:
        current = int(current)

        if current == blank_id:
            previous = None
            continue

        if current != previous:
            sequence.append(current)

        previous = current

    return sequence


@torch.no_grad()
def ctc_decode(logits, input_lengths, blank_id=BLANK_ID):
    best = logits.argmax(dim=-1).cpu()
    lengths = input_lengths.detach().cpu().tolist()

    return [
        collapse(ids[:length], blank_id) for ids, length in zip(best, lengths)
    ]


class CTCLoss(nn.Module):
    """Plain frame-synchronous CTC over a flat vocabulary.

    Targets are the transform ids untouched, because blank already sits at
    blank_id. MCTCWELoss has to shift its targets instead, since its blank
    lives in a channel of its own rather than in the vocabulary.
    """

    def __init__(self, blank_id=BLANK_ID):
        super().__init__()
        self.blank_id = blank_id
        self.ctc = nn.CTCLoss(blank=blank_id, reduction="none", zero_infinity=True)

    def forward(self, logits, labels, input_lengths, label_lengths):
        label_mask = make_non_pad_mask(
            label_lengths.to(labels.device),
            labels.size(1),
        )
        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)

        loss = self.ctc(
            log_probs,
            labels[label_mask],
            input_lengths,
            label_lengths,
        )
        valid_alignment = ctc_alignment_mask(
            labels,
            input_lengths,
            label_lengths,
        )
        loss = torch.where(valid_alignment, loss, torch.zeros_like(loss))

        return loss.sum() / labels.size(0)
