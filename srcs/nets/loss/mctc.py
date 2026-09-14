import torch
import torch.nn as nn
import torch.nn.functional as F

from srcs.nets.backend.nets_utils import ctc_alignment_mask, make_non_pad_mask


def to_log_probs(logits):
    return {
        "blank": logits["blank"].float(),
        "rhyme": F.log_softmax(logits["rhyme"].float(), dim=-1),
        "initial": F.log_softmax(logits["initial"].float(), dim=-1),
        "tone": F.log_softmax(logits["tone"].float(), dim=-1),
    }


def pick_syllables(log_probs, syllables):
    initial_ids, rhyme_ids, tone_ids = syllables.unbind(-1)

    rhyme = log_probs["rhyme"][:, :, rhyme_ids]
    initial = log_probs["initial"][:, :, rhyme_ids, initial_ids]
    tone = log_probs["tone"][:, :, rhyme_ids, tone_ids]

    return rhyme + initial + tone


def collapse(syllable_ids, blank_mask, length):
    sequence = []
    previous = None

    for current_ids, is_blank in zip(syllable_ids[:length], blank_mask[:length]):
        if is_blank:
            previous = None
            continue

        current_ids = current_ids.tolist()

        if current_ids != previous:
            sequence.append(current_ids)

        previous = current_ids

    return sequence


@torch.no_grad()
def mctc_decode(logits, input_lengths):
    log_probs = to_log_probs(logits)

    initial_log_probs, initial_ids = log_probs["initial"].max(dim=-1)
    tone_log_probs, tone_ids = log_probs["tone"].max(dim=-1)

    syllable_log_probs = log_probs["rhyme"] + initial_log_probs + tone_log_probs
    syllable_log_probs, rhyme_ids = syllable_log_probs.max(dim=-1)

    blank_mask = (log_probs["blank"] >= syllable_log_probs).cpu()

    rhyme_ids = rhyme_ids.unsqueeze(-1)
    syllable_ids = torch.cat(
        [
            initial_ids.gather(-1, rhyme_ids),
            rhyme_ids,
            tone_ids.gather(-1, rhyme_ids),
        ],
        dim=-1,
    ).cpu()

    lengths = input_lengths.detach().cpu().tolist()

    return [
        collapse(ids, mask, length)
        for ids, mask, length in zip(syllable_ids, blank_mask, lengths)
    ]


class MCTCWELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)

    def forward(self, logits, labels, input_lengths, label_lengths):
        label_mask = make_non_pad_mask(
            label_lengths.to(labels.device),
            labels.size(1),
        )
        syllables, targets = torch.unique(
            labels[label_mask],
            dim=0,
            return_inverse=True,
        )

        log_probs = to_log_probs(logits)
        blank_logits = log_probs["blank"]

        blank_scores = F.logsigmoid(blank_logits).unsqueeze(-1)
        emit_scores = F.logsigmoid(-blank_logits).unsqueeze(-1)
        emit_scores = emit_scores + pick_syllables(log_probs, syllables)

        scores = torch.cat([blank_scores, emit_scores], dim=-1)
        normalizer = scores.logsumexp(dim=-1)

        loss = self.ctc(
            (scores - normalizer.unsqueeze(-1)).transpose(0, 1),
            targets + 1,
            input_lengths,
            label_lengths,
        )

        frame_mask = make_non_pad_mask(
            input_lengths.to(scores.device),
            scores.size(1),
        )
        correction = (normalizer * frame_mask).sum(dim=1)
        valid_alignment = ctc_alignment_mask(
            labels,
            input_lengths,
            label_lengths,
        )
        losses = torch.where(
            valid_alignment,
            loss - correction,
            torch.zeros_like(loss),
        )

        return losses.sum() / labels.size(0)
