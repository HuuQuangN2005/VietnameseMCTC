import argparse
import json
import os

import torch

from srcs.datasets.collator import Collator
from srcs.datasets.vicocktail import load_vicocktail
from srcs.nlp.text_transform import PhonemeTransform, WordTransform
from srcs.nets.e2e import CTCVSR, MCTCVSR, get_model
from srcs.nets.loss.ctc import ctc_decode
from srcs.nets.loss.mctc import mctc_decode
from srcs.trainer.utils import (
    create_loader,
    create_metrics,
    get_metric_results,
    load_configs,
    move_batch,
    set_seed,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIGS_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
MODEL_NAMES = ("MCTCVSR", "CTCVSR")
DEFAULT_MIN_FRAMES = 20
DEFAULT_MAX_FRAMES = 260


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score a checkpoint against the raw transcript."
    )
    parser.add_argument("--configs", default=DEFAULT_CONFIGS_PATH)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--min_frames", type=int)
    parser.add_argument("--max_frames", type=int)

    return parser.parse_args()


def apply_run_args(args, dir_path):
    """Fill in whatever training recorded, so the split matches by default."""
    path = os.path.join(dir_path, "args.json")

    if not os.path.exists(path):
        if args.model is None:
            raise SystemExit(f"--model is required: no args.json found in {dir_path}")
        return args

    with open(path, encoding="utf-8") as file:
        saved = json.load(file)

    for name in ("model", "min_frames", "max_frames"):
        if getattr(args, name) is None:
            setattr(args, name, saved.get(name))

    return args


def main():
    args = parse_args()
    configs = load_configs(args.configs)
    seed = configs["training"]["seed"]
    set_seed(seed)

    ckpt = os.path.abspath(args.ckpt)
    dir_path = ckpt if os.path.isdir(ckpt) else os.path.dirname(ckpt)
    args = apply_run_args(args, dir_path)
    args.min_frames = (
        DEFAULT_MIN_FRAMES if args.min_frames is None else args.min_frames
    )
    args.max_frames = (
        DEFAULT_MAX_FRAMES if args.max_frames is None else args.max_frames
    )

    if args.model == "MCTCVSR":
        model_cls = MCTCVSR
        transform_cls = PhonemeTransform
        decode = mctc_decode
    else:
        model_cls = CTCVSR
        transform_cls = WordTransform
        decode = ctc_decode

    transform = transform_cls(
        **transform_cls.vocab_paths(os.path.join(dir_path, "vocab"))
    )

    model = get_model(model_cls, transform.vocab_size, ckpt=ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    source_split = "train" if args.split == "val" else "test"
    dataset = load_vicocktail(
        split=source_split,
        fraction=args.fraction,
        seed=seed,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )[args.split]

    if len(dataset) == 0:
        raise ValueError(f"{args.split} split is empty.")

    loader = create_loader(
        dataset,
        Collator(args.split, transform),
        configs["evaluation"],
    )

    metrics = create_metrics(transform)
    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            model_inputs = {
                name: value for name, value in batch.items() if name != "texts"
            }
            outputs = model(**model_inputs)

            batch_size = batch["videos"].size(0)
            total_loss += outputs["loss"].detach().float().item() * batch_size
            sample_count += batch_size

            predictions = decode(outputs["logits"], outputs["input_lengths"])

            for ids, text in zip(predictions, batch["texts"]):
                hypothesis = transform.decode_for_metrics(ids)
                reference = transform.reference_for_metrics(text)

                for name, metric in metrics.items():
                    metric.update([hypothesis[name]], [reference[name]])

    print(f"{args.model}  {args.split}  {dataset.num_rows:,} clips")
    print(f"  {'loss':13s} {total_loss / max(1, sample_count):.4f}")

    for name, value in get_metric_results(metrics).items():
        print(f"  {name:13s} {value:.4f}")


if __name__ == "__main__":
    main()
