import argparse
import json
import os

from srcs.datasets.collator import Collator
from srcs.datasets.vicocktail import load_vicocktail
from srcs.nlp.text_transform import WordTransform
from srcs.nets.e2e import CTCVSR, get_model
from srcs.trainer.trainer import CTCTrainer
from srcs.trainer.utils import create_loader, load_configs, set_seed

MODEL_NAME = "CTCVSR"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIGS_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", MODEL_NAME)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CTCVSR on ViCocktail.")
    parser.add_argument("--configs", default=DEFAULT_CONFIGS_PATH)
    parser.add_argument("--ckpt")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--visual_pretrained")
    parser.add_argument("--min_frames", type=int, default=20)
    parser.add_argument("--max_frames", type=int, default=260)
    parser.add_argument("--epochs", type=int, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    for path in (args.ckpt, args.visual_pretrained):
        if path and not os.path.exists(path):
            raise FileNotFoundError(path)

    train_configs = load_configs(args.configs)["training"]
    seed = train_configs["seed"]
    set_seed(seed)

    splits = load_vicocktail(
        split="train",
        fraction=args.fraction,
        seed=seed,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )

    for name in ("train", "val"):
        if len(splits[name]) == 0:
            raise ValueError(f"{name} split is empty.")

    output_dir = os.path.abspath(args.output_dir)
    vocab_dir = os.path.join(output_dir, "vocab")
    os.makedirs(vocab_dir, exist_ok=True)

    with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as file:
        json.dump({"model": MODEL_NAME, **vars(args)}, file, indent=2)

    text_transform = WordTransform(
        train_dataset=splits["train"],
        min_word_frequency=train_configs.get("min_word_frequency", 1),
        **WordTransform.vocab_paths(vocab_dir),
    )
    model = get_model(
        CTCVSR,
        text_transform.vocab_size,
        ckpt=args.ckpt,
        visual_pretrained=args.visual_pretrained,
    )
    train_loader = create_loader(
        splits["train"],
        Collator("train", text_transform),
        train_configs,
        shuffle=True,
    )
    val_loader = create_loader(
        splits["val"],
        Collator("val", text_transform),
        train_configs,
    )
    trainer = CTCTrainer(model, text_transform, train_configs)
    trainer.train(train_loader, val_loader, args.epochs, output_dir)


if __name__ == "__main__":
    main()
