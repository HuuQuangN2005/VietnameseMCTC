import os

from abc import ABC, abstractmethod

import torch
from tqdm.auto import tqdm

from srcs.nets.loss.ctc import ctc_decode
from srcs.nets.loss.mctc import mctc_decode

from srcs.trainer.utils import (
    create_lr_scheduler,
    create_metrics,
    get_metric_results,
    move_batch,
    save_ckpt,
    save_history,
)

WORST_SCORE = (float("inf"), float("inf"))


class Trainer(ABC):
    def __init__(self, model, text_transform, configs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.text_transform = text_transform
        self.configs = configs
        self.use_amp = configs.get("amp", True) and self.device.type == "cuda"
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

    def setup_training(self, loader, epoch_count):
        parameters = [param for param in self.model.parameters() if param.requires_grad]

        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.configs["lr"],
            weight_decay=self.configs.get("weight_decay", 0.0),
        )

        self.scheduler = create_lr_scheduler(
            self.optimizer,
            loader,
            epoch_count,
            self.configs.get("warmup_steps", 0),
        )

        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.use_amp,
            init_scale=1024.0,
        )

    @staticmethod
    @abstractmethod
    def decode(outputs):
        pass

    def score(self, metrics):
        return metrics["wer"], metrics["loss"]

    def update_metrics(self, metrics, outputs, batch):
        preds = [
            self.text_transform.decode_for_metrics(ids) for ids in self.decode(outputs)
        ]
        references = [
            self.text_transform.reference_for_metrics(text) for text in batch["texts"]
        ]

        for name, metric in metrics.items():
            metric.update(
                [prediction[name] for prediction in preds],
                [reference[name] for reference in references],
            )

    def update_model(self, loss):
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        max_gradient_norm = self.configs.get("max_grad_norm", 0.0)

        if max_gradient_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_gradient_norm)

        previous_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if self.scaler.get_scale() >= previous_scale:
            self.scheduler.step()

    def run_epoch(self, loader, training, desc):
        if len(loader) == 0:
            raise ValueError(f"{desc} loader is empty.")

        self.model.train(training)

        logging_steps = self.configs.get("logging_steps", 25)
        batch_count = len(loader)
        sample_count = 0
        total_loss = 0.0

        metrics = create_metrics(self.text_transform)
        progress = tqdm(loader, desc=desc)

        with torch.set_grad_enabled(training):
            for batch_number, batch in enumerate(progress, start=1):
                batch = move_batch(batch, self.device)

                with torch.amp.autocast(
                    self.device.type, dtype=torch.float16, enabled=self.use_amp
                ):
                    model_inputs = {
                        name: value for name, value in batch.items() if name != "texts"
                    }
                    outputs = self.model(**model_inputs)
                    loss = outputs["loss"]

                if training:
                    self.update_model(loss)

                batch_size = batch["videos"].size(0)
                total_loss += loss.detach().float().item() * batch_size
                sample_count += batch_size

                self.update_metrics(metrics, outputs, batch)

                if batch_number % logging_steps and batch_number != batch_count:
                    continue

                progress_values = {"loss": total_loss / sample_count}
                progress_values.update(get_metric_results(metrics))

                if training:
                    progress_values["lr"] = self.optimizer.param_groups[0]["lr"]

                progress.set_postfix(progress_values)

        return {"loss": total_loss / sample_count, **get_metric_results(metrics)}

    def train(self, train_loader, val_loader, epoch_count, output_dir):
        self.setup_training(train_loader, epoch_count)
        os.makedirs(output_dir, exist_ok=True)

        best_ckpt_path = os.path.join(output_dir, "best.pt")
        last_ckpt_path = os.path.join(output_dir, "last.pt")
        history_path = os.path.join(output_dir, "history.json")

        best_score = WORST_SCORE
        history = []

        for epoch in tqdm(range(1, epoch_count + 1), desc="Epochs"):
            train_metrics = self.run_epoch(train_loader, True, "Training")
            val_metrics = self.run_epoch(val_loader, False, "Validation")

            epoch_metrics = {
                "epoch": epoch,
                **{f"training_{name}": value for name, value in train_metrics.items()},
                **{f"validation_{name}": value for name, value in val_metrics.items()},
            }

            history.append(epoch_metrics)
            save_history(history, history_path)
            save_ckpt(self.model, last_ckpt_path, epoch, epoch_metrics)

            val_score = self.score(val_metrics)

            if val_score < best_score:
                best_score = val_score
                save_ckpt(self.model, best_ckpt_path, epoch, epoch_metrics)

        return history


class MCTCTrainer(Trainer):
    @staticmethod
    def decode(outputs):
        return mctc_decode(outputs["logits"], outputs["input_lengths"])


class CTCTrainer(Trainer):
    @staticmethod
    def decode(outputs):
        return ctc_decode(outputs["logits"], outputs["input_lengths"])
