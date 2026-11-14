"""
Module: training/trainer.py
Robust training engine for protein graph classification with early stopping, checkpointing, and adaptive regularization.
"""

import time
import os
import torch
import torch.nn as nn
import numpy as np
from evaluation.metrics import compute_classification_metrics


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ProteinTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = None,
        device: torch.device = None,
        scheduler = None,
        regularizer_fn = None,
        checkpoint_dir: str = "results/checkpoints"
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.device = device if device is not None else torch.device("cpu")
        self.scheduler = scheduler
        self.regularizer_fn = regularizer_fn
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.model.to(self.device)

    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0.0
        total_graphs = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Apply graph edge regularization if configured
            if self.regularizer_fn is not None:
                train_batch = self.regularizer_fn(batch)
            else:
                train_batch = batch

            edge_attr = getattr(train_batch, "edge_attr", None)
            logits = self.model(train_batch.x, train_batch.edge_index, batch=train_batch.batch, edge_attr=edge_attr)

            loss = self.criterion(logits, batch.y.squeeze())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / max(total_graphs, 1)

    def evaluate(self, loader, perturb_fn = None) -> tuple:
        self.model.eval()
        total_loss = 0.0
        total_graphs = 0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                if perturb_fn is not None:
                    batch = perturb_fn(batch)

                edge_attr = getattr(batch, "edge_attr", None)
                logits = self.model(batch.x, batch.edge_index, batch=batch.batch, edge_attr=edge_attr)

                loss = self.criterion(logits, batch.y.squeeze())
                total_loss += loss.item() * batch.num_graphs
                total_graphs += batch.num_graphs

                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(batch.y.squeeze().cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        mean_loss = total_loss / max(total_graphs, 1)
        metrics = compute_classification_metrics(np.array(all_targets), np.array(all_preds))
        metrics["loss"] = mean_loss
        return metrics

    def fit(
        self,
        train_loader,
        val_loader,
        test_loader,
        epochs: int = 150,
        patience: int = 30,
        model_name: str = "best_model.pt",
        perturb_test_fn = None
    ) -> dict:
        best_val_f1 = -1.0
        best_epoch = 0
        counter = 0
        best_checkpoint_path = os.path.join(self.checkpoint_dir, model_name)

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": []
        }

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])

            val_f1 = val_metrics["macro_f1"]

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                counter = 0
                torch.save(self.model.state_dict(), best_checkpoint_path)
            else:
                counter += 1

            if counter >= patience:
                break

        elapsed_time = time.time() - start_time

        # Load best model checkpoint for final test evaluation
        if os.path.exists(best_checkpoint_path):
            self.model.load_state_dict(torch.load(best_checkpoint_path, map_location=self.device))

        test_metrics = self.evaluate(test_loader, perturb_fn=perturb_test_fn)
        test_metrics["best_epoch"] = best_epoch
        test_metrics["total_epochs_trained"] = len(history["train_loss"])
        test_metrics["training_time_seconds"] = elapsed_time
        test_metrics["num_parameters"] = count_parameters(self.model)

        return {
            "test_metrics": test_metrics,
            "history": history
        }
