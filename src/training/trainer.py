"""Training loop for GNN models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .losses import FocalLoss


class GNNTrainer:
    """Handles training, validation, early stopping and checkpointing."""

    def __init__(
        self,
        model,
        data,
        learning_rate=0.001,
        weight_decay=1e-5,
        class_weights=None,
        use_focal_loss=True,
        focal_gamma=2.0,
        device="auto",
    ):
        # TODO: add gradient clipping as a parameter
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.data = data.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Loss function
        if use_focal_loss:
            self.criterion = FocalLoss(gamma=focal_gamma, weight=class_weights)
        elif class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Training state
        self.best_val_auc = 0
        self.patience_counter = 0
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_f1": [],
        }

    def train(
        self,
        epochs=100,
        patience=20,
        min_delta=0.001,
        checkpoint_path=None,
        verbose=True,
    ):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5, verbose=verbose
        )

        for epoch in range(epochs):
            # Training step
            train_loss = self._train_epoch()

            # Validation step
            val_loss, val_metrics = self._validate()

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_metrics["auc_roc"])
            self.history["val_f1"].append(val_metrics["f1"])

            # Learning rate scheduling
            scheduler.step(val_metrics["auc_roc"])

            # Early stopping check
            if val_metrics["auc_roc"] > self.best_val_auc + min_delta:
                self.best_val_auc = val_metrics["auc_roc"]
                self.patience_counter = 0

                if checkpoint_path:
                    self.save(checkpoint_path)
            else:
                self.patience_counter += 1

            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val AUC: {val_metrics['auc_roc']:.4f} | "
                    f"Val F1: {val_metrics['f1']:.4f}"
                )

            if self.patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        return self.history

    def _train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()

        self.optimizer.zero_grad()

        # Forward pass
        out = self.model(
            self.data.x,
            self.data.edge_index,
            getattr(self.data, "edge_attr", None),
        )

        # Get training mask
        if hasattr(self.data, "train_mask"):
            mask = self.data.train_mask
            labels = self.data.y[: len(mask)]  # For edge classification
            out = out[mask]
            labels = labels[mask]
        else:
            labels = self.data.y

        loss = self.criterion(out, labels)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def _validate(self) -> tuple[float, dict[str, float]]:
        """Validate the model."""
        self.model.eval()

        with torch.no_grad():
            out = self.model(
                self.data.x,
                self.data.edge_index,
                getattr(self.data, "edge_attr", None),
            )

            # Get validation mask
            if hasattr(self.data, "val_mask"):
                mask = self.data.val_mask
                labels = self.data.y[: len(mask)]
                out = out[mask]
                labels = labels[mask]
            else:
                labels = self.data.y

            loss = self.criterion(out, labels)

            # Compute metrics
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            metrics = self._compute_metrics(labels, preds, probs)

        return loss.item(), metrics

    def evaluate(self, test_mask=None):
        self.model.eval()

        with torch.no_grad():
            out = self.model(
                self.data.x,
                self.data.edge_index,
                getattr(self.data, "edge_attr", None),
            )

            if test_mask is None:
                test_mask = getattr(self.data, "test_mask", None)

            if test_mask is not None:
                labels = self.data.y[: len(test_mask)]
                out = out[test_mask]
                labels = labels[test_mask]
            else:
                labels = self.data.y

            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            return self._compute_metrics(labels, preds, probs)

    def _compute_metrics(self, labels, preds, probs):
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        }

        # AUC metrics (need try-except for edge cases)
        try:
            metrics["auc_roc"] = roc_auc_score(labels, probs)
        except ValueError:
            metrics["auc_roc"] = 0.5

        try:
            metrics["auc_pr"] = average_precision_score(labels, probs)
        except ValueError:
            metrics["auc_pr"] = 0.0

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        metrics["true_positives"] = int(tp)
        metrics["false_positives"] = int(fp)
        metrics["true_negatives"] = int(tn)
        metrics["false_negatives"] = int(fn)

        return metrics

    def predict(self, x, edge_index, edge_attr=None):
        self.model.eval()

        with torch.no_grad():
            out = self.model(
                x.to(self.device),
                edge_index.to(self.device),
                edge_attr.to(self.device) if edge_attr is not None else None,
            )

            probs = F.softmax(out, dim=1)[:, 1]
            preds = out.argmax(dim=1)

        return preds, probs

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_auc": self.best_val_auc,
                "history": self.history,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_auc = checkpoint.get("best_val_auc", 0)
        self.history = checkpoint.get("history", {})


def compute_class_weights(labels):
    """Inverse frequency weighting for imbalanced classes."""
    unique, counts = torch.unique(labels, return_counts=True)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum() * len(unique)

    return weights
