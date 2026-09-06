"""Huấn luyện MLP theo từng epoch và dừng sớm bằng validation loss."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier

from config.defaults import (
    EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    MINIMUM_LOSS_IMPROVEMENT,
    MLP_PARAMS,
)


@dataclass
class TrainingResult:
    """Kết quả cần thiết để đánh giá và vẽ learning curve."""

    model: MLPClassifier
    history: dict[str, list[float]]
    best_epoch: int
    best_validation_loss: float


class MLPTrainer:
    """Huấn luyện MLPClassifier bằng partial_fit và early stopping."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(MLP_PARAMS if params is None else params)

    def train(
        self,
        train_matrix,
        train_labels: np.ndarray,
        validation_matrix,
        validation_labels: np.ndarray,
        class_names: list[str],
    ) -> TrainingResult:
        """Huấn luyện và khôi phục model có validation loss tốt nhất."""
        model = MLPClassifier(**self.params)
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }
        best_model = None
        best_validation_loss = np.inf
        epochs_without_improvement = 0

        for _ in range(MAX_EPOCHS):
            model.partial_fit(train_matrix, train_labels, classes=class_names)
            train_probabilities = model.predict_proba(train_matrix)
            validation_probabilities = model.predict_proba(validation_matrix)
            validation_loss = log_loss(
                validation_labels,
                validation_probabilities,
                labels=class_names,
            )
            history["train_loss"].append(
                log_loss(train_labels, train_probabilities, labels=class_names)
            )
            history["val_loss"].append(validation_loss)
            history["train_accuracy"].append(
                accuracy_score(train_labels, model.predict(train_matrix))
            )
            history["val_accuracy"].append(
                accuracy_score(validation_labels, model.predict(validation_matrix))
            )

            if validation_loss < best_validation_loss - MINIMUM_LOSS_IMPROVEMENT:
                best_validation_loss = validation_loss
                best_model = deepcopy(model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    break

        if best_model is None:
            raise RuntimeError("Không tạo được model trong quá trình huấn luyện.")
        best_epoch = int(np.argmin(history["val_loss"])) + 1
        return TrainingResult(
            model=best_model,
            history=history,
            best_epoch=best_epoch,
            best_validation_loss=float(best_validation_loss),
        )
