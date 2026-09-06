"""Metric và biểu đồ đánh giá mô hình phân loại văn bản."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)

from config.defaults import (
    CLASSIFICATION_REPORT_DIGITS,
    CONFUSION_MATRIX_CMAP,
    CONFUSION_MATRIX_FIGSIZE,
    CONFUSION_MATRIX_ROTATION,
    HISTORY_FIGSIZE,
)


class ClassificationEvaluator:
    """Đánh giá định lượng và trực quan hóa kết quả MLP."""

    @staticmethod
    def metrics(
        true_labels: np.ndarray,
        predictions: np.ndarray,
        class_names: list[str],
    ) -> tuple[float, str]:
        """Trả về accuracy và classification report."""
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(
            true_labels,
            predictions,
            labels=class_names,
            digits=CLASSIFICATION_REPORT_DIGITS,
            zero_division=0,
        )
        return float(accuracy), report

    @staticmethod
    def plot_history(history: dict[str, list[float]], best_epoch: int):
        """Vẽ loss và accuracy trên train/validation theo epoch."""
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        figure, axes = plt.subplots(1, 2, figsize=HISTORY_FIGSIZE)
        axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
        axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation")
        axes[0].axvline(best_epoch, color="gray", linestyle="--", label="Best epoch")
        axes[0].set(title="Loss theo epoch", xlabel="Epoch", ylabel="Cross-entropy loss")
        axes[0].legend()
        axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Train")
        axes[1].plot(epochs, history["val_accuracy"], marker="o", label="Validation")
        axes[1].axvline(best_epoch, color="gray", linestyle="--", label="Best epoch")
        axes[1].set(
            title="Accuracy theo epoch",
            xlabel="Epoch",
            ylabel="Accuracy",
            ylim=(0, 1),
        )
        axes[1].legend()
        figure.tight_layout()
        return figure

    @staticmethod
    def plot_confusion_matrix(
        true_labels: np.ndarray,
        predictions: np.ndarray,
        class_names: list[str],
    ):
        """Vẽ confusion matrix đã chuẩn hóa theo nhãn thật."""
        figure, axis = plt.subplots(figsize=CONFUSION_MATRIX_FIGSIZE)
        ConfusionMatrixDisplay.from_predictions(
            true_labels,
            predictions,
            labels=class_names,
            normalize="true",
            values_format=".2f",
            cmap=CONFUSION_MATRIX_CMAP,
            xticks_rotation=CONFUSION_MATRIX_ROTATION,
            ax=axis,
            colorbar=False,
        )
        axis.set_title("Confusion matrix chuẩn hóa theo nhãn thật")
        figure.tight_layout()
        return figure

    @staticmethod
    def correct_example_indices(
        true_labels: np.ndarray,
        predictions: np.ndarray,
        class_names: list[str],
    ) -> list[int]:
        """Chọn một dự đoán đúng cho mỗi lớp nếu có."""
        correct = np.flatnonzero(true_labels == predictions)
        return [
            next(index for index in correct if true_labels[index] == class_name)
            for class_name in class_names
        ]

    @staticmethod
    def error_indices(
        true_labels: np.ndarray,
        predictions: np.ndarray,
    ) -> np.ndarray:
        """Trả về vị trí các dự đoán sai."""
        return np.flatnonzero(true_labels != predictions)
