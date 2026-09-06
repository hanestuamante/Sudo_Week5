"""Chia validation set và tạo đặc trưng TF-IDF."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from config.defaults import RANDOM_SEED, TFIDF_PARAMS, VALIDATION_SIZE


class TfidfFeatures:
    """Quản lý việc chia train/validation và vector hóa không rò rỉ dữ liệu."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.vectorizer = TfidfVectorizer(
            **(TFIDF_PARAMS if params is None else params)
        )

    @staticmethod
    def split_training_data(
        texts: list[str],
        labels: np.ndarray,
        validation_size: float = VALIDATION_SIZE,
    ) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
        """Tách validation có stratify từ train gốc."""
        return train_test_split(
            texts,
            labels,
            test_size=validation_size,
            random_state=RANDOM_SEED,
            stratify=labels,
        )

    def fit_transform(
        self,
        train_texts: list[str],
        validation_texts: list[str],
        test_texts: list[str],
    ):
        """Fit vocabulary trên train rồi transform validation và test."""
        train_matrix = self.vectorizer.fit_transform(train_texts)
        validation_matrix = self.vectorizer.transform(validation_texts)
        test_matrix = self.vectorizer.transform(test_texts)
        return train_matrix, validation_matrix, test_matrix
