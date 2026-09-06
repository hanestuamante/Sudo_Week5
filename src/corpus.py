"""Giải nén, làm sạch và đọc corpus VNTC."""

from __future__ import annotations

import re
import subprocess
import unicodedata
from pathlib import Path

import numpy as np

from config.defaults import (
    DOCUMENT_GLOB,
    PROCESSED_DATA_DIR,
    TEST_ARCHIVE_PATH,
    TEXT_ENCODINGS,
    TRAIN_ARCHIVE_PATH,
    UNICODE_NORMALIZATION_FORM,
    WHITESPACE_PATTERN,
)


class VNTCCorpus:
    """Chuẩn bị và đọc train/test split của VNTC 10 Topics."""

    def __init__(
        self,
        train_archive: str | Path = TRAIN_ARCHIVE_PATH,
        test_archive: str | Path = TEST_ARCHIVE_PATH,
        extract_dir: str | Path = PROCESSED_DATA_DIR,
    ) -> None:
        self.archives = {
            "Train_Full": Path(train_archive),
            "Test_Full": Path(test_archive),
        }
        self.extract_dir = Path(extract_dir)
        self.train_data_dir = self.extract_dir / "Train_Full"
        self.test_data_dir = self.extract_dir / "Test_Full"

    def extract(self) -> None:
        """Giải nén các archive chưa có trong processed data."""
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        for folder_name, archive_path in self.archives.items():
            if (self.extract_dir / folder_name).exists():
                continue
            if not archive_path.exists():
                raise FileNotFoundError(f"Không tìm thấy archive: {archive_path}")
            subprocess.run(
                ["bsdtar", "-xf", str(archive_path), "-C", str(self.extract_dir)],
                check=True,
            )

    @staticmethod
    def clean_text(text: str) -> str:
        """Chuẩn hóa Unicode, lowercase và gộp khoảng trắng."""
        normalized = unicodedata.normalize(UNICODE_NORMALIZATION_FORM, text)
        return re.sub(WHITESPACE_PATTERN, " ", normalized.lower()).strip()

    def read_document(self, path: Path) -> str:
        """Đọc một document bằng các encoding phổ biến của corpus."""
        for encoding in TEXT_ENCODINGS:
            try:
                return self.clean_text(path.read_text(encoding=encoding))
            except UnicodeError:
                continue
        return self.clean_text(
            path.read_text(encoding=TEXT_ENCODINGS[-1], errors="ignore")
        )

    def load_split(self, split_dir: str | Path) -> tuple[list[str], np.ndarray]:
        """Đọc văn bản và lấy tên thư mục chủ đề làm nhãn."""
        texts: list[str] = []
        labels: list[str] = []
        directory = Path(split_dir)
        for topic_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
            for text_path in sorted(topic_dir.glob(DOCUMENT_GLOB)):
                text = self.read_document(text_path)
                if text:
                    texts.append(text)
                    labels.append(topic_dir.name)
        return texts, np.asarray(labels)

    def load(self) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
        """Đọc cả train và test split sau khi giải nén."""
        train_texts, train_labels = self.load_split(self.train_data_dir)
        test_texts, test_labels = self.load_split(self.test_data_dir)
        return train_texts, train_labels, test_texts, test_labels
