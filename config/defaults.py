"""Các đường dẫn và tham số mặc định dùng chung trong project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "10Topics" / "Ver1.1"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_ARCHIVE_PATH = RAW_DATA_DIR / "Train_Full.rar"
TEST_ARCHIVE_PATH = RAW_DATA_DIR / "Test_Full.rar"

TEXT_ENCODINGS = ("utf-16", "utf-8-sig", "utf-8")
UNICODE_NORMALIZATION_FORM = "NFKC"
WHITESPACE_PATTERN = r"\s+"
DOCUMENT_GLOB = "*.txt"

RANDOM_SEED = 42
VALIDATION_SIZE = 0.10
TFIDF_PARAMS = {
    "max_features": 5_000,
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_df": 0.95,
    "sublinear_tf": True,
}
MLP_PARAMS = {
    "hidden_layer_sizes": (64,),
    "activation": "relu",
    "solver": "adam",
    "batch_size": 256,
    "learning_rate_init": 1e-3,
    "alpha": 1e-4,
    "random_state": RANDOM_SEED,
}

MAX_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3
MINIMUM_LOSS_IMPROVEMENT = 1e-4
CLASSIFICATION_REPORT_DIGITS = 4
CORRECT_PREVIEW_LENGTH = 220
ERROR_PREVIEW_LENGTH = 180
ERROR_EXAMPLE_COUNT = 10

HISTORY_FIGSIZE = (12, 4)
CONFUSION_MATRIX_FIGSIZE = (10, 9)
CONFUSION_MATRIX_CMAP = "Blues"
CONFUSION_MATRIX_ROTATION = 45
