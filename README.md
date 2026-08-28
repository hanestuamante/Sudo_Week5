# Vietnamese News Text Classification

Phân loại 50.373 bài báo tiếng Việt thuộc 10 chủ đề bằng một mạng neural đơn giản. Pipeline sử dụng TF-IDF và MLP có một hidden layer, được trình bày và chạy hoàn chỉnh trong một notebook duy nhất.

## Kết quả chính

- Test accuracy: **92,30%**
- Macro F1-score: **90,64%**
- Weighted F1-score: **92,24%**
- Epoch được chọn: **9** theo validation loss

Kết quả chi tiết, learning curve, confusion matrix và ví dụ dự đoán sai đã được lưu trong [notebook](notebooks/text_classification.ipynb). Phần phân tích nằm tại [training report](Report/training_report.md).

## Cấu trúc repo

```text
VNTC/
├── Data/                              # Corpus gốc ở định dạng RAR
├── notebooks/
│   └── text_classification.ipynb      # Toàn bộ pipeline và output
├── Report/
│   ├── training_report.md             # Báo cáo huấn luyện và kết quả
│   └── ...                            # Tài liệu gốc của corpus
├── Source/                            # Source gốc đi kèm corpus
├── requirements.txt
├── LICENSE
└── README.md
```

## Chạy lại

Yêu cầu Python 3.12 và `bsdtar` (có sẵn mặc định trên macOS; trên Linux thường thuộc gói `libarchive-tools`).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Mở `notebooks/text_classification.ipynb` bằng VS Code hoặc một giao diện Jupyter, chọn kernel trong `.venv`, rồi chọn **Run All**. Notebook tự giải nén corpus vào `Data/10Topics/Ver1.1/extracted/`; thư mục này đã được bỏ qua bởi Git. Lần chạy đầu cần thêm dung lượng đĩa và thời gian để giải nén 84.132 file.

## Dữ liệu

Corpus được dùng trong bài báo:

> Cong Duy Vu Hoang, Dien Dinh, Le Nguyen Nguyen, Quoc Hung Ngo. *A Comparative Study on Vietnamese Text Classification Methods*. Proceedings of IEEE RIVF, 2007.

Repo sử dụng phiên bản 10 chủ đề: 33.759 văn bản train và 50.373 văn bản test.
