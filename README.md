# Vietnamese News Text Classification

Phân loại bài báo tiếng Việt thuộc 10 chủ đề bằng TF-IDF và một mạng MLP có
một hidden layer. Pipeline gồm chuẩn bị corpus VNTC, trích xuất đặc trưng,
huấn luyện có early stopping và đánh giá trên test set.

## Kết quả tham khảo

- Test accuracy: **92,30%**
- Macro F1-score: **90,64%**
- Weighted F1-score: **92,24%**
- Epoch được chọn: **9** theo validation loss

Kết quả chi tiết của lần chạy trước nằm trong
[reports/training_report.md](reports/training_report.md).

## Cấu trúc project

```text
.
├── config/
│   └── defaults.py                  # Đường dẫn và tham số mặc định
├── data/
│   ├── raw/                         # Archive VNTC (không commit)
│   ├── processed/                   # Dữ liệu giải nén (không commit)
│   └── README.md                    # Hướng dẫn tải dữ liệu
├── notebook/
│   └── text_classification.ipynb    # Notebook bài nộp
├── src/
│   ├── corpus.py                    # Giải nén, làm sạch và đọc corpus
│   ├── features.py                  # Chia validation và tạo TF-IDF
│   ├── training.py                  # Huấn luyện MLP, early stopping
│   └── evaluation.py                # Metric và biểu đồ đánh giá
├── reports/                         # Báo cáo và tài liệu tham khảo
├── README.md
└── requirements.txt
```

Các file trong `src` được chia theo pipeline cụ thể của bài Week 5, không dựa
trên một danh sách tên module cố định.

## Yêu cầu

- Python 3.12
- `bsdtar` để giải nén RAR (`libarchive-tools` trên Ubuntu/Debian)
- Đủ dung lượng cho khoảng 84.000 file văn bản sau giải nén

## Cài đặt

```bash
git clone https://github.com/hanestuamante/Sudo_Week5.git
cd Sudo_Week5
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Trên Windows, kích hoạt môi trường bằng `.venv\Scripts\activate`.

## Chuẩn bị dữ liệu

Làm theo [data/README.md](data/README.md), sau đó đặt `Train_Full.rar` và
`Test_Full.rar` vào `data/raw/10Topics/Ver1.1/`.

## Thực thi

Từ thư mục gốc của project:

```bash
jupyter lab notebook/text_classification.ipynb
```

Chạy lần lượt các cell. Notebook gọi logic từ `src`, còn toàn bộ tham số mặc
định được quản lý tại `config/defaults.py`. Lần chạy đầu sẽ giải nén corpus vào
`data/processed/`.

## Nguồn dữ liệu

Corpus được giới thiệu trong bài báo:

> Cong Duy Vu Hoang, Dien Dinh, Le Nguyen Nguyen, Quoc Hung Ngo. *A Comparative
> Study on Vietnamese Text Classification Methods*. Proceedings of IEEE RIVF,
> 2007.
