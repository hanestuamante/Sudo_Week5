# Dữ liệu VNTC

Dataset có dung lượng lớn nên không được lưu trực tiếp trong Git.

## Tải dữ liệu

Dataset và source tham khảo có tại repository gốc
[duyvuleo/VNTC](https://github.com/duyvuleo/VNTC).

Sau khi tải, đặt hai archive của phiên bản 10 chủ đề vào đúng vị trí:

```text
data/raw/10Topics/Ver1.1/Train_Full.rar
data/raw/10Topics/Ver1.1/Test_Full.rar
```

Notebook sử dụng `bsdtar` để tự động giải nén hai archive vào:

```text
data/processed/Train_Full/
data/processed/Test_Full/
```

Không commit archive, file đã giải nén hoặc dữ liệu phát sinh trở lại repository.
