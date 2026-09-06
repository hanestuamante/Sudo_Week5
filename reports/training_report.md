# Báo cáo phân loại văn bản tiếng Việt bằng mạng neural

## 1. Mục tiêu

Xây dựng một mô hình neural network đơn giản để phân loại bài báo tiếng Việt vào 10 chủ đề, đồng thời quan sát rõ quá trình mô hình học thông qua loss và accuracy theo từng epoch. Toàn bộ kết quả trong báo cáo được lấy từ lần chạy notebook trên corpus thật với random seed 42.

## 2. Dữ liệu và tiền xử lý

Corpus VNTC 10 Topics gồm:

| Tập dữ liệu | Số văn bản | Mục đích |
|---|---:|---|
| Train gốc | 33.759 | Học mô hình và tạo validation set |
| Train sau khi chia | 30.383 | Cập nhật trọng số |
| Validation | 3.376 | Chọn epoch tốt nhất |
| Test | 50.373 | Đánh giá cuối cùng |

Mỗi file được đọc theo UTF-16, chuẩn hóa Unicode NFKC, chuyển thành chữ thường và gộp khoảng trắng. Văn bản sau đó được biểu diễn bằng TF-IDF với tối đa 5.000 đặc trưng unigram và bigram. Vocabulary và trọng số IDF chỉ được học từ train; validation và test chỉ gọi `transform`, vì vậy không có rò rỉ thông tin từ hai tập này.

TF-IDF biến văn bản có độ dài khác nhau thành vector số thưa. Từ xuất hiện nhiều trong một văn bản nhưng ít phổ biến trong toàn corpus nhận trọng số cao hơn. Đây là đầu vào phù hợp cho mạng nhỏ và dễ hiểu hơn embedding học sâu.

## 3. Mô hình và các khái niệm chính

Kiến trúc mạng là **5.000 input → 64 hidden neurons → 10 output classes**.

Một perceptron nhận vector đầu vào, tính tổng có trọng số cộng bias rồi truyền qua activation. Nếu chỉ có một perceptron hay một lớp tuyến tính, ranh giới quyết định bị giới hạn ở dạng tuyến tính. Hidden layer gồm 64 perceptron dùng ReLU, nhờ đó mạng có thể kết hợp các dấu hiệu từ/cụm từ thành biểu diễn phi tuyến trước khi phân loại.

Lớp output dùng softmax để tạo phân phối xác suất trên 10 chủ đề. Loss function là cross-entropy: dự đoán càng ít xác suất cho nhãn đúng thì loss càng lớn. Loss cung cấp tín hiệu liên tục và giàu thông tin hơn chỉ số đúng/sai.

Backpropagation tính gradient của loss theo từng trọng số. Adam dùng gradient này với learning rate 0,001 để cập nhật trọng số theo hướng làm giảm loss. Mỗi lượt đi qua toàn bộ train set là một epoch; batch size 256 giúp cập nhật nhiều lần trong một epoch mà vẫn tiết kiệm bộ nhớ.

Memory-based learning, chẳng hạn k-nearest neighbors, lưu các mẫu huấn luyện và tìm mẫu gần nhất khi dự đoán. MLP trong bài này không làm vậy: sau huấn luyện, tri thức được nén vào trọng số của các perceptron. Vì thế dự đoán nhanh hơn trên test set lớn, đổi lại mô hình phải trải qua quá trình tối ưu bằng gradient descent.

## 4. Quá trình huấn luyện

Mạng được huấn luyện tối đa 15 epoch. Nếu validation loss không cải thiện trong 3 epoch liên tiếp, quá trình dừng sớm. Mô hình ở epoch có validation loss thấp nhất được giữ lại.

| Epoch | Train loss | Validation loss | Train accuracy | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0,5866 | 0,6237 | 87,85% | 86,52% |
| 2 | 0,3255 | 0,3776 | 91,27% | 88,92% |
| 3 | 0,2448 | 0,3128 | 93,12% | 90,17% |
| 4 | 0,1999 | 0,2844 | 94,42% | 90,91% |
| 5 | 0,1686 | 0,2691 | 95,39% | 91,14% |
| 6 | 0,1443 | 0,2604 | 96,22% | 91,14% |
| 7 | 0,1247 | 0,2556 | 96,90% | 91,29% |
| 8 | 0,1082 | 0,2534 | 97,45% | 91,32% |
| **9** | **0,0942** | **0,2529** | **98,02%** | **91,38%** |
| 10 | 0,0823 | 0,2538 | 98,37% | 91,38% |
| 11 | 0,0720 | 0,2558 | 98,73% | 91,29% |
| 12 | 0,0631 | 0,2585 | 99,00% | 91,29% |

Train loss giảm liên tục, cho thấy gradient descent đang tối ưu đúng mục tiêu. Validation loss đạt thấp nhất 0,2529 ở epoch 9 rồi tăng nhẹ, trong khi train loss vẫn giảm. Khoảng cách này là dấu hiệu overfitting bắt đầu xuất hiện. Early stopping dừng tại epoch 12 và khôi phục mô hình epoch 9.

## 5. Kết quả test

Kết quả tổng quát trên 50.373 văn bản test:

| Metric | Giá trị |
|---|---:|
| Accuracy | **92,30%** |
| Macro precision | 91,57% |
| Macro recall | 89,97% |
| Macro F1-score | **90,64%** |
| Weighted F1-score | **92,24%** |

Kết quả theo lớp:

| Chủ đề | Precision | Recall | F1-score | Số mẫu |
|---|---:|---:|---:|---:|
| Chính trị Xã hội | 84,49% | 92,52% | 88,32% | 7.567 |
| Đời sống | 83,21% | 67,68% | 74,65% | 2.036 |
| Khoa học | 87,46% | 82,20% | 84,75% | 2.096 |
| Kinh doanh | 94,55% | 87,85% | 91,08% | 5.276 |
| Pháp luật | 92,31% | 92,19% | 92,25% | 3.788 |
| Sức khỏe | 93,28% | 95,11% | 94,19% | 5.417 |
| Thế giới | 95,89% | 93,69% | 94,77% | 6.716 |
| Thể thao | 98,55% | 97,99% | **98,27%** | 6.667 |
| Văn hóa | 93,05% | 94,88% | 93,96% | 6.250 |
| Vi tính | 92,92% | 95,55% | 94,22% | 4.560 |

Thể thao là lớp tốt nhất với F1 98,27%, phù hợp với việc lớp này thường có vocabulary đặc trưng. Đời sống khó nhất, đặc biệt recall chỉ 67,68%. Nội dung đời sống dễ giao thoa với văn hóa, sức khỏe, xã hội hoặc kinh doanh; ngoài ra đây là một trong các lớp có ít mẫu train hơn. Macro F1 thấp hơn weighted F1 cho thấy hiệu năng chưa đồng đều giữa các lớp, dù accuracy tổng thể cao.

## 6. Kết luận

Một hidden-layer MLP kết hợp TF-IDF đã đạt accuracy 92,30% mà không cần kiến trúc phức tạp. Thí nghiệm minh họa trực tiếp vai trò của perceptron và hidden layer trong biểu diễn phi tuyến, loss function trong việc đo sai số, gradient descent/Adam trong cập nhật trọng số, cũng như sự khác biệt giữa mô hình học tham số và memory-based learning. Learning curve đồng thời cho thấy vì sao phải theo dõi validation loss và chọn epoch trước khi overfitting tăng lên.

Hướng cải thiện hợp lý tiếp theo là thử tokenizer tiếng Việt, điều chỉnh số neuron/regularization, hoặc cân bằng lớp Đời sống. Các thay đổi này cần được chọn bằng validation set; test set nên tiếp tục chỉ dành cho lần đánh giá cuối.
