# Giải Thích Kết Quả Mô Hình

## Training có phải là 1/2 không?

Không, không phải training chỉ có `1/2`.

Trong project này có `2` file dữ liệu tách sẵn:

- `data/customer_churn_dataset-training-master.csv`: dùng để train và tune model
- `data/customer_churn_dataset-testing-master.csv`: dùng để test holdout cuối cùng

Còn giá trị `0/1` trong cột `Churn` là nhãn lớp, không phải "một nửa dữ liệu".

- `Churn = 1`: khách hàng có nguy cơ rời bỏ
- `Churn = 0`: khách hàng không rời bỏ

## Dữ liệu của project này

Sau khi đọc file và loại bỏ các dòng bị missing:

- Train có `440,832` dòng
- Test có `64,374` dòng

Tỷ lệ churn trong train:

- `train_churn_rate = 0.5671`
- Nghĩa là khoảng `56.71%` khách trong tập train có `Churn = 1`

Tỷ lệ churn trong test:

- `test_churn_rate = 0.4737`
- Nghĩa là khoảng `47.37%` khách trong tập test có `Churn = 1`

Vì vậy, nếu hỏi "training có 1/2 hả" thì có 3 cách hiểu:

- Nếu ý bạn là tập train có bị chia đôi không: `Không`
- Nếu ý bạn là target có 2 class không: `Đúng`, gồm `0` và `1`
- Nếu ý bạn là tỷ lệ churn có gần một nửa không: `Gần đúng`, vì train khoảng `56.7%`, test khoảng `47.4%`

## Vì sao trong code lại có sample 60,000 dòng?

Dữ liệu train gốc vẫn là `440,832` dòng.

Tuy nhiên, trong lúc tune model với `GridSearchCV`, project chỉ lấy mẫu `60,000` dòng từ tập train để:

- Giảm thời gian chạy
- Vẫn giữ được tỷ lệ nhãn gần giống ban đầu
- Để project portfolio có thể chạy được trên máy cá nhân mà không quá nặng

Điều quan trọng:

- Train gốc vẫn là full dataset
- Tune hyperparameter dùng sample đại diện
- Đánh giá cuối cùng vẫn dùng tập test holdout đầy đủ

## Cách hiểu cho dễ nhớ

Bạn có thể hiểu ngắn gọn như sau:

- `0/1` là nhãn churn
- `train/test` là cách tách dữ liệu
- `60,000 sample` là phần dữ liệu train được lấy ra để tune nhanh hơn

## Một câu để trả lời khi phỏng vấn

"Dataset của em đã có sẵn file train và test. Cột `Churn` là nhãn nhị phân với `1` là churn và `0` là non-churn. Tập train gốc có hơn 440K dòng, nhưng em sample 60K dòng để `GridSearchCV` chạy nhanh hơn; còn tập test holdout vẫn được giữ nguyên để đánh giá cuối cùng."
