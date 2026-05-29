# 🚀 Hướng Dẫn Chạy Pipeline Dự Án (End-to-End)

Tài liệu này hướng dẫn các thành viên trong nhóm cách thiết lập môi trường và chạy toàn bộ mã nguồn của dự án "Nhận diện Mạng lưới Lừa đảo" từ đầu đến cuối.

## 📌 Bước 0: Yêu cầu hệ thống
- Đã cài đặt **Python 3.8** trở lên.
- Đã cài đặt Git và Clone repository này về máy.

---

## 🛠️ Bước 1: Thiết lập Môi trường Ảo (Virtual Environment)
Để tránh xung đột thư viện với các môn học khác, chúng ta BẮT BUỘC phải dùng môi trường ảo.
Mở Terminal của VSCode (chọn PowerShell hoặc Command Prompt) và chạy:

```bash
# 1. Tạo môi trường ảo có tên là .venv
python -m venv .venv

# 2. Kích hoạt môi trường ảo
# Nếu dùng Windows (PowerShell):
.\.venv\Scripts\activate
# Nếu dùng Mac/Linux:
source .venv/bin/activate
```

*(Thành công khi bạn thấy chữ `(.venv)` màu xanh hiện ở đầu dòng lệnh trong Terminal).*

---

## 📦 Bước 2: Cài đặt Thư viện

Đảm bảo bạn đã kích hoạt `.venv`, sau đó chạy lệnh cài đặt toàn bộ package cần thiết:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Bước 3: Chạy Pipeline theo Thứ tự (Dataflow)

Vì dữ liệu có tính kế thừa, bạn **PHẢI CHẠY LẦN LƯỢT** các script dưới đây từ trên xuống dưới. Đảm bảo bạn đang đứng ở thư mục gốc của dự án (`PTMXH_SNA_Fraud_Detection`).

### 🗄️ Phase 1: Tiền xử lý dữ liệu (Tâm)

Script này sẽ tải file CSV từ Stanford SNAP, làm sạch và xuất ra các file GraphML.

```bash
python src/data/make_dataset.py
```

*Đầu ra mong đợi:* Thấy dòng chữ `Sanity Check PASSED ✓` và 4 file xuất hiện trong `data/processed/`.

### 📊 Phase 2: Chạy Baseline Model (Huyền)

Áp dụng trực tiếp thuật toán Louvain và PageRank lên đồ thị thô.

```bash
python src/models/baseline_model.py
```

*Đầu ra mong đợi:* File `baseline_masterminds.csv`, `baseline_communities.csv` và `baseline_model_metrics.txt` lưu vào `results/metrics/`.

### 🔬 Phase 3: Chạy Proposed Model (Trí)

Áp dụng pipeline bóc tách K-core -> Louvain -> PageRank.

```bash
python src/models/proposed_model.py
```

*Đầu ra mong đợi:* File metrics k-core và Masterminds lưu vào `results/metrics/`.

### 📈 Phase 4: Đánh giá & Đo lường (Hoàng - Sắp ra mắt)

Sau khi có dữ liệu từ Phase 2 và 3, script này sẽ vẽ biểu đồ so sánh.

```bash
python src/evaluation/evaluate_models.py
```

---

## ⚠️ Lưu ý rà lỗi (Troubleshooting)

* Nếu VSCode báo lỗi gạch chân vàng ở các dòng `import networkx`, hãy bấm `Ctrl + Shift + P` -> gõ `Python: Select Interpreter` -> Chọn đường dẫn có chữ `('.venv': venv)`.
* Tuyệt đối không push thư mục `.venv/` và `data/raw/` lên GitHub.