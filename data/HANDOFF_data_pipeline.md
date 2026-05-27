# Bàn giao: Data Pipeline — Tiền xử lý Dataset

> **Người thực hiện:** Tâm  
> **Nhiệm vụ gốc:** Task 1 — Thu thập & tiền xử lý dữ liệu  
> **Trạng thái:** ✅ Hoàn thành

---

## 1. Tổng quan nhiệm vụ đã làm

Xây dựng pipeline hoàn chỉnh để tải, làm sạch và xuất dataset Bitcoin OTC
thành các định dạng chuẩn, sẵn sàng cho các bước phân tích tiếp theo
(K-core, Louvain, PageRank).

---

## 2. Cấu trúc thư mục liên quan

```
PTMXH_SNA_Fraud_Detection/
│
├── data/
│   ├── make_dataset.py                    ← file code duy nhất của task này
│   ├── HANDOFF_data_pipeline.md           ← file bàn giao (file này)
│   │
│   ├── raw/
│   │   └── soc-sign-bitcoinotc.csv        ← dataset gốc từ Stanford SNAP
│   │
│   └── processed/                         ← ĐẦU RA của pipeline (4 file)
│       ├── bitcoin_otc_directed.graphml   ← đồ thị CÓ HƯỚNG  → dùng cho PageRank
│       ├── bitcoin_otc_undirected.graphml ← đồ thị VÔ HƯỚNG → dùng cho K-core + Louvain
│       ├── bitcoin_otc_edgelist.csv       ← danh sách cạnh nhẹ → import Gephi
│       └── dataset_stats.txt             ← thống kê mạng lưới → dùng viết báo cáo
│
├── src/                                   ← code phân tích (baseline, proposed, visualize)
└── requirements.txt
```

> ⚠️ **Lưu ý:** `data/raw/` không được commit lên Git (file nặng ~3MB).
> Xem hướng dẫn tải ở mục 4.

---

## 3. Thông tin về dataset

| Thuộc tính | Giá trị |
|---|---|
| Tên | Bitcoin OTC Trust Weighted Signed Network |
| Nguồn | [Stanford SNAP](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html) |
| Số nodes (sau xử lý) | **5.881** |
| Số cạnh DiGraph | **35.592** |
| Số cạnh Undirected | **21.492** |
| Rating gốc | -10 đến +10 |
| Weight sau chuẩn hóa | 0.0 đến 1.0 |

**Ý nghĩa dữ liệu:** Mỗi dòng ghi nhận người dùng A đánh giá mức độ tin
cậy của người dùng B trên sàn giao dịch Bitcoin OTC.
Bài toán đặt ra là tìm các nhóm tự đánh giá khống cho nhau (fraud rings).

---

## 4. Cách chạy lại từ đầu

### Bước 1 — Cài thư viện

```bash
pip install -r requirements.txt
```

### Bước 2 — Tải dataset

SNAP đôi khi chặn tải tự động. Nếu script báo lỗi 403, tải thủ công:

1. Vào: https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz
2. Giải nén file `.gz`
3. Đặt vào: `data/raw/soc-sign-bitcoinotc.csv`

### Bước 3 — Chạy pipeline

```bash
# Từ thư mục gốc project
python data/make_dataset.py
```

Nếu file CSV ở chỗ khác:

```bash
python data/make_dataset.py --raw_path /đường/dẫn/tới/file.csv
```

### Kết quả khi chạy thành công

```
14:12:42 [INFO]   ✓ DiGraph    : 5881 nodes, 35592 edges
14:12:42 [INFO]   ✓ Undirected : 5881 nodes, 21492 edges
14:12:42 [INFO]   ✓ Edge list  : 21492 edges, weight ∈ [0,1]
14:12:42 [INFO] ─── Sanity Check PASSED ✓ ───
```

> Nếu không thấy dòng **PASSED** ở cuối thì pipeline chưa thành công.

---

## 5. Giải thích các quyết định kỹ thuật quan trọng

### 5.1 Tại sao xuất ra 2 file GraphML khác nhau?

| File | Kiểu đồ thị | Dùng cho | Lý do |
|---|---|---|---|
| `directed.graphml` | DiGraph | PageRank | PageRank cần chiều cạnh: "A trust B" ≠ "B trust A" |
| `undirected.graphml` | Graph | K-core, Louvain | Hai thuật toán này không xử lý đồ thị có hướng |

### 5.2 Công thức chuẩn hóa trọng số

```
weight = (rating + 10) / 20
```

Rating gốc âm (-10 đến -1) nghĩa là *distrust*. K-core và Louvain yêu cầu
trọng số dương. Công thức này map tuyến tính toàn bộ về [0.0, 1.0] mà
không mất thông tin tương đối giữa các cạnh.

| Rating gốc | Weight sau chuẩn hóa | Ý nghĩa |
|---|---|---|
| -10 | 0.00 | Hoàn toàn không tin |
| 0 | 0.50 | Trung lập |
| +10 | 1.00 | Hoàn toàn tin tưởng |

### 5.3 Cạnh ngược chiều trong Undirected được xử lý thế nào?

Nếu A→B (rating=8) và B→A (rating=4) cùng tồn tại trong DiGraph,
khi chuyển sang Undirected thì gộp thành một cạnh A-B với:

```
weight = (weight_AB + weight_BA) / 2 = (0.9 + 0.7) / 2 = 0.8
```

Đây là lý do số cạnh Undirected (21.492) ít hơn DiGraph (35.592).

### 5.4 Các bước làm sạch đã thực hiện

1. **Loại self-loop** — node tự đánh giá chính mình (vô nghĩa)
2. **Loại duplicate edge** — nếu A đánh giá B nhiều lần, chỉ giữ lần gần nhất
3. **Loại isolated node** — node không có cạnh nào (không góp phần vào cộng đồng)

---

## 6. Cách các thành viên tiếp theo load dữ liệu

Chỉ cần copy đoạn code này vào đầu file của mình:

```python
import networkx as nx
import pandas as pd
from pathlib import Path

# Tìm thư mục gốc project (chứa data/, src/, requirements.txt)
# Cách 1: Nếu file của bạn nằm trong src/models/ hoặc src/visualization/
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/models/file.py → parents[2] = project root

# Cách 2: Nếu không chắc cấu trúc, dùng đường dẫn tuyệt đối
# PROJECT_ROOT = Path(r"C:/Users/.../PTMXH_SNA_Fraud_Detection")

DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Load đồ thị vô hướng — dùng cho K-core và Louvain
G_und = nx.read_graphml(DATA_DIR / "bitcoin_otc_undirected.graphml")

# Load đồ thị có hướng — dùng cho PageRank
G_dir = nx.read_graphml(DATA_DIR / "bitcoin_otc_directed.graphml")

# Load edge list — dùng cho Gephi hoặc pandas analysis
df = pd.read_csv(DATA_DIR / "bitcoin_otc_edgelist.csv")

print(f"Undirected: {G_und.number_of_nodes()} nodes, {G_und.number_of_edges()} edges")
print(f"Directed  : {G_dir.number_of_nodes()} nodes, {G_dir.number_of_edges()} edges")
```

> ⚠️ Chú ý: `nx.read_graphml()` trả về node ID dạng **string**, không phải int.
> Nếu cần so sánh node ID thì phải convert: `int(node_id)`.

---