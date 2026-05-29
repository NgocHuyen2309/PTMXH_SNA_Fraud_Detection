"""
make_dataset.py
───────────────────────────────────────────────────────────────────────────────
Pipeline tiền xử lý dữ liệu cho đồ án Phân tích Mạng Xã Hội.
Dataset : Bitcoin OTC Trust Weighted Signed Network (Stanford SNAP)
URL gốc : https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html

Đầu ra (thư mục data/):
  data/raw/
      soc-sign-bitcoinotc.csv          ← file CSV gốc (tải thủ công từ SNAP)
  data/processed/
      bitcoin_otc_directed.graphml     ← DiGraph đầy đủ (có weight, rating)
      bitcoin_otc_undirected.graphml   ← UndirectedGraph (cho K-core, Louvain)
      bitcoin_otc_edgelist.csv         ← Edge list nhẹ (src, tgt, weight)
      dataset_stats.txt                ← Thống kê tóm tắt

Ràng buộc:
  - Có thể load lại bằng NetworkX / igraph mà không lỗi.
  - Trọng số (weight) chuẩn hóa về [0, 1].
  - Không tồn tại isolated node, self-loop, hay duplicate edge.
───────────────────────────────────────────────────────────────────────────────
Cách dùng:
  python src/data/make_dataset.py

  Tuỳ chọn (thêm --raw_path nếu đặt file CSV ở chỗ khác):
  python src/data/make_dataset.py --raw_path /path/to/soc-sign-bitcoinotc.csv
───────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

# ─── Cấu hình logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Đường dẫn mặc định ────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR      = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RAW_CSV      = RAW_DIR / "soc-sign-bitcoinotc.csv"

# ─── Tên file đầu ra ────────────────────────────────────────────────────────
OUT_DIRECTED    = PROCESSED_DIR / "bitcoin_otc_directed.graphml"
OUT_UNDIRECTED  = PROCESSED_DIR / "bitcoin_otc_undirected.graphml"
OUT_EDGELIST    = PROCESSED_DIR / "bitcoin_otc_edgelist.csv"
OUT_STATS       = PROCESSED_DIR / "dataset_stats.txt"


# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 0 – Tải dữ liệu
# ═══════════════════════════════════════════════════════════════════════════

def download_dataset(dest: Path) -> None:
    """
    Tải file CSV gốc từ Stanford SNAP (nếu chưa có).

    Lưu ý: SNAP đôi khi chặn automated download.
    Nếu hàm này thất bại, hãy tải thủ công tại:
        https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz
    rồi giải nén vào data/raw/soc-sign-bitcoinotc.csv
    """
    import urllib.request
    import gzip
    import shutil

    gz_url  = "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz"
    gz_path = dest.parent / (dest.name + ".gz")

    log.info("Đang tải dataset từ SNAP: %s", gz_url)
    try:
        urllib.request.urlretrieve(gz_url, gz_path)
    except Exception as e:
        raise RuntimeError(
            f"Không thể tải tự động ({e}).\n"
            f"Hãy tải thủ công tại {gz_url} rồi giải nén vào {dest}"
        )

    log.info("Giải nén %s -> %s", gz_path.name, dest.name)
    with gzip.open(gz_path, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()  # xoá file .gz tạm


# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 1 – Đọc CSV & Làm sạch
# ═══════════════════════════════════════════════════════════════════════════

def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """
    Đọc CSV gốc và thực hiện làm sạch cơ bản.

    Cột CSV gốc (không có header):
        source, target, rating, time
        • source / target : ID người dùng (integer)
        • rating          : mức tin cậy, [-10 .. +10]
        • time            : Unix timestamp

    Các bước làm sạch:
        1. Loại bỏ self-loop (src == tgt)
        2. Loại bỏ duplicate edge (giữ lại lần đánh giá mới nhất)
        3. Chuẩn hóa rating -> weight ∈ [0, 1]
           công thức: weight = (rating + 10) / 20
    """
    log.info("Đọc CSV: %s", csv_path)
    # Không ép dtype ngay khi đọc — file SNAP thực tế có cột time dạng float.
    # Sẽ convert về int sau khi đã load xong.
    df = pd.read_csv(
        csv_path,
        names=["source", "target", "rating", "time"],
    )
    # Convert về kiểu số nguyên (dùng Int64 nullable để an toàn với NaN)
    for col in ["source", "target", "rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["time"] = pd.to_numeric(df["time"], errors="coerce").round(0).astype("Int64")

    # Loại bỏ dòng có giá trị null (dữ liệu lỗi)
    n_before = len(df)
    df = df.dropna()
    if len(df) < n_before:
        log.warning("  Loại %d dòng null/lỗi khi parse", n_before - len(df))

    # Chuyển sang int64 thông thường để xử lý tiếp
    df = df.astype({"source": int, "target": int, "rating": int, "time": int})
    log.info("  Dữ liệu thô  : %d dòng, %d cột", *df.shape)

    # 1. Loại self-loop
    n_self = (df["source"] == df["target"]).sum()
    df = df[df["source"] != df["target"]]
    log.info("  Loại self-loop: -%d dòng", n_self)

    # 2. Loại duplicate (giữ lần đánh giá mới nhất - timestamp lớn nhất)
    df = df.sort_values("time", ascending=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=["source", "target"], keep="last")
    log.info("  Loại duplicate: -%d dòng", n_before - len(df))

    # 3. Chuẩn hóa rating -> weight ∈ [0, 1]
    #    rating=-10 -> weight=0.00  (hoàn toàn không tin)
    #    rating=  0 -> weight=0.50  (trung lập)
    #    rating=+10 -> weight=1.00  (hoàn toàn tin tưởng)
    df["weight"] = (df["rating"] + 10) / 20.0
    df["weight"] = df["weight"].round(4)

    log.info("  Dữ liệu sạch : %d dòng còn lại", len(df))
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 2 – Xây dựng đồ thị NetworkX
# ═══════════════════════════════════════════════════════════════════════════

def build_graphs(df: pd.DataFrame):
    """
    Xây dựng hai phiên bản đồ thị từ DataFrame sạch:

        G_dir  : DiGraph  (đồ thị có hướng)
                 → dùng cho PageRank (Bài 4)

        G_und  : Graph    (đồ thị vô hướng)
                 → dùng cho K-core Decomposition (Bài 5)
                 → dùng cho Louvain Community Detection (Bài 6)

    Trọng số cạnh chứa cả 'rating' (gốc) lẫn 'weight' (đã chuẩn hóa).
    """
    log.info("Xây dựng DiGraph...")
    G_dir = nx.from_pandas_edgelist(
        df,
        source="source",
        target="target",
        edge_attr=["rating", "weight", "time"],
        create_using=nx.DiGraph(),
    )

    # Xoá isolated node (không có cạnh nào sau khi lọc)
    isolated_dir = list(nx.isolates(G_dir))
    G_dir.remove_nodes_from(isolated_dir)
    log.info(
        "  DiGraph   : %d nodes, %d edges (loại %d isolated)",
        G_dir.number_of_nodes(), G_dir.number_of_edges(), len(isolated_dir)
    )

    log.info("Xây dựng Undirected Graph (cho K-core & Louvain)...")
    # to_undirected: cạnh ngược chiều sẽ được gộp (merge) lấy trọng số
    # lớn hơn để ưu tiên quan hệ tích cực
    G_und = nx.Graph()
    G_und.add_nodes_from(G_dir.nodes())
    for u, v, data in G_dir.edges(data=True):
        if G_und.has_edge(u, v):
            # Gộp cạnh ngược: lấy weight trung bình
            existing = G_und[u][v]
            existing["weight"] = round(
                (existing["weight"] + data["weight"]) / 2, 4
            )
            existing["rating"] = round(
                (existing["rating"] + data["rating"]) / 2, 2
            )
        else:
            G_und.add_edge(u, v, **data)

    isolated_und = list(nx.isolates(G_und))
    G_und.remove_nodes_from(isolated_und)
    log.info(
        "  Undirected: %d nodes, %d edges",
        G_und.number_of_nodes(), G_und.number_of_edges()
    )

    return G_dir, G_und


# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 3 – Thống kê mạng lưới
# ═══════════════════════════════════════════════════════════════════════════

def compute_stats(G_dir: nx.DiGraph, G_und: nx.Graph) -> dict:
    """
    Tính các chỉ số thống kê cơ bản:
        • Số node, số cạnh
        • Average degree
        • Density
        • Phân bố rating (tỉ lệ trust/distrust)
    """
    log.info("Tính thống kê mạng lưới...")

    weights = [d["weight"] for _, _, d in G_und.edges(data=True)]
    ratings = [d["rating"] for _, _, d in G_und.edges(data=True)]

    stats = {
        # --- DiGraph ---
        "directed_nodes"       : G_dir.number_of_nodes(),
        "directed_edges"       : G_dir.number_of_edges(),
        "directed_density"     : round(nx.density(G_dir), 6),
        "directed_avg_degree"  : round(
            sum(d for _, d in G_dir.degree()) / G_dir.number_of_nodes(), 4
        ),
        "directed_avg_in_deg"  : round(
            sum(d for _, d in G_dir.in_degree()) / G_dir.number_of_nodes(), 4
        ),

        # --- Undirected ---
        "undirected_nodes"     : G_und.number_of_nodes(),
        "undirected_edges"     : G_und.number_of_edges(),
        "undirected_density"   : round(nx.density(G_und), 6),
        "undirected_avg_degree": round(
            sum(d for _, d in G_und.degree()) / G_und.number_of_nodes(), 4
        ),

        # --- Trọng số ---
        "weight_mean"          : round(float(np.mean(weights)), 4),
        "weight_std"           : round(float(np.std(weights)), 4),
        "weight_min"           : round(float(np.min(weights)), 4),
        "weight_max"           : round(float(np.max(weights)), 4),

        # --- Tỉ lệ trust/distrust ---
        "pct_positive_trust"   : round(
            sum(1 for r in ratings if r > 0) / len(ratings) * 100, 2
        ),
        "pct_negative_trust"   : round(
            sum(1 for r in ratings if r < 0) / len(ratings) * 100, 2
        ),
        "pct_neutral_trust"    : round(
            sum(1 for r in ratings if r == 0) / len(ratings) * 100, 2
        ),
    }
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 4 – Xuất file
# ═══════════════════════════════════════════════════════════════════════════

def export_files(
    G_dir  : nx.DiGraph,
    G_und  : nx.Graph,
    stats  : dict,
    df     : pd.DataFrame,
) -> None:
    """
    Xuất 4 file đầu ra vào data/processed/:
        1. bitcoin_otc_directed.graphml    (DiGraph)
        2. bitcoin_otc_undirected.graphml  (Undirected)
        3. bitcoin_otc_edgelist.csv        (nhẹ, dễ đọc)
        4. dataset_stats.txt               (tóm tắt số liệu)
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. GraphML – DiGraph
    log.info("Xuất %s", OUT_DIRECTED.name)
    nx.write_graphml(G_dir, str(OUT_DIRECTED))

    # 2. GraphML – Undirected
    log.info("Xuất %s", OUT_UNDIRECTED.name)
    nx.write_graphml(G_und, str(OUT_UNDIRECTED))

    # 3. Edge list CSV: source, target, weight
    log.info("Xuất %s", OUT_EDGELIST.name)
    rows = [
        {"source": u, "target": v, "weight": d["weight"], "rating": d["rating"]}
        for u, v, d in G_und.edges(data=True)
    ]
    pd.DataFrame(rows).to_csv(OUT_EDGELIST, index=False)

    # 4. Thống kê
    log.info("Xuất %s", OUT_STATS.name)
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  BITCOIN OTC – DATASET STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        f.write("[DiGraph – Có hướng]\n")
        for k, v in stats.items():
            if k.startswith("directed"):
                f.write(f"  {k:<35} {v}\n")
        f.write("\n[Undirected Graph – Vô hướng]\n")
        for k, v in stats.items():
            if k.startswith("undirected"):
                f.write(f"  {k:<35} {v}\n")
        f.write("\n[Thống kê trọng số (weight)]\n")
        for k, v in stats.items():
            if k.startswith("weight"):
                f.write(f"  {k:<35} {v}\n")
        f.write("\n[Phân bố đánh giá (Trust)]\n")
        for k, v in stats.items():
            if k.startswith("pct"):
                f.write(f"  {k:<35} {v}%\n")
        f.write("\n[Ghi chú]\n")
        f.write("  Trọng số (weight) = (rating + 10) / 20\n")
        f.write("  Undirected: cạnh ngược chiều được gộp (avg weight)\n")


# ═══════════════════════════════════════════════════════════════════════════
# BƯỚC 5 – Kiểm tra tính toàn vẹn (Sanity Check)
# ═══════════════════════════════════════════════════════════════════════════

def sanity_check() -> None:
    """
    Load lại các file đã xuất và kiểm tra tính toàn vẹn.
    Đảm bảo NetworkX đọc được mà không báo lỗi.
    """
    log.info("─── Sanity Check: Load lại các file ───")

    G1 = nx.read_graphml(str(OUT_DIRECTED))
    assert G1.number_of_nodes() > 0, "DiGraph load rỗng!"
    log.info(
        "  ✓ DiGraph    : %d nodes, %d edges",
        G1.number_of_nodes(), G1.number_of_edges()
    )

    G2 = nx.read_graphml(str(OUT_UNDIRECTED))
    assert G2.number_of_nodes() > 0, "Undirected load rỗng!"
    log.info(
        "  ✓ Undirected : %d nodes, %d edges",
        G2.number_of_nodes(), G2.number_of_edges()
    )

    df_el = pd.read_csv(OUT_EDGELIST)
    assert "weight" in df_el.columns, "Edge list thiếu cột weight!"
    assert df_el["weight"].between(0, 1).all(), "weight nằm ngoài [0,1]!"
    log.info("  ✓ Edge list  : %d edges, weight ∈ [0,1]", len(df_el))

    log.info("─── Sanity Check PASSED ✓ ───")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Tiền xử lý dataset Bitcoin OTC cho đồ án SNA."
    )
    parser.add_argument(
        "--raw_path",
        type=Path,
        default=RAW_CSV,
        help=f"Đường dẫn tới file CSV gốc (mặc định: {RAW_CSV})"
    )
    args = parser.parse_args()

    t_start = time.time()
    log.info("══════ Bitcoin OTC – Data Pipeline ══════")

    # ── Tải dữ liệu nếu chưa có ──────────────────────────────────────────
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not args.raw_path.exists():
        log.warning("Không tìm thấy file CSV: %s", args.raw_path)
        log.info("Thử tải tự động từ SNAP...")
        try:
            download_dataset(args.raw_path)
        except RuntimeError as e:
            log.error(str(e))
            sys.exit(1)
    else:
        log.info("Dùng file có sẵn: %s", args.raw_path)

    # ── Pipeline chính ────────────────────────────────────────────────────
    df           = load_and_clean(args.raw_path)
    G_dir, G_und = build_graphs(df)
    stats        = compute_stats(G_dir, G_und)
    export_files(G_dir, G_und, stats, df)
    sanity_check()

    elapsed = time.time() - t_start
    log.info("══════ Hoàn thành trong %.2f giây ══════", elapsed)
    log.info("Đầu ra: %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()