"""
baseline_model.py
───────────────────────────────────────────────────────────────────────────────
Triển khai mô hình cơ sở (Baseline Model):
Louvain Community Detection trực tiếp trên toàn bộ đồ thị thô -> PageRank Mastermind

- Bước 1: Load đồ thị thô chưa qua K-core peeling.
- Bước 2: Gom cụm bằng thuật toán Louvain.
- Bước 3: Áp dụng PageRank cho từng cụm để tìm Mastermind.
- Xuất kết quả và đo thời gian để làm đối chứng.
"""

import time
import logging
from pathlib import Path

import networkx as nx
import pandas as pd
import community as community_louvain

# ─── Cấu hình logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Thư mục làm việc ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

def load_graphs():
    log.info("Đang tải đồ thị từ %s", DATA_DIR)
    
    undirected_path = DATA_DIR / "bitcoin_otc_undirected.graphml"
    directed_path = DATA_DIR / "bitcoin_otc_directed.graphml"
    
    if not undirected_path.exists() or not directed_path.exists():
        raise FileNotFoundError("Không tìm thấy dữ liệu đã tiền xử lý. Hãy chạy make_dataset.py trước.")
        
    G_und = nx.read_graphml(undirected_path)
    G_dir = nx.read_graphml(directed_path)
    
    log.info("Undirected: %d nodes, %d edges", G_und.number_of_nodes(), G_und.number_of_edges())
    log.info("Directed: %d nodes, %d edges", G_dir.number_of_nodes(), G_dir.number_of_edges())
    
    return G_und, G_dir

def detect_communities(graph: nx.Graph) -> dict:
    """
    Sử dụng thuật toán Louvain để phát hiện cộng đồng trên toàn bộ đồ thị.
    """
    log.info("─── Bắt đầu phân cụm Louvain trên toàn bộ đồ thị ───")
    # Sử dụng trọng số (weight) để tối ưu Modularity và cố định seed
    partition = community_louvain.best_partition(graph, weight='weight', random_state=42)
    
    num_communities = len(set(partition.values()))
    modularity = community_louvain.modularity(partition, graph, weight='weight')
    
    log.info("Đã gom cụm thành công. Số lượng cộng đồng: %d", num_communities)
    log.info("Chỉ số Modularity (Q): %.4f", modularity)
    
    return partition, modularity

def find_masterminds(partition: dict, G_dir: nx.DiGraph) -> pd.DataFrame:
    """
    Sử dụng PageRank trên mỗi cộng đồng (với đồ thị có hướng G_dir) để
    tìm kẻ cầm đầu (Mastermind).
    """
    log.info("─── Tìm Mastermind bằng PageRank ───")
    
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)
        
    masterminds = []
    
    for comm_id, members in communities.items():
        sub_dir = G_dir.subgraph(members)
        
        if sub_dir.number_of_nodes() > 1 and sub_dir.number_of_edges() > 0:
            pr_scores = nx.pagerank(sub_dir, alpha=0.85, weight='weight')
            mastermind = max(pr_scores, key=pr_scores.get)
            max_score = pr_scores[mastermind]
            
            masterminds.append({
                "Community_ID": comm_id,
                "Size": len(members),
                "Mastermind_ID": mastermind,
                "PageRank_Score": max_score
            })
            
    df_masterminds = pd.DataFrame(masterminds).sort_values(by="Size", ascending=False)
    log.info("Đã tìm thấy Mastermind cho %d cộng đồng hợp lệ.", len(df_masterminds))
    
    return df_masterminds

def main():
    t_start = time.time()
    log.info("══════ Chạy mô hình Cơ sở (Baseline) ══════")
    
    # Bước 1: Load đồ thị
    G_und, G_dir = load_graphs()
    
    # Bước 2: Phân cụm Louvain trực tiếp trên G_und (không có K-core)
    partition, modularity = detect_communities(G_und)
    
    # Bước 3: Tìm Mastermind bằng PageRank
    df_masterminds = find_masterminds(partition, G_dir)
    
    # Xuất kết quả
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_METRICS_DIR / "baseline_masterminds.csv"
    df_masterminds.to_csv(out_csv, index=False)
    log.info("Đã lưu thông tin Masterminds tại: %s", out_csv)
    
    # Lưu partition (mapping node -> cộng đồng)
    partition_df = pd.DataFrame(list(partition.items()), columns=["Node_ID", "Community_ID"])
    partition_csv = RESULTS_METRICS_DIR / "baseline_communities.csv"
    partition_df.to_csv(partition_csv, index=False)
    
    t_end = time.time()
    elapsed = t_end - t_start
    log.info("══════ Hoàn thành trong %.2f giây ══════", elapsed)
    
    # Lưu số liệu Modularity & Execution time
    with open(RESULTS_METRICS_DIR / "baseline_model_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Pipeline: Baseline (Louvain -> PageRank)\n")
        f.write(f"Execution Time (s): {elapsed:.2f}\n")
        f.write(f"Baseline Modularity (Q): {modularity:.4f}\n")

if __name__ == "__main__":
    main()
