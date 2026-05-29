"""
proposed_model.py
───────────────────────────────────────────────────────────────────────────────
Triển khai mô hình đề xuất (Hybrid Pipeline):
K-core Decomposition -> Louvain Community Detection -> PageRank Mastermind

- Bước 1: Lọc nhiễu bằng K-core, in chi tiết số lượng node/edge qua từng lớp vỏ k.
- Bước 2: Gom cụm bằng thuật toán Louvain trên đồ thị lõi đặc (max k-core).
- Bước 3: Áp dụng PageRank cho từng cụm (sử dụng DiGraph) để tìm Mastermind.
- Xuất kết quả và đo thời gian.
"""

import time
import logging
from pathlib import Path

import networkx as nx
import pandas as pd
import community as community_louvain  # python-louvain

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

def k_core_peeling(G_und: nx.Graph) -> nx.Graph:
    """
    Tiến hành bóc tách các lớp K-core và ghi nhận (log) số lượng node/edge
    còn lại ở từng bước. Lấy lõi đặc nhất (max k-core) làm đầu ra.
    """
    log.info("─── Bắt đầu phân rã K-core ───")
    
    # Tính core_number cho tất cả các node trong đồ thị
    core_numbers = nx.core_number(G_und)
    max_k = max(core_numbers.values())
    
    k_core_trace = []
    
    # Sử dụng view thay vì .copy() để tiết kiệm bộ nhớ trong vòng lặp
    current_graph = G_und
    
    # Lặp qua từng k-shell từ 1 đến max_k
    for k in range(1, max_k + 1):
        # Lọc ra các node có core_number >= k
        nodes_to_keep = [n for n, c in core_numbers.items() if c >= k]
        current_graph = current_graph.subgraph(nodes_to_keep)
        
        num_nodes = current_graph.number_of_nodes()
        num_edges = current_graph.number_of_edges()
        
        k_core_trace.append({"k": k, "nodes": num_nodes, "edges": num_edges})
        log.info("K-core step k=%2d : %5d nodes, %6d edges còn lại", k, num_nodes, num_edges)
        
    # Lưu bảng log k-core
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(k_core_trace).to_csv(RESULTS_METRICS_DIR / "k_core_trace.csv", index=False)
    
    log.info("Đã đạt lõi k-core lớn nhất (k=%d).", max_k)
    return current_graph.copy()

def detect_communities(core_graph: nx.Graph) -> dict:
    """
    Sử dụng thuật toán Louvain để phát hiện các băng đảng/cụm trên lõi K-core.
    """
    log.info("─── Bắt đầu phân cụm Louvain ───")
    # Sử dụng trọng số (weight) để tối ưu Modularity và cố định seed (random_state)
    partition = community_louvain.best_partition(core_graph, weight='weight', random_state=42)
    
    num_communities = len(set(partition.values()))
    modularity = community_louvain.modularity(partition, core_graph, weight='weight')
    
    log.info("Đã gom cụm thành công. Số lượng cộng đồng: %d", num_communities)
    log.info("Chỉ số Modularity (Q): %.4f", modularity)
    
    return partition, modularity

def find_masterminds(partition: dict, G_dir: nx.DiGraph) -> pd.DataFrame:
    """
    Sử dụng PageRank trên mỗi cộng đồng (với đồ thị có hướng G_dir) để
    tìm kẻ cầm đầu (Mastermind).
    """
    log.info("─── Tìm Mastermind bằng PageRank ───")
    
    # Gom danh sách node theo từng cụm
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)
        
    masterminds = []
    
    for comm_id, members in communities.items():
        # Lọc ra subgraph có hướng của cộng đồng hiện tại
        sub_dir = G_dir.subgraph(members)
        
        # Chỉ lấy những cộng đồng có từ 2 node trở lên và có cạnh
        if sub_dir.number_of_nodes() > 1 and sub_dir.number_of_edges() > 0:
            # Tính PageRank (chú ý: pagerank mặc định hỗ trợ DiGraph, dùng 'weight' làm trọng số cạnh, alpha=0.85)
            pr_scores = nx.pagerank(sub_dir, alpha=0.85, weight='weight')
            
            # Mastermind là người có PageRank cao nhất trong cụm
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
    log.info("══════ Chạy mô hình Hybrid Đề xuất ══════")
    
    # Bước 1: Load đồ thị
    G_und, G_dir = load_graphs()
    
    # Bước 2: K-core peeling
    core_graph = k_core_peeling(G_und)
    
    # Bước 3: Phân cụm Louvain
    partition, modularity = detect_communities(core_graph)
    
    # Bước 4: Tìm Mastermind bằng PageRank
    df_masterminds = find_masterminds(partition, G_dir)
    
    # Xuất kết quả
    out_csv = RESULTS_METRICS_DIR / "proposed_masterminds.csv"
    df_masterminds.to_csv(out_csv, index=False)
    log.info("Đã lưu thông tin Masterminds tại: %s", out_csv)
    
    # Lưu partition (mapping node -> cộng đồng)
    partition_df = pd.DataFrame(list(partition.items()), columns=["Node_ID", "Community_ID"])
    partition_csv = RESULTS_METRICS_DIR / "proposed_communities.csv"
    partition_df.to_csv(partition_csv, index=False)
    
    t_end = time.time()
    elapsed = t_end - t_start
    log.info("══════ Hoàn thành trong %.2f giây ══════", elapsed)
    
    # Lưu số liệu Modularity & Execution time
    with open(RESULTS_METRICS_DIR / "proposed_model_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Pipeline: K-core -> Louvain -> PageRank\n")
        f.write(f"Execution Time (s): {elapsed:.2f}\n")
        f.write(f"Final Core Modularity (Q): {modularity:.4f}\n")
        f.write(f"Max K-core used: {max(nx.core_number(G_und).values())}\n")

if __name__ == "__main__":
    main()
