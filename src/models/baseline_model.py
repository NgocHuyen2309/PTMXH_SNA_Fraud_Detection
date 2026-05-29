"""
baseline_model.py
───────────────────────────────────────────────────────────────────────────────
Triển khai mô hình cơ sở (Baseline Model) theo mô hình Hướng đối tượng (OOP).
Pipeline: Louvain Community Detection trực tiếp trên raw graph -> PageRank Mastermind
───────────────────────────────────────────────────────────────────────────────
"""

import time
import logging
from pathlib import Path
import networkx as nx
import pandas as pd
import community as community_louvain

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class BaselineModel:
    """
    Lớp đóng gói mô hình đối chứng (Baseline Model).
    Thiết kế theo chuẩn OOP giúp dễ dàng quản lý trạng thái đồ thị và mở rộng cấu trúc.
    """
    def __init__(self):
        # Thiết lập các thuộc tính đường dẫn dựa trên cấu trúc thư mục chuẩn
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_dir = self.project_root / "data" / "processed"
        self.results_metrics_dir = self.project_root / "results" / "metrics"
        
        # Các thuộc tính lưu trữ trạng thái đồ thị và kết quả phân tích
        self.G_und = None
        self.G_dir = None
        self.partition = None
        self.modularity = 0.0
        self.elapsed_time = 0.0

    def load_graphs(self):
        """Nạp các file đồ thị định dạng GraphML từ tầng tiền xử lý dữ liệu"""
        log.info("Đang tải đồ thị dữ liệu từ %s", self.data_dir)
        undirected_path = self.data_dir / "bitcoin_otc_undirected.graphml"
        directed_path = self.data_dir / "bitcoin_otc_directed.graphml"
        
        if not undirected_path.exists() or not directed_path.exists():
            raise FileNotFoundError("Không tìm thấy dữ liệu. Hãy chạy make_dataset.py trước.")
            
        self.G_und = nx.read_graphml(undirected_path)
        self.G_dir = nx.read_graphml(directed_path)
        
        log.info("Undirected Graph: %d nodes, %d edges Loaded.", self.G_und.number_of_nodes(), self.G_und.number_of_edges())
        log.info("Directed Graph  : %d nodes, %d edges Loaded.", self.G_dir.number_of_nodes(), self.G_dir.number_of_edges())

    def detect_communities(self):
        """Áp dụng trực tiếp giải thuật Louvain không qua bộ lọc K-core"""
        log.info("─── Bắt đầu phân cụm Louvain trên toàn bộ đồ thị thô ───")
        # Sử dụng trọng số để tối ưu Modularity cục bộ và đóng băng tính ngẫu nhiên
        self.partition = community_louvain.best_partition(self.G_und, weight='weight', random_state=42)
        
        num_communities = len(set(self.partition.values()))
        self.modularity = community_louvain.modularity(self.partition, self.G_und, weight='weight')
        
        log.info("Phân cụm hoàn tất. Tìm thấy: %d cộng đồng.", num_communities)
        log.info("Chỉ số Modularity đạt được (Q): %.4f", self.modularity)

    def find_masterminds(self) -> pd.DataFrame:
        """Khai thác PageRank cục bộ trên từng phân vùng đồ thị có hướng"""
        log.info("─── Định vị Mastermind trong các phân cụm bằng PageRank ───")
        
        # Gom danh sách các node ID theo từng Community ID tương ứng
        communities = {}
        for node, comm_id in self.partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
            
        masterminds = []
        for comm_id, members in communities.items():
            # Trích xuất phân vùng cấu trúc có hướng (subgraph view)
            sub_dir = self.G_dir.subgraph(members)
            
            # Chỉ xử lý các cụm có cấu trúc liên kết hợp lệ
            if sub_dir.number_of_nodes() > 1 and sub_dir.number_of_edges() > 0:
                pr_scores = nx.pagerank(sub_dir, alpha=0.85, weight='weight')
                
                # Tìm phần tử có điểm xác suất chuyển đổi trạng thái cao nhất cụm
                mastermind = max(pr_scores, key=pr_scores.get)
                max_score = pr_scores[mastermind]
                
                masterminds.append({
                    "Community_ID": comm_id,
                    "Size": len(members),
                    "Mastermind_ID": mastermind,
                    "PageRank_Score": max_score
                })
                
        df_masterminds = pd.DataFrame(masterminds).sort_values(by="Size", ascending=False)
        log.info("Đã trích xuất Mastermind thành công cho %d cộng đồng.", len(df_masterminds))
        return df_masterminds

    def export_results(self, df_masterminds: pd.DataFrame):
        """Lưu trữ toàn bộ tài nguyên đầu ra phục vụ ma trận đánh giá thực nghiệm"""
        self.results_metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Xuất file xếp hạng thực thể cầm đầu
        out_csv = self.results_metrics_dir / "baseline_masterminds.csv"
        df_masterminds.to_csv(out_csv, index=False)
        log.info("Đã lưu thông tin Masterminds tại: %s", out_csv)
        
        # 2. Xuất file ánh xạ phân cụm để vẽ Gephi mức vi mô
        partition_df = pd.DataFrame(list(self.partition.items()), columns=["Node_ID", "Community_ID"])
        partition_csv = self.results_metrics_dir / "baseline_communities.csv"
        partition_df.to_csv(partition_csv, index=False)
        log.info("Đã lưu phân vùng cấu trúc tại: %s", partition_csv)
        
        # 3. Ghi nhận các chỉ số đo lường hiệu suất
        with open(self.results_metrics_dir / "baseline_model_metrics.txt", "w", encoding="utf-8") as f:
            f.write(f"Pipeline: Baseline (Louvain -> PageRank)\n")
            f.write(f"Execution Time (s): {self.elapsed_time:.2f}\n")
            f.write(f"Baseline Modularity (Q): {self.modularity:.4f}\n")
        log.info("Đã cập nhật bảng số liệu định lượng baseline_model_metrics.txt")

    def run(self):
        """Kích hoạt và điều phối toàn bộ đường ống xử lý dữ liệu của mô hình"""
        t_start = time.time()
        log.info("══════ Khởi động mô hình Cơ sở (Baseline Model) ══════")
        
        self.load_graphs()
        self.detect_communities()
        df_masterminds = self.find_masterminds()
        
        self.elapsed_time = time.time() - t_start
        self.export_results(df_masterminds)
        log.info("══════ Mô hình hoàn thành trong %.2f giây ══════", self.elapsed_time)


if __name__ == "__main__":
    # Khởi tạo đối tượng mô hình và chạy toàn bộ pipeline cấu trúc
    model = BaselineModel()
    model.run()