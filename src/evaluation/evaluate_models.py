"""
evaluate_models.py
───────────────────────────────────────────────────────────────────────────────
Phase 4 - Evaluation & Metrics

Mục tiêu:
1. Tính lại Modularity (Q) cho Baseline và Proposed.
2. Tổng hợp Execution Time từ file metrics của từng mô hình.
3. Vẽ biểu đồ so sánh Modularity, Execution Time.
4. Vẽ Degree Distribution trước và sau lọc K-core.
5. Xuất kết quả vào results/metrics và results/figures.

Yêu cầu đầu vào theo cấu trúc repo:
data/processed/bitcoin_otc_undirected.graphml
results/metrics/baseline_communities.csv
results/metrics/baseline_model_metrics.txt
results/metrics/proposed_communities.csv
results/metrics/proposed_model_metrics.txt
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx.algorithms.community.quality import modularity


# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class EvaluationMetrics:
    """Đánh giá và so sánh Baseline Model với Proposed Model."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_dir = self.project_root / "data" / "processed"
        self.results_metrics_dir = self.project_root / "results" / "metrics"
        self.results_figures_dir = self.project_root / "results" / "figures"

        self.graph_path = self.data_dir / "bitcoin_otc_undirected.graphml"

        self.baseline_partition_path = self.results_metrics_dir / "baseline_communities.csv"
        self.baseline_metrics_path = self.results_metrics_dir / "baseline_model_metrics.txt"

        self.proposed_partition_path = self.results_metrics_dir / "proposed_communities.csv"
        self.proposed_metrics_path = self.results_metrics_dir / "proposed_model_metrics.txt"

        self.G_raw: Optional[nx.Graph] = None

    # ─── Load dữ liệu ───────────────────────────────────────────────────────
    def load_graph(self) -> nx.Graph:
        """Đọc đồ thị vô hướng đã tiền xử lý."""
        if not self.graph_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file graph: {self.graph_path}\n"
                "Hãy kiểm tra data/processed/bitcoin_otc_undirected.graphml"
            )

        log.info("Đang đọc graph từ: %s", self.graph_path)
        self.G_raw = nx.read_graphml(self.graph_path)
        log.info(
            "Raw graph: %d nodes, %d edges",
            self.G_raw.number_of_nodes(),
            self.G_raw.number_of_edges(),
        )
        return self.G_raw

    @staticmethod
    def read_partition_csv(path: Path) -> Dict[str, int]:
        """
        Đọc file ánh xạ node -> community.

        Hỗ trợ tên cột:
        - Node_ID, Community_ID
        - node, community
        - id, label
        """
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file phân cụm: {path}")

        df = pd.read_csv(path)

        # Chuẩn hóa tên cột để tránh lỗi khác format giữa các thành viên.
        col_map = {c.lower().strip(): c for c in df.columns}

        node_col = None
        for candidate in ["node_id", "node", "id", "source"]:
            if candidate in col_map:
                node_col = col_map[candidate]
                break

        comm_col = None
        for candidate in ["community_id", "community", "modularity_class", "label", "cluster"]:
            if candidate in col_map:
                comm_col = col_map[candidate]
                break

        if node_col is None or comm_col is None:
            raise ValueError(
                f"File {path.name} cần có cột node và community.\n"
                f"Các cột hiện có: {list(df.columns)}"
            )

        # GraphML thường đọc node id dưới dạng string, nên ép Node_ID về str.
        partition = {
            str(row[node_col]): int(row[comm_col])
            for _, row in df[[node_col, comm_col]].dropna().iterrows()
        }

        log.info(
            "Đã đọc %s: %d nodes, %d communities",
            path.name,
            len(partition),
            len(set(partition.values())),
        )
        return partition

    @staticmethod
    def partition_to_communities(partition: Dict[str, int]) -> List[Set[str]]:
        """Chuyển dict node -> community thành list[set[node]] để tính modularity."""
        grouped: Dict[int, Set[str]] = {}
        for node, comm_id in partition.items():
            grouped.setdefault(comm_id, set()).add(node)
        return list(grouped.values())

    @staticmethod
    def parse_execution_time(path: Path) -> Optional[float]:
        """Lấy Execution Time từ file .txt của Baseline/Proposed."""
        if not path.exists():
            log.warning("Không tìm thấy file metrics: %s", path)
            return None

        text = path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"Execution\s*Time\s*\(s\)\s*:\s*([0-9]*\.?[0-9]+)", text, re.I)
        if match:
            return float(match.group(1))

        log.warning("Không đọc được Execution Time trong file: %s", path)
        return None

    @staticmethod
    def compute_partition_modularity(
        G: nx.Graph,
        partition: Dict[str, int],
        model_name: str,
    ) -> dict:
        """
        Tính Modularity Q trên graph tương ứng với tập node của partition.

        Với Baseline: partition thường chứa toàn bộ raw graph.
        Với Proposed: partition chỉ chứa các node còn lại sau K-core, vì vậy cần lấy induced subgraph.
        """
        partition_nodes = set(partition.keys())
        graph_nodes = set(G.nodes())

        valid_nodes = partition_nodes.intersection(graph_nodes)
        missing_nodes = partition_nodes.difference(graph_nodes)

        if missing_nodes:
            log.warning(
                "%s: Có %d node trong partition không tồn tại trong graph. Bỏ qua các node này.",
                model_name,
                len(missing_nodes),
            )

        if not valid_nodes:
            raise ValueError(f"{model_name}: Không có node hợp lệ để tính modularity.")

        G_eval = G.subgraph(valid_nodes).copy()

        # Lọc lại partition chỉ giữ node có trong graph.
        filtered_partition = {n: c for n, c in partition.items() if n in valid_nodes}
        communities = EvaluationMetrics.partition_to_communities(filtered_partition)

        q_score = modularity(G_eval, communities, weight="weight")

        return {
            "Model": model_name,
            "Num_Nodes": G_eval.number_of_nodes(),
            "Num_Edges": G_eval.number_of_edges(),
            "Num_Communities": len(communities),
            "Modularity_Q": round(q_score, 6),
        }

    # ─── Degree Distribution ────────────────────────────────────────────────
    @staticmethod
    def degree_distribution(G: nx.Graph) -> pd.DataFrame:
        """Tính bảng phân phối bậc: degree -> count."""
        degrees = [degree for _, degree in G.degree()]
        df = pd.Series(degrees).value_counts().sort_index().reset_index()
        df.columns = ["Degree", "Count"]
        return df

    def build_kcore_graph(self, proposed_partition: Optional[Dict[str, int]] = None) -> nx.Graph:
        """
        Tạo graph sau K-core để vẽ degree distribution.

        Ưu tiên dùng chính các node trong proposed_communities.csv để khớp với mô hình Proposed.
        Nếu chưa có proposed_communities.csv, dùng max k-core từ raw graph.
        """
        if self.G_raw is None:
            raise RuntimeError("Graph chưa được load.")

        if proposed_partition:
            nodes = set(proposed_partition.keys()).intersection(set(self.G_raw.nodes()))
            if nodes:
                log.info("Dùng node trong proposed_communities.csv làm graph sau K-core.")
                return self.G_raw.subgraph(nodes).copy()

        core_numbers = nx.core_number(self.G_raw)
        max_k = max(core_numbers.values())
        log.info("Chưa có proposed partition. Tự lấy max K-core với k=%d.", max_k)
        return nx.k_core(self.G_raw, k=max_k).copy()

    def export_degree_distribution(
        self,
        G_raw: nx.Graph,
        G_kcore: nx.Graph,
    ) -> pd.DataFrame:
        """Xuất bảng degree distribution trước/sau K-core."""
        raw_df = self.degree_distribution(G_raw).rename(columns={"Count": "Raw_Count"})
        kcore_df = self.degree_distribution(G_kcore).rename(columns={"Count": "Kcore_Count"})

        merged = pd.merge(raw_df, kcore_df, on="Degree", how="outer").fillna(0)
        merged["Raw_Count"] = merged["Raw_Count"].astype(int)
        merged["Kcore_Count"] = merged["Kcore_Count"].astype(int)
        merged = merged.sort_values("Degree")

        out_path = self.results_metrics_dir / "degree_distribution_before_after_kcore.csv"
        merged.to_csv(out_path, index=False)
        log.info("Đã lưu degree distribution: %s", out_path)
        return merged

    # ─── Visualization ─────────────────────────────────────────────────────
    def plot_modularity(self, benchmark_df: pd.DataFrame) -> None:
        plt.figure(figsize=(8, 5))
        plt.bar(benchmark_df["Model"], benchmark_df["Modularity_Q"])
        plt.title("Modularity Comparison: Baseline vs Proposed")
        plt.xlabel("Model")
        plt.ylabel("Modularity (Q)")
        plt.ylim(0, max(benchmark_df["Modularity_Q"]) * 1.2 if len(benchmark_df) else 1)

        for i, value in enumerate(benchmark_df["Modularity_Q"]):
            plt.text(i, value, f"{value:.4f}", ha="center", va="bottom")

        plt.tight_layout()
        out_path = self.results_figures_dir / "modularity_comparison.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        log.info("Đã lưu biểu đồ Modularity: %s", out_path)

    def plot_execution_time(self, benchmark_df: pd.DataFrame) -> None:
        if "Execution_Time_s" not in benchmark_df.columns:
            return

        plot_df = benchmark_df.dropna(subset=["Execution_Time_s"])
        if plot_df.empty:
            log.warning("Không có Execution Time để vẽ biểu đồ.")
            return

        plt.figure(figsize=(8, 5))
        plt.bar(plot_df["Model"], plot_df["Execution_Time_s"])
        plt.title("Execution Time Comparison: Baseline vs Proposed")
        plt.xlabel("Model")
        plt.ylabel("Execution Time (seconds)")

        for i, value in enumerate(plot_df["Execution_Time_s"]):
            plt.text(i, value, f"{value:.2f}s", ha="center", va="bottom")

        plt.tight_layout()
        out_path = self.results_figures_dir / "execution_time_comparison.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        log.info("Đã lưu biểu đồ Execution Time: %s", out_path)

    def plot_degree_distribution(self, degree_df: pd.DataFrame) -> None:
        plt.figure(figsize=(9, 5))
        
        # SỬA LẠI: Lọc bỏ các Degree có Count = 0 để tránh lỗi log(0)
        plot_raw = degree_df[degree_df["Raw_Count"] > 0]
        plot_kcore = degree_df[degree_df["Kcore_Count"] > 0]

        # SỬA LẠI: Thay plt.plot bằng plt.loglog để vẽ thang đo Logarit
        plt.loglog(plot_raw["Degree"], plot_raw["Raw_Count"], marker="o", linestyle="", alpha=0.6, label="Before K-core (Raw)")
        plt.loglog(plot_kcore["Degree"], plot_kcore["Kcore_Count"], marker="x", linestyle="", color="red", label="After K-core")

        plt.title("Degree Distribution (Log-Log Scale)")
        plt.xlabel("Degree (log)")
        plt.ylabel("Number of Nodes (log)")
        plt.legend()
        plt.grid(True, alpha=0.3, which="both", ls="--") # Lưới cho cả trục log

        plt.tight_layout()
        out_path = self.results_figures_dir / "degree_distribution_before_after_kcore.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        log.info("Đã lưu biểu đồ Degree Distribution (Log-Log): %s", out_path)


    # ─── Main pipeline ─────────────────────────────────────────────────────
    def run(self) -> None:
        self.results_metrics_dir.mkdir(parents=True, exist_ok=True)
        self.results_figures_dir.mkdir(parents=True, exist_ok=True)

        G_raw = self.load_graph()

        benchmark_rows = []
        proposed_partition = None

        # Baseline
        if self.baseline_partition_path.exists():
            baseline_partition = self.read_partition_csv(self.baseline_partition_path)
            baseline_row = self.compute_partition_modularity(G_raw, baseline_partition, "Baseline")
            baseline_row["Execution_Time_s"] = self.parse_execution_time(self.baseline_metrics_path)
            benchmark_rows.append(baseline_row)
        else:
            log.warning("Bỏ qua Baseline vì chưa có %s", self.baseline_partition_path)

        # Proposed
        if self.proposed_partition_path.exists():
            proposed_partition = self.read_partition_csv(self.proposed_partition_path)
            proposed_row = self.compute_partition_modularity(G_raw, proposed_partition, "Proposed")
            proposed_row["Execution_Time_s"] = self.parse_execution_time(self.proposed_metrics_path)
            benchmark_rows.append(proposed_row)
        else:
            log.warning(
                "Chưa có proposed_communities.csv. "
                "Hãy chạy src/models/proposed_model.py trước để so sánh đủ Baseline vs Proposed."
            )

        if not benchmark_rows:
            raise RuntimeError("Không có kết quả phân cụm nào để đánh giá.")

        benchmark_df = pd.DataFrame(benchmark_rows)
        out_benchmark = self.results_metrics_dir / "benchmark_metrics.csv"
        benchmark_df.to_csv(out_benchmark, index=False)
        log.info("Đã lưu benchmark metrics: %s", out_benchmark)

        # Degree distribution
        G_kcore = self.build_kcore_graph(proposed_partition)
        degree_df = self.export_degree_distribution(G_raw, G_kcore)

        # Charts
        self.plot_modularity(benchmark_df)
        self.plot_execution_time(benchmark_df)
        self.plot_degree_distribution(degree_df)

        log.info("══════ Phase 4 Evaluation & Metrics hoàn tất ══════")
        print("\nKết quả Benchmark:")
        print(benchmark_df.to_string(index=False))


if __name__ == "__main__":
    evaluator = EvaluationMetrics()
    evaluator.run()
