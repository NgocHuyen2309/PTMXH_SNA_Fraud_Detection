import time
import logging
import argparse
from pathlib import Path
from collections import Counter

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR          = PROJECT_ROOT / "data"    / "processed"
RESULTS_METRICS   = PROJECT_ROOT / "results" / "metrics"
RESULTS_FIGURES   = PROJECT_ROOT / "results" / "figures"

# Input 
COMMUNITIES_CSV   = RESULTS_METRICS / "proposed_communities.csv"
MASTERMINDS_CSV   = RESULTS_METRICS / "proposed_masterminds.csv"
GRAPH_UNDIRECTED  = DATA_DIR / "bitcoin_otc_undirected.graphml"
GRAPH_DIRECTED    = DATA_DIR / "bitcoin_otc_directed.graphml"

# Output
GEXF_OUTPUT       = DATA_DIR / "fraud_network_micro.gexf"
FIG_COMMUNITIES   = RESULTS_FIGURES / "micro_communities.png"
FIG_MASTERMINDS   = RESULTS_FIGURES / "micro_masterminds.png"

# Hằng số đồ hoạ
NODE_SIZE_MIN     = 80
NODE_SIZE_MAX     = 1800
FONT_SIZE_MM      = 9
FIG_DPI           = 180
PALETTE           = list(mcolors.TABLEAU_COLORS.values())   # 10 màu rõ ràng

class MicroVisualizer:
    """
    Lớp đóng gói toàn bộ pipeline trực quan hóa micro-level.
    Thiết kế OOP nhất quán với BaselineModel và ProposedModel.
    """

    def __init__(self):
        self.G_und   = None   
        self.G_dir   = None   
        self.G_core  = None   
        self.partition      = {}  
        self.pagerank_global= {}   
        self.mastermind_df  = None 
        self.color_map      = {}   
        RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def load_graphs(self):
        log.info("Đang tải đồ thị từ %s", DATA_DIR)
        if not GRAPH_UNDIRECTED.exists() or not GRAPH_DIRECTED.exists():
            raise FileNotFoundError(
                "Không tìm thấy file GraphML. Hãy chạy make_dataset.py trước."
            )
        self.G_und = nx.read_graphml(str(GRAPH_UNDIRECTED))
        self.G_dir = nx.read_graphml(str(GRAPH_DIRECTED))
        log.info("Undirected : %d nodes, %d edges",
                 self.G_und.number_of_nodes(), self.G_und.number_of_edges())
        log.info("Directed   : %d nodes, %d edges",
                 self.G_dir.number_of_nodes(), self.G_dir.number_of_edges())

    def load_proposed_results(self):
        """
        Đọc hai file CSV do proposed_model.py xuất ra:
          - proposed_communities.csv  : Node_ID, Community_ID
          - proposed_masterminds.csv  : Community_ID, Size, Mastermind_ID, PageRank_Score
        """
        log.info("Đọc kết quả proposed_model từ %s", RESULTS_METRICS)
        if not COMMUNITIES_CSV.exists() or not MASTERMINDS_CSV.exists():
            raise FileNotFoundError(
                "Không tìm thấy file CSV kết quả. Hãy chạy proposed_model.py trước."
            )

        df_comm = pd.read_csv(COMMUNITIES_CSV, dtype={"Node_ID": str, "Community_ID": int})
        self.partition = dict(zip(df_comm["Node_ID"], df_comm["Community_ID"]))
        log.info("  Đã đọc partition: %d nodes, %d cộng đồng",
                 len(self.partition), len(set(self.partition.values())))

        self.mastermind_df = pd.read_csv(
            MASTERMINDS_CSV,
            dtype={"Community_ID": int, "Mastermind_ID": str, "PageRank_Score": float}
        )
        log.info("  Đã đọc mastermind: %d cộng đồng hợp lệ", len(self.mastermind_df))

    def compute_global_pagerank(self):
        """
        Tính PageRank trên toàn bộ G_dir (không phân theo cụm).
        Dùng giá trị này để scale kích thước node trên Gephi và matplotlib.
        """
        log.info("─── Tính PageRank toàn cục ───")
        weight_attr = "weight" if nx.is_weighted(self.G_dir) else None
        self.pagerank_global = nx.pagerank(
            self.G_dir, alpha=0.85, max_iter=200, weight=weight_attr
        )
        top5 = sorted(self.pagerank_global.items(), key=lambda x: x[1], reverse=True)[:5]
        log.info("  Top-5 PageRank: %s", [(n, round(s, 6)) for n, s in top5])

    def build_core_subgraph(self):
        """
        Lấy subgraph từ G_und chỉ gồm các node xuất hiện trong partition
        (tức là các node đã qua K-core filtering của proposed_model.py).
        """
        core_nodes = set(self.partition.keys())
        self.G_core = self.G_und.subgraph(core_nodes).copy()
        log.info("Core subgraph: %d nodes, %d edges",
                 self.G_core.number_of_nodes(), self.G_core.number_of_edges())

    def annotate_and_export_gexf(self):
        """
        Gán vào mỗi node của G_core:
          - modularity_class (int)   ← từ Louvain partition
          - pagerank_score   (float) ← từ PageRank global
          - is_mastermind    (int)   ← 1 nếu là đầu não, 0 nếu không
          - label            (str)   ← tên node, mastermind có tiền tố ★

        Xuất file .gexf để dùng trong Gephi.
        """
        log.info("─── Gán thuộc tính node & xuất GEXF ───")
        mastermind_ids = set(self.mastermind_df["Mastermind_ID"].astype(str))

        for node in self.G_core.nodes():
            node_str = str(node)
            comm_id  = self.partition.get(node_str, -1)
            pr_score = self.pagerank_global.get(node_str, 0.0)
            is_mm    = 1 if node_str in mastermind_ids else 0

            self.G_core.nodes[node]["modularity_class"] = int(comm_id)
            self.G_core.nodes[node]["pagerank_score"]   = float(pr_score)
            self.G_core.nodes[node]["is_mastermind"]    = int(is_mm)

            label = f"USER_{node_str}"
            if is_mm:
                label = f"[MASTERMIND] USER_{node_str}"
            self.G_core.nodes[node]["label"] = label

        nx.write_gexf(self.G_core, str(GEXF_OUTPUT))
        log.info("  ✓ GEXF xuất tại: %s", GEXF_OUTPUT)

    def _build_color_map(self):
        unique_comms = sorted(set(self.partition.values()))
        self.color_map = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(unique_comms)}

    def _scale_sizes(self, nodes: list) -> list:
        scores = np.array([self.pagerank_global.get(str(n), 0.0) for n in nodes])
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-12:
            return [300] * len(nodes)
        scaled = (scores - s_min) / (s_max - s_min)
        return list(NODE_SIZE_MIN + scaled * (NODE_SIZE_MAX - NODE_SIZE_MIN))

    def plot_communities(self):
        log.info("Vẽ hình tổng quan cộng đồng ...")
        self._build_color_map()

        mastermind_ids = set(self.mastermind_df["Mastermind_ID"].astype(str))
        nodes          = list(self.G_core.nodes())
        n              = len(nodes)

        # Layout
        if n <= 800:
            pos = nx.spring_layout(self.G_core, seed=42, k=1.5 / np.sqrt(max(n, 1)))
        else:
            pos = nx.kamada_kawai_layout(self.G_core)

        fig, ax = plt.subplots(figsize=(16, 12), dpi=FIG_DPI)
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")

        # Cạnh
        nx.draw_networkx_edges(
            self.G_core, pos, ax=ax,
            edge_color="#ffffff", alpha=0.10, width=0.3, arrows=False,
        )

        # Node thường
        normal = [nd for nd in nodes if str(nd) not in mastermind_ids]
        nx.draw_networkx_nodes(
            self.G_core, pos, nodelist=normal, ax=ax,
            node_color=[self.color_map.get(self.partition.get(str(nd), -1), "#888") for nd in normal],
            node_size=self._scale_sizes(normal),
            alpha=0.85, linewidths=0.3, edgecolors="#ffffff",
        )

        # Node mastermind (viền đỏ, lớn hơn)
        mm_nodes = [nd for nd in nodes if str(nd) in mastermind_ids]
        mm_sizes = [max(self._scale_sizes([nd])[0] * 1.6, 700) for nd in mm_nodes]
        nx.draw_networkx_nodes(
            self.G_core, pos, nodelist=mm_nodes, ax=ax,
            node_color=[self.color_map.get(self.partition.get(str(nd), -1), "#f00") for nd in mm_nodes],
            node_size=mm_sizes, edgecolors="#ff4444", linewidths=2.8, alpha=1.0,
        )

        # Nhãn mastermind
        mm_labels = {nd: f"★ {nd}" for nd in mm_nodes}
        nx.draw_networkx_labels(self.G_core, pos, labels=mm_labels, ax=ax,
                                font_size=FONT_SIZE_MM, font_color="#ffdd00", font_weight="bold")

        # Legend
        unique_comms = sorted(set(self.partition.values()))
        patches = [mpatches.Patch(color=self.color_map[c], label=f"Community {c}")
                   for c in unique_comms]
        patches.append(mpatches.Patch(color="#ff4444", label="★ Mastermind"))
        ax.legend(handles=patches, loc="upper left", fontsize=7,
                  framealpha=0.3, labelcolor="white", facecolor="#2d2d2d")

        ax.set_title(
            f"Fraud Network – Micro-level Community Detection\n"
            f"Bitcoin OTC · K-core + Louvain · {len(unique_comms)} cộng đồng · "
            f"Kích thước node ∝ PageRank",
            color="white", fontsize=13, pad=12,
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(str(FIG_COMMUNITIES), bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("  ✓ Lưu → %s", FIG_COMMUNITIES)

    def plot_mastermind_closeup(self, top_comms: int = 6):
        log.info("Vẽ hình close-up mastermind ...")
        self._build_color_map()

        comm_sizes   = Counter(self.partition.values())
        top          = [c for c, _ in comm_sizes.most_common(top_comms)]
        mastermind_ids = set(self.mastermind_df["Mastermind_ID"].astype(str))

        # Map: community_id -> mastermind_id
        mm_by_comm = dict(zip(
            self.mastermind_df["Community_ID"],
            self.mastermind_df["Mastermind_ID"].astype(str)
        ))
        mm_pr_by_comm = dict(zip(
            self.mastermind_df["Community_ID"],
            self.mastermind_df["PageRank_Score"]
        ))

        ncols = 3
        nrows = int(np.ceil(len(top) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), dpi=FIG_DPI)
        fig.patch.set_facecolor("#12121f")
        axes_flat = np.array(axes).flatten()

        for idx, comm_id in enumerate(top):
            ax = axes_flat[idx]
            ax.set_facecolor("#1e1e3a")

            comm_nodes = [n for n, c in self.partition.items() if c == comm_id]
            subG       = self.G_core.subgraph(comm_nodes).copy()

            if subG.number_of_nodes() == 0:
                ax.axis("off")
                continue

            pos = nx.spring_layout(
                subG, seed=42 + comm_id,
                k=2.0 / np.sqrt(max(len(comm_nodes), 1))
            )

            mm_id = mm_by_comm.get(comm_id)

            # Cạnh
            nx.draw_networkx_edges(subG, pos, ax=ax,
                                   edge_color="#aaaaff", alpha=0.22,
                                   width=0.6, arrows=False)

            # Node thường
            normal_sub = [nd for nd in subG.nodes() if str(nd) != mm_id]
            nx.draw_networkx_nodes(
                subG, pos, nodelist=normal_sub, ax=ax,
                node_color=self.color_map.get(comm_id, "#4488cc"),
                node_size=[max(80, self.pagerank_global.get(str(nd), 0) * 30000)
                           for nd in normal_sub],
                edgecolors="#ffffff", linewidths=0.4, alpha=0.88,
            )

            # Mastermind node
            mm_nodes_sub = [nd for nd in subG.nodes() if str(nd) == mm_id]
            if mm_nodes_sub:
                nx.draw_networkx_nodes(
                    subG, pos, nodelist=mm_nodes_sub, ax=ax,
                    node_color="#ff4444", node_size=900,
                    edgecolors="#ffff00", linewidths=2.5, alpha=1.0,
                )

            # Nhãn
            if len(comm_nodes) <= 25:
                labels = {nd: str(nd) for nd in subG.nodes()}
                nx.draw_networkx_labels(subG, pos, labels=labels, ax=ax,
                                        font_size=6, font_color="#cccccc")
            else:
                # Chỉ label top-5 PageRank
                top5_pr = sorted(subG.nodes(),
                                 key=lambda nd: self.pagerank_global.get(str(nd), 0),
                                 reverse=True)[:5]
                labels = {nd: str(nd) for nd in top5_pr}
                nx.draw_networkx_labels(subG, pos, labels=labels, ax=ax,
                                        font_size=7, font_color="#ffee88")

            # Nhãn mastermind nổi bật
            if mm_nodes_sub:
                nx.draw_networkx_labels(
                    subG, pos, labels={mm_nodes_sub[0]: f"★ {mm_id}"},
                    ax=ax, font_size=FONT_SIZE_MM,
                    font_color="#ff4444", font_weight="bold",
                )

            mm_pr = mm_pr_by_comm.get(comm_id, 0.0)
            ax.set_title(
                f"Community {comm_id}  ({len(comm_nodes)} nodes)\n"
                f"★ Mastermind: {mm_id}  [PR = {mm_pr:.5f}]",
                fontsize=8, color="white", pad=6,
            )
            ax.axis("off")

        for j in range(idx + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(
            "Fraud Community Close-up  ·  Nút đỏ = Mastermind (đầu não)  ·  Kích thước ∝ PageRank",
            color="white", fontsize=11, y=1.01,
        )
        plt.tight_layout()
        plt.savefig(str(FIG_MASTERMINDS), bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("  ✓ Lưu → %s", FIG_MASTERMINDS)

    def run(self, skip_plot: bool = False):
        t_start = time.time()
        log.info("══════ Khởi động MicroVisualizer (Quang) ══════")

        self.load_graphs()
        self.load_proposed_results()
        self.compute_global_pagerank()
        self.build_core_subgraph()
        self.annotate_and_export_gexf()

        if not skip_plot:
            self.plot_communities()
            self.plot_mastermind_closeup()

        elapsed = time.time() - t_start
        log.info("══════ Hoàn thành trong %.2f giây ══════", elapsed)
        log.info("  GEXF      → %s", GEXF_OUTPUT)
        log.info("  Hình 1    → %s", FIG_COMMUNITIES)
        log.info("  Hình 2    → %s", FIG_MASTERMINDS)

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Micro-level fraud visualization – Quang (SNA Project)"
    )
    parser.add_argument("--no-plot", action="store_true",
                        help="Chỉ xuất GEXF, bỏ qua vẽ ảnh matplotlib")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    vis = MicroVisualizer()
    vis.run(skip_plot=args.no_plot)
