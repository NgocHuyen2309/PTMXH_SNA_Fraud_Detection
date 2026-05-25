# Enhancing Community Detection Accuracy in Fraud Networks via K-core Decomposition

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![NetworkX](https://img.shields.io/badge/NetworkX-Latest-orange.svg)
![Gephi](https://img.shields.io/badge/Gephi-Visualization-success.svg)
![Course](https://img.shields.io/badge/Academic_Project-Social_Network_Analysis-purple.svg)

## 📌 1. Project Overview
This repository contains the source code, data pipelines, and evaluation metrics for the **Social Network Analysis** course's final project. 

**Research Problem:** E-commerce platforms and social trust networks frequently suffer from coordinated fraud rings and botnets. Traditional community detection algorithms applied to massive raw networks often yield low Modularity scores and consume excessive computational resources due to the interference of millions of peripheral, low-degree spam accounts.

**Objective:** To design, implement, and benchmark a hybrid pipeline that integrates structural noise filtering with community optimization to accurately identify fraud rings and their masterminds.

## 🛠️ 2. Methodology & Core Algorithms
Our system architecture branches into two comparative pipelines:

### Baseline Model
* **Direct Community Detection:** Direct application of the **Louvain Algorithm (Fast community unfolding)** on the entire raw graph.

### Proposed Model (Hybrid Pipeline)
1. **Noise Reduction (K-core Decomposition):** Sequentially peeling away peripheral nodes (shells) with degrees less than $k$ to isolate the dense core graph.
2. **Community Detection (Louvain):** Executing the Louvain algorithm exclusively on the core graph to find highly cohesive fraudulent communities.
3. **Mastermind Identification (PageRank):** Applying Eigenvector-based **PageRank Centrality** within each detected community to pinpoint the most influential node orchestrating the operation.

## 🗂️ 3. Project Directory Structure
The repository strictly follows a standard Data Science dataflow architecture. All evaluation processes are implemented as automated Python scripts rather than notebooks.

```text
PTMXH_SNA_Fraud_Detection/
├── data/
│   ├── raw/                 # Immutable original data from Stanford SNAP (Ignored by Git)
│   └── processed/           # Cleaned and standardized graph datasets (EdgeList, GraphML)
├── src/                     # Source code modules
│   ├── data/                # Scripts to fetch, clean, and preprocess data
│   ├── models/              # Algorithm implementations (baseline_model.py, proposed_model.py)
│   ├── evaluation/          # Scripts to calculate Modularity, Execution Time, and Degree Distribution
│   └── visualization/       # Scripts to export graph topological data for Gephi
├── results/
│   ├── figures/             # High-resolution network visualizations exported from Gephi
│   └── metrics/             # CSV files containing Execution Time, Modularity (Q), and Degree distribution
├── .gitignore               # Ignored files and directories
├── README.md                # Project guideline and documentation
└── requirements.txt         # Project dependencies (NetworkX, Pandas, Matplotlib, etc.)
```

## 📊 4. Dataset
* **Source:** Stanford Large Network Dataset Collection (SNAP).
* **Network Type:** Trust weighted signed network (e.g., Bitcoin OTC / Epinions).
* **Characteristics:** Directed graphs where nodes represent platform users and edges represent weighted trust/distrust evaluation scores.

## 👥 5. Team Members & Responsibilities
This project is collaboratively developed by a team of 6 members. Tasks are strictly divided according to the linear dataflow pipeline:

| Member | Role | Key Responsibilities |
| :--- | :--- | :--- |
| **Tâm** | Data Engineer | Data collection (SNAP), isolated nodes removal, weight normalization, and GraphML conversion. |
| **Huyền** | Tech Lead | Repository architecture, Git management, and Baseline Model (Louvain) implementation. |
| **Trí** | Algorithm Engineer | Proposed Model implementation (K-core peeling -> Louvain -> PageRank pipeline). |
| **Hoàng** | Data Analyst | Performance evaluation (Modularity Q, Execution Time benchmark, Degree distribution analysis). |
| **Linh** | Visualization Specialist | Macro-level topological visualization (Raw Graph vs. K-core structures) using Gephi ForceAtlas2. |
| **Quang** | Visualization Specialist | Micro-level community visualization (Modularity coloring, PageRank sizing) using Gephi. |

## 🚀 6. How to Run the Pipeline
**Step 1: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2: Execute the pipeline sequentially**
```bash
# 1. Preprocess data
python src/data/make_dataset.py

# 2. Run Baseline Model
python src/models/baseline_model.py

# 3. Run Proposed Pipeline
python src/models/proposed_model.py

# 4. Evaluate and Benchmark Models
python src/evaluation/evaluate_models.py
```
*(Check the `results/` folder for output metrics and visualization assets after execution).*