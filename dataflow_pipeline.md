# System Architecture & Overall Dataflow Pipeline

Sơ đồ dưới đây mô tả luồng chạy dữ liệu của toàn bộ dự án, đối chiếu trực tiếp giữa Baseline Model và Proposed Model, kèm theo các file script tham chiếu.

```mermaid
graph TD
    %% Define Styles
    classDef rawData fill:#f9d0c4,stroke:#333,stroke-width:2px;
    classDef process fill:#d4e157,stroke:#333,stroke-width:2px;
    classDef data fill:#bbdefb,stroke:#333,stroke-width:2px;
    classDef baseline fill:#ffcc80,stroke:#333,stroke-width:2px;
    classDef proposed fill:#81d4fa,stroke:#333,stroke-width:2px;
    classDef evaluate fill:#ce93d8,stroke:#333,stroke-width:2px;
    classDef visual fill:#ffab91,stroke:#333,stroke-width:2px;

    %% Step 1: Raw Data & Preprocessing
    A[Raw Dataset<br>soc-sign-bitcoinotc.csv]:::rawData -->|src/data/make_dataset.py| B(Data Cleaning & Normalization):::process
    
    B --> C[(Undirected Graph<br>bitcoin_otc_undirected.graphml)]:::data
    B --> D[(Directed Graph<br>bitcoin_otc_directed.graphml)]:::data
    
    %% Branch 1: Baseline Model
    subgraph Baseline Pipeline
        C -.->|src/models/baseline_model.py| E[Direct Louvain<br>Community Detection]:::baseline
        D -.->|src/models/baseline_model.py| F[PageRank<br>Mastermind Identification]:::baseline
        E -.-> F
    end
    
    %% Branch 2: Proposed Model
    subgraph Proposed Hybrid Pipeline
        C ==>|src/models/proposed_model.py| G[K-core Decomposition<br>Noise Filtering]:::proposed
        G ==> H[Louvain on Core Graph<br>Community Detection]:::proposed
        D ==>|src/models/proposed_model.py| I[PageRank<br>Mastermind Identification]:::proposed
        H ==> I
    end
    
    %% Step 4: Output Metrics
    F -.-> J{Metrics & Output CSVs}:::data
    I ==> J
    
    %% Step 5: Evaluation & Visualization
    J -->|src/evaluation/evaluate_models.py| K[Modularity & Execution Time<br>Benchmarking]:::evaluate
    J -->|src/visualization/macro_vis.py| L[Macro Visualization<br>Gephi GEXF Export]:::visual
    J -->|src/visualization/micro_vis.py| M[Micro Visualization<br>Communities & Masterminds]:::visual
```
