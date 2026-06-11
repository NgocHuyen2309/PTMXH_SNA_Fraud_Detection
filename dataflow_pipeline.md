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

    %% --- GIAI ĐOẠN 1 ---
    subgraph Phase 1: Data Preprocessing
        direction TB
        A[Raw Dataset<br>soc-sign-bitcoinotc.csv]:::rawData -->|src/data/make_dataset.py| B(Data Cleaning & Normalization):::process
        B --> C[(Undirected Graph)]:::data
        B --> D[(Directed Graph)]:::data
    end

    %% --- GIAI ĐOẠN 2 ---
    subgraph Phase 2: Bifurcated Models
        direction TB
        
        subgraph Baseline Pipeline
            direction TB
            E[Direct Louvain<br>Community Detection]:::baseline
            F[PageRank<br>Mastermind Identification]:::baseline
            E -.-> F
        end
        
        subgraph Proposed Hybrid Pipeline
            direction TB
            G[K-core Decomposition<br>Noise Filtering]:::proposed
            H[Louvain on Core Graph<br>Community Detection]:::proposed
            I[PageRank<br>Mastermind Identification]:::proposed
            G ==> H
            H ==> I
        end
    end
    
    %% Nối Data vào 2 luồng
    C -.->|src/models/baseline_model.py| E
    C ==>|src/models/proposed_model.py| G
    
    D -.->|src/models/baseline_model.py| F
    D ==>|src/models/proposed_model.py| I

    %% --- GIAI ĐOẠN 3 ---
    subgraph Phase 3: Evaluation & Visualization
        direction TB
        J{Metrics & Output CSVs}:::data
        K[Modularity & Execution Time<br>Benchmarking]:::evaluate
        L[Macro Visualization<br>Gephi GEXF Export]:::visual
        M[Micro Visualization<br>Communities & Masterminds]:::visual
        
        J -->|src/evaluation/evaluate_models.py| K
        J -->|src/visualization/macro_vis.py| L
        J -->|src/visualization/micro_vis.py| M
    end
    
    %% Nối 2 luồng vào Output
    F -.-> J
    I ==> J
```
