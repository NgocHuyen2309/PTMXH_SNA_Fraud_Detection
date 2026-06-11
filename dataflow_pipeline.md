# System Architecture & Overall Dataflow Pipeline

Sơ đồ dưới đây mô tả luồng chạy dữ liệu của toàn bộ dự án, đối chiếu trực tiếp giữa Baseline Model và Proposed Model, kèm theo các file script tham chiếu.

```mermaid
flowchart TD
    %% Define Styles
    classDef rawData fill:#f9d0c4,stroke:#333,stroke-width:2px;
    classDef process fill:#d4e157,stroke:#333,stroke-width:2px;
    classDef data fill:#bbdefb,stroke:#333,stroke-width:2px;
    classDef baseline fill:#ffcc80,stroke:#333,stroke-width:2px;
    classDef proposed fill:#81d4fa,stroke:#333,stroke-width:2px;
    classDef evaluate fill:#ce93d8,stroke:#333,stroke-width:2px;
    classDef visual fill:#ffab91,stroke:#333,stroke-width:2px;

    %% Data Preprocessing
    A[Raw Dataset<br>soc-sign-bitcoinotc.csv]:::rawData -->|make_dataset.py| B(Data Cleaning & Normalization):::process

    %% Models
    subgraph Baseline Pipeline
        direction TB
        C1[(Undirected Graph)]:::data -.->|baseline_model.py| E[Direct Louvain<br>Community Detection]:::baseline
        D1[(Directed Graph)]:::data -.->|baseline_model.py| F[PageRank<br>Mastermind Identification]:::baseline
        E -.-> F
    end
    
    subgraph Proposed Hybrid Pipeline
        direction TB
        C2[(Undirected Graph)]:::data ==>|proposed_model.py| G[K-core Decomposition<br>Noise Filtering]:::proposed
        G ==> H[Louvain on Core Graph<br>Community Detection]:::proposed
        D2[(Directed Graph)]:::data ==>|proposed_model.py| I[PageRank<br>Mastermind Identification]:::proposed
        H ==> I
    end
    
    %% Phân phối Data sạch vào 2 luồng độc lập (Đảm bảo 100% không đè mũi tên)
    B --> C1
    B --> D1
    B --> C2
    B --> D2

    %% Evaluation & Visualization
    J{Metrics & Output CSVs}:::data
    
    %% Nối 2 luồng vào Output
    F -.-> J
    I ==> J

    J -->|evaluate_models.py| K[Modularity & Execution Time<br>Benchmarking]:::evaluate
    J -->|macro_vis.py| L[Macro Visualization<br>Gephi GEXF Export]:::visual
    J -->|micro_vis.py| M[Micro Visualization<br>Communities & Masterminds]:::visual
```
