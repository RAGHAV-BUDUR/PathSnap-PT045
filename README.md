# PathSnap: Distributed Map-Matching & Route Reconstruction
**UE24CS352A: Machine Learning Hackathon | PES University**  
**Team Number:** PT-045  

---

## Team Roster & Module Ownership

| Name | SRN | Assigned Module |
|---|---|---|
| **Raghav Budur** (P1) | PES1UG23AM225 | Data Processing, Metric Projection, Spatial Indexing (`cKDTree`) & EDA |
| **Pabbisetty Harshitha Sai** (P2) | PES1UG24AM182 | Feature Engineering, Candidate Extraction & 5 ML Models Benchmark |
| **Palleneni Sloka** (P3) | PES1UG24AM183 | Bayesian Hyperparameter Optimisation & HMM Viterbi Sequential Decoder |
| **Parth Pandurang Kale** (P4) | PES1UG24AM184 | Folium Interactive Maps, FastAPI Backend & Evaluator Dashboard UI |

---

## 1. System Overview

PathSnap is an end-to-end distributed machine learning system that matches noisy, low-frequency GPS trajectories onto complex road networks. It employs a two-stage hybrid architecture:
1. **Stage 1 (Discriminative Candidate Classifier):** Scores candidate road segments using a Bayesian-tuned Random Forest classifier across 11 spatial, kinematic, and topological features.
2. **Stage 2 (Generative Sequence Reconstruction):** Connects the classifier's emission probabilities with road-network topological adjacency in a Hidden Markov Model (HMM) solved via Viterbi dynamic programming.

### Key Results
- **Route Continuity:** Jumped from **89.33% to 99.17%** (+9.84% gain) with HMM decoding.
- **Top-1 Observation Accuracy:** Increased from **79.90% to 81.41%** (+1.51% gain).
- **Valid Graph Transitions:** Increased from 17,107 to **18,991 / 19,150** (eliminating 1,884 invalid segment hops).
- **Inference Latency:** Sub-millisecond candidate inference and **< 0.98s** per full trip Viterbi decoding.

---

## 2. Repository Structure

```
PathSnap/
├── backend/
│   ├── app.py                     # FastAPI application (Laptop 1 Server)
│   └── requirements.txt           # Python dependencies
├── frontend/
│   ├── index.html                 # Evaluator-facing Dashboard UI (Laptop 2 Client)
│   ├── style.css                  # Responsive styling & contrast layout
│   └── app.js                     # Async client logic & API communication
├── src/
│   ├── data_processing.py         # Equirectangular projection & spatial KD-Tree
│   ├── eda.py                     # Structured exploratory data analysis
│   ├── features.py                # Multi-point candidate extraction & 11 features
│   ├── train_models.py            # Training & cross-validation of 5 models
│   ├── bayesian_opt.py            # Optuna Bayesian hyperparameter optimisation
│   ├── hmm_viterbi.py             # HMM formulation & Viterbi dynamic programming
│   ├── map_viz.py                 # Folium interactive HTML map generator
│   └── generate_static_maps.py    # Static trajectory plot generator
├── team/
│   ├── commit_p1.sh / .bat        # P1 commit script (Raghav Budur)
│   ├── commit_p2.sh / .bat        # P2 commit script (Pabbisetty Harshitha Sai)
│   ├── commit_p3.sh / .bat        # P3 commit script (Palleneni Sloka)
│   ├── commit_p4.sh / .bat        # P4 commit script (Parth Pandurang Kale)
│   └── README_TEAM_COMMITS.md     # Step-by-step instructions for team members
├── models/                        # Serialised joblib model artifacts & scalers
├── results/                       # CSV benchmarks, logs, and experiment outputs
├── plots/                         # Evaluation charts, confusion matrices, and maps
├── folium_maps/                   # Interactive Leaflet/Folium HTML maps
├── Technical_Report_PT045.md      # Final Consolidated Technical Report (Deliverable 2)
├── Project_Documentation.md       # Technical decisions, hyperparams & HMM connection
├── TWO_LAPTOP_DEMO_GUIDE.md       # Demonstration plan on 2 laptops vs 1 laptop test
├── Dockerfile                     # Docker deployment specification
├── docker-compose.yml             # Container orchestration
└── run_all.py                     # Master execution runner
```

---

## 3. Installation & Quick Start

### Prerequisites
Python 3.10 or 3.11 with pip.

```bash
# Install dependencies
pip install -r backend/requirements.txt
```

### Running on ONE Laptop (For Testing):
```bash
python run_all.py --server
```
Open your browser at `http://localhost:8000/ui`.

---

## 4. Two-Laptop Deployment Architecture (Evaluation Mode)

The system strictly follows the required distributed architecture:
- **Laptop 1 (Model and API Server):**
  - Connect Laptop 1 to local Wi-Fi or mobile hotspot.
  - Find IPv4 address (`ipconfig` on Windows or `ifconfig` on Linux/Mac).
  - Start API server:
    ```bash
    python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
    ```
  - Keep this terminal open in view of the evaluator to display real-time inference request logs.
- **Laptop 2 (Evaluator UI Client):**
  - Open web browser on Laptop 2 and navigate to:
    ```
    http://<LAPTOP_1_IP>:8000/ui
    ```
  - The UI connects to Laptop 1, displays all 10 reserved trips, and allows the evaluator to trigger live map-matching and inspect server-verified accuracy metrics.

*(See [TWO_LAPTOP_DEMO_GUIDE.md](TWO_LAPTOP_DEMO_GUIDE.md) for full details)*

---

## 5. Model Comparison Summary Table

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test F1 | Log Loss | Inference Time |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear / Probabilistic | C=1.0, penalty=l2, balanced | 0.8428 | 79.00% | 87.89% | 0.7675 | 0.2685 | 0.000 ms |
| **K-Nearest Neighbors** | Instance-based | n_neighbors=9, distance-weighted | 0.8595 | 76.78% | 86.78% | 0.8521 | 0.2693 | 0.074 ms |
| **Decision Tree** | Tree-based | max_depth=12, min_samples_split=20 | 0.8742 | 79.08% | 87.83% | 0.8679 | 0.1734 | 0.000 ms |
| **Random Forest (Chosen)**| Ensemble | n_estimators=100, max_depth=14 | 0.8806 | **79.81%** | **89.11%** | **0.8746** | 0.1414 | 0.001 ms |
| **HistGradientBoosting** | Boosting | lr=0.08, max_iter=120, max_depth=8 | **0.8814** | 79.68% | 88.97% | 0.8729 | **0.1409** | 0.002 ms |

---

## 6. Team Commit Workflow

Each team member commits their assigned files independently using isolated scripts in `team/`:

```bash
# Person 1 (Raghav Budur)
bash team/commit_p1.sh "P1: Data processing and EDA updates"

# Person 2 (Pabbisetty Harshitha Sai)
bash team/commit_p2.sh "P2: Feature extraction and model benchmarking"

# Person 3 (Palleneni Sloka)
bash team/commit_p3.sh "P3: Optuna Bayes opt and HMM Viterbi decoder"

# Person 4 (Parth Pandurang Kale)
bash team/commit_p4.sh "P4: FastAPI backend and Evaluator UI"

# Auto-commit watch mode (every 10 minutes):
bash team/commit_p1.sh --watch
```

*(See [team/README_TEAM_COMMITS.md](team/README_TEAM_COMMITS.md) for full details)*
