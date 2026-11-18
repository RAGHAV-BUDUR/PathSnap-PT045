# Team Commit Instructions - Team PT-045

This folder contains isolated commit scripts for each team member to independently commit their designated components with their official git identity.

---

## Team Roster & Module Allocation

| Member Code | Name | SRN | Assigned Project Modules & Files |
|---|---|---|---|
| **P1** | **Raghav Budur** | PES1UG23AM225 | Data Processing, Metric Projection, Spatial Indexing (`cKDTree`), and EDA (`src/data_processing.py`, `src/eda.py`) |
| **P2** | **Pabbisetty Harshitha Sai** | PES1UG24AM182 | Feature Engineering, Candidate Extraction, 5 Models Training & CV Benchmarks (`src/features.py`, `src/train_models.py`, `models/*.joblib`) |
| **P3** | **Palleneni Sloka** | PES1UG24AM183 | Bayesian Hyperparameter Optimisation (`src/bayesian_opt.py`), HMM Formulation & Viterbi Decoder (`src/hmm_viterbi.py`) |
| **P4** | **Parth Pandurang Kale** | PES1UG24AM184 | Folium Interactive Maps (`src/map_viz.py`), FastAPI Backend (`backend/app.py`), and Evaluator Dashboard (`frontend/`) |

---

## 1. Quick One-Time Git Setup with Personal Access Token (PAT)

If you haven't set up your remote repository credentials yet:

```bash
# Clone or configure the origin repository with your GitHub Token in one go:
git remote set-url origin https://<YOUR_GITHUB_TOKEN>@github.com/<USERNAME>/<REPO_NAME>.git
```

---

## 2. How to Execute Your Commit

### For Linux / macOS / Git Bash Users:

Run from the root of the repository:

```bash
# Person 1 (Raghav Budur)
bash team/commit_p1.sh "P1: Complete spatial KD-tree and dataset EDA analysis"

# Person 2 (Pabbisetty Harshitha Sai)
bash team/commit_p2.sh "P2: Train 5 machine learning models and record cross-validation table"

# Person 3 (Palleneni Sloka)
bash team/commit_p3.sh "P3: Implement Optuna Bayesian optimization and HMM Viterbi route decoder"

# Person 4 (Parth Pandurang Kale)
bash team/commit_p4.sh "P4: Deploy FastAPI backend and evaluator UI dashboard"
```

### Auto-Commit Watch Mode (Every 10 Minutes):
To keep committing your assigned components automatically in the background:

```bash
bash team/commit_p1.sh --watch   # P1 - checks & commits every 10 minutes (Ctrl+C to stop)
bash team/commit_p2.sh --watch   # P2
bash team/commit_p3.sh --watch   # P3
bash team/commit_p4.sh --watch   # P4
```

---

### For Windows Command Prompt (cmd.exe) / PowerShell Users:

```cmd
team\commit_p1.bat "Your commit message here"
team\commit_p2.bat "Your commit message here"
team\commit_p3.bat "Your commit message here"
team\commit_p4.bat "Your commit message here"
```

Each script automatically stages only that member's assigned folders, sets the commit author, and creates the commit.
