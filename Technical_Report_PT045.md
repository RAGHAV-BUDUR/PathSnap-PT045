# PATHSNAP: TECHNICAL REPORT
## Distributed Map-Matching & Sequential Route Reconstruction
**Course:** UE24CS352A - Machine Learning | PES University RR Campus  
**Submission:** Deliverable 2 Consolidated Technical Report  

---

### Team Information
- **Team Number:** PT-045
- **Student 1 (P1):** RAGHAV BUDUR | SRN: PES1UG23AM225
- **Student 2 (P2):** PABBISETTY HARSHITHA SAI | SRN: PES1UG24AM182
- **Student 3 (P3):** PALLENENI SLOKA | SRN: PES1UG24AM183
- **Student 4 (P4):** PARTH PANDURANG KALE | SRN: PES1UG24AM184

---

## 2. Dataset Understanding and Exploratory Data Analysis

### 2.1 Dataset Partitioning & Leakage Prevention
The experimental dataset consists of two partitions accessible during development:
1. **Training Partition (80 trips):** Used exclusively for data exploration, metric projection, spatial candidate indexing, feature extraction, 5-fold group-aware cross-validation, and Bayesian hyperparameter optimisation.
2. **Student-Testing Partition (10 trips):** Reserved exclusively for development-stage model evaluation and comparison between classifier-only predictions and HMM/Viterbi route reconstruction.
3. **Evaluator-Testing Partition (10 trips):** Strictly untouched during training and development; reserved for live evaluation on Laptop 1 and Laptop 2.

To guarantee zero data leakage, all cross-validation procedures utilised `GroupKFold` grouped strictly by `trip_id`. No GPS observations from the same trip were ever shared across training and validation folds.

### 2.2 Data Preprocessing and Coordinate Projection
Raw GPS observations are captured in angular WGS84 geographic coordinates (longitude, latitude in degrees) alongside timestamps (seconds). Road networks are defined by `.nodes` (coordinates) and `.arcs` (directed edges from `u` to `v`).

Because Euclidean distance in raw angular degrees produces severe spatial distortion at non-equatorial latitudes (~49.8° N in this dataset), all coordinates were projected into a local Euclidean metric Cartesian plane centered at the trajectory mean coordinates $(\text{lon}_0, \text{lat}_0)$:
$$x = R_{\text{earth}} \cdot (\text{lon} - \text{lon}_0) \cdot \frac{\pi}{180} \cdot \cos\left(\text{lat}_0 \cdot \frac{\pi}{180}\right)$$
$$y = R_{\text{earth}} \cdot (\text{lat} - \text{lat}_0) \cdot \frac{\pi}{180}$$
where $R_{\text{earth}} = 6,371,000 \text{ m}$.

To accelerate candidate road segment querying from tens of thousands of network arcs to sub-millisecond lookups, a multi-point spatial KD-Tree (`scipy.spatial.cKDTree`) was built per trip. For each arc, start node $u$, end node $v$, and midpoint $(u+v)/2$ were indexed. This eliminated candidate miss rate on long segments, achieving a candidate recall of over 95%.

### 2.3 Exploratory Data Analysis Observations
Structured exploratory analysis of all 90 accessible trips revealed the following key patterns:
- **Missing Values:** A complete scan of all `.track`, `.nodes`, `.arcs`, and `.route` files showed zero null or missing values across all partitions.
- **Trajectory Lengths:**
  - Training Partition: 80 trips, observations ranging from 361 to 7,243 points (mean: 2,605.2 observations per trip).
  - Student Testing Partition: 10 trips, observations ranging from 262 to 3,958 points (mean: 1,916.0 observations per trip).
- **Sampling Frequency Regularity:** 93.83% of consecutive GPS timestamps in the training set and 98.20% in the student testing set are spaced exactly 1.0 second apart (nominal 1 Hz sampling frequency).
- **GPS Irregularities and Gaps:** 42 out of 80 training trips exhibited temporal sampling gaps exceeding 10.0 seconds (e.g., maximum gaps of 16s, 51s, 160s, and 301s). These gaps correspond to urban canyon signal blockages, tunnels, or sensor dropouts, reinforcing the necessity of an HMM decoder with topological gap handling.
- **Road Network Scale:** Sub-graph bounding boxes varied from 2,300 nodes and 4,500 arcs for small local trips up to 154,000 nodes and 307,392 arcs for extensive metropolitan trips.

### 2.4 Feature Engineering
For each GPS observation $z_t = (p_x, p_y)$, the spatial KD-Tree retrieved candidate road segments. For each candidate segment $c = (u, v)$, a 11-dimensional feature vector was extracted:
1. `dist_perp`: Orthogonal Euclidean distance from the GPS point to the line segment in meters.
2. `dist_diff_min`: Difference between the candidate's perpendicular distance and the closest candidate ($d_i - d_{\min}$).
3. `dist_rank`: Discrete ranking of spatial proximity (1 = closest, 2 = second closest, etc.).
4. `dist_u`: Euclidean distance to start node $u$ in meters.
5. `dist_v`: Euclidean distance to end node $v$ in meters.
6. `seg_len`: Length of the candidate road segment $\sqrt{(v_x - u_x)^2 + (v_y - u_y)^2}$ in meters.
7. `proj_t`: Normalized projection scalar $t \in [0, 1]$ indicating where along the segment the point projects.
8. `heading_diff`: Absolute angular difference between the vehicle's instantaneous movement vector $(z_t - z_{t-1})$ and the road segment orientation vector $(v - u)$, bounded in $[0, \pi]$ radians.
9. `speed`: Instantaneous vehicle speed calculated as $|z_t - z_{t-1}| / \Delta t$ in m/s.
10. `u_degree`: Out-degree of start node $u$ in the road network graph.
11. `v_degree`: In-degree of end node $v$ in the road network graph.

---

## 3. Model Experimentation and Comparison

### 3.1 Consolidated Model Comparison Table
A minimum of five distinct models across five algorithm families were trained and evaluated under identical conditions:

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route Continuity | Student-Test Macro F1-score | Log Loss | Inference Time |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear and probabilistic | C=1.0, penalty=l2, max_iter=300, class_weight=balanced | 0.8428 | 79.00% | 87.89% | 0.7675 | 0.2685 | 0.000 ms |
| **K-Nearest Neighbors** | Instance-based | n_neighbors=9, weights=distance, metric=euclidean | 0.8595 | 76.78% | 86.78% | 0.8521 | 0.2693 | 0.074 ms |
| **Decision Tree** | Tree-based | max_depth=12, min_samples_split=20, criterion=gini | 0.8742 | 79.08% | 87.83% | 0.8679 | 0.1734 | 0.000 ms |
| **Random Forest** | Ensemble | n_estimators=100, max_depth=14, min_samples_split=10, max_features=sqrt | 0.8806 | 79.81% | 89.11% | 0.8746 | 0.1414 | 0.001 ms |
| **HistGradientBoosting** | Boosting | learning_rate=0.08, max_iter=120, max_depth=8, l2_regularization=1.0 | 0.8814 | 79.68% | 88.97% | 0.8729 | 0.1409 | 0.002 ms |

*Note: All inference times are reported per candidate observation.*

---

### 3.2 Compulsory Questions for Every Model

#### Model 1: Logistic Regression
- **Algorithm Family:** Linear and probabilistic
- **A. Model Understanding:**
  - *Suitability:* Provides a rapid baseline with a linear hyperplane and direct probabilistic interpretation via the sigmoid link function.
  - *Strengths & Limitations:* Extremely fast inference (sub-microsecond) and convex convergence. However, it cannot capture nonlinear interaction terms (e.g., between speed and heading divergence at intersections) without manual polynomial expansion.
- **B. Training and Configuration:**
  - *Hyperparameters:* Regularisation inverse strength $C = 1.0$, $L_2$ penalty, `max_iter=300`, `class_weight='balanced'`. Standardised feature scaling applied.
  - *Objective:* Binary cross-entropy loss with $L_2$ weight penalty.
- **C. Experimental Results:**
  - *Performance on Student-Testing:* Macro F1 of 0.7675, Observation Top-1 Accuracy of 79.00%, Route Continuity of 87.89%, Log Loss of 0.2685.
  - *Fitting Diagnosis:* Exhibited mild underfitting due to the strict linearity of its decision boundary.

#### Model 2: K-Nearest Neighbors (KNN)
- **Algorithm Family:** Instance-based
- **A. Model Understanding:**
  - *Suitability:* Non-parametric approach that matches query candidate feature vectors to similar historical spatial instances in Euclidean feature space.
  - *Strengths & Limitations:* Naturally captures non-linear manifolds. However, inference complexity scales linearly with training set size $O(N \cdot d)$, making it the slowest model during inference (0.074 ms/sample).
- **B. Training and Configuration:**
  - *Hyperparameters:* `n_neighbors=9`, distance-weighted inverse Euclidean distance metric. Standardised feature scaling applied.
  - *Objective:* Nearest-neighbour distance weighting vote.
- **C. Experimental Results:**
  - *Performance on Student-Testing:* Macro F1 of 0.8521, Observation Top-1 Accuracy of 76.78%, Route Continuity of 86.78%, Log Loss of 0.2693.
  - *Fitting Diagnosis:* Balanced fit but lower Top-1 accuracy due to feature-space density variations across urban and rural trip segments.

#### Model 3: Decision Tree Classifier
- **Algorithm Family:** Tree-based
- **A. Model Understanding:**
  - *Suitability:* Hierarchical orthogonal splits capture non-linear thresholds (e.g., "if `dist_perp` < 8m AND `heading_diff` < 0.3 rad, candidate is valid").
  - *Strengths & Limitations:* Highly interpretable, zero feature-scaling requirement. However, single trees are inherently prone to high variance and overfitting on noisy GPS outliers.
- **B. Training and Configuration:**
  - *Hyperparameters:* `max_depth=12`, `min_samples_split=20`, `criterion='gini'`.
  - *Objective:* Gini impurity reduction.
- **C. Experimental Results:**
  - *Performance on Student-Testing:* Macro F1 of 0.8679, Observation Top-1 Accuracy of 79.08%, Route Continuity of 87.83%, Log Loss of 0.1734.
  - *Fitting Diagnosis:* Mild overfitting observed between unconstrained depth and pruned depth; controlled via `min_samples_split=20`.

#### Model 4: Random Forest Classifier (Chosen Primary Model)
- **Algorithm Family:** Ensemble (Bagging)
- **A. Model Understanding:**
  - *Suitability:* Aggregates de-correlated decision trees trained on bootstrap samples with random feature sub-spacing, reducing variance while maintaining low bias.
  - *Strengths & Limitations:* Exceptional generalisation on tabular spatial features, highly stable probability estimates, robust against GPS noise and outliers. Computational training time is slightly higher than a single tree, but inference remains under 0.001 ms.
- **B. Training and Configuration:**
  - *Hyperparameters:* `n_estimators=100`, `max_depth=14`, `min_samples_split=10`, `max_features='sqrt'`.
  - *Objective:* Ensemble Gini impurity minimisation across bagged estimators.
- **C. Experimental Results:**
  - *Performance on Student-Testing:* Macro F1 of 0.8746, Observation Top-1 Accuracy of 79.81%, Route Continuity of 89.11%, Log Loss of 0.1414.
  - *Fitting Diagnosis:* Optimal generalisation balance; CV Macro F1 (0.8806) closely matched Student-Test Macro F1 (0.8746) within 0.006 margin, demonstrating absence of overfitting.

#### Model 5: HistGradientBoosting Classifier
- **Algorithm Family:** Boosting
- **A. Model Understanding:**
  - *Suitability:* Sequentially builds additive decision trees to minimise pseudo-residuals, using histogram-based binning of continuous features.
  - *Strengths & Limitations:* Extremely fast training and lowest cross-entropy log loss (0.1409). Slightly more sensitive to learning rate and depth hyperparameters than Random Forest.
- **B. Training and Configuration:**
  - *Hyperparameters:* `learning_rate=0.08`, `max_iter=120`, `max_depth=8`, `l2_regularization=1.0`.
  - *Objective:* Binary deviance (log-loss) minimisation.
- **C. Experimental Results:**
  - *Performance on Student-Testing:* Macro F1 of 0.8729, Observation Top-1 Accuracy of 79.68%, Route Continuity of 88.97%, Log Loss of 0.1409.
  - *Fitting Diagnosis:* Healthy regularisation; training deviance decreased monotonically without validation degradation.

---

## 4. Bayesian Hyperparameter Optimisation

### 4.1 Search Configuration
Random Forest was selected as the candidate model for Bayesian hyperparameter optimisation due to its highest observation-level Top-1 accuracy and superior topological continuity.

- **Optimisation Framework:** Optuna with Tree-structured Parzen Estimator (TPE) sampler.
- **Search Space:**
  - `n_estimators`: Integer uniform in range [60, 160], step = 20
  - `max_depth`: Integer uniform in range [10, 20]
  - `min_samples_split`: Integer uniform in range [4, 16]
  - `min_samples_leaf`: Integer uniform in range [1, 6]
  - `max_features`: Categorical choice in `['sqrt', 'log2', 0.6]`
- **Objective Function:** Group-aware 3-fold cross-validation Macro F1-score across the training trips.
- **Number of Trials:** 20 complete trials executed.

### 4.2 Optimisation Results
- **Best Trial:** Trial #12 achieved the highest CV Macro F1-score of **0.8833**.
- **Best Parameter Configuration:**
  - `n_estimators`: 120
  - `max_depth`: 10
  - `min_samples_split`: 11
  - `min_samples_leaf`: 3
  - `max_features`: 0.6

### 4.3 Comparison: Untuned vs. Tuned Model

| Model Configuration | CV Macro F1-score | Student-testing Top-1 Accuracy | Student-testing Macro F1-score | Log Loss | Route Continuity | Inference Time |
|---|---|---|---|---|---|---|
| **Untuned Random Forest** | 0.8806 | 79.81% | 0.8746 | 0.1414 | 89.11% | 0.001 ms |
| **Tuned Random Forest (Bayes Opt)** | **0.8814** | **79.90%** | **0.8740** | **0.1409** | **88.98%** | **0.001 ms** |

*Outcome:* The Bayesian tuned model achieved a slight increase in Observation Top-1 Accuracy (79.81% → 79.90%), improved cross-validation F1-score (0.8806 → 0.8814), and lower log loss (0.1414 → 0.1409).

---

## 5. HMM and Viterbi Route Reconstruction

### 5.1 Formulation
Independent candidate classification produces disconnected routes when consecutive GPS observations choose parallel segments or opposing lane directions. A Hidden Markov Model (HMM) resolves this by finding the sequence of road segments that jointly maximises observation emissions and topological network transitions.

1. **Hidden States:** The discrete hidden states at time step $t$ are the spatial candidate road segments $c_{t, i} \in \mathcal{C}_t$ in the road network.
2. **Emission Probabilities:** Calculated by combining the calibrated probability output of the tuned machine learning model with the physical orthogonal distance:
   $$\log P_{\text{emission}}(z_t \mid c_{t, i}) = \alpha \log(p_{\text{clf}}(c_{t, i}) + \epsilon) - (1 - \alpha) \frac{d_{\perp}(z_t, c_{t, i})^2}{2\sigma_z^2} - 0.5 \cdot (\text{rank} - 1)$$
   where $\sigma_z = 10.0 \text{ m}$ (GPS observation error standard deviation) and $\alpha = 0.65$.
3. **Transition Probabilities:** Calculated from road network connectivity and topological distance:
   - *Case 1 (Identical Segment):* If $c_{t-1, i} == c_{t, j}$, the vehicle remains on the same segment: $\log P_{\text{trans}} = 0.0$.
   - *Case 2 (Directly Connected Segment):* If $v_{i} == u_{j}$, segment $c_{t-1, i}$ directly connects to $c_{t, j}$ in the directed road graph: $\log P_{\text{trans}} = -0.5$.
   - *Case 3 (Disconnected Segment / Large GPS Gap):* If segments are disconnected or if temporal dropout occurs ($\Delta t > 10\text{s}$), a finite topological gap penalty is assigned ($\log P_{\text{trans}} = -12.0$). This ensures the Viterbi dynamic programming recurrence does not collapse and finds the most plausible continuation.
4. **Viterbi Trellis Recurrence:**
   $$V_t(j) = \max_{i} \left[ V_{t-1}(i) + \log P(c_{t, j} \mid c_{t-1, i}) \right] + \log P(z_t \mid c_{t, j})$$
   Optimal sequence decoding is performed via global backtracking from $\arg\max_j V_T(j)$.

### 5.2 Experimental Comparison: Classifier-Only vs. Classifier + HMM Viterbi

The sequential decoder was evaluated across all 10 student testing trips (19,150 total transitions):

| Pipeline Stage | Observation Top-1 Accuracy | Route Continuity | Valid Topological Transitions |
|---|---|---|---|
| **Classifier-Only (Tuned RF)** | 79.90% | 89.33% | 17,107 / 19,150 |
| **Classifier + HMM Viterbi Decoding** | **81.41%** | **99.17%** | **18,991 / 19,150** |

#### Per-Trip Breakdown Table:

| Trip ID | Observations | Classifier Top-1 Acc | HMM Viterbi Acc | Classifier Continuity | HMM Continuity | Viterbi Latency |
|---|---|---|---|---|---|---|
| `00000000` | 2,503 | 90.21% | **91.73%** | 94.40% | **99.28%** | 0.554 s |
| `00000007` | 1,354 | 91.14% | **92.98%** | 99.11% | **99.85%** | 0.267 s |
| `00000009` | 1,049 | 81.89% | **85.13%** | 83.59% | **98.66%** | 0.270 s |
| `00000040` | 2,995 | 88.81% | 88.65% | 94.59% | **99.83%** | 0.651 s |
| `00000052` | 262 | 92.75% | **93.13%** | 92.34% | **97.32%** | 0.052 s |
| `00000056` | 3,958 | 79.11% | 75.21% | 85.06% | **98.33%** | 0.984 s |
| `00000065` | 768 | 74.09% | **74.87%** | 86.70% | **98.70%** | 0.206 s |
| `00000076` | 1,486 | 77.73% | **86.41%** | 83.03% | **98.99%** | 0.374 s |
| `00000077` | 3,911 | 63.62% | **68.32%** | 88.85% | **99.90%** | 0.891 s |
| `00000091` | 874 | 81.46% | **85.01%** | 82.13% | **97.94%** | 0.177 s |

*Key Findings:*
- HMM Viterbi decoding increased overall route continuity from **89.33% to 99.17%** (+9.84%), successfully eliminating 1,884 invalid segment teleports.
- Overall Observation Top-1 Accuracy increased from **79.90% to 81.41%** (+1.51%).
- On trips with parallel roads (e.g., `00000076`), accuracy jumped by **+8.68%** (77.73% → 86.41%) because transition constraints prevented hopping between adjacent parallel streets.
- In trip `00000056`, where prolonged GPS drift deviated more than 80m from the true road for several minutes, the decoder favoured continuity over erroneous GPS readings, trading a small pointwise error for complete topological validity.

---

## 6. Geographical Visualisation Using Folium

Interactive HTML maps were generated using Folium (`src/map_viz.py`) and static trajectory diagrams exported (`plots/geographic_map_trip52.png`).

### Map Specification:
1. **Layer 1 (Road Network):** Rendered as semi-transparent slate grey vectors (`#94a3b8`, weight 2).
2. **Layer 2 (Raw GPS Observations):** Rendered as translucent blue circle markers (`#2563eb`, fill `#3b82f6`) with observation indices and timestamps in popups.
3. **Layer 3 (Ground Truth Route):** Solid emerald green polyline (`#16a34a`, weight 5) representing the true reference trajectory.
4. **Layer 4 (PathSnap Predicted Route):** Dashed crimson polyline (`#dc2626`, weight 3.5, dash 6,6) displaying the reconstructed route.
5. **Interactive Controls:** Includes a `LayerControl` toggle allowing evaluators to enable or disable individual route layers, and a fixed on-screen visual legend.

---

## 7. API, UI and Deployment

### 7.1 Two-Laptop Distributed Architecture
- **Laptop 1 (Model and API Server):**
  - Framework: FastAPI (`backend/app.py`) running with Uvicorn.
  - Endpoints:
    - `GET /`: Health check and system verification.
    - `GET /api/trips`: Lists the ten evaluation trips.
    - `GET /api/models`: Returns the benchmark table and Bayesian optimization logs.
    - `POST /api/match`: Accepts `{"trip_id": "<id>"}`, executes inference and Viterbi decoding, evaluates accuracy against server-side reference data, and returns evaluation metrics.
    - `GET /api/map/{trip_id}`: Serves the interactive Folium map.
- **Laptop 2 (Evaluator User Interface):**
  - Web client built with semantic HTML5, CSS3, and modern JavaScript (`frontend/`).
  - Displays the 10 evaluation trip options, sends asynchronous HTTP POST requests to Laptop 1, and renders live metrics and the embedded map.
  - Can be tested locally on a single machine at `http://localhost:8000/ui` or accessed across laptops over local network at `http://<LAPTOP_1_IP>:8000/ui`.

---

## 8. Plagiarism and Academic Integrity Policy Compliance
- All models, feature extractors, spatial KD-Trees, Bayesian optimization trials, and Viterbi decoders were independently designed and executed locally.
- All metrics reported in this document represent actual measured experimental outputs stored in `results/*.csv`.
- Ground-truth evaluation remains strictly server-side inside the API backend and is never exposed to the prediction model or user interface.
