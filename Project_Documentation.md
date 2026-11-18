# PathSnap: Technical Project Documentation & Design Decisions
**Team Number:** PT-045  
**Institution:** PES University | UE24CS352A Machine Learning  
**Team Members:**
- **P1:** Raghav Budur (PES1UG23AM225) - Data Processing, Metric Projection & Spatial Indexing
- **P2:** Pabbisetty Harshitha Sai (PES1UG24AM182) - Feature Engineering & 5 Machine Learning Models
- **P3:** Palleneni Sloka (PES1UG24AM183) - Bayesian Hyperparameter Optimisation & HMM Viterbi Decoder
- **P4:** Parth Pandurang Kale (PES1UG24AM184) - Folium Visualisations, FastAPI Backend & Evaluator Dashboard

---

## 1. Problem Statement and Architectural Overview

Urban vehicle tracking via GPS suffers from significant signal distortion caused by high-rise buildings, multipath reflections, sensor noise, and irregular sampling intervals. In consequence, raw GPS points frequently appear on parallel adjacent streets, jump across city blocks, or lie off the actual road network.

**PathSnap** solves this through a distributed two-stage machine learning system:
1. **Stage 1 (Observation-Level Candidate Classification):** A supervised machine learning model scores nearby candidate road segments for each GPS observation using spatial, kinematic, and topological features.
2. **Stage 2 (Sequence-Level Route Reconstruction):** A Hidden Markov Model (HMM) coupled with Viterbi dynamic programming decodes the globally most coherent sequence of connected road segments, resolving ambiguous parallel streets and eliminating teleportation artifacts.

```
+---------------------------------------------------------------------------------------------------+
|                                     PATHSNAP SYSTEM PIPELINE                                      |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [Raw GPS Observations]  +  [Road Network Graph]                                                 |
|            |                        |                                                             |
|            v                        v                                                             |
|  +--------------------+   +------------------------------------+                                  |
|  | Equirectangular    |   | Multi-Point Spatial KD-Tree Index  |                                  |
|  | Metric Projection  |   | (Start Node, End Node, Midpoint)   |                                  |
|  +--------------------+   +------------------------------------+                                  |
|            \                        /                                                             |
|             \                      /                                                              |
|              v                    v                                                               |
|        +-----------------------------------+                                                      |
|        | Spatial Candidate Generation      | (Recall > 95% within search buffer)                  |
|        +-----------------------------------+                                                      |
|                          |                                                                        |
|                          v                                                                        |
|        +-----------------------------------+                                                      |
|        | 11-Dimensional Feature Vector     | (Distance, Heading Diff, Projection t, Speed, etc.) |
|        +-----------------------------------+                                                      |
|                          |                                                                        |
|                          v                                                                        |
|        +-----------------------------------+                                                      |
|        | Tuned Random Forest Classifier    | (Bayesian TPE Optimised)                             |
|        | Predicts Candidate Probability:   |                                                      |
|        |   p_clf(c_{t, i})                 |                                                      |
|        +-----------------------------------+                                                      |
|                          |                                                                        |
|                          v                                                                        |
|        +-------------------------------------------------------------+                            |
|        | Hidden Markov Model (HMM) Trellis                           |                            |
|        | - Emission Probabilities: log P(z_t | c) ~ alpha*log(p_clf) |                            |
|        | - Transition Probabilities: A_ij ~ Road Network Adjacency   |                            |
|        | - GPS Gap & Disconnect Handling: Smooth penalty             |                            |
|        +-------------------------------------------------------------+                            |
|                          |                                                                        |
|                          v                                                                        |
|        +-----------------------------------+                                                      |
|        | Viterbi Dynamic Programming       | (Global Trellis Backtracking)                        |
|        +-----------------------------------+                                                      |
|                          |                                                                        |
|                          v                                                                        |
|        +-----------------------------------+                                                      |
|        | Reconstructed Route & Metrics     | (Continuity: 99.17% | Top-1 Accuracy: 81.41%)        |
|        +-----------------------------------+                                                      |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Technical Decisions & Mathematical Rationale

### 2.1 Coordinate System & Metric Projection
- **Problem:** Raw coordinates are given in WGS84 angular degrees $(\text{lon}, \text{lat})$. At the latitude of the dataset ($\approx 49.8^\circ \text{N}$), one degree of longitude corresponds to $\approx 71.7 \text{ km}$, while one degree of latitude corresponds to $\approx 111.2 \text{ km}$. Calculating Euclidean distance directly in degrees treats longitude and latitude symmetrically, introducing severe aspect ratio distortions ($\approx 35\%$ error in distance and angles).
- **Decision:** All coordinates are projected to a local metric Euclidean plane using an equirectangular projection centered at the trajectory mean $(\text{lon}_0, \text{lat}_0)$:
  $$x = R \cdot (\text{lon} - \text{lon}_0) \cdot \frac{\pi}{180} \cdot \cos\left(\text{lat}_0 \cdot \frac{\pi}{180}\right)$$
  $$y = R \cdot (\text{lat} - \text{lat}_0) \cdot \frac{\pi}{180}$$
  where $R = 6,371,000\text{ m}$. Over a localized trajectory bounding box ($< 15\text{ km}$), projection distortion is $< 0.01\%$, providing millimeter-level accuracy for orthogonal distance calculations.

### 2.2 Spatial Indexing & Candidate Generation
- **Problem:** A typical trip sub-network contains between 15,000 and 165,000 directed arcs. Naively searching all arcs for each of 2,000 GPS points requires $\approx 3.3 \times 10^8$ point-to-line calculations per trip, taking $> 30$ seconds.
- **Decision:** A multi-point spatial KD-Tree (`scipy.spatial.cKDTree`) indexes three points for every arc: start node $u$, end node $v$, and midpoint $(u+v)/2$. Indexing multiple points prevents candidate miss rates on long road segments where the midpoint may be distant while the segment edge passes directly beneath the vehicle.
- **Result:** Querying 15-25 nearest candidate segments executes in **0.003 ms per point**, reducing candidate extraction for an entire 1,500-point trip to under **0.5 seconds** while maintaining $> 95\%$ candidate recall.

### 2.3 Feature Engineering Strategy
For every candidate segment $c = (u, v)$ relative to GPS observation $z_t$, we engineer 11 features:
1. `dist_perp`: Orthogonal Euclidean distance from GPS point to the line segment. Primary indicator of spatial proximity.
2. `dist_diff_min`: Margin between this candidate's distance and the closest candidate ($d_i - d_{\min}$). High positive values signify uncompetitive candidates.
3. `dist_rank`: Discrete ranking (1 = nearest, 2 = second nearest). Invariant to GPS noise magnitude.
4. `dist_u`, `dist_v`: Euclidean distances to the segment's junction endpoints.
5. `seg_len`: Road segment length. Helps normalize projection and distinguishes highways from dense urban alleys.
6. `proj_t`: Normalized projection scalar $t \in [0, 1]$. Segments where $0 < t < 1$ represent true parallel travel; values clamped at 0 or 1 indicate the vehicle is approaching or departing a junction.
7. `heading_diff`: Absolute angular difference between vehicle travel vector $(z_t - z_{t-1})$ and arc direction $(v - u)$, mapped to $[0, \pi]$ radians. Critical for disambiguating opposing lanes on divided highways.
8. `speed`: Instantaneous GPS speed.
9. `u_degree`, `v_degree`: Out-degree and in-degree of junction nodes. Segments connected to major intersections have higher transition entropy.

### 2.4 Leakage Prevention with Group-Aware Cross-Validation
- **Problem:** GPS observations from the same trip exhibit strong temporal autocorrelation. If random K-Fold cross-validation is used, consecutive points from the same trip end up in both training and test folds, artificially inflating accuracy scores (data leakage).
- **Decision:** All cross-validation was performed using `GroupKFold(n_splits=5)` grouped strictly by `trip_id`. Every validation fold evaluated entirely unseen trips, ensuring reported metrics represent genuine real-world generalization.

---

## 3. Algorithm Families and Experimental Comparison

We evaluated five distinct algorithms across five families under identical group-aware cross-validation and student-testing splits:

1. **Logistic Regression (Linear & Probabilistic):**
   - *Strengths:* Ultra-fast inference, convex loss surface.
   - *Limitations:* Unable to model non-linear boundaries (e.g., combinations of low heading diff but high distance).
   - *CV Macro F1:* 0.8428 | *Student-Test Top-1:* 79.00% | *Log Loss:* 0.2685

2. **K-Nearest Neighbors (Instance-Based):**
   - *Strengths:* Non-parametric, models arbitrary manifold shapes.
   - *Limitations:* $O(N \cdot d)$ query cost makes it computationally heavy (0.074 ms/sample); sensitive to local density variations.
   - *CV Macro F1:* 0.8595 | *Student-Test Top-1:* 76.78% | *Log Loss:* 0.2693

3. **Decision Tree (Tree-Based):**
   - *Strengths:* Non-linear orthogonal splitting, zero scaling required.
   - *Limitations:* Prone to variance and overfitting on noisy GPS outliers.
   - *CV Macro F1:* 0.8742 | *Student-Test Top-1:* 79.08% | *Log Loss:* 0.1734

4. **Random Forest (Ensemble) - CHOSEN PRIMARY MODEL:**
   - *Strengths:* Bagging reduces variance; random feature sub-spacing decorrelates individual trees; highly resilient to GPS outliers; produces well-calibrated probabilities.
   - *CV Macro F1:* 0.8806 | *Student-Test Top-1:* **79.81%** | *Route Continuity:* **89.11%** | *Log Loss:* 0.1414

5. **HistGradientBoosting (Boosting):**
   - *Strengths:* Histogram binning provides rapid training; sequential gradient descent directly optimizes cross-entropy log loss.
   - *CV Macro F1:* **0.8814** | *Student-Test Top-1:* 79.68% | *Route Continuity:* 88.97% | *Log Loss:* **0.1409**

*Selection Decision:* **Random Forest** was selected as the final candidate model because it achieved the highest observation Top-1 accuracy (79.81%) and highest baseline route continuity (89.11%).

---

## 4. Bayesian Hyperparameter Optimisation

We tuned the Random Forest using **Optuna** with the **Tree-structured Parzen Estimator (TPE)** algorithm:
- **Search Space:**
  - `n_estimators`: [60, 160] (step 20)
  - `max_depth`: [10, 20]
  - `min_samples_split`: [4, 16]
  - `min_samples_leaf`: [1, 6]
  - `max_features`: `['sqrt', 'log2', 0.6]`
- **Objective:** 3-fold GroupKFold Macro F1 on the training partition.
- **Trial History:** 20 trials executed. Trial #12 yielded the optimal configuration:
  `{'n_estimators': 120, 'max_depth': 10, 'min_samples_split': 11, 'min_samples_leaf': 3, 'max_features': 0.6}`
- **Gains:**
  - Cross-Validation Macro F1: **0.8806 → 0.8814** (Trial peak: 0.8833)
  - Student-Test Top-1 Accuracy: **79.81% → 79.90%**
  - Log Loss: **0.1414 → 0.1409** (improved calibration)

---

## 5. Connecting the Best Model to HMM Viterbi Route Reconstruction

### 5.1 The Fundamental Challenge
Pointwise classifiers predict road segments independently for each timestamp $t$. Even with an 80% accuracy rate, the 20% erroneous predictions cause the vehicle to frequently "teleport" onto parallel access roads, across river bridges, or into opposing lanes, yielding fragmented routes with low continuity ($\approx 89\%$).

### 5.2 Mathematical Formulation of HMM for Map Matching

```
   Observation t-1                   Observation t
      [ z_{t-1} ]                      [ z_t ]
           ^                               ^
           | Emission                      | Emission
           |                               |
    [ State c_{t-1, i} ] ---- Transition ---> [ State c_{t, j} ]
```

1. **Hidden States:** The set of candidate road segments $\mathcal{C}_t = \{c_{t, 1}, c_{t, 2}, \dots, c_{t, K}\}$ available at time step $t$.
2. **Connecting the Classifier to Emission Probabilities:**
   The emission probability $P(z_t \mid c_{t, i})$ models how likely the GPS observation $z_t$ is given that the vehicle is on segment $c_{t, i}$.
   We directly feed the Bayesian-tuned Random Forest's calibrated probability $p_{\text{clf}}(c_{t, i}) = P(y=1 \mid \mathbf{x}_{t, i})$ into the emission model:
   $$\log P(z_t \mid c_{t, i}) = \alpha \cdot \log(p_{\text{clf}}(c_{t, i}) + \epsilon) - (1 - \alpha) \frac{d_{\perp}(z_t, c_{t, i})^2}{2\sigma_z^2} - 0.5 \cdot (\text{rank} - 1)$$
   where:
   - $\alpha = 0.65$ balances the learned multidimensional classifier features (speed, heading, projection, node degrees) with physical orthogonal distance.
   - $\sigma_z = 10.0\text{ m}$ is the standard deviation of typical GPS noise.
3. **Transition Probabilities from Road Network Topology:**
   The transition probability $P(c_{t, j} \mid c_{t-1, i})$ models the physical feasibility of moving from candidate $c_{t-1, i} = (u_1, v_1)$ to $c_{t, j} = (u_2, v_2)$:
   - **Segment Continuation ($c_{t-1, i} == c_{t, j}$):** Vehicle stays on the same road segment:
     $$\log P_{\text{trans}} = 0.0$$
   - **Direct Network Connection ($v_1 == u_2$):** Road segment $c_{t-1, i}$ directly terminates at the start node of $c_{t, j}$:
     $$\log P_{\text{trans}} = -0.5$$
   - **Disconnected Segments or GPS Dropouts ($\Delta t > 10\text{s}$):**
     If the road segments are not directly adjacent, instead of setting probability to 0 (which would terminate the trellis), a smooth gap penalty is applied:
     $$\log P_{\text{trans}} = -12.0$$
     This allows Viterbi to bridge over tunnels, signal dropouts, and occasional missing candidate intervals without crashing.
4. **Viterbi Dynamic Programming Recurrence:**
   The trellis is solved in log-space:
   - Initialization:
     $$V_0(i) = \log P(z_0 \mid c_{0, i})$$
   - Recurrence for $t = 1, \dots, T-1$:
     $$V_t(j) = \max_{1 \le i \le K_{t-1}} \left[ V_{t-1}(i) + \log P(c_{t, j} \mid c_{t-1, i}) \right] + \log P(z_t \mid c_{t, j})$$
     $$BP_t(j) = \arg\max_{1 \le i \le K_{t-1}} \left[ V_{t-1}(i) + \log P(c_{t, j} \mid c_{t-1, i}) \right]$$
   - Backtracking:
     $$S_T^* = \arg\max_j V_T(j)$$
     $$S_{t-1}^* = BP_t(S_t^*)$$

### 5.3 Quantitative Proof of Improvement

| Metric | Classifier-Only (Tuned RF) | Classifier + HMM Viterbi | Absolute Change |
|---|---|---|---|
| **Route Continuity** | 89.33% | **99.17%** | **+9.84%** |
| **Observation Top-1 Accuracy** | 79.90% | **81.41%** | **+1.51%** |
| **Valid Graph Transitions** | 17,107 / 19,150 | **18,991 / 19,150** | **+1,884 transitions** |
| **Decoding Speed** | 0.001 ms / point | **0.05 to 0.98 s / trip** | Real-time capable |

*Takeaway:* The sequential HMM decoder almost completely eliminates disconnected road hops (continuity reaching 99.17%) while simultaneously improving pointwise accuracy by 1.51%, validating the mathematical synergy between discriminative candidate classifiers and generative sequence models.
