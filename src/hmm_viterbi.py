"""
PathSnap - HMM and Viterbi Route Reconstruction Module
Team: PT-045 | P3: Palleneni Sloka (PES1UG24AM183)
Implements Hidden Markov Model decoding with Viterbi dynamic programming.
Combines tuned machine learning candidate probabilities (emissions) with
road network topological connectivity and distance continuity (transitions).
Handles large GPS gaps and disconnected segments gracefully.
"""

import sys, os, time, joblib
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.features import FEATURE_COLS, extract_candidates_and_features_for_trip
from src.data_processing import load_trip_raw, project_wgs84_to_metric

def build_adjacency(arcs_df):
    """
    Builds forward and backward graph adjacency maps.
    """
    out_map = {}
    u_vals = arcs_df['u'].values
    v_vals = arcs_df['v'].values
    for arc_id, (u, v) in enumerate(zip(u_vals, v_vals)):
        if u not in out_map:
            out_map[u] = []
        out_map[u].append((v, arc_id))
    return out_map, u_vals, v_vals

def viterbi_route_decode(trip_cands_df, arcs_df, sigma_z=10.0, beta=15.0, alpha=0.65, gap_penalty=-12.0):
    """
    Performs Viterbi decoding over observation candidate trellis.
    """
    obs_groups = [g for _, g in trip_cands_df.groupby('obs_idx')]
    T = len(obs_groups)
    if T == 0:
        return []

    u_vals = arcs_df['u'].values
    v_vals = arcs_df['v'].values

    # Pre-index out_edges per node: node -> set of arc_ids
    node_out_arcs = {}
    for arc_id, u in enumerate(u_vals):
        if u not in node_out_arcs:
            node_out_arcs[u] = set()
        node_out_arcs[u].add(arc_id)

    # Initialize Viterbi trellis: V[t] is array of log probs for candidates at step t
    # Backpointers: bp[t] is array of predecessor indices at step t-1
    V = []
    BP = []

    # Initial step t = 0
    g0 = obs_groups[0]
    p_clf_0 = np.clip(g0['clf_prob'].values, 1e-6, 1.0)
    d_perp_0 = g0['dist_perp'].values
    log_emit_0 = alpha * np.log(p_clf_0) - (1.0 - alpha) * (d_perp_0**2 / (2.0 * sigma_z**2))
    # Rank bias
    rank_bias_0 = -0.5 * (g0['dist_rank'].values - 1)
    v0 = log_emit_0 + rank_bias_0
    V.append(v0)
    BP.append(np.zeros(len(v0), dtype=int))

    # Recurrence steps t = 1 ... T-1
    for t in range(1, T):
        gt = obs_groups[t]
        cands_t = gt['arc_id'].values
        p_clf_t = np.clip(gt['clf_prob'].values, 1e-6, 1.0)
        d_perp_t = gt['dist_perp'].values
        ranks_t = gt['dist_rank'].values

        log_emit_t = alpha * np.log(p_clf_t) - (1.0 - alpha) * (d_perp_t**2 / (2.0 * sigma_z**2)) - 0.5 * (ranks_t - 1)

        prev_v = V[t - 1]
        cands_prev = obs_groups[t - 1]['arc_id'].values
        K_prev = len(cands_prev)
        K_curr = len(cands_t)

        v_curr = np.zeros(K_curr)
        bp_curr = np.zeros(K_curr, dtype=int)

        # Transition matrix between prev cands and curr cands
        for j in range(K_curr):
            c_j = cands_t[j]
            u_j = u_vals[c_j]

            # Vectorized score from all prev candidates
            scores = np.zeros(K_prev)
            for i in range(K_prev):
                c_i = cands_prev[i]
                v_i = v_vals[c_i]

                # Transition logic:
                if c_i == c_j:
                    # Vehicle stayed on same road segment
                    trans_score = 0.0
                elif v_i == u_j:
                    # Directly connected adjacent road segment
                    trans_score = -0.5
                elif c_j in node_out_arcs.get(v_i, set()):
                    trans_score = -0.5
                else:
                    # Disconnected transition / gap
                    trans_score = gap_penalty

                scores[i] = prev_v[i] + trans_score

            best_prev_idx = np.argmax(scores)
            bp_curr[j] = best_prev_idx
            v_curr[j] = scores[best_prev_idx] + log_emit_t[j]

        V.append(v_curr)
        BP.append(bp_curr)

    # Backtracking
    best_last = np.argmax(V[-1])
    path_indices = [best_last]
    for t in range(T - 1, 0, -1):
        prev_idx = BP[t][path_indices[-1]]
        path_indices.append(prev_idx)
    path_indices.reverse()

    decoded_arcs = [obs_groups[t]['arc_id'].values[path_indices[t]] for t in range(T)]
    return decoded_arcs

def evaluate_hmm_vs_classifier():
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    print("Loading best tuned model...")
    clf = joblib.load('models/best_tuned_model.joblib')

    test_trips = sorted(os.listdir('student_test'))
    print(f"Evaluating on {len(test_trips)} student test trips...")

    results = []
    overall_clf_correct = 0
    overall_hmm_correct = 0
    overall_total_obs = 0

    overall_clf_connected = 0
    overall_hmm_connected = 0
    overall_total_transitions = 0

    trip_detail_records = []

    for tid in test_trips:
        track_df, nodes_df, arcs_df, route_df = load_trip_raw('student_test', tid)
        cands_df = extract_candidates_and_features_for_trip('student_test', tid, is_train=False)

        # Classifier predictions
        probs = clf.predict_proba(cands_df[FEATURE_COLS].values)[:, 1]
        cands_df['clf_prob'] = probs

        # Classifier-only Top 1
        top1_idx = cands_df.groupby('obs_idx')['clf_prob'].idxmax()
        top1_df = cands_df.loc[top1_idx].sort_values('obs_idx').reset_index(drop=True)
        clf_pred_arcs = top1_df['arc_id'].values
        gt_arcs = top1_df['gt_arc_id'].values

        # HMM Viterbi decoding
        t0 = time.time()
        hmm_pred_arcs = viterbi_route_decode(cands_df, arcs_df)
        hmm_time = time.time() - t0

        # Accuracies
        N_obs = len(gt_arcs)
        clf_acc = (clf_pred_arcs == gt_arcs).mean()
        hmm_acc = (np.array(hmm_pred_arcs) == gt_arcs).mean()

        overall_clf_correct += (clf_pred_arcs == gt_arcs).sum()
        overall_hmm_correct += (np.array(hmm_pred_arcs) == gt_arcs).sum()
        overall_total_obs += N_obs

        # Continuity
        u_map = arcs_df['u'].values
        v_map = arcs_df['v'].values
        clf_conn = 0
        hmm_conn = 0
        N_trans = N_obs - 1

        for i in range(N_trans):
            # clf
            c_p, c_c = clf_pred_arcs[i], clf_pred_arcs[i+1]
            if c_p == c_c or (c_p < len(v_map) and c_c < len(u_map) and v_map[c_p] == u_map[c_c]):
                clf_conn += 1
            # hmm
            h_p, h_c = hmm_pred_arcs[i], hmm_pred_arcs[i+1]
            if h_p == h_c or (h_p < len(v_map) and h_c < len(u_map) and v_map[h_p] == u_map[h_c]):
                hmm_conn += 1

        overall_clf_connected += clf_conn
        overall_hmm_connected += hmm_conn
        overall_total_transitions += N_trans

        clf_cont_pct = (clf_conn / N_trans) * 100 if N_trans > 0 else 100.0
        hmm_cont_pct = (hmm_conn / N_trans) * 100 if N_trans > 0 else 100.0

        trip_detail_records.append({
            'trip_id': tid,
            'obs_count': N_obs,
            'clf_top1_acc': round(clf_acc * 100, 2),
            'hmm_viterbi_acc': round(hmm_acc * 100, 2),
            'clf_continuity': round(clf_cont_pct, 2),
            'hmm_continuity': round(hmm_cont_pct, 2),
            'viterbi_time_s': round(hmm_time, 3)
        })

    df_details = pd.DataFrame(trip_detail_records)
    df_details.to_csv('results/hmm_viterbi_trip_breakdown.csv', index=False)

    summary = {
        'Pipeline Stage': ['Classifier-Only (Tuned RF)', 'Classifier + HMM Viterbi Decoding'],
        'Observation-Level Top-1 Accuracy': [
            f"{(overall_clf_correct / overall_total_obs)*100:.2f}%",
            f"{(overall_hmm_correct / overall_total_obs)*100:.2f}%"
        ],
        'Route Continuity': [
            f"{(overall_clf_connected / overall_total_transitions)*100:.2f}%",
            f"{(overall_hmm_connected / overall_total_transitions)*100:.2f}%"
        ],
        'Valid Topological Transitions': [
            f"{overall_clf_connected} / {overall_total_transitions}",
            f"{overall_hmm_connected} / {overall_total_transitions}"
        ]
    }
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv('results/hmm_viterbi_comparison.csv', index=False)

    print("\n================ HMM VITERBI COMPARISON SUMMARY ================")
    print(df_summary.to_string(index=False))
    print("\nPer-Trip Breakdown:\n", df_details.to_string(index=False))

    # Plot Comparison
    plt.figure(figsize=(8, 4.5))
    metrics = ['Top-1 Accuracy (%)', 'Route Continuity (%)']
    clf_vals = [(overall_clf_correct / overall_total_obs)*100, (overall_clf_connected / overall_total_transitions)*100]
    hmm_vals = [(overall_hmm_correct / overall_total_obs)*100, (overall_hmm_connected / overall_total_transitions)*100]

    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, clf_vals, width, label='Classifier-Only', color='cornflowerblue', edgecolor='black')
    plt.bar(x + width/2, hmm_vals, width, label='Classifier + HMM Viterbi', color='mediumseagreen', edgecolor='black')
    plt.xticks(x, metrics)
    plt.ylabel('Percentage (%)')
    plt.ylim(70, 105)
    for i in range(len(metrics)):
        plt.text(x[i] - width/2, clf_vals[i] + 0.8, f"{clf_vals[i]:.2f}%", ha='center', fontsize=9)
        plt.text(x[i] + width/2, hmm_vals[i] + 0.8, f"{hmm_vals[i]:.2f}%", ha='center', fontsize=9, fontweight='bold')
    plt.title('Impact of HMM Viterbi Route Reconstruction on Accuracy and Topological Continuity')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/hmm_viterbi_comparison.png', dpi=200)
    plt.close()

    print("\nSaved comparison plot in plots/hmm_viterbi_comparison.png")
    return df_summary

if __name__ == '__main__':
    evaluate_hmm_vs_classifier()
