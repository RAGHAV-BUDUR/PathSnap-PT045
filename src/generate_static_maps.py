"""
PathSnap - Static Geographical Map Export
Team: PT-045 | P4: Parth Pandurang Kale (PES1UG24AM184)
"""
import sys, os, joblib
sys.path.insert(0, os.path.abspath('.'))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_processing import load_trip_raw
from src.features import extract_candidates_and_features_for_trip, FEATURE_COLS
from src.hmm_viterbi import viterbi_route_decode

def generate_static_map_plot(trip_id='00000052', out_path='plots/geographic_map_trip52.png'):
    track_df, nodes_df, arcs_df, route_df = load_trip_raw('student_test', trip_id)
    cands_df = extract_candidates_and_features_for_trip('student_test', trip_id, is_train=False)

    clf = joblib.load('models/best_tuned_model.joblib')
    probs = clf.predict_proba(cands_df[FEATURE_COLS].values)[:, 1]
    cands_df['clf_prob'] = probs
    pred_arcs = viterbi_route_decode(cands_df, arcs_df)

    plt.figure(figsize=(10, 8))
    u_vals = arcs_df['u'].values
    v_vals = arcs_df['v'].values
    node_lats = nodes_df['lat'].values
    node_lons = nodes_df['lon'].values

    # Bounding box
    pad = 0.002
    min_lat, max_lat = track_df['lat'].min() - pad, track_df['lat'].max() + pad
    min_lon, max_lon = track_df['lon'].min() - pad, track_df['lon'].max() + pad

    # Plot road network
    for a in range(len(arcs_df)):
        u, v = u_vals[a], v_vals[a]
        la1, lo1 = node_lats[u], node_lons[u]
        la2, lo2 = node_lats[v], node_lons[v]
        if min_lat <= la1 <= max_lat and min_lon <= lo1 <= max_lon:
            plt.plot([lo1, lo2], [la1, la2], color='#cbd5e1', linewidth=1.2, zorder=1)

    # Plot Ground truth route
    gt_lons, gt_lats = [], []
    for a in route_df['arc_id'].values:
        u, v = u_vals[a], v_vals[a]
        gt_lons.extend([node_lons[u], node_lons[v]])
        gt_lats.extend([node_lats[u], node_lats[v]])
    plt.plot(gt_lons, gt_lats, color='#16a34a', linewidth=4.0, label='Ground Truth Reference Route', zorder=2)

    # Plot Predicted route
    pred_lons, pred_lats = [], []
    for a in pred_arcs:
        u, v = u_vals[a], v_vals[a]
        pred_lons.extend([node_lons[u], node_lons[v]])
        pred_lats.extend([node_lats[u], node_lats[v]])
    plt.plot(pred_lons, pred_lats, color='#dc2626', linestyle='--', linewidth=2.5, label='PathSnap Predicted Route (HMM + RF)', zorder=3)

    # Plot GPS points
    plt.scatter(track_df['lon'], track_df['lat'], c='#2563eb', s=25, label='Raw GPS Observations (Noisy)', edgecolors='black', linewidths=0.5, zorder=4)

    # Start and End
    plt.scatter(track_df['lon'].iloc[0], track_df['lat'].iloc[0], c='green', marker='^', s=120, label='Trip Start', zorder=5)
    plt.scatter(track_df['lon'].iloc[-1], track_df['lat'].iloc[-1], c='red', marker='v', s=120, label='Trip End', zorder=5)

    plt.xlabel('Longitude (Degrees East)')
    plt.ylabel('Latitude (Degrees North)')
    plt.title(f'Geographical Map-Matching Trajectory Visualisation: Trip {trip_id}')
    plt.legend(loc='lower right', framealpha=0.95)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(min_lon, max_lon)
    plt.ylim(min_lat, max_lat)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved static map to {out_path}")

if __name__ == '__main__':
    generate_static_map_plot('00000052', 'plots/geographic_map_trip52.png')
    generate_static_map_plot('00000000', 'plots/geographic_map_trip00.png')
