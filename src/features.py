import sys, os
sys.path.insert(0, os.path.abspath('.'))
"""
PathSnap - Feature Engineering Module
Team: PT-045 | P2: Pabbisetty Harshitha Sai (PES1UG24AM182)
Generates candidate road segments for each GPS observation using spatial KD-Trees,
extracts spatial, kinematic, and topological features, and constructs training/evaluation datasets.
"""

import os, sys, time
import pandas as pd
import numpy as np
from src.data_processing import project_wgs84_to_metric, load_trip_raw, build_spatial_network

FEATURE_COLS = [
    'dist_perp',
    'dist_diff_min',
    'dist_rank',
    'dist_u',
    'dist_v',
    'seg_len',
    'proj_t',
    'heading_diff',
    'speed',
    'u_degree',
    'v_degree'
]

def extract_candidates_and_features_for_trip(folder, trip_id, k_query=25, max_cands=8, is_train=True):
    """
    Extracts candidate segments and features for each observation in a single trip.
    """
    track_df, nodes_df, arcs_df, route_df = load_trip_raw(folder, trip_id)
    lon0, lat0 = track_df['lon'].mean(), track_df['lat'].mean()

    net = build_spatial_network(nodes_df, arcs_df, lon0, lat0)
    tx, ty = project_wgs84_to_metric(track_df['lon'].values, track_df['lat'].values, lon0, lat0)
    times = track_df['time'].values

    # GPS movement vector and velocity
    N = len(track_df)
    gps_dx = np.zeros(N)
    gps_dy = np.zeros(N)
    speed = np.zeros(N)
    for i in range(1, N):
        dt = max(times[i] - times[i-1], 0.1)
        gps_dx[i] = tx[i] - tx[i-1]
        gps_dy[i] = ty[i] - ty[i-1]
        speed[i] = np.sqrt(gps_dx[i]**2 + gps_dy[i]**2) / dt
    if N > 1:
        gps_dx[0], gps_dy[0], speed[0] = gps_dx[1], gps_dy[1], speed[1]

    gps_heading = np.arctan2(gps_dy, gps_dx)

    # Determine ground-truth arc for each observation if route exists
    gt_arcs = np.full(N, -1, dtype=int)
    if route_df is not None:
        route_arc_ids = route_df['arc_id'].values
        r_u = net['ux'][route_arc_ids]
        r_v = net['vx'][route_arc_ids]
        r_uy = net['uy'][route_arc_ids]
        r_vy = net['vy'][route_arc_ids]
        r_dx = r_v - r_u
        r_dy = r_vy - r_uy
        r_len_sq = np.maximum(r_dx**2 + r_dy**2, 1e-9)

        for i in range(N):
            px, py = tx[i], ty[i]
            t = ((px - r_u)*r_dx + (py - r_uy)*r_dy) / r_len_sq
            t = np.clip(t, 0.0, 1.0)
            proj_x = r_u + t * r_dx
            proj_y = r_uy + t * r_dy
            best_idx = np.argmin((px - proj_x)**2 + (py - proj_y)**2)
            gt_arcs[i] = route_arc_ids[best_idx]

    # Query multi-point spatial KD-tree
    query_pts = np.column_stack([tx, ty])
    _, query_idx = net['tree'].query(query_pts, k=min(k_query, len(net['tree_arc_map'])))

    rows = []
    tree_map = net['tree_arc_map']
    ux, uy, vx, vy = net['ux'], net['uy'], net['vx'], net['vy']
    dx, dy = net['dx'], net['dy']
    seg_lens = net['seg_lens']
    seg_len_sq = net['seg_len_sq']
    arc_heading = net['arc_heading']
    out_degree = net['out_degree']
    in_degree = net['in_degree']
    u_ids = net['u_ids']
    v_ids = net['v_ids']

    for i in range(N):
        px, py = tx[i], ty[i]
        cands = list(dict.fromkeys(tree_map[query_idx[i]]))
        
        cand_dists = []
        cand_ts = []
        for c in cands:
            t = ((px - ux[c])*dx[c] + (py - uy[c])*dy[c]) / seg_len_sq[c]
            t = np.clip(t, 0.0, 1.0)
            proj_x = ux[c] + t * dx[c]
            proj_y = uy[c] + t * dy[c]
            d = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
            cand_dists.append(d)
            cand_ts.append(t)

        order = np.argsort(cand_dists)[:max_cands]
        min_d = cand_dists[order[0]] if len(order) > 0 else 0.0

        for rank, idx in enumerate(order):
            c = cands[idx]
            d = cand_dists[idx]
            t = cand_ts[idx]
            h_diff = np.abs(gps_heading[i] - arc_heading[c])
            h_diff = np.where(h_diff > np.pi, 2*np.pi - h_diff, h_diff)

            d_u = np.sqrt((px - ux[c])**2 + (py - uy[c])**2)
            d_v = np.sqrt((px - vx[c])**2 + (py - vy[c])**2)

            label = 1 if (gt_arcs[i] == c) else 0

            rows.append({
                'trip_id': trip_id,
                'obs_idx': i,
                'arc_id': c,
                'gt_arc_id': gt_arcs[i],
                'dist_perp': d,
                'dist_diff_min': d - min_d,
                'dist_rank': rank + 1,
                'dist_u': d_u,
                'dist_v': d_v,
                'seg_len': seg_lens[c],
                'proj_t': t,
                'heading_diff': h_diff,
                'speed': speed[i],
                'u_degree': out_degree[u_ids[c]],
                'v_degree': in_degree[v_ids[c]],
                'label': label
            })

    return pd.DataFrame(rows)

def build_dataset(folder, trip_ids, out_csv=None, max_cands=8):
    dfs = []
    t0 = time.time()
    for idx, tid in enumerate(trip_ids):
        df_trip = extract_candidates_and_features_for_trip(folder, tid, max_cands=max_cands)
        dfs.append(df_trip)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(trip_ids):
            print(f"Processed {idx+1}/{len(trip_ids)} trips ({time.time()-t0:.1f}s elapsed)")
    full_df = pd.concat(dfs, ignore_index=True)
    if out_csv:
        full_df.to_csv(out_csv, index=False)
        print(f"Saved dataset to {out_csv} ({len(full_df)} candidate rows)")
    return full_df

if __name__ == '__main__':
    train_trips = sorted(os.listdir('train'))
    test_trips = sorted(os.listdir('student_test'))

    # Build dataset using 30 diverse training trips for rich representation and quick iteration
    print("Building training dataset from 30 training trips...")
    train_sample = train_trips[:30]
    df_train = build_dataset('train', train_sample, 'results/train_features.csv')

    print("\nBuilding student-test dataset from all 10 student-test trips...")
    df_test = build_dataset('student_test', test_trips, 'results/student_test_features.csv')

    print(f"\nTraining set: {len(df_train)} rows, {df_train['label'].sum()} positive ({df_train['label'].mean()*100:.2f}%)")
    print(f"Test set: {len(df_test)} rows, {df_test['label'].sum()} positive ({df_test['label'].mean()*100:.2f}%)")
