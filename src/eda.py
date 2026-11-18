"""
PathSnap - Exploratory Data Analysis Module
Team: PT-045 | P1: Raghav Budur (PES1UG23AM225)
Produces structured exploratory data analysis, missing-value checks, GPS trajectory stats,
road-network graph statistics, and generates figures for Deliverable 2 (Technical Report).
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_eda(train_folder='train', test_folder='student_test', out_dir='results', plot_dir='plots'):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    train_trips = sorted(os.listdir(train_folder))
    test_trips = sorted(os.listdir(test_folder))

    print(f"EDA: Analyzing {len(train_trips)} train trips and {len(test_trips)} student test trips...")

    records = []
    gap_trips = []

    for split, trips, folder in [('train', train_trips, train_folder), ('student_test', test_trips, test_folder)]:
        for tid in trips:
            tdir = os.path.join(folder, tid)
            track_path = os.path.join(tdir, f"{tid}.track")
            nodes_path = os.path.join(tdir, f"{tid}.nodes")
            arcs_path = os.path.join(tdir, f"{tid}.arcs")
            route_path = os.path.join(tdir, f"{tid}.route")

            track = pd.read_csv(track_path, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
            nodes = pd.read_csv(nodes_path, sep=r'\s+', header=None, names=['lon', 'lat'])
            arcs = pd.read_csv(arcs_path, sep=r'\s+', header=None, names=['u', 'v'])
            route_len = len(pd.read_csv(route_path, sep=r'\s+', header=None)) if os.path.exists(route_path) else 0

            # Missing value checks
            missing_gps = track.isnull().sum().sum()
            missing_nodes = nodes.isnull().sum().sum()
            missing_arcs = arcs.isnull().sum().sum()

            dt = track['time'].diff().dropna()
            max_dt = dt.max() if len(dt) > 0 else 0
            mean_dt = dt.mean() if len(dt) > 0 else 0
            frac_1s = (dt == 1.0).mean() if len(dt) > 0 else 0

            lon_min, lon_max = track['lon'].min(), track['lon'].max()
            lat_min, lat_max = track['lat'].min(), track['lat'].max()

            records.append({
                'partition': split,
                'trip_id': tid,
                'num_obs': len(track),
                'duration_s': track['time'].iloc[-1] - track['time'].iloc[0],
                'missing_gps': missing_gps,
                'missing_nodes': missing_nodes,
                'missing_arcs': missing_arcs,
                'num_nodes': len(nodes),
                'num_arcs': len(arcs),
                'route_arcs': route_len,
                'max_dt': max_dt,
                'mean_dt': mean_dt,
                'frac_1s': frac_1s,
                'lon_span': lon_max - lon_min,
                'lat_span': lat_max - lat_min
            })

            if max_dt > 10.0:
                gap_trips.append((tid, split, max_dt))

    df_eda = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, 'eda_inventory_summary.csv')
    df_eda.to_csv(csv_path, index=False)
    print(f"EDA summary saved to {csv_path}")

    # Summary table
    summary_by_split = df_eda.groupby('partition').agg({
        'trip_id': 'count',
        'num_obs': ['min', 'mean', 'max', 'sum'],
        'duration_s': ['min', 'mean', 'max'],
        'num_nodes': ['min', 'mean', 'max'],
        'num_arcs': ['min', 'mean', 'max'],
        'frac_1s': 'mean',
        'max_dt': 'max'
    })
    print("\nEDA Aggregate Summary:\n", summary_by_split)

    # 1. Plot GPS Observations Distribution
    plt.figure(figsize=(8, 4))
    plt.hist(df_eda[df_eda['partition']=='train']['num_obs'], bins=20, alpha=0.7, color='steelblue', label='Train (80 trips)', edgecolor='black')
    plt.hist(df_eda[df_eda['partition']=='student_test']['num_obs'], bins=10, alpha=0.8, color='darkorange', label='Student Test (10 trips)', edgecolor='black')
    plt.xlabel('Number of GPS Observations per Trip')
    plt.ylabel('Frequency (Count of Trips)')
    plt.title('Distribution of GPS Observations per Trip across Partitions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'eda_gps_distribution.png'), dpi=200)
    plt.close()

    # 2. Plot Sampling Interval Analysis (Time Delta dt)
    plt.figure(figsize=(7, 4))
    sample_trip_track = pd.read_csv(os.path.join(train_folder, '00000001', '00000001.track'), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    dt_sample = sample_trip_track['time'].diff().dropna()
    dt_counts = dt_sample.value_counts().head(5)
    plt.bar([f"{k:.0f}s" for k in dt_counts.index], dt_counts.values, color='teal', edgecolor='black', width=0.5)
    plt.xlabel('Sampling Time Interval (Δt)')
    plt.ylabel('Observation Count')
    plt.title('Sampling Rate Regularity: Nominal 1 Hz GPS Frequency (Trip 00000001)')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'eda_sampling_rate.png'), dpi=200)
    plt.close()

    # 3. Plot Road Network Scale: Nodes vs Arcs
    plt.figure(figsize=(7, 4))
    plt.scatter(df_eda['num_nodes'], df_eda['num_arcs'], c='indigo', alpha=0.6, edgecolors='none', s=40)
    plt.xlabel('Number of Road Network Nodes')
    plt.ylabel('Number of Road Network Arcs (Directed Edges)')
    plt.title('Road Network Graph Dimensions Across Trajectory Bounding Boxes')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'eda_network_scale.png'), dpi=200)
    plt.close()

    print(f"EDA plots saved in {plot_dir}/")
    print(f"Identified {len(gap_trips)} trips with sampling gaps > 10s: {gap_trips[:5]}")
    return df_eda

if __name__ == '__main__':
    run_eda()
