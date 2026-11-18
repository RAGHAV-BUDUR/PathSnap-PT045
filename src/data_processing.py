"""
PathSnap - Data Processing Module
Team: PT-045 | P1: Raghav Budur (PES1UG23AM225)
Handles raw data ingestion, geographical coordinate projection (WGS84 to local metric projection),
road network topology graph construction, and spatial indexing via cKDTree.
"""

import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

R_EARTH = 6371000.0  # Earth radius in meters

def project_wgs84_to_metric(lons, lats, lon0, lat0):
    """
    Project angular WGS84 coordinates (lon, lat) to local Euclidean metric coordinates (x, y).
    Uses equirectangular projection centered at (lon0, lat0), accurate for localized city tracks.
    """
    cos_lat0 = np.cos(np.radians(lat0))
    x = np.radians(lons - lon0) * R_EARTH * cos_lat0
    y = np.radians(lats - lat0) * R_EARTH
    return x, y

def load_trip_raw(folder_path, trip_id):
    """
    Load raw trip data: .track, .nodes, .arcs, and optional .route.
    """
    trip_dir = os.path.join(folder_path, trip_id)
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    track_df = pd.read_csv(track_path, sep=r'\s+', header=None, names=['lon', 'lat', 'time'], dtype=float)
    nodes_df = pd.read_csv(nodes_path, sep=r'\s+', header=None, names=['lon', 'lat'], dtype=float)
    arcs_df = pd.read_csv(arcs_path, sep=r'\s+', header=None, names=['u', 'v'], dtype=int)

    route_df = None
    if os.path.exists(route_path):
        route_df = pd.read_csv(route_path, sep=r'\s+', header=None, names=['arc_id'], dtype=int)

    return track_df, nodes_df, arcs_df, route_df

def build_spatial_network(nodes_df, arcs_df, lon0, lat0):
    """
    Projects road network nodes and arcs to local metric space,
    computes segment lengths, vectors, degrees, and constructs a multi-point spatial KD-Tree.
    """
    nx, ny = project_wgs84_to_metric(nodes_df['lon'].values, nodes_df['lat'].values, lon0, lat0)
    u_ids = arcs_df['u'].values
    v_ids = arcs_df['v'].values

    ux, uy = nx[u_ids], ny[u_ids]
    vx, vy = nx[v_ids], ny[v_ids]
    dx, dy = vx - ux, vy - uy
    seg_lens = np.sqrt(dx**2 + dy**2)
    seg_len_sq = np.maximum(seg_lens**2, 1e-9)

    out_degree = np.bincount(u_ids, minlength=len(nodes_df))
    in_degree = np.bincount(v_ids, minlength=len(nodes_df))

    # Multi-point index: endpoints and midpoint to ensure high candidate recall on long arcs
    pts_u = np.column_stack([ux, uy])
    pts_v = np.column_stack([vx, vy])
    pts_mid = np.column_stack([(ux + vx)/2.0, (uy + vy)/2.0])
    all_pts = np.vstack([pts_u, pts_v, pts_mid])
    num_arcs = len(arcs_df)
    arc_indices = np.concatenate([np.arange(num_arcs), np.arange(num_arcs), np.arange(num_arcs)])

    tree = cKDTree(all_pts)

    net_info = {
        'nx': nx, 'ny': ny,
        'u_ids': u_ids, 'v_ids': v_ids,
        'ux': ux, 'uy': uy, 'vx': vx, 'vy': vy,
        'dx': dx, 'dy': dy,
        'seg_lens': seg_lens,
        'seg_len_sq': seg_len_sq,
        'arc_heading': np.arctan2(dy, dx),
        'out_degree': out_degree,
        'in_degree': in_degree,
        'num_arcs': num_arcs,
        'tree': tree,
        'tree_arc_map': arc_indices
    }
    return net_info
