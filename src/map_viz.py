"""
PathSnap - Folium Geographical Visualisation Module
Team: PT-045 | P4: Parth Pandurang Kale (PES1UG24AM184)
Generates interactive Folium maps comparing:
1. Raw GPS observations (blue circle markers)
2. Road network geometry (light slate lines)
3. Ground truth reference route (green line)
4. Model + HMM Viterbi predicted route (crimson dashed line)
Includes LayerControl, tooltips, popups, and visual legend.
"""

import sys, os, time, joblib
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
import folium

from src.data_processing import load_trip_raw
from src.features import extract_candidates_and_features_for_trip, FEATURE_COLS
from src.hmm_viterbi import viterbi_route_decode

def generate_trip_folium_map(folder, trip_id, out_html_path, model_path='models/best_tuned_model.joblib', max_network_arcs=2500):
    """
    Builds a high-fidelity interactive Folium map for a given trip.
    """
    track_df, nodes_df, arcs_df, route_df = load_trip_raw(folder, trip_id)
    cands_df = extract_candidates_and_features_for_trip(folder, trip_id, is_train=False)

    clf = joblib.load(model_path)
    probs = clf.predict_proba(cands_df[FEATURE_COLS].values)[:, 1]
    cands_df['clf_prob'] = probs

    predicted_arcs = viterbi_route_decode(cands_df, arcs_df)

    center_lat = track_df['lat'].mean()
    center_lon = track_df['lon'].mean()

    # Initialize Folium Map with CartoDB Positron for clean contrast
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='OpenStreetMap')

    # 1. FeatureGroup: Road Network (Local bounding box)
    fg_network = folium.FeatureGroup(name="Road Network (Sub-graph)", show=True)
    lat_min, lat_max = track_df['lat'].min() - 0.005, track_df['lat'].max() + 0.005
    lon_min, lon_max = track_df['lon'].min() - 0.005, track_df['lon'].max() + 0.005

    u_vals = arcs_df['u'].values
    v_vals = arcs_df['v'].values
    node_lats = nodes_df['lat'].values
    node_lons = nodes_df['lon'].values

    rendered_arcs = 0
    for arc_id in range(len(arcs_df)):
        u, v = u_vals[arc_id], v_vals[arc_id]
        lat1, lon1 = node_lats[u], node_lons[u]
        lat2, lon2 = node_lats[v], node_lons[v]
        if (lat_min <= lat1 <= lat_max and lon_min <= lon1 <= lon_max) or (lat_min <= lat2 <= lat_max and lon_min <= lon2 <= lon_max):
            folium.PolyLine([(lat1, lon1), (lat2, lon2)], color="#94a3b8", weight=2, opacity=0.6).add_to(fg_network)
            rendered_arcs += 1
            if rendered_arcs >= max_network_arcs:
                break
    fg_network.add_to(m)

    # 2. FeatureGroup: Ground Truth Route
    fg_gt = folium.FeatureGroup(name="Ground Truth Route", show=True)
    if route_df is not None:
        gt_coords = []
        for arc_id in route_df['arc_id'].values:
            u, v = u_vals[arc_id], v_vals[arc_id]
            gt_coords.append((node_lats[u], node_lons[u]))
            gt_coords.append((node_lats[v], node_lons[v]))
        if gt_coords:
            folium.PolyLine(gt_coords, color="#16a34a", weight=5, opacity=0.85, tooltip="Ground Truth Reference Route").add_to(fg_gt)
    fg_gt.add_to(m)

    # 3. FeatureGroup: Predicted Route (HMM Viterbi)
    fg_pred = folium.FeatureGroup(name="PathSnap Predicted Route (HMM + RF)", show=True)
    pred_coords = []
    for arc_id in predicted_arcs:
        u, v = u_vals[arc_id], v_vals[arc_id]
        pred_coords.append((node_lats[u], node_lons[u]))
        pred_coords.append((node_lats[v], node_lons[v]))
    if pred_coords:
        folium.PolyLine(pred_coords, color="#dc2626", weight=3.5, dash_array="6, 6", opacity=0.9, tooltip="Model + Viterbi Predicted Route").add_to(fg_pred)
    fg_pred.add_to(m)

    # 4. FeatureGroup: Raw GPS Points
    fg_gps = folium.FeatureGroup(name="Raw GPS Observations (Noisy)", show=True)
    step = 1 if len(track_df) < 1500 else 2
    for i in range(0, len(track_df), step):
        lat = track_df['lat'].iloc[i]
        lon = track_df['lon'].iloc[i]
        t = track_df['time'].iloc[i]
        folium.CircleMarker(
            location=[lat, lon],
            radius=3.5,
            color="#2563eb",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.7,
            popup=f"Obs #{i} | Time: {t:.1f}s | Lat: {lat:.5f}, Lon: {lon:.5f}"
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    # Start and End Markers
    start_pt = [track_df['lat'].iloc[0], track_df['lon'].iloc[0]]
    end_pt = [track_df['lat'].iloc[-1], track_df['lon'].iloc[-1]]
    folium.Marker(start_pt, popup="Trajectory Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(end_pt, popup="Trajectory End", icon=folium.Icon(color="red", icon="flag")).add_to(m)

    # Layer Control
    folium.LayerControl(position='topright', collapsed=False).add_to(m)

    # HTML Legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 30px; left: 30px; width: 290px; height: 175px; 
                background-color: white; border:2px solid #cbd5e1; z-index:9999; 
                font-size:13px; font-family: sans-serif; padding: 12px; border-radius: 8px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
      <h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: bold; color: #1e293b;">PathSnap Map: Trip {trip_id}</h4>
      <div style="margin-bottom: 5px;"><i style="background: #2563eb; width: 10px; height: 10px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i> <b>Raw GPS Track</b> (Blue Dots)</div>
      <div style="margin-bottom: 5px;"><i style="background: #94a3b8; width: 18px; height: 3px; display: inline-block; margin-right: 6px;"></i> <b>Road Network</b> (Slate Lines)</div>
      <div style="margin-bottom: 5px;"><i style="background: #16a34a; width: 18px; height: 4px; display: inline-block; margin-right: 6px;"></i> <b>Ground Truth Route</b> (Solid Green)</div>
      <div style="margin-bottom: 5px;"><i style="background: #dc2626; width: 18px; height: 4px; display: inline-block; border-top: 2px dashed #dc2626; margin-right: 6px;"></i> <b>Predicted Route</b> (Crimson Dash)</div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">Team PT-045 | HMM Viterbi Decoder</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    m.save(out_html_path)
    print(f"Interactive Folium map for Trip {trip_id} saved to {out_html_path}")
    return out_html_path

if __name__ == '__main__':
    generate_trip_folium_map('student_test', '00000000', 'folium_maps/trip_00000000_map.html')
    generate_trip_folium_map('student_test', '00000007', 'folium_maps/trip_00000007_map.html')
    generate_trip_folium_map('student_test', '00000052', 'folium_maps/trip_00000052_map.html')
    print("All representative Folium maps generated successfully!")
