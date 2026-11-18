"""
PathSnap - Distributed Map-Matching Backend API
Team: PT-045 | P4: Parth Pandurang Kale (PES1UG24AM184)
FastAPI application designed for the Two-Laptop Deployment Architecture (Laptop 1: Model & API).
Executes candidate generation, machine learning inference, HMM Viterbi route decoding,
and server-side ground-truth evaluation.
"""

import sys, os, time, joblib
sys.path.insert(0, os.path.abspath('.'))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np

from src.data_processing import load_trip_raw
from src.features import extract_candidates_and_features_for_trip, FEATURE_COLS
from src.hmm_viterbi import viterbi_route_decode
from src.map_viz import generate_trip_folium_map

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Distributed Map-Matching and Sequential Route Reconstruction API | Team PT-045",
    version="1.0.0"
)

# Enable CORS for cross-laptop requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
MODEL_PATH = 'models/best_tuned_model.joblib'
print(f"[SERVER STARTUP] Loading tuned model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
print("[SERVER STARTUP] Model loaded successfully.")

DATA_FOLDER = 'student_test'

class MatchRequest(BaseModel):
    trip_id: str

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "PathSnap Model & API Server",
        "role": "Laptop 1 (Model and Inference Server)",
        "team_number": "PT-045",
        "team_members": [
            {"srn": "PES1UG23AM225", "name": "RAGHAV BUDUR"},
            {"srn": "PES1UG24AM182", "name": "PABBISETTY HARSHITHA SAI"},
            {"srn": "PES1UG24AM183", "name": "PALLENENI SLOKA"},
            {"srn": "PES1UG24AM184", "name": "PARTH PANDURANG KALE"}
        ],
        "active_model": "Tuned Random Forest + HMM Viterbi Decoder"
    }

@app.get("/api/trips")
def get_trips():
    """
    Returns list of the ten reserved evaluation trips.
    """
    if not os.path.exists(DATA_FOLDER):
        raise HTTPException(status_code=404, detail="Dataset folder not found")

    trips = sorted(os.listdir(DATA_FOLDER))
    trip_list = []
    for tid in trips:
        track_path = os.path.join(DATA_FOLDER, tid, f"{tid}.track")
        if os.path.exists(track_path):
            track_df = pd.read_csv(track_path, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
            duration = track_df['time'].iloc[-1] - track_df['time'].iloc[0]
            trip_list.append({
                "trip_id": tid,
                "num_observations": len(track_df),
                "duration_seconds": round(duration, 1),
                "start_coords": [float(track_df['lat'].iloc[0]), float(track_df['lon'].iloc[0])],
                "end_coords": [float(track_df['lat'].iloc[-1]), float(track_df['lon'].iloc[-1])]
            })
    return {"trips": trip_list}

@app.get("/api/models")
def get_model_benchmarks():
    """
    Returns the 5-model benchmark comparison and Bayesian optimization results.
    """
    comp_file = 'results/model_comparison.csv'
    bayes_file = 'results/bayesian_comparison.csv'
    hmm_file = 'results/hmm_viterbi_comparison.csv'

    comp_data = pd.read_csv(comp_file).to_dict(orient='records') if os.path.exists(comp_file) else []
    bayes_data = pd.read_csv(bayes_file).to_dict(orient='records') if os.path.exists(bayes_file) else []
    hmm_data = pd.read_csv(hmm_file).to_dict(orient='records') if os.path.exists(hmm_file) else []

    return {
        "model_comparison": comp_data,
        "bayesian_optimization": bayes_data,
        "hmm_viterbi": hmm_data
    }

@app.post("/api/match")
def match_trip(req: MatchRequest):
    """
    Processes selected evaluation trip through the complete prediction pipeline:
    1. Spatial candidate generation
    2. Feature engineering
    3. Tuned ML Classifier emission probabilities
    4. HMM Viterbi route decoding
    5. Server-side evaluation against reference route
    """
    tid = req.trip_id
    trip_dir = os.path.join(DATA_FOLDER, tid)
    if not os.path.exists(trip_dir):
        raise HTTPException(status_code=404, detail=f"Trip {tid} not found in {DATA_FOLDER}")

    t_start = time.time()
    track_df, nodes_df, arcs_df, route_df = load_trip_raw(DATA_FOLDER, tid)

    # Candidate extraction & features
    cands_df = extract_candidates_and_features_for_trip(DATA_FOLDER, tid, is_train=False)

    # ML Classifier scoring
    probs = model.predict_proba(cands_df[FEATURE_COLS].values)[:, 1]
    cands_df['clf_prob'] = probs

    # Raw classifier Top-1
    top1_idx = cands_df.groupby('obs_idx')['clf_prob'].idxmax()
    top1_df = cands_df.loc[top1_idx].sort_values('obs_idx').reset_index(drop=True)
    clf_arcs = top1_df['arc_id'].values

    # HMM Viterbi Sequential Decoding
    t_viterbi = time.time()
    hmm_arcs = viterbi_route_decode(cands_df, arcs_df)
    viterbi_duration_ms = (time.time() - t_viterbi) * 1000.0

    total_duration_ms = (time.time() - t_start) * 1000.0

    # Server-Side Ground Truth Evaluation (Labels remain strictly server-side)
    gt_arcs = top1_df['gt_arc_id'].values
    N_obs = len(gt_arcs)

    clf_acc = float((clf_arcs == gt_arcs).mean() * 100) if N_obs > 0 else 0.0
    hmm_acc = float((np.array(hmm_arcs) == gt_arcs).mean() * 100) if N_obs > 0 else 0.0

    # Route Continuity Calculation
    u_map = arcs_df['u'].values
    v_map = arcs_df['v'].values
    N_trans = N_obs - 1

    clf_conn = 0
    hmm_conn = 0
    for i in range(N_trans):
        cp, cc = clf_arcs[i], clf_arcs[i+1]
        if cp == cc or (cp < len(v_map) and cc < len(u_map) and v_map[cp] == u_map[cc]):
            clf_conn += 1
        hp, hc = hmm_arcs[i], hmm_arcs[i+1]
        if hp == hc or (hp < len(v_map) and hc < len(u_map) and v_map[hp] == u_map[hc]):
            hmm_conn += 1

    clf_continuity = float((clf_conn / N_trans) * 100) if N_trans > 0 else 100.0
    hmm_continuity = float((hmm_conn / N_trans) * 100) if N_trans > 0 else 100.0

    # Prepare coordinates for frontend Leaflet/Folium visualization
    node_lats = nodes_df['lat'].values
    node_lons = nodes_df['lon'].values

    # Predicted route polyline
    pred_polyline = []
    for a in hmm_arcs:
        u, v = u_map[a], v_map[a]
        pred_polyline.append([float(node_lats[u]), float(node_lons[u])])
        pred_polyline.append([float(node_lats[v]), float(node_lons[v])])

    # Ground truth reference route polyline
    gt_polyline = []
    if route_df is not None:
        for a in route_df['arc_id'].values:
            u, v = u_map[a], v_map[a]
            gt_polyline.append([float(node_lats[u]), float(node_lons[u])])
            gt_polyline.append([float(node_lats[v]), float(node_lons[v])])

    # Raw GPS points (subsample if huge for network transfer)
    gps_step = 1 if len(track_df) < 1500 else 2
    gps_points = [
        {"lat": float(track_df['lat'].iloc[i]), "lon": float(track_df['lon'].iloc[i]), "t": float(track_df['time'].iloc[i])}
        for i in range(0, len(track_df), gps_step)
    ]

    print(f"[API INFERENCE] Evaluated Trip {tid}: HMM Top-1 Acc={hmm_acc:.2f}%, Continuity={hmm_continuity:.2f}%, Time={total_duration_ms:.1f}ms")

    return {
        "trip_id": tid,
        "num_observations": N_obs,
        "observation_top1_accuracy": round(hmm_acc, 2),
        "raw_classifier_accuracy": round(clf_acc, 2),
        "accuracy_improvement": round(hmm_acc - clf_acc, 2),
        "route_continuity": round(hmm_continuity, 2),
        "raw_classifier_continuity": round(clf_continuity, 2),
        "total_inference_time_ms": round(total_duration_ms, 1),
        "viterbi_decoding_time_ms": round(viterbi_duration_ms, 1),
        "predicted_route_arc_count": len(hmm_arcs),
        "gps_points": gps_points,
        "predicted_polyline": pred_polyline,
        "ground_truth_polyline": gt_polyline,
        "map_url": f"/api/map/{tid}"
    }

@app.get("/api/map/{trip_id}", response_class=HTMLResponse)
def get_map_html(trip_id: str):
    """
    Renders and serves the interactive Folium HTML map for the selected trip.
    """
    html_path = f"folium_maps/trip_{trip_id}_map.html"
    if not os.path.exists(html_path):
        generate_trip_folium_map(DATA_FOLDER, trip_id, html_path)
    with open(html_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

# Mount frontend directory for single-laptop demo or local viewing
if os.path.exists('frontend'):
    app.mount("/ui", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
