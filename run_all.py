"""
PathSnap Master Runner Script
Team: PT-045 | PES University UE24CS352A
Usage:
  python run_all.py --pipeline    # Re-runs EDA, 5-model training, Bayes Opt, and HMM benchmarks
  python run_all.py --server      # Starts the FastAPI backend and serves the Evaluator UI
  python run_all.py               # Checks artifacts and launches server (Default)
"""

import os, sys, argparse, subprocess

def check_artifacts():
    required = [
        'models/best_tuned_model.joblib',
        'results/model_comparison.csv',
        'results/bayesian_comparison.csv',
        'results/hmm_viterbi_comparison.csv'
    ]
    missing = [f for f in required if not os.path.exists(f)]
    return len(missing) == 0

def run_pipeline():
    print("==================================================")
    print("Running Full PathSnap Machine Learning Pipeline")
    print("==================================================")
    scripts = [
        ('src/eda.py', '1. Exploratory Data Analysis & GPS Stats'),
        ('src/features.py', '2. Feature Engineering & Candidate Extraction'),
        ('src/train_models.py', '3. Training & Evaluating 5 ML Models'),
        ('src/bayesian_opt.py', '4. Bayesian Hyperparameter Optimisation'),
        ('src/hmm_viterbi.py', '5. HMM Viterbi Route Reconstruction'),
        ('src/map_viz.py', '6. Folium Map Generation')
    ]
    for script, desc in scripts:
        print(f"\n--- Running: {desc} ({script}) ---")
        ret = subprocess.run([sys.executable, script])
        if ret.returncode != 0:
            print(f"[ERROR] Failed executing {script}. Halting.")
            sys.exit(ret.returncode)
    print("\n[SUCCESS] Entire pipeline completed successfully!")

def start_server(port=8000):
    print("==================================================")
    print(f"Starting PathSnap FastAPI Backend on port {port}")
    print("==================================================")
    print(f"Local Evaluator UI: http://localhost:{port}/ui")
    print(f"API Docs:           http://localhost:{port}/docs")
    print("Press Ctrl+C to stop the server.\n")
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PathSnap Master Runner")
    parser.add_argument('--pipeline', action='store_true', help="Re-run the complete training & evaluation pipeline")
    parser.add_argument('--server', action='store_true', help="Launch the FastAPI server and UI")
    parser.add_argument('--port', type=int, default=8000, help="Port to run the FastAPI server on (default: 8000)")
    args = parser.parse_args()

    if args.pipeline:
        run_pipeline()
    elif args.server:
        start_server(args.port)
    else:
        if not check_artifacts():
            print("[INFO] Model artifacts missing. Running pipeline first...")
            run_pipeline()
        start_server(args.port)
