#!/usr/bin/env bash
# ==============================================================================
# PathSnap - Team Commit Script for Person 2 (P2)
# Member: PABBISETTY HARSHITHA SAI (PES1UG24AM182)
# Role: Feature Engineering, 5 Machine Learning Models & Group-Aware Evaluation
# ==============================================================================

MEMBER_NAME="Pabbisetty Harshitha Sai"
MEMBER_EMAIL="pes1ug24am182@pes.edu"

P2_FILES=(
    "src/features.py"
    "src/train_models.py"
    "results/train_features.csv"
    "results/student_test_features.csv"
    "results/model_comparison.csv"
    "models/scaler.joblib"
    "models/logistic_regression.joblib"
    "models/k_nearest_neighbors.joblib"
    "models/decision_tree.joblib"
    "models/random_forest.joblib"
    "models/histgradientboosting.joblib"
    "plots/model_f1_comparison.png"
    "plots/model_top1_continuity.png"
    "plots/model_log_loss.png"
    "plots/confusion_matrices.png"
)

do_commit() {
    local msg="$1"
    if [ -z "$msg" ]; then
        msg="P2 (Pabbisetty Harshitha Sai): Update feature engineering, 5 models training and evaluation"
    fi

    git config user.name "$MEMBER_NAME"
    git config user.email "$MEMBER_EMAIL"

    local staged=0
    for f in "${P2_FILES[@]}"; do
        if [ -e "$f" ]; then
            git add "$f"
            staged=1
        fi
    done

    if [ $staged -eq 1 ]; then
        if git diff --cached --quiet; then
            echo "[P2: $MEMBER_NAME] No changes detected in P2's files."
        else
            git commit --author="$MEMBER_NAME <$MEMBER_EMAIL>" -m "$msg"
            echo "[P2: $MEMBER_NAME] Successfully committed P2 files with message: '$msg'"
            if git remote | grep -q 'origin'; then
                git push origin main || git push origin master || echo "[INFO] Run git push when ready."
            fi
        fi
    else
        echo "[P2: $MEMBER_NAME] Warning: Assigned P2 files not found yet."
    fi
}

if [ "$1" == "--watch" ]; then
    echo "=================================================================="
    echo "Starting Watch Mode for P2: $MEMBER_NAME"
    echo "Committing P2 changes every 10 minutes. Press [Ctrl+C] to stop."
    echo "=================================================================="
    while true; do
        timestamp=$(date "+%Y-%m-%d %H:%M:%S")
        do_commit "Auto-commit (P2: Pabbisetty Harshitha Sai) at $timestamp"
        echo "[Watch] Sleeping for 10 minutes (600s)..."
        sleep 600
    done
else
    do_commit "$1"
fi
