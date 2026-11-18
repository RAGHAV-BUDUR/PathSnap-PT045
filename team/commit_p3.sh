#!/usr/bin/env bash
# ==============================================================================
# PathSnap - Team Commit Script for Person 3 (P3)
# Member: PALLENENI SLOKA (PES1UG24AM183)
# Role: Bayesian Hyperparameter Optimisation & HMM Viterbi Sequential Decoder
# ==============================================================================

MEMBER_NAME="Palleneni Sloka"
MEMBER_EMAIL="pes1ug24am183@pes.edu"

P3_FILES=(
    "src/bayesian_opt.py"
    "src/hmm_viterbi.py"
    "models/best_tuned_model.joblib"
    "models/best_hyperparameters.json"
    "results/bayesian_trials.csv"
    "results/bayesian_comparison.csv"
    "results/hmm_viterbi_trip_breakdown.csv"
    "results/hmm_viterbi_comparison.csv"
    "plots/bayes_opt_history.png"
    "plots/hmm_viterbi_comparison.png"
)

do_commit() {
    local msg="$1"
    if [ -z "$msg" ]; then
        msg="P3 (Palleneni Sloka): Update Bayesian hyperparameter optimisation and HMM Viterbi decoder"
    fi

    git config user.name "$MEMBER_NAME"
    git config user.email "$MEMBER_EMAIL"

    local staged=0
    for f in "${P3_FILES[@]}"; do
        if [ -e "$f" ]; then
            git add "$f"
            staged=1
        fi
    done

    if [ $staged -eq 1 ]; then
        if git diff --cached --quiet; then
            echo "[P3: $MEMBER_NAME] No changes detected in P3's files."
        else
            git commit --author="$MEMBER_NAME <$MEMBER_EMAIL>" -m "$msg"
            echo "[P3: $MEMBER_NAME] Successfully committed P3 files with message: '$msg'"
            if git remote | grep -q 'origin'; then
                git push origin main || git push origin master || echo "[INFO] Run git push when ready."
            fi
        fi
    else
        echo "[P3: $MEMBER_NAME] Warning: Assigned P3 files not found yet."
    fi
}

if [ "$1" == "--watch" ]; then
    echo "=================================================================="
    echo "Starting Watch Mode for P3: $MEMBER_NAME"
    echo "Committing P3 changes every 10 minutes. Press [Ctrl+C] to stop."
    echo "=================================================================="
    while true; do
        timestamp=$(date "+%Y-%m-%d %H:%M:%S")
        do_commit "Auto-commit (P3: Palleneni Sloka) at $timestamp"
        echo "[Watch] Sleeping for 10 minutes (600s)..."
        sleep 600
    done
else
    do_commit "$1"
fi
