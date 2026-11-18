#!/usr/bin/env bash
# ==============================================================================
# PathSnap - Team Commit Script for Person 1 (P1)
# Member: RAGHAV BUDUR (PES1UG23AM225)
# Role: Data Processing, Coordinate Projections, Spatial Indexing & EDA
# ==============================================================================

MEMBER_NAME="Raghav Budur"
MEMBER_EMAIL="pes1ug23am225@pes.edu"

P1_FILES=(
    "src/data_processing.py"
    "src/eda.py"
    "results/eda_inventory_summary.csv"
    "plots/eda_gps_distribution.png"
    "plots/eda_sampling_rate.png"
    "plots/eda_network_scale.png"
)

do_commit() {
    local msg="$1"
    if [ -z "$msg" ]; then
        msg="P1 (Raghav Budur): Update data processing, spatial indexing and EDA pipeline"
    fi

    # Set local author config
    git config user.name "$MEMBER_NAME"
    git config user.email "$MEMBER_EMAIL"

    # Add only P1's assigned files
    local staged=0
    for f in "${P1_FILES[@]}"; do
        if [ -e "$f" ]; then
            git add "$f"
            staged=1
        fi
    done

    if [ $staged -eq 1 ]; then
        if git diff --cached --quiet; then
            echo "[P1: $MEMBER_NAME] No changes detected in P1's files."
        else
            git commit --author="$MEMBER_NAME <$MEMBER_EMAIL>" -m "$msg"
            echo "[P1: $MEMBER_NAME] Successfully committed P1 files with message: '$msg'"
            # Optional push if remote is configured
            if git remote | grep -q 'origin'; then
                git push origin main || git push origin master || echo "[INFO] Run git push when ready."
            fi
        fi
    else
        echo "[P1: $MEMBER_NAME] Warning: Assigned P1 files not found yet."
    fi
}

# Check for --watch argument
if [ "$1" == "--watch" ]; then
    echo "=================================================================="
    echo "Starting Watch Mode for P1: $MEMBER_NAME"
    echo "Committing P1 changes every 10 minutes. Press [Ctrl+C] to stop."
    echo "=================================================================="
    while true; do
        timestamp=$(date "+%Y-%m-%d %H:%M:%S")
        do_commit "Auto-commit (P1: Raghav Budur) at $timestamp"
        echo "[Watch] Sleeping for 10 minutes (600s)..."
        sleep 600
    done
else
    do_commit "$1"
fi
