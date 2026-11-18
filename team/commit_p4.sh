#!/usr/bin/env bash
# ==============================================================================
# PathSnap - Team Commit Script for Person 4 (P4)
# Member: PARTH PANDURANG KALE (PES1UG24AM184)
# Role: Folium Maps, FastAPI Backend, Evaluator UI & Deployment Architecture
# ==============================================================================

MEMBER_NAME="Parth Pandurang Kale"
MEMBER_EMAIL="pes1ug24am184@pes.edu"

P4_FILES=(
    "src/map_viz.py"
    "src/generate_static_maps.py"
    "backend/app.py"
    "backend/requirements.txt"
    "frontend/index.html"
    "frontend/style.css"
    "frontend/app.js"
    "folium_maps/trip_00000000_map.html"
    "folium_maps/trip_00000007_map.html"
    "folium_maps/trip_00000052_map.html"
    "plots/geographic_map_trip00.png"
    "plots/geographic_map_trip52.png"
    "run_all.py"
    "Dockerfile"
    "docker-compose.yml"
)

do_commit() {
    local msg="$1"
    if [ -z "$msg" ]; then
        msg="P4 (Parth Pandurang Kale): Update Folium visualisations, FastAPI backend, and Evaluator UI"
    fi

    git config user.name "$MEMBER_NAME"
    git config user.email "$MEMBER_EMAIL"

    local staged=0
    for f in "${P4_FILES[@]}"; do
        if [ -e "$f" ]; then
            git add "$f"
            staged=1
        fi
    done

    if [ $staged -eq 1 ]; then
        if git diff --cached --quiet; then
            echo "[P4: $MEMBER_NAME] No changes detected in P4's files."
        else
            git commit --author="$MEMBER_NAME <$MEMBER_EMAIL>" -m "$msg"
            echo "[P4: $MEMBER_NAME] Successfully committed P4 files with message: '$msg'"
            if git remote | grep -q 'origin'; then
                git push origin main || git push origin master || echo "[INFO] Run git push when ready."
            fi
        fi
    else
        echo "[P4: $MEMBER_NAME] Warning: Assigned P4 files not found yet."
    fi
}

if [ "$1" == "--watch" ]; then
    echo "=================================================================="
    echo "Starting Watch Mode for P4: $MEMBER_NAME"
    echo "Committing P4 changes every 10 minutes. Press [Ctrl+C] to stop."
    echo "=================================================================="
    while true; do
        timestamp=$(date "+%Y-%m-%d %H:%M:%S")
        do_commit "Auto-commit (P4: Parth Pandurang Kale) at $timestamp"
        echo "[Watch] Sleeping for 10 minutes (600s)..."
        sleep 600
    done
else
    do_commit "$1"
fi
