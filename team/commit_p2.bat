@echo off
set PATH=C:\Program Files\Git\cmd;%PATH%
REM ==============================================================================
REM PathSnap - Team Commit Batch Script for Person 2 (P2)
REM Member: PABBISETTY HARSHITHA SAI (PES1UG24AM182)
REM ==============================================================================

set MEMBER_NAME=Pabbisetty Harshitha Sai
set MEMBER_EMAIL=pes1ug24am182@pes.edu

set MSG=%~1
if "%MSG%"=="" set MSG=P2 (Pabbisetty Harshitha Sai): Update feature engineering, 5 models training and evaluation

git config user.name "%MEMBER_NAME%"
git config user.email "%MEMBER_EMAIL%"

echo Adding P2 files...
if exist src\features.py git add src\features.py
if exist src\train_models.py git add src\train_models.py
if exist results\train_features.csv git add results\train_features.csv
if exist results\student_test_features.csv git add results\student_test_features.csv
if exist results\model_comparison.csv git add results\model_comparison.csv
if exist models\scaler.joblib git add models\scaler.joblib
if exist models\logistic_regression.joblib git add models\logistic_regression.joblib
if exist models\k_nearest_neighbors.joblib git add models\k_nearest_neighbors.joblib
if exist models\decision_tree.joblib git add models\decision_tree.joblib
if exist models\random_forest.joblib git add models\random_forest.joblib
if exist models\histgradientboosting.joblib git add models\histgradientboosting.joblib
if exist plots\model_f1_comparison.png git add plots\model_f1_comparison.png
if exist plots\model_top1_continuity.png git add plots\model_top1_continuity.png
if exist plots\model_log_loss.png git add plots\model_log_loss.png
if exist plots\confusion_matrices.png git add plots\confusion_matrices.png

git commit --author="%MEMBER_NAME% <%MEMBER_EMAIL%>" -m "%MSG%"
echo [P2] Done.
