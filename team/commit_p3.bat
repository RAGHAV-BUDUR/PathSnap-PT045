@echo off
set PATH=C:\Program Files\Git\cmd;%PATH%
REM ==============================================================================
REM PathSnap - Team Commit Batch Script for Person 3 (P3)
REM Member: PALLENENI SLOKA (PES1UG24AM183)
REM ==============================================================================

set MEMBER_NAME=Palleneni Sloka
set MEMBER_EMAIL=pes1ug24am183@pes.edu

set MSG=%~1
if "%MSG%"=="" set MSG=P3 (Palleneni Sloka): Update Bayesian hyperparameter optimisation and HMM Viterbi decoder

git config user.name "%MEMBER_NAME%"
git config user.email "%MEMBER_EMAIL%"

echo Adding P3 files...
if exist src\bayesian_opt.py git add src\bayesian_opt.py
if exist src\hmm_viterbi.py git add src\hmm_viterbi.py
if exist models\best_tuned_model.joblib git add models\best_tuned_model.joblib
if exist models\best_hyperparameters.json git add models\best_hyperparameters.json
if exist results\bayesian_trials.csv git add results\bayesian_trials.csv
if exist results\bayesian_comparison.csv git add results\bayesian_comparison.csv
if exist results\hmm_viterbi_trip_breakdown.csv git add results\hmm_viterbi_trip_breakdown.csv
if exist results\hmm_viterbi_comparison.csv git add results\hmm_viterbi_comparison.csv
if exist plots\bayes_opt_history.png git add plots\bayes_opt_history.png
if exist plots\hmm_viterbi_comparison.png git add plots\hmm_viterbi_comparison.png

git commit --author="%MEMBER_NAME% <%MEMBER_EMAIL%>" -m "%MSG%"
echo [P3] Done.
