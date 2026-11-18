@echo off
set PATH=C:\Program Files\Git\cmd;%PATH%
REM ==============================================================================
REM PathSnap - Team Commit Batch Script for Person 1 (P1)
REM Member: RAGHAV BUDUR (PES1UG23AM225)
REM ==============================================================================

set MEMBER_NAME=Raghav Budur
set MEMBER_EMAIL=pes1ug23am225@pes.edu

set MSG=%~1
if "%MSG%"=="" set MSG=P1 (Raghav Budur): Update data processing, spatial indexing and EDA pipeline

git config user.name "%MEMBER_NAME%"
git config user.email "%MEMBER_EMAIL%"

echo Adding P1 files...
if exist src\data_processing.py git add src\data_processing.py
if exist src\eda.py git add src\eda.py
if exist results\eda_inventory_summary.csv git add results\eda_inventory_summary.csv
if exist plots\eda_gps_distribution.png git add plots\eda_gps_distribution.png
if exist plots\eda_sampling_rate.png git add plots\eda_sampling_rate.png
if exist plots\eda_network_scale.png git add plots\eda_network_scale.png

git commit --author="%MEMBER_NAME% <%MEMBER_EMAIL%>" -m "%MSG%"
echo [P1] Done.
