@echo off
set PATH=C:\Program Files\Git\cmd;%PATH%
REM ==============================================================================
REM PathSnap - Team Commit Batch Script for Person 4 (P4)
REM Member: PARTH PANDURANG KALE (PES1UG24AM184)
REM ==============================================================================

set MEMBER_NAME=Parth Pandurang Kale
set MEMBER_EMAIL=pes1ug24am184@pes.edu

set MSG=%~1
if "%MSG%"=="" set MSG=P4 (Parth Pandurang Kale): Update Folium visualisations, FastAPI backend, and Evaluator UI

git config user.name "%MEMBER_NAME%"
git config user.email "%MEMBER_EMAIL%"

echo Adding P4 files...
if exist src\map_viz.py git add src\map_viz.py
if exist src\generate_static_maps.py git add src\generate_static_maps.py
if exist backend\app.py git add backend\app.py
if exist backend\requirements.txt git add backend\requirements.txt
if exist frontend\index.html git add frontend\index.html
if exist frontend\style.css git add frontend\style.css
if exist frontend\app.js git add frontend\app.js
if exist folium_maps git add folium_maps
if exist plots\geographic_map_trip00.png git add plots\geographic_map_trip00.png
if exist plots\geographic_map_trip52.png git add plots\geographic_map_trip52.png
if exist run_all.py git add run_all.py
if exist Dockerfile git add Dockerfile
if exist docker-compose.yml git add docker-compose.yml

git commit --author="%MEMBER_NAME% <%MEMBER_EMAIL%>" -m "%MSG%"
echo [P4] Done.
