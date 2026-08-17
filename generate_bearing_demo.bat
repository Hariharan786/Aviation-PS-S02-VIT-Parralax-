@echo off
setlocal
cd /d "%~dp0"
echo [AeroGuard] Generating physics telemetry + bearing vibration sample...
python generate_fd001_physics_telemetry.py
echo.
echo [AeroGuard] Converting all cycles to raw bearing time-series...
python generate_bearing_timeseries.py --input FD001_physics_live_test.txt --output-dir bearing_output
echo.
echo Done. See bearing_output\
epause
