@echo off
REM AeroGuard PS-S02 — Start backend API + Streamlit frontend
REM ---------------------------------------------------------------
SETLOCAL
SET "ROOT=%~dp0"
SET "ROOT=%ROOT:~0,-1%"

echo [AeroGuard] Starting telemetry API server on port 8000...
start "AeroGuard API Server" cmd /k "cd /d %ROOT% && python api_server.py"

REM Give the API a moment to boot before Streamlit tries to connect
timeout /t 3 /nobreak >nul

echo [AeroGuard] Starting Streamlit frontend on port 8501...
start "AeroGuard Streamlit" cmd /k "cd /d %ROOT% && streamlit run app\app.py --server.port 8501"

echo.
echo [AeroGuard] Both services are starting.
echo   API  : http://localhost:8000
echo   UI   : http://localhost:8501
echo.
echo Close the two terminal windows to stop the system.
ENDLOCAL
