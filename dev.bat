@echo off
cd /d "%~dp0"

echo [1/3] Installing frontend dependencies...
cd frontend
call npm install --silent
cd /d "%~dp0"

echo [2/3] Starting Backend (port 8001)...
start "Backend - FastAPI" cmd /k "cd /d %~dp0 && call venv\Scripts\activate && set PYTHONIOENCODING=utf-8 && python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8001 --reload"

echo [3/3] Starting Frontend (port 5173)...
start "Frontend - Vite" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Both servers are starting:
echo   Backend  -^>  http://localhost:8001
echo   Frontend -^>  http://localhost:5173
echo.
echo Close the opened windows to stop the servers.
pause
