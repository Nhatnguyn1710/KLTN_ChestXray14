@echo off
cd /d "%~dp0"
call venv\Scripts\activate

echo [1/2] Building frontend...
cd frontend
call npm install --silent
call npm run build
cd /d "%~dp0"

echo [2/2] Starting backend...
start "APP" cmd /k "cd /d %~dp0 && call venv\Scripts\activate && python -m src.api.app --config configs/config.yaml --port 8001"
