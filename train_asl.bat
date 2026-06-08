@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
set CONFIG_PATH=configs\config_asl.yaml
echo Training ASL — config: %CONFIG_PATH%
python src\cnn\train.py --config %CONFIG_PATH%
pause
