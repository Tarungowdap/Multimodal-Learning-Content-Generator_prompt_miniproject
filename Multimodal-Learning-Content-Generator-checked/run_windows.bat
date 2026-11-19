@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
python -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Setup or startup failed. Review the error above. Python 3.13 was used for validation.
pause
exit /b 1
