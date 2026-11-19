@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
python -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
if errorlevel 1 goto failed
if exist ".venv\local_video_ready.txt" goto start
if exist ".venv\local_video_skipped.txt" goto start
echo.
echo The app can make REAL text-to-video clips on this computer with no API key.
echo This needs a one-time download of about 7 GB. An NVIDIA GPU makes it fast.
choice /C YN /M "Install the local text-to-video model now"
if errorlevel 2 goto skip_video
call setup_local_video.bat
goto start
:skip_video
echo skipped> ".venv\local_video_skipped.txt"
echo Skipped. Run setup_local_video.bat later to add it.
:start
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Setup or startup failed. Review the error above. Python 3.13 was used for validation.
pause
exit /b 1
