@echo off
setlocal
cd /d "%~dp0"
rem One-time setup of the LOCAL text-to-video model (AnimateDiff-Lightning). No API key, no account.
rem Downloads PyTorch (about 2.5 GB with CUDA) and the model weights (about 4.5 GB), then works offline.
if exist ".venv\Scripts\python.exe" goto venv_ok
python -m venv .venv
if errorlevel 1 goto failed
:venv_ok
set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip >nul
where nvidia-smi >nul 2>nul
if errorlevel 1 goto cpu_torch
echo NVIDIA GPU found - installing PyTorch with CUDA support...
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if not errorlevel 1 goto torch_ok
echo CUDA 12.8 build not available, trying CUDA 12.6...
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu126
if not errorlevel 1 goto torch_ok
:cpu_torch
echo Installing the CPU version of PyTorch - video generation will take several minutes per lesson.
"%PY%" -m pip install torch
if errorlevel 1 goto failed
:torch_ok
"%PY%" -m pip install -r requirements-video-lock.txt
if errorlevel 1 goto failed
echo Downloading the model weights - one time only...
"%PY%" -m lcg.local_t2v
if errorlevel 1 goto failed
echo ready> ".venv\local_video_ready.txt"
echo.
echo Local text-to-video is ready.
exit /b 0
:failed
echo.
echo Local video setup did not finish - see the error above. The app still works and makes
echo an image-sequence video instead. You can run setup_local_video.bat again at any time.
exit /b 1
