@echo off
setlocal

REM Go to this script's folder
cd /d "%~dp0"

REM Choose a Python launcher (py on most Windows, else python)
where py >nul 2>&1
if %ERRORLEVEL%==0 (set "PY=py") else (set "PY=python")

REM Create venv if missing
if not exist ".venv\Scripts\python.exe" (
  echo [Setup] Creating virtual environment...
  %PY% -m venv .venv
)

echo [Setup] Ensuring dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo [Run] Launching app...
REM Streamlit will open the browser automatically; if not, uncomment the next line:
REM start "" "http://localhost:8501"

".venv\Scripts\python.exe" -m streamlit run app.py

echo [Done] Close this window to exit.
endlocal
