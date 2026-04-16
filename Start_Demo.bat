@echo off
echo [*] Checking for Python Environment...

:: If the venv folder doesn't exist, create it and install packages
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [*] Building portable virtual environment - This takes 1-2 minutes...
    python -m venv venv
    call venv\Scripts\activate
    echo [*] Installing required packages...
    pip install -r requirements.txt
) ELSE (
    call venv\Scripts\activate
)

echo [*] Starting the V4 Dashboard Backend on Port 5001...
:: FIX: Move into the dashboard folder to launch Flask, then step back out
cd dashboard
start /B python app_v4.py
cd ..

echo [*] Launching Dashboard in Browser...
timeout /t 3 /nobreak >nul
start http://127.0.0.1:5001

echo [*] Environment Ready.
echo [*] When you are ready to begin the simulation, type:
echo [*] python test_pcap_v4.py
cmd /k