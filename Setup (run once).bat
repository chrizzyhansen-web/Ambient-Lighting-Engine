@echo off
echo Setting up the Ambient Lighting Engine...
echo (This only needs to run once.)
echo.

py -3 --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found on this computer.
    echo Please install it from https://www.python.org/downloads/
    echo IMPORTANT: on the first install screen, check the box that says
    echo "Add python.exe to PATH" before clicking Install.
    echo.
    pause
    exit /b 1
)

echo Installing required pieces, this can take a minute...
py -3 -m pip install -r requirements.txt

echo.
echo Done! You can now double-click "Start App.bat" any time to launch it.
pause
