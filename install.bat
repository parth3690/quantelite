@echo off
REM Stock Analysis Platform Installation Script for Windows

echo 🚀 Installing Stock Analysis Platform...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.9 or higher.
    pause
    exit /b 1
)

echo ✅ Python detected

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install Python dependencies
echo 📚 Installing Python dependencies...
pip install -r requirements.txt

REM Try to install TA-Lib
echo 📊 Installing TA-Lib...
pip install TA-Lib
if errorlevel 1 (
    echo ⚠️  Could not install TA-Lib automatically. Please install it manually.
    echo    The application will still work with basic functionality.
    echo    For Windows, you can download TA-Lib from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
)

echo.
echo 🎉 Installation completed successfully!
echo.
echo To run the application:
echo 1. Activate the virtual environment: venv\Scripts\activate
echo 2. Start the application: python app.py
echo 3. Open your browser to: http://localhost:8080
echo.
echo Happy trading! 📈
pause
