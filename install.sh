#!/bin/bash

# Stock Analysis Platform Installation Script
# This script will set up the application on macOS/Linux

echo "🚀 Installing Stock Analysis Platform..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python version $python_version is too old. Please install Python 3.9 or higher."
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install TA-Lib (if available)
echo "📊 Installing TA-Lib..."
if command -v brew &> /dev/null; then
    echo "🍺 Installing TA-Lib via Homebrew..."
    brew install ta-lib
    pip install TA-Lib
elif command -v apt-get &> /dev/null; then
    echo "📦 Installing TA-Lib via apt..."
    sudo apt-get update
    sudo apt-get install -y build-essential
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make
    sudo make install
    cd ..
    pip install TA-Lib
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
else
    echo "⚠️  Could not install TA-Lib automatically. Please install it manually."
    echo "   The application will still work with basic functionality."
fi

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "To run the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Start the application: python app.py"
echo "3. Open your browser to: http://localhost:8080"
echo ""
echo "Happy trading! 📈"
