# 🚀 Advanced Stock Analysis Platform

A comprehensive web application that combines **Technical Analysis** and **Quantitative Trading Strategies** for stock market analysis. Built with Flask, Python, and modern web technologies.

![Stock Analysis Platform](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

### 📊 Technical Analysis
- **Interactive Stock Charts** with Plotly.js
- **Moving Averages** (50-day and 200-day)
- **Golden Cross & Death Cross** detection
- **Fibonacci Retracement** levels
- **RSI (Relative Strength Index)** analysis
- **P/E Ratios** (Trailing and Forward)
- **Market Cap** and volume analysis

### 🎯 Quantitative Trading Strategies
- **Time-Series Trend Following** - Identifies market trends using moving averages and MACD
- **Multi-Factor Equity Model** - Combines value, momentum, quality, and low-volatility factors
- **Cross-Sectional Momentum** - Analyzes relative strength and momentum indicators
- **Statistical Arbitrage** - Mean reversion and pairs trading opportunities
- **Machine Learning Alpha** - AI-driven signal generation using multiple features

### 🎨 User Interface
- **Modern Dark Theme** with gradient backgrounds
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Tabbed Navigation** - Switch between Technical and Quantitative analysis
- **Real-time Data** from Yahoo Finance
- **Interactive Charts** with hover tooltips
- **Signal Indicators** with confidence levels

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package installer)
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/stock-analysis-app.git
   cd stock-analysis-app
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   **On Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **On macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8080`

## 📋 Requirements

The following Python packages are required (automatically installed with `pip install -r requirements.txt`):

```
Flask==2.3.3
yfinance==0.2.65
pandas==2.0.3
numpy==1.24.3
talib==0.6.7
```

## 🚀 Usage

### Technical Analysis

1. **Navigate to the Technical Analysis tab**
2. **Enter a stock ticker** (e.g., AAPL, MSFT, GOOGL)
3. **Select date range** for analysis
4. **Click "Analyze Stock"**
5. **Toggle chart indicators** using checkboxes:
   - 50-Day Moving Average
   - 200-Day Moving Average
   - Golden Cross Signals
   - Death Cross Signals
   - Fibonacci Retracement

### Quantitative Strategies

1. **Navigate to the Quantitative Strategies tab**
2. **Enter a stock ticker** (e.g., AAPL, MSFT, GOOGL)
3. **Click "Run Quantitative Analysis"**
4. **Review the results:**
   - Stock information (price, change, volume)
   - 5 different strategy analyses
   - BUY/SELL/NEUTRAL signals with confidence levels
   - Detailed metrics for each strategy

## 📊 Strategy Explanations

### 1. Time-Series Trend Following
- **Purpose**: Identifies market trends using moving averages and MACD
- **Signals**: BUY when 20-day SMA > 50-day SMA and MACD histogram > 0
- **Use Case**: Trend-following traders

### 2. Multi-Factor Equity Model
- **Purpose**: Combines value, momentum, quality, and low-volatility factors
- **Signals**: Based on composite score from multiple factors
- **Use Case**: Long-term investors

### 3. Cross-Sectional Momentum
- **Purpose**: Analyzes relative strength and momentum indicators
- **Signals**: Based on RSI and relative strength scores
- **Use Case**: Momentum traders

### 4. Statistical Arbitrage
- **Purpose**: Mean reversion and pairs trading opportunities
- **Signals**: Based on z-scores and Bollinger Bands
- **Use Case**: Mean reversion traders

### 5. Machine Learning Alpha
- **Purpose**: AI-driven signal generation using multiple features
- **Signals**: Based on ensemble of technical indicators
- **Use Case**: Algorithmic traders

## 🔧 Configuration

### Environment Variables
You can customize the application using environment variables:

```bash
export PORT=8080                    # Server port (default: 8080)
export FLASK_ENV=development        # Development mode
export SECRET_KEY=your-secret-key   # Flask secret key
```

### Port Configuration
If port 8080 is already in use, you can change it by:

1. **Using environment variable:**
   ```bash
   export PORT=9000
   python app.py
   ```

2. **Or modify the code in `app.py`:**
   ```python
   port = int(os.environ.get('PORT', 9000))  # Change default port
   ```

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
# Kill existing processes
pkill -f "python app.py"

# Or use a different port
export PORT=9000
python app.py
```

**Python Command Not Found:**
```bash
# Make sure Python is installed and in PATH
python --version

# On some systems, use python3
python3 app.py
```

**Virtual Environment Issues:**
```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Yahoo Finance Data Issues:**
- Some tickers may not have data available
- Check if the ticker symbol is correct
- Try different tickers (AAPL, MSFT, GOOGL work well)

**TA-Lib Installation Issues:**
```bash
# On macOS with Homebrew
brew install ta-lib
pip install TA-Lib

# On Ubuntu/Debian
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

## 📁 Project Structure

```
stock-analysis-app/
├── app.py                          # Main Flask application
├── templates/
│   └── index.html                  # Main HTML template
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── venv/                          # Virtual environment (created after setup)
└── .gitignore                     # Git ignore file
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/new-feature
   ```
3. **Make your changes**
4. **Commit your changes**
   ```bash
   git commit -m "Add new feature"
   ```
5. **Push to the branch**
   ```bash
   git push origin feature/new-feature
   ```
6. **Open a Pull Request**

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free stock data
- **Plotly** for interactive charting capabilities
- **Flask** for the web framework
- **TA-Lib** for technical analysis indicators

## 📞 Support

If you encounter any issues or have questions:

1. **Check the troubleshooting section** above
2. **Search existing issues** on GitHub
3. **Create a new issue** with detailed information
4. **Include error messages** and system information

## 🔮 Future Enhancements

- [ ] Real-time data streaming
- [ ] More technical indicators
- [ ] Portfolio analysis
- [ ] Backtesting capabilities
- [ ] User authentication
- [ ] Data export features
- [ ] Mobile app version
- [ ] Advanced ML models

---

**Happy Trading! 📈**

*Disclaimer: This application is for educational and research purposes only. It is not intended as financial advice. Always do your own research before making investment decisions.*
