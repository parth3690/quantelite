from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import logging
import time
import warnings

# Suppress yfinance warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

class QuantStrategies:
    """Quantitative Trading Strategy Analyzer"""
    
    def __init__(self, data, ticker_info=None):
        self.data = data
        self.ticker_info = ticker_info or {}
        self.prices = data['Close'].values
        self.volumes = data['Volume'].values
        self.dates = data.index
        
    def calculate_sma(self, period):
        """Calculate Simple Moving Average"""
        return self.data['Close'].rolling(window=period, min_periods=1).mean()
    
    def calculate_ema(self, period):
        """Calculate Exponential Moving Average"""
        return self.data['Close'].ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, period=14):
        """Calculate RSI"""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 else 50
    
    def calculate_macd(self):
        """Calculate MACD"""
        ema12 = self.calculate_ema(12)
        ema26 = self.calculate_ema(26)
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            'macd': macd_line.iloc[-1] if len(macd_line) > 0 else 0,
            'signal': signal_line.iloc[-1] if len(signal_line) > 0 else 0,
            'histogram': histogram.iloc[-1] if len(histogram) > 0 else 0
        }
    
    def calculate_volatility(self, period=20):
        """Calculate volatility"""
        returns = self.data['Close'].pct_change().dropna()
        if len(returns) >= period:
            return returns.tail(period).std()
        return 0.02
    
    def trend_following_strategy(self):
        """Time-Series Trend Following Strategy"""
        try:
            if len(self.data) < 50:
                return {
                    'signal': 'NEUTRAL',
                    'confidence': 30,
                    'metrics': {
                        'Status': 'Insufficient data',
                        'Required': '50+ days',
                        'Available': f'{len(self.data)} days'
                    }
                }
            
            sma20 = self.calculate_sma(20)
            sma50 = self.calculate_sma(50)
            macd_data = self.calculate_macd()
            current_price = self.prices[-1]
            
            # Trend strength
            trend_strength = ((current_price - sma50.iloc[-1]) / sma50.iloc[-1]) * 100
            
            # Generate signal
            signal = 'NEUTRAL'
            confidence = 30
            
            if sma20.iloc[-1] > sma50.iloc[-1] and macd_data['histogram'] > 0:
                signal = 'BUY'
                confidence = min(90, 50 + abs(trend_strength) * 2)
            elif sma20.iloc[-1] < sma50.iloc[-1] and macd_data['histogram'] < 0:
                signal = 'SELL'
                confidence = min(90, 50 + abs(trend_strength) * 2)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'metrics': {
                    'SMA 20': f"${sma20.iloc[-1]:.2f}",
                    'SMA 50': f"${sma50.iloc[-1]:.2f}",
                    'MACD': f"{macd_data['macd']:.3f}",
                    'Trend Strength': f"{trend_strength:.2f}%"
                }
            }
        except Exception as e:
            logger.error(f"Error in trend following strategy: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0, 'metrics': {'Error': str(e)}}
    
    def multi_factor_strategy(self):
        """Multi-Factor Equity Strategy"""
        try:
            current_price = self.prices[-1]
            
            # Value Factor (using P/E and P/B ratios if available)
            pe_ratio = self.ticker_info.get('trailingPE', 20)
            pb_ratio = self.ticker_info.get('priceToBook', 3)
            value_factor = 1 if (pe_ratio and pe_ratio < 15) or (pb_ratio and pb_ratio < 2) else -1
            
            # Momentum Factor
            if len(self.data) >= 30:
                returns_30d = (current_price - self.prices[-30]) / self.prices[-30]
                momentum_factor = 1 if returns_30d > 0.05 else -1 if returns_30d < -0.05 else 0
            else:
                returns_30d = 0
                momentum_factor = 0
            
            # Quality Factor (using volume stability)
            avg_volume = np.mean(self.volumes)
            current_volume = self.volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            quality_factor = 1 if 0.8 < volume_ratio < 1.5 else -1
            
            # Low Volatility Factor
            volatility = self.calculate_volatility()
            low_vol_factor = 1 if volatility < 0.02 else -1 if volatility > 0.04 else 0
            
            # Composite score
            composite_score = (value_factor + momentum_factor * 2 + quality_factor + low_vol_factor) / 5
            
            signal = 'NEUTRAL'
            confidence = abs(composite_score) * 100
            
            if composite_score > 0.3:
                signal = 'BUY'
            elif composite_score < -0.3:
                signal = 'SELL'
            
            return {
                'signal': signal,
                'confidence': min(85, confidence),
                'metrics': {
                    'Value Score': f"{value_factor * 100:.0f}%",
                    'Momentum (30d)': f"{returns_30d * 100:.2f}%",
                    'Quality Score': f"{quality_factor * 100:.0f}%",
                    'Composite Score': f"{composite_score * 100:.1f}%"
                }
            }
        except Exception as e:
            logger.error(f"Error in multi-factor strategy: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0, 'metrics': {'Error': str(e)}}
    
    def momentum_strategy(self):
        """Cross-Sectional Momentum Strategy"""
        try:
            # Calculate returns for different periods
            returns = {}
            if len(self.data) >= 5:
                returns['1W'] = (self.prices[-1] - self.prices[-5]) / self.prices[-5]
            else:
                returns['1W'] = 0
                
            if len(self.data) >= 21:
                returns['1M'] = (self.prices[-1] - self.prices[-21]) / self.prices[-21]
            else:
                returns['1M'] = 0
                
            if len(self.data) >= 63:
                returns['3M'] = (self.prices[-1] - self.prices[-63]) / self.prices[-63]
            else:
                returns['3M'] = 0
            
            # RSI
            rsi = self.calculate_rsi()
            
            # Relative strength score
            rs_score = (returns['1W'] * 0.5 + returns['1M'] * 0.3 + returns['3M'] * 0.2) * 100
            
            signal = 'NEUTRAL'
            confidence = 50
            
            if rsi > 70 and rs_score > 10:
                signal = 'SELL'
                confidence = min(80, 50 + abs(rsi - 70))
            elif rsi < 30 and rs_score < -10:
                signal = 'BUY'
                confidence = min(80, 50 + abs(30 - rsi))
            elif rs_score > 5 and rsi > 50:
                signal = 'BUY'
                confidence = 60
            elif rs_score < -5 and rsi < 50:
                signal = 'SELL'
                confidence = 60
            
            return {
                'signal': signal,
                'confidence': confidence,
                'metrics': {
                    'RSI': f"{rsi:.2f}",
                    'RS Score': f"{rs_score:.2f}%",
                    '1W Return': f"{returns['1W'] * 100:.2f}%",
                    '1M Return': f"{returns['1M'] * 100:.2f}%"
                }
            }
        except Exception as e:
            logger.error(f"Error in momentum strategy: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0, 'metrics': {'Error': str(e)}}
    
    def stat_arb_strategy(self):
        """Statistical Arbitrage Strategy"""
        try:
            if len(self.data) < 20:
                return {
                    'signal': 'NEUTRAL',
                    'confidence': 30,
                    'metrics': {'Status': 'Insufficient data for analysis'}
                }
            
            # Calculate mean and standard deviation
            sma20 = self.calculate_sma(20)
            mean = sma20.iloc[-1]
            std_dev = self.data['Close'].tail(20).std()
            current_price = self.prices[-1]
            
            # Calculate z-score
            z_score = (current_price - mean) / std_dev if std_dev > 0 else 0
            
            # Bollinger Bands
            upper_band = mean + (std_dev * 2)
            lower_band = mean - (std_dev * 2)
            
            signal = 'NEUTRAL'
            confidence = abs(z_score) * 25
            
            if z_score < -2:
                signal = 'BUY'
                confidence = min(85, 50 + abs(z_score) * 15)
            elif z_score > 2:
                signal = 'SELL'
                confidence = min(85, 50 + abs(z_score) * 15)
            elif z_score < -1:
                signal = 'BUY'
                confidence = 40
            elif z_score > 1:
                signal = 'SELL'
                confidence = 40
            
            return {
                'signal': signal,
                'confidence': confidence,
                'metrics': {
                    'Z-Score': f"{z_score:.2f}",
                    'Upper Band': f"${upper_band:.2f}",
                    'Lower Band': f"${lower_band:.2f}",
                    'Mean Price': f"${mean:.2f}"
                }
            }
        except Exception as e:
            logger.error(f"Error in stat arb strategy: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0, 'metrics': {'Error': str(e)}}
    
    def ml_alpha_strategy(self):
        """Machine Learning Alpha Strategy (Simplified)"""
        try:
            # Get various technical indicators
            rsi = self.calculate_rsi()
            macd_data = self.calculate_macd()
            volatility = self.calculate_volatility()
            
            # Detect volume anomaly
            if len(self.volumes) >= 20:
                avg_volume = np.mean(self.volumes[-20:-1])
                current_volume = self.volumes[-1]
                volume_anomaly = current_volume > avg_volume * 1.5 if avg_volume > 0 else False
            else:
                volume_anomaly = False
            
            # Feature engineering
            features = {
                'momentum': (self.prices[-1] - self.prices[-5]) / self.prices[-5] if len(self.prices) >= 5 else 0,
                'rsi_signal': 1 if rsi < 30 else -1 if rsi > 70 else 0,
                'macd_signal': 1 if macd_data['histogram'] > 0 else -1,
                'vol_signal': 1 if volatility < 0.02 else -1 if volatility > 0.04 else 0,
                'volume_signal': 1 if volume_anomaly else 0
            }
            
            # Simplified ML prediction (weighted ensemble)
            ml_score = (
                features['momentum'] * 0.3 +
                features['rsi_signal'] * 0.25 +
                features['macd_signal'] * 0.2 +
                features['vol_signal'] * 0.15 +
                features['volume_signal'] * 0.1
            )
            
            signal = 'NEUTRAL'
            confidence = abs(ml_score) * 100
            
            if ml_score > 0.3:
                signal = 'BUY'
            elif ml_score < -0.3:
                signal = 'SELL'
            
            # Simulate sentiment (in real implementation, this would use NLP on news/social media)
            sentiment_score = np.random.uniform(-1, 1)
            sentiment = 'Positive' if sentiment_score > 0.3 else 'Negative' if sentiment_score < -0.3 else 'Neutral'
            
            return {
                'signal': signal,
                'confidence': min(75, confidence),
                'metrics': {
                    'ML Score': f"{ml_score * 100:.1f}%",
                    'Feature Count': '5 active',
                    'Sentiment': sentiment,
                    'Volume Anomaly': 'Detected' if volume_anomaly else 'Normal'
                }
            }
        except Exception as e:
            logger.error(f"Error in ML alpha strategy: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0, 'metrics': {'Error': str(e)}}

def format_large_number(num):
    """Format large numbers with appropriate suffixes"""
    if num == 'N/A' or not isinstance(num, (int, float)):
        return 'N/A'
    
    if num >= 1e12:
        return f"${num/1e12:.2f}T"
    elif num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"

def format_pe_ratio(ratio):
    """Format P/E ratio for display"""
    if ratio == 'N/A' or not isinstance(ratio, (int, float)):
        return 'N/A'
    return f"{ratio:.2f}"

def format_rsi(rsi):
    """Format RSI value for display"""
    if rsi == 'N/A' or pd.isna(rsi):
        return 'N/A'
    return f"{rsi:.2f}"

def calculate_rsi(data, period=14):
    """Calculate RSI manually if TA-Lib is not available"""
    try:
        import talib
        return talib.RSI(data['Close'].values, timeperiod=period)
    except:
        # Manual RSI calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

def fetch_stock_data_with_retry(ticker, start, end, max_retries=3):
    """Fetch stock data with retry logic and multiple methods"""
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1} to fetch data for {ticker}")
            
            # Method 1: Try with yfinance download function (more reliable)
            data = yf.download(
                ticker, 
                start=start, 
                end=end, 
                progress=False,
                auto_adjust=True,
                prepost=True,
                threads=False
            )
            
            if not data.empty:
                logger.info(f"Successfully fetched data using yf.download for {ticker}")
                # Create a simple Ticker object for info
                stock = yf.Ticker(ticker)
                return data, stock
            
            # Method 2: Try with Ticker object
            stock = yf.Ticker(ticker)
            data = stock.history(start=start, end=end, auto_adjust=True)
            
            if not data.empty:
                logger.info(f"Successfully fetched data using Ticker.history for {ticker}")
                return data, stock
            
            # If still empty, wait and retry
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                logger.warning(f"No data found, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            continue
    
    raise ValueError(f"Unable to fetch data for {ticker} after {max_retries} attempts. The ticker might be invalid or the service might be temporarily unavailable.")

def fetch_stock_data(ticker, start=None, end=None, period=None):
    """Main function to fetch stock data"""
    try:
        # Handle period-based fetching
        if period:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if data.empty:
                raise ValueError(f"No historical data found for {ticker}")
            
            # Ensure the index is a DatetimeIndex
            data.index = pd.to_datetime(data.index)
            
            # If multi-level columns, flatten them
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if col[0] != '' else col[1] for col in data.columns.values]
            
            # Calculate Moving Averages with sufficient data check
            if len(data) >= 50:
                data['50-DMA'] = data['Close'].rolling(window=50, min_periods=1).mean()
            else:
                data['50-DMA'] = data['Close'].rolling(window=len(data), min_periods=1).mean()
            
            if len(data) >= 200:
                data['200-DMA'] = data['Close'].rolling(window=200, min_periods=1).mean()
            else:
                data['200-DMA'] = data['Close'].rolling(window=len(data), min_periods=1).mean()
            
            # Calculate RSI
            if len(data) >= 14:
                data['RSI'] = calculate_rsi(data, 14)
            else:
                data['RSI'] = np.nan
            
            return data, stock
        
        # Handle date range fetching
        if not start or not end:
            raise ValueError("Either period or start/end dates must be provided")
            
        # Get the data with retry logic
        data, stock = fetch_stock_data_with_retry(ticker, start, end)
        
        if data.empty:
            raise ValueError(f"No historical data found for {ticker}")
        
        # Ensure the index is a DatetimeIndex
        data.index = pd.to_datetime(data.index)
        
        # If multi-level columns, flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[0] != '' else col[1] for col in data.columns.values]
        
        # Calculate Moving Averages with sufficient data check
        if len(data) >= 50:
            data['50-DMA'] = data['Close'].rolling(window=50, min_periods=1).mean()
        else:
            data['50-DMA'] = data['Close'].rolling(window=len(data), min_periods=1).mean()
        
        if len(data) >= 200:
            data['200-DMA'] = data['Close'].rolling(window=200, min_periods=1).mean()
        else:
            data['200-DMA'] = data['Close'].rolling(window=len(data), min_periods=1).mean()
        
        # Calculate RSI
        if len(data) >= 14:
            data['RSI'] = calculate_rsi(data, 14)
        else:
            data['RSI'] = np.nan
        
        return data, stock
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        raise

def identify_crosses(data):
    """Identify Golden Cross and Death Cross patterns"""
    try:
        # Ensure we have valid moving average data
        if '50-DMA' not in data or '200-DMA' not in data:
            return pd.DataFrame(), pd.DataFrame()
        
        # Create boolean masks for crosses
        golden_cross_mask = (
            (data['50-DMA'] > data['200-DMA']) & 
            (data['50-DMA'].shift(1) <= data['200-DMA'].shift(1))
        )
        death_cross_mask = (
            (data['50-DMA'] < data['200-DMA']) & 
            (data['50-DMA'].shift(1) >= data['200-DMA'].shift(1))
        )
        
        # Filter data based on masks
        golden_cross = data[golden_cross_mask]
        death_cross = data[death_cross_mask]
        
        return golden_cross, death_cross
        
    except Exception as e:
        logger.error(f"Error identifying crosses: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def calculate_fibonacci_levels(data):
    """Calculate Fibonacci retracement levels"""
    try:
        max_price = data['Close'].max()
        min_price = data['Close'].min()
        difference = max_price - min_price
        
        # Match the expected format in the HTML template
        fib_levels = {
            '0.0%': max_price,
            '23.6%': max_price - difference * 0.236,
            '38.2%': max_price - difference * 0.382,
            '50.0%': max_price - difference * 0.5,
            '61.8%': max_price - difference * 0.618,
            '100.0%': min_price
        }
        
        return fib_levels
        
    except Exception as e:
        logger.error(f"Error calculating Fibonacci levels: {str(e)}")
        return {}

def get_stock_info_safe(ticker):
    """Safely retrieve stock information with fallback"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Check if info is valid
        if not info or 'symbol' not in info:
            logger.warning(f"Limited info available for {ticker}")
            return {}
            
        return info
        
    except Exception as e:
        logger.warning(f"Could not retrieve full stock info for {ticker}: {str(e)}")
        return {}

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route for the stock analysis application"""
    if request.method == 'POST':
        try:
            # Get form data and sanitize ticker
            ticker = request.form.get('ticker', '').upper().strip().lstrip('$')
            start = request.form.get('start')
            end = request.form.get('end')
            
            # Validate inputs
            if not ticker:
                raise ValueError("Please enter a valid stock ticker symbol")
            
            if not start or not end:
                raise ValueError("Please select both start and end dates")
            
            # Convert date strings to datetime objects for validation
            start_date = datetime.strptime(start, '%Y-%m-%d')
            end_date = datetime.strptime(end, '%Y-%m-%d')
            
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
            
            if end_date > datetime.now():
                end_date = datetime.now()
                end = end_date.strftime('%Y-%m-%d')
                logger.info(f"Adjusted end date to today: {end}")
            
            # Fetch stock data
            logger.info(f"Fetching data for {ticker} from {start} to {end}")
            data, stock = fetch_stock_data(ticker, start, end)
            
            # Identify technical patterns
            golden_cross, death_cross = identify_crosses(data)
            
            # Calculate Fibonacci levels
            fib_levels = calculate_fibonacci_levels(data)
            
            # Prepare data for frontend plotting
            ticker_data = {
                'ticker': ticker,
                'dates': data.index.strftime('%Y-%m-%d').tolist(),
                'close_prices': data['Close'].round(2).tolist(),
                'dma50': data['50-DMA'].round(2).tolist() if '50-DMA' in data else [],
                'dma200': data['200-DMA'].round(2).tolist() if '200-DMA' in data else [],
                'golden_cross_dates': golden_cross.index.strftime('%Y-%m-%d').tolist() if not golden_cross.empty else [],
                'golden_cross_prices': golden_cross['Close'].round(2).tolist() if not golden_cross.empty else [],
                'death_cross_dates': death_cross.index.strftime('%Y-%m-%d').tolist() if not death_cross.empty else [],
                'death_cross_prices': death_cross['Close'].round(2).tolist() if not death_cross.empty else [],
                'fib_levels': fib_levels
            }
            
            # Get stock info separately to avoid blocking data fetch
            info = get_stock_info_safe(ticker)
            
            # Extract and format metrics with defaults
            trailing_pe_ratio = format_pe_ratio(info.get('trailingPE', 'N/A'))
            forward_pe_ratio = format_pe_ratio(info.get('forwardPE', 'N/A'))
            market_cap = format_large_number(info.get('marketCap', 'N/A'))
            
            # Get RSI value
            if 'RSI' in data and not data['RSI'].isnull().all():
                rsi_level = format_rsi(data['RSI'].iloc[-1])
            else:
                rsi_level = 'N/A'
            
            # Get last occurrence dates for crosses
            golden_cross_date = golden_cross.index[-1].strftime('%Y-%m-%d') if not golden_cross.empty else "N/A"
            death_cross_date = death_cross.index[-1].strftime('%Y-%m-%d') if not death_cross.empty else "N/A"
            
            # Log successful data retrieval
            logger.info(f"Successfully retrieved data for {ticker}")
            
            return render_template('index.html',
                                 ticker=ticker,
                                 trailing_pe_ratio=trailing_pe_ratio,
                                 forward_pe_ratio=forward_pe_ratio,
                                 market_cap=market_cap,
                                 rsi_level=rsi_level,
                                 golden_cross_date=golden_cross_date,
                                 death_cross_date=death_cross_date,
                                 ticker_data=ticker_data,
                                 start=start,
                                 end=end)
                                 
        except ValueError as ve:
            # Handle validation errors
            logger.warning(f"Validation error: {str(ve)}")
            return render_template('index.html', 
                                 error_message=str(ve),
                                 start=request.form.get('start', ''),
                                 end=request.form.get('end', ''))
                                 
        except Exception as e:
            # Handle other errors
            logger.error(f"Error processing request: {str(e)}")
            error_msg = f"Error fetching data for {ticker}: {str(e)}" if 'ticker' in locals() else str(e)
            
            # Provide helpful error message
            if "No timezone found" in str(e) or "No data found" in str(e):
                error_msg = f"Unable to fetch data for {ticker}. This could be due to: 1) Invalid ticker symbol, 2) Yahoo Finance service issues, or 3) The stock may be delisted. Please try again or use a different ticker."
            
            return render_template('index.html', 
                                 error_message=error_msg,
                                 start=request.form.get('start', ''),
                                 end=request.form.get('end', ''))
    else:
        # GET request - show empty form with default dates
        default_end = datetime.now().strftime('%Y-%m-%d')
        default_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        return render_template('index.html',
                             start=default_start,
                             end=default_end)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze stock with quantitative strategies"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper().strip()
        
        if not ticker:
            return jsonify({'error': 'Please enter a valid ticker symbol'}), 400
        
        # Fetch stock data
        stock_data, stock_info = fetch_stock_data(ticker, period="3mo")
        
        # Initialize strategy analyzer
        stock_info_dict = stock_info.info if hasattr(stock_info, 'info') else {}
        strategies = QuantStrategies(stock_data, stock_info_dict)
        
        # Get basic stock info
        current_price = stock_data['Close'].iloc[-1]
        prev_close = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
        change = current_price - prev_close
        change_percent = (change / prev_close * 100) if prev_close != 0 else 0
        
        
        # Run all strategies
        results = {
            'stockInfo': {
                'symbol': ticker,
                'name': stock_info_dict.get('longName', ticker),
                'price': round(current_price, 2),
                'change': round(change, 2),
                'changePercent': round(change_percent, 2),
                'volume': int(stock_data['Volume'].iloc[-1]),
                'marketCap': stock_info_dict.get('marketCap', 'N/A')
            },
            'strategies': [
                {
                    'name': '📈 Time-Series Trend Following',
                    'description': 'Identifies market trends using moving averages and MACD',
                    **strategies.trend_following_strategy()
                },
                {
                    'name': '⚖️ Multi-Factor Equity Model',
                    'description': 'Combines value, momentum, quality, and low-volatility factors',
                    **strategies.multi_factor_strategy()
                },
                {
                    'name': '🚀 Cross-Sectional Momentum',
                    'description': 'Analyzes relative strength and momentum indicators',
                    **strategies.momentum_strategy()
                },
                {
                    'name': '🎯 Statistical Arbitrage',
                    'description': 'Mean reversion and pairs trading opportunities',
                    **strategies.stat_arb_strategy()
                },
                {
                    'name': '🤖 Machine Learning Alpha',
                    'description': 'AI-driven signal generation using multiple features',
                    **strategies.ml_alpha_strategy()
                }
            ],
            'chartData': {
                'dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
                'prices': stock_data['Close'].round(2).tolist()
            }
        }
        
        return jsonify(results)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return jsonify({'error': f"Error analyzing {ticker}: {str(e)}"}), 500

@app.route('/test')
def test():
    """Test route to verify server is running"""
    return jsonify({'status': 'Server is running', 'timestamp': datetime.now().isoformat()})

@app.route('/quant-analyzer')
def quant_analyzer():
    """Serve the standalone quantitative strategy analyzer"""
    return app.send_static_file('quant-strategy-analyzer.html')

@app.route('/test-ticker/<ticker>')
def test_ticker(ticker):
    """Test endpoint to check if a ticker is valid"""
    try:
        ticker = ticker.upper()
        # Try simple download first
        data = yf.download(ticker, period="1d", progress=False)
        
        if data.empty:
            return jsonify({
                'success': False,
                'message': f'{ticker} returned no data'
            })
        
        return jsonify({
            'success': True,
            'message': f'{ticker} is valid',
            'last_price': float(data['Close'].iloc[-1]) if 'Close' in data else 'N/A'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('index.html', 
                         error_message="Page not found. Please return to the main page."), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}")
    return render_template('index.html', 
                         error_message="An internal error occurred. Please try again later."), 500

if __name__ == '__main__':
    # Ensure templates folder exists
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        logger.info(f"Created templates directory at {templates_dir}")
    
    # Get port from environment or use default (avoid macOS AirPlay conflict)
    port = int(os.environ.get('PORT', 8080))
    
    # Development vs Production settings
    is_development = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    if is_development:
        logger.info(f"Starting Flask app in development mode on port {port}")
        logger.info("Visit http://localhost:5000 to access the application")
        app.run(host='127.0.0.1', port=port, debug=True)
    else:
        logger.info(f"Starting Flask app in production mode on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)