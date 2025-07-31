import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import lightgbm as lgb
import pandas_ta as ta
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from binance.client import Client
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from flask import Flask
from threading import Thread
import warnings
warnings.filterwarnings('ignore')

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_ml_trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EnhancedMLTrading')

# إضافة handler منفصل للأخطاء
error_handler = logging.FileHandler('enhanced_ml_trading_errors.log', encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(error_handler)

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    TELEGRAM_TOKEN: Optional[str] = config('TELEGRAM_BOT_TOKEN', default=None)
    CHAT_ID: Optional[str] = config('TELEGRAM_CHAT_ID', default=None)
except Exception as e:
    logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
    exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
class Config:
    # Data Parameters
    DATA_LOOKBACK_DAYS = 180
    SIGNAL_TIMEFRAME = '5m'
    SYMBOLS_FILE = 'crypto_list.txt'
    MAX_SYMBOLS = 50
    
    # Target Parameters
    TARGET_PERIODS = 5
    TARGET_CHANGE_THRESHOLD = 0.01
    
    # Feature Engineering
    VOLUME_LOOKBACK = 5
    RSI_PERIOD = 14
    ADX_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_WINDOW = 20
    VWAP_WINDOW = 15
    ATR_WINDOW = 14
    
    # SMC Parameters
    LIQUIDITY_WINDOW = 20
    SWING_WINDOW = 5
    ORDER_BLOCK_VOLUME_MULTIPLIER = 1.5
    FVG_STRENGTH_THRESHOLD = 0.003
    
    # Triple-Barrier Method Parameters
    TP_ATR_MULTIPLIER: float = 2.0
    SL_ATR_MULTIPLIER: float = 1.5
    MAX_HOLD_PERIOD: int = 24
    
    # Model Training
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_STATE = 42
    EARLY_STOPPING_ROUNDS = 50
    CV_FOLDS = 5
    N_JOBS = -1
    
    # Risk Management
    MAX_POSITION_SIZE = 0.1
    STOP_LOSS_PCT = 0.02
    TAKE_PROFIT_PCT = 0.03
    
    # API Configuration
    FLASK_PORT = 10000
    TELEGRAM_TIMEOUT = 30
    BINANCE_RETRIES = 3
    BINANCE_RATE_LIMIT_DELAY = 1

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
btc_data_cache: Optional[pd.DataFrame] = None
current_models = {}
training_status = "Idle"
last_training_time = None
training_metrics = {}
symbols = []

# ---------------------- Database & API Setup ----------------------
class DatabaseManager:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.conn = None
        self.cur = None
        self._connect()
    
    def _connect(self, retries=5, delay=5):
        for attempt in range(retries):
            try:
                self.conn = psycopg2.connect(
                    DB_URL,
                    cursor_factory=RealDictCursor,
                    connect_timeout=10
                )
                self.conn.autocommit = False
                self.cur = self.conn.cursor()
                self._create_tables()
                logger.info("✅ Database connection established")
                return True
            except OperationalError as e:
                logger.error(f"❌ Connection attempt {attempt+1} failed: {e}")
                time.sleep(delay)
            except Exception as e:
                logger.error(f"❌ Database initialization error: {e}")
                raise
        logger.critical("❌ Failed to connect to database after retries")
        return False
    
    def _create_tables(self):
        tables = [
            """CREATE TABLE IF NOT EXISTS enhanced_signals (
                id SERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                prediction INT NOT NULL,
                confidence FLOAT NOT NULL,
                features JSONB NOT NULL,
                model_version TEXT NOT NULL,
                processed BOOLEAN DEFAULT FALSE,
                entry_price FLOAT,
                target_price FLOAT,
                stop_loss_price FLOAT,
                closed BOOLEAN DEFAULT FALSE,
                profit_loss FLOAT,
                closed_at TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )""",
            """CREATE TABLE IF NOT EXISTS enhanced_models (
                id SERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_data BYTEA NOT NULL,
                metrics JSONB NOT NULL,
                feature_importance JSONB,
                trained_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(symbol, model_name)
            )""",
            """CREATE TABLE IF NOT EXISTS training_metadata (
                id SERIAL PRIMARY KEY,
                training_date TIMESTAMP DEFAULT NOW(),
                symbols TEXT[] NOT NULL,
                model_types TEXT[] NOT NULL,
                avg_metrics JSONB NOT NULL,
                duration_seconds FLOAT NOT NULL,
                parameters JSONB NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS market_conditions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT NOW(),
                btc_dominance FLOAT,
                total_market_cap FLOAT,
                fear_greed_index FLOAT,
                volume_24h FLOAT,
                trending_coins JSONB
            )"""
        ]
        
        try:
            for table in tables:
                self.cur.execute(table)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Failed to create tables: {e}")
            raise
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = False):
        try:
            if not self.conn or self.conn.closed:
                self._connect()
                
            with self._lock:
                self.cur.execute(query, params)
                if fetch:
                    return self.cur.fetchall()
                self.conn.commit()
                return True
        except (OperationalError, InterfaceError) as e:
            logger.error(f"Database connection error: {e}")
            self._connect()  # Attempt to reconnect
            return False
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Query failed: {e}\nQuery: {query}\nParams: {params}")
            return False
    
    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("✅ Database connection closed")

# ---------------------- Enhanced Data Fetching ----------------------
class BinanceDataFetcher:
    def __init__(self):
        self.client = Client(
            API_KEY,
            API_SECRET,
            {"timeout": 20}
        )
        self.rate_limit_queue = Queue()
        self.last_request_time = 0
    
    def _rate_limit(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < Config.BINANCE_RATE_LIMIT_DELAY:
            time.sleep(Config.BINANCE_RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid on Binance"""
        try:
            self._rate_limit()
            self.client.get_symbol_info(symbol)
            return True
        except BinanceAPIException as e:
            if e.code == -1121:  # Invalid symbol
                return False
            raise
    
    def fetch_klines(self, symbol: str, interval: str, start_str: str, end_str: str = None):
        self._rate_limit()
        for attempt in range(Config.BINANCE_RETRIES):
            try:
                return self.client.get_historical_klines(
                    symbol, interval, start_str, end_str
                )
            except BinanceAPIException as e:
                if e.status_code == 429:  # Rate limit
                    wait_time = min(2 ** attempt, 10)
                    logger.warning(f"⚠️ Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise
            except Exception as e:
                logger.error(f"❌ Fetch error for {symbol}: {e}")
                return None
        return None
    
    def fetch_enhanced_data(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        if not self.validate_symbol(symbol):
            logger.warning(f"⚠️ Invalid symbol: {symbol}")
            return None
            
        start_date = datetime.utcnow() - timedelta(days=days)
        start_str = start_date.strftime("%d %b %Y %H:%M:%S")
        
        klines = self.fetch_klines(symbol, interval, start_str)
        if not klines:
            return None
            
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
            'taker_buy_quote', 'ignore'
        ])
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        df = df.set_index('timestamp')[numeric_cols]
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        df = df.dropna(subset=['close'])
        
        return df

# ---------------------- Feature Engineering ----------------------
class FeatureEngineer:
    @staticmethod
    def add_smc_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add Smart Money Concept features"""
        df = df.copy()
        
        # Liquidity Pools
        df['high_range'] = df['high'].rolling(Config.LIQUIDITY_WINDOW).max()
        df['low_range'] = df['low'].rolling(Config.LIQUIDITY_WINDOW).min()
        df['liquidity_pool_high'] = ((df['high'] >= df['high_range'] * 0.998) & 
                                    (df['high'] == df['high'].rolling(3, center=True).max())).astype(int)
        df['liquidity_pool_low'] = ((df['low'] <= df['low_range'] * 1.002) & 
                                   (df['low'] == df['low'].rolling(3, center=True).min())).astype(int)
        
        # Order Blocks
        volume_threshold = df['volume'].rolling(5).mean() * Config.ORDER_BLOCK_VOLUME_MULTIPLIER
        df['bullish_ob'] = 0
        df['bearish_ob'] = 0
        
        for i in range(1, len(df)-1):
            prev_candle = df.iloc[i-1]
            current_candle = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Bullish Order Block
            if (prev_candle['close'] < prev_candle['open'] and
                current_candle['close'] > current_candle['open'] and
                current_candle['volume'] > volume_threshold.iloc[i] and
                current_candle['close'] > prev_candle['close'] and
                next_candle['close'] > current_candle['close']):
                df.loc[df.index[i], 'bullish_ob'] = 1
            
            # Bearish Order Block
            if (prev_candle['close'] > prev_candle['open'] and
                current_candle['close'] < current_candle['open'] and
                current_candle['volume'] > volume_threshold.iloc[i] and
                current_candle['close'] < prev_candle['close'] and
                next_candle['close'] < current_candle['close']):
                df.loc[df.index[i], 'bearish_ob'] = 1
        
        # Fair Value Gaps
        df['bullish_fvg'] = 0
        df['bearish_fvg'] = 0
        
        for i in range(1, len(df)):
            prev_high = df.iloc[i-1]['high']
            prev_low = df.iloc[i-1]['low']
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            if (current_low - prev_high) > (prev_high * Config.FVG_STRENGTH_THRESHOLD):
                df.loc[df.index[i], 'bullish_fvg'] = 1
            elif (prev_low - current_high) > (prev_low * Config.FVG_STRENGTH_THRESHOLD):
                df.loc[df.index[i], 'bearish_fvg'] = 1
        
        # Break of Structure (BOS) and Change of Character (CHOCH)
        df['bullish_bos'] = 0
        df['bearish_bos'] = 0
        df['bullish_choch'] = 0
        df['bearish_choch'] = 0
        
        if len(df) > Config.SWING_WINDOW + 2:
            df['swing_high'] = df['high'].rolling(Config.SWING_WINDOW, center=True).max()
            df['swing_low'] = df['low'].rolling(Config.SWING_WINDOW, center=True).min()
            
            for i in range(Config.SWING_WINDOW, len(df)-1):
                current_high = df.iloc[i]['high']
                current_low = df.iloc[i]['low']
                prev_swing_high = df.iloc[i-Config.SWING_WINDOW]['swing_high']
                prev_swing_low = df.iloc[i-Config.SWING_WINDOW]['swing_low']
                
                if current_high > prev_swing_high:
                    df.loc[df.index[i], 'bullish_bos'] = 1
                if current_low < prev_swing_low:
                    df.loc[df.index[i], 'bearish_bos'] = 1
                if (df.iloc[i]['close'] > df.iloc[i]['open'] and
                    current_low > prev_swing_low and
                    df.iloc[i-1]['close'] < df.iloc[i-1]['open'] and
                    df.iloc[i-1]['low'] < prev_swing_low):
                    df.loc[df.index[i], 'bullish_choch'] = 1
                if (df.iloc[i]['close'] < df.iloc[i]['open'] and
                    current_high < prev_swing_high and
                    df.iloc[i-1]['close'] > df.iloc[i-1]['open'] and
                    df.iloc[i-1]['high'] > prev_swing_high):
                    df.loc[df.index[i], 'bearish_choch'] = 1
        
        return df

    @staticmethod
    def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        if len(df) < 50:
            logger.warning("Insufficient data for TA features")
            return df
            
        df = df.copy()
        
        try:
            # Price Transformations
            df['log_price'] = np.log(df['close'])
            df['returns'] = df['close'].pct_change()
            df['cumulative_returns'] = (1 + df['returns']).cumprod()
            
            # Momentum Indicators
            df['rsi'] = ta.rsi(df['close'], length=Config.RSI_PERIOD)
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
            df['tsi'] = ta.tsi(df['close'], 25, 13)['TSI_25_13']
            df['ultimate_osc'] = ta.uo(df['high'], df['low'], df['close'], 7, 14, 28)['UO_7_14_28']
            df['roc'] = ta.roc(df['close'], 12)['ROC_12']
            
            # Trend Indicators
            macd = ta.macd(df['close'], fast=Config.MACD_FAST, slow=Config.MACD_SLOW, signal=Config.MACD_SIGNAL)
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_diff'] = macd['MACDh_12_26_9']
            
            df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=Config.ADX_PERIOD)['ADX_14']
            df['plus_di'] = ta.adx(df['high'], df['low'], df['close'], length=Config.ADX_PERIOD)['DMP_14']
            df['minus_di'] = ta.adx(df['high'], df['low'], df['close'], length=Config.ADX_PERIOD)['DMN_14']
            
            df['ema_20'] = ta.ema(df['close'], length=20)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['ema_200'] = ta.ema(df['close'], length=200)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['sma_200'] = ta.sma(df['close'], length=200)
            
            ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
            df['ichimoku_conv'] = ichimoku['ITS_9']
            df['ichimoku_base'] = ichimoku['IKS_26']
            df['ichimoku_a'] = ichimoku['ISA_9']
            df['ichimoku_b'] = ichimoku['ISB_26']
            
            aroon = ta.aroon(df['high'], df['low'], length=14)
            df['aroon_up'] = aroon['AROONU_14']
            df['aroon_down'] = aroon['AROOND_14']
            
            # Volume Indicators
            df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'], length=Config.VWAP_WINDOW)['VWAP_D']
            df['adi'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])['AD']
            df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)['CMF_20']
            df['eom'] = ta.eom(df['high'], df['low'], df['close'], df['volume'], length=14)['EOM_14']
            
            # Volatility Indicators
            bb = ta.bbands(df['close'], length=Config.BB_WINDOW)
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=Config.ATR_WINDOW)['ATRr_14']
            
            dc = ta.donchian(df['high'], df['low'], lower_length=20, upper_length=20)
            df['dc_upper'] = dc['DCU_20_20']
            df['dc_lower'] = dc['DCL_20_20']
            
            kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=2)
            df['kc_upper'] = kc['KCUe_20_2']
            df['kc_middle'] = kc['KCLe_20_2']
            df['kc_lower'] = kc['KCL_20_2']
            
            # Other Indicators
            df['daily_log_return'] = ta.percent_return(df['close'], cumulative=False)['PCTRET_1']
            df['daily_return'] = ta.percent_return(df['close'], cumulative=False)['PCTRET_1']
            df['cumulative_return'] = ta.percent_return(df['close'], cumulative=True)['CUMRET_1']
            
            # Derived Features
            df['price_vwap_diff'] = df['close'] - df['vwap']
            df['ema_20_50_diff'] = df['ema_20'] - df['ema_50']
            df['ema_50_200_diff'] = df['ema_50'] - df['ema_200']
            df['high_low_spread'] = df['high'] - df['low']
            df['open_close_spread'] = df['close'] - df['open']
            df['volume_spike'] = (df['volume'] / df['volume'].rolling(20).mean() - 1)
            
            # Lag Features
            for lag in [1, 2, 3, 5, 10, 20]:
                df[f'price_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            
            # Rolling Features
            windows = [5, 10, 20, 50]
            for window in windows:
                df[f'rolling_mean_{window}'] = df['close'].rolling(window).mean()
                df[f'rolling_std_{window}'] = df['close'].rolling(window).std()
                df[f'rolling_min_{window}'] = df['low'].rolling(window).min()
                df[f'rolling_max_{window}'] = df['high'].rolling(window).max()
                df[f'rolling_volume_mean_{window}'] = df['volume'].rolling(window).mean()
                df[f'rolling_volume_std_{window}'] = df['volume'].rolling(window).std()
            
            # Window-based Features
            df['z_score_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['volatility_20'] = df['returns'].rolling(20).std()
            
            # Date/Time Features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['is_weekend'] = df.index.dayofweek >= 5
            
            return df
        except Exception as e:
            logger.error(f"Error calculating TA features: {e}")
            return df

    @staticmethod
    def add_market_context(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
        """Add Bitcoin dominance and market context features"""
        if btc_df is None or len(btc_df) < 20:
            logger.warning("Insufficient BTC data for market context")
            return df
            
        btc_df = btc_df.copy()
        btc_df['btc_returns'] = btc_df['close'].pct_change()
        btc_df['btc_volatility'] = btc_df['btc_returns'].rolling(20).std()
        
        # Merge BTC features
        merged = df.join(btc_df[['close', 'btc_returns', 'btc_volatility']], 
                         rsuffix='_btc', how='left')
        merged['price_btc_ratio'] = merged['close'] / merged['close_btc']
        
        # Calculate correlation with BTC
        rolling_corr = df['close'].rolling(20).corr(btc_df['close'])
        merged['btc_correlation_20'] = rolling_corr
        
        return merged

# ---------------------- Target Engineering ----------------------
class TargetGenerator:
    @staticmethod
    def create_multiclass_target(df: pd.DataFrame, periods: int = Config.TARGET_PERIODS) -> pd.Series:
        """Create 3-class target: 0 (neutral), 1 (up), 2 (down)"""
        future_max = df['close'].shift(-periods).rolling(periods).max()
        future_min = df['close'].shift(-periods).rolling(periods).min()
        
        target = pd.Series(0, index=df.index)  # Default to neutral
        target[(future_max / df['close'] - 1) > Config.TARGET_CHANGE_THRESHOLD] = 1  # Up
        target[(1 - future_min / df['close']) > Config.TARGET_CHANGE_THRESHOLD] = 2  # Down
        
        return target

    @staticmethod
    def create_regression_target(df: pd.DataFrame, periods: int = Config.TARGET_PERIODS) -> pd.Series:
        """Create continuous target for regression"""
        future_price = df['close'].shift(-periods).rolling(periods).mean()
        return (future_price / df['close'] - 1)  # Percentage change

# ---------------------- Feature Selection ----------------------
class FeatureSelector:
    @staticmethod
    def get_feature_groups() -> Dict[str, List[str]]:
        """Define feature groups for selective inclusion"""
        return {
            'price': ['open', 'high', 'low', 'close', 'log_price'],
            'volume': ['volume', 'volume_spike', 'quote_volume'],
            'returns': ['returns', 'cumulative_returns', 'daily_log_return'],
            'smc': [
                'liquidity_pool_high', 'liquidity_pool_low',
                'bullish_ob', 'bearish_ob', 'bullish_fvg', 'bearish_fvg',
                'bullish_bos', 'bearish_bos', 'bullish_choch', 'bearish_choch'
            ],
            'momentum': [
                'rsi', 'stoch_k', 'stoch_d', 'tsi', 'ultimate_osc', 'roc',
                'macd', 'macd_signal', 'macd_diff'
            ],
            'trend': [
                'adx', 'plus_di', 'minus_di', 'ema_20', 'ema_50', 'ema_200',
                'sma_50', 'sma_200', 'ichimoku_conv', 'ichimoku_base',
                'ichimoku_a', 'ichimoku_b', 'aroon_up', 'aroon_down'
            ],
            'volatility': [
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct',
                'atr', 'dc_upper', 'dc_lower', 'kc_upper', 'kc_middle', 'kc_lower'
            ],
            'volume_ta': ['vwap', 'adi', 'cmf', 'eom'],
            'derived': [
                'price_vwap_diff', 'ema_20_50_diff', 'ema_50_200_diff',
                'high_low_spread', 'open_close_spread', 'z_score_20',
                'momentum_10', 'volatility_20'
            ],
            'market_context': [
                'close_btc', 'btc_returns', 'btc_volatility',
                'price_btc_ratio', 'btc_correlation_20'
            ],
            'time': ['hour', 'day_of_week', 'day_of_month', 'is_weekend']
        }

    @staticmethod
    def select_features(df: pd.DataFrame, groups: List[str] = None) -> pd.DataFrame:
        """Select features based on specified groups"""
        if groups is None:
            groups = ['price', 'volume', 'smc', 'momentum', 'trend']
            
        feature_groups = FeatureSelector.get_feature_groups()
        selected_features = []
        
        for group in groups:
            if group in feature_groups:
                selected_features.extend(feature_groups[group])
        
        # Ensure features exist in dataframe
        available_features = [f for f in selected_features if f in df.columns]
        return df[available_features].copy()

# ---------------------- Data Preprocessing ----------------------
class DataPreprocessor:
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """Handle missing values with specified strategy"""
        if strategy == 'drop':
            return df.dropna()
        elif strategy == 'ffill':
            return df.ffill().bfill()
        elif strategy == 'interpolate':
            return df.interpolate().bfill()
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")

    @staticmethod
    def remove_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using Z-score method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        z_scores = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        return df[(z_scores.abs() < threshold).all(axis=1)]

    @staticmethod
    def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      scaler_type: str = 'robust') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features using specified scaler"""
        if scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal')
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return (
            pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index),
            pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        )

# ---------------------- Model Training ----------------------
class ModelTrainer:
    @staticmethod
    def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame = None, y_val: pd.Series = None,
                      params: Dict = None) -> lgb.LGBMClassifier:
        """Train LightGBM model with optional validation set"""
        default_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': Config.RANDOM_STATE,
            'n_jobs': Config.N_JOBS
        }
        
        if params:
            default_params.update(params)
        
        model = lgb.LGBMClassifier(**default_params)
        
        if X_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
                verbose=50
            )
        else:
            model.fit(X_train, y_train)
        
        return model

    @staticmethod
    def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame = None, y_val: pd.Series = None,
                     params: Dict = None) -> xgb.XGBClassifier:
        """Train XGBoost model with optional validation set"""
        default_params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
            'random_state': Config.RANDOM_STATE,
            'n_jobs': Config.N_JOBS
        }
        
        if params:
            default_params.update(params)
        
        eval_set = [(X_val, y_val)] if X_val is not None else None
        early_stopping = Config.EARLY_STOPPING_ROUNDS if eval_set else None
        
        model = xgb.XGBClassifier(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping,
            verbose=50
        )
        
        return model

    @staticmethod
    def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                           params: Dict = None) -> RandomForestClassifier:
        """Train Random Forest model"""
        default_params = {
            'n_estimators': 500,
            'max_depth': 10,
            'min_samples_split': 5,
            'class_weight': 'balanced',
            'random_state': Config.RANDOM_STATE,
            'n_jobs': Config.N_JOBS,
            'verbose': 1
        }
        
        if params:
            default_params.update(params)
        
        model = RandomForestClassifier(**default_params)
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def train_balanced_rf(X_train: pd.DataFrame, y_train: pd.Series,
                         params: Dict = None) -> BalancedRandomForestClassifier:
        """Train Balanced Random Forest (handles class imbalance)"""
        default_params = {
            'n_estimators': 500,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': Config.RANDOM_STATE,
            'n_jobs': Config.N_JOBS,
            'verbose': 1
        }
        
        if params:
            default_params.update(params)
        
        model = BalancedRandomForestClassifier(**default_params)
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def create_ensemble(models: Dict[str, Any], voting: str = 'soft') -> VotingClassifier:
        """Create voting ensemble of trained models"""
        estimators = [(name, model) for name, model in models.items()]
        return VotingClassifier(estimators=estimators, voting=voting, n_jobs=Config.N_JOBS)

    @staticmethod
    def create_stacking_ensemble(base_models: Dict[str, Any], 
                                meta_model: Any = None) -> StackingClassifier:
        """Create stacking ensemble"""
        if meta_model is None:
            meta_model = LogisticRegression()
            
        return StackingClassifier(
            estimators=list(base_models.items()),
            final_estimator=meta_model,
            n_jobs=Config.N_JOBS
        )

# ---------------------- Model Evaluation ----------------------
class ModelEvaluator:
    @staticmethod
    def replace_nan_with_none(metrics):
        """Replace NaN values with None in metrics dictionary"""
        return {k: (None if isinstance(v, float) and math.isnan(v) else v) 
                for k, v in metrics.items()}

    @staticmethod
    def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'roc_auc': roc_auc_score(y_test, y_proba, multi_class='ovr') if y_proba is not None else None,
            'pr_auc': average_precision_score(y_test, y_proba[:,1]) if y_proba is not None and len(np.unique(y_test)) == 2 else None,
            'probabilities': y_proba.tolist() if y_proba is not None else None
        }
        
        return ModelEvaluator.replace_nan_with_none(metrics)

    @staticmethod
    def cross_validate(model: Any, X: pd.DataFrame, y: pd.Series, 
                       cv: int = Config.CV_FOLDS) -> Dict[str, Any]:
        """Cross-validate model performance"""
        tscv = TimeSeriesSplit(n_splits=cv)
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred, average='weighted'))
            scores['recall'].append(recall_score(y_test, y_pred, average='weighted'))
            scores['f1'].append(f1_score(y_test, y_pred, average='weighted'))
        
        return {k: np.nanmean(v) for k, v in scores.items()}

    @staticmethod
    def calculate_feature_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance for various model types"""
        if isinstance(model, lgb.Booster):
            importance = model.feature_importance(importance_type='gain')
            return dict(zip(feature_names, importance))
        elif hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        else:
            logger.warning("Feature importance not available for this model type")
            return {}

    @staticmethod
    def analyze_shap_values(model: Any, X: pd.DataFrame) -> Dict[str, Any]:
        """Calculate SHAP values for model interpretation"""
        try:
            if isinstance(model, lgb.Booster):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model)
                
            shap_values = explainer(X)
            return {
                'shap_values': shap_values.values.tolist(),
                'base_values': shap_values.base_values.tolist(),
                'data': X.values.tolist(),
                'feature_names': X.columns.tolist()
            }
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return {}

# ---------------------- Main Training Pipeline ----------------------
class TradingModelPipeline:
    def __init__(self):
        self.db = DatabaseManager()
        self.data_fetcher = BinanceDataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.target_generator = TargetGenerator()
        self.feature_selector = FeatureSelector()
        self.preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        self.current_models = {}
        self.training_status = "Idle"
        self.last_training_time = None
        self.training_metrics = {}
        self.symbols = []

    def load_symbols(self) -> List[str]:
        """Load and validate symbols from configuration file"""
        try:
            with open(Config.SYMBOLS_FILE, 'r') as f:
                raw_symbols = [line.strip().upper() for line in f 
                             if line.strip() and not line.startswith('#')]
                
            # التحقق من صحة الرموز مع Binance
            valid_symbols = []
            invalid_symbols = []
            
            for symbol in raw_symbols[:Config.MAX_SYMBOLS]:
                try:
                    # إضافة USDT افتراضياً إذا لم يكن موجوداً
                    full_symbol = symbol if 'USDT' in symbol else f"{symbol}USDT"
                    if self.data_fetcher.validate_symbol(full_symbol):
                        valid_symbols.append(full_symbol)
                    else:
                        invalid_symbols.append(symbol)
                except Exception as e:
                    logger.warning(f"Error validating symbol {symbol}: {e}")
                    invalid_symbols.append(symbol)
            
            if invalid_symbols:
                logger.warning(f"Removed invalid symbols: {invalid_symbols}")
            
            self.symbols = valid_symbols
            return valid_symbols
            
        except Exception as e:
            logger.error(f"Failed to load symbols: {e}")
            return []

    def prepare_data(self, symbol: str, btc_data: pd.DataFrame = None) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """Prepare data for a single symbol with enhanced error handling"""
        try:
            logger.info(f"📊 Preparing data for {symbol}")
            
            # 1. Fetch data with retries
            df = self.data_fetcher.fetch_enhanced_data(symbol, Config.SIGNAL_TIMEFRAME, Config.DATA_LOOKBACK_DAYS)
            if df is None or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # 2. Feature engineering
            df = self.feature_engineer.add_smc_features(df)
            df = self.feature_engineer.add_ta_features(df)
            
            if btc_data is not None:
                df = self.feature_engineer.add_market_context(df, btc_data)
            
            # 3. Target generation
            target = self.target_generator.create_multiclass_target(df)
            if target.nunique() < 2:
                logger.warning(f"Insufficient target classes for {symbol}")
                return None
            
            # 4. Feature selection
            features = self.feature_selector.select_features(df)
            
            # 5. Handle missing values and outliers
            features = self.preprocessor.handle_missing_values(features, strategy='interpolate')
            features = self.preprocessor.remove_outliers(features)
            target = target[features.index]
            
            return features, target
            
        except Exception as e:
            logger.error(f"Error preparing data for {symbol}: {e}")
            return None

    def train_symbol_model(self, symbol: str, btc_data: pd.DataFrame = None) -> Optional[Dict[str, Any]]:
        """Complete training pipeline for a single symbol with enhanced robustness"""
        try:
            logger.info(f"🚀 Starting training for {symbol}")
            self.training_status = f"Training {symbol}"
            
            # 1. Prepare data
            prepared_data = self.prepare_data(symbol, btc_data)
            if prepared_data is None:
                return None
                
            features, target = prepared_data
            
            # 2. Train-test split with time series awareness
            split_idx = int(len(features) * (1 - Config.TRAIN_TEST_SPLIT))
            X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
            
            # 3. Handle class imbalance
            smote = SMOTE(random_state=Config.RANDOM_STATE)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            
            # 4. Feature scaling
            X_train_scaled, X_test_scaled = self.preprocessor.scale_features(
                X_train_res, X_test, scaler_type='robust'
            )
            
            # 5. Train models with error handling
            models = {}
            try:
                models['lightgbm'] = self.model_trainer.train_lightgbm(X_train_scaled, y_train_res)
            except Exception as e:
                logger.error(f"Failed to train LightGBM for {symbol}: {e}")
            
            try:
                models['xgboost'] = self.model_trainer.train_xgboost(X_train_scaled, y_train_res)
            except Exception as e:
                logger.error(f"Failed to train XGBoost for {symbol}: {e}")
            
            try:
                models['random_forest'] = self.model_trainer.train_random_forest(X_train_scaled, y_train_res)
            except Exception as e:
                logger.error(f"Failed to train Random Forest for {symbol}: {e}")
            
            if not models:
                logger.error(f"No models trained successfully for {symbol}")
                return None
            
            # 6. Create and evaluate ensemble
            try:
                ensemble = self.model_trainer.create_ensemble(models)
                ensemble.fit(X_train_scaled, y_train_res)
                models['ensemble'] = ensemble
            except Exception as e:
                logger.error(f"Failed to create ensemble for {symbol}: {e}")
                models['ensemble'] = next(iter(models.values()))  # Use first available model
            
            # 7. Evaluate all models
            results = {}
            for name, model in models.items():
                try:
                    metrics = self.model_evaluator.evaluate_model(model, X_test_scaled, y_test)
                    feature_importance = self.model_evaluator.calculate_feature_importance(
                        model, features.columns
                    )
                    
                    results[name] = {
                        'model': model,
                        'metrics': metrics,
                        'feature_importance': feature_importance
                    }
                    
                    # Save model to database
                    config_dict = {k: v for k, v in vars(Config).items() if not k.startswith('__')}
                    self.db.execute_query(
                        """
                        INSERT INTO enhanced_models 
                        (symbol, model_name, model_type, model_data, metrics, feature_importance)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, model_name) 
                        DO UPDATE SET 
                            model_data = EXCLUDED.model_data,
                            metrics = EXCLUDED.metrics,
                            feature_importance = EXCLUDED.feature_importance,
                            trained_at = NOW()
                        """,
                        (
                            symbol,
                            f"enhanced_{name}",
                            type(model).__name__,
                            pickle.dumps(model),
                            json.dumps(metrics),
                            json.dumps(feature_importance)
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to evaluate/save model {name} for {symbol}: {e}")
            
            # 8. SHAP analysis for best model
            if results:
                best_model_name = max(results.keys(), 
                                    key=lambda k: results[k]['metrics']['f1'])
                try:
                    shap_analysis = self.model_evaluator.analyze_shap_values(
                        results[best_model_name]['model'], X_test_scaled
                    )
                    results['shap_analysis'] = shap_analysis
                except Exception as e:
                    logger.error(f"SHAP analysis failed for {symbol}: {e}")
            
            logger.info(f"✅ Completed training for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error training {symbol}: {e}")
            return None
        finally:
            self.training_status = "Idle"

    def run_pipeline(self):
        """Run complete training pipeline with comprehensive error handling"""
        start_time = time.time()
        self.training_status = "Running"
        self.last_training_time = datetime.now()
        overall_results = {}
        
        try:
            # 1. Load and validate symbols
            symbols = self.load_symbols()
            if not symbols:
                raise ValueError("No valid symbols found")
            
            logger.info(f"Starting training for {len(symbols)} valid symbols: {symbols}")
            
            # 2. Pre-fetch BTC data for market context
            try:
                btc_data = self.data_fetcher.fetch_enhanced_data(
                    "BTCUSDT", Config.SIGNAL_TIMEFRAME, Config.DATA_LOOKBACK_DAYS
                )
                if btc_data is None or len(btc_data) < 100:
                    logger.warning("Insufficient BTC data, proceeding without market context")
                    btc_data = None
            except Exception as e:
                logger.error(f"Failed to fetch BTC data: {e}")
                btc_data = None
            
            # 3. Train models in parallel with error handling
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self.train_symbol_model, symbol, btc_data): symbol
                    for symbol in symbols
                }
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        result = future.result()
                        if result:
                            overall_results[symbol] = result
                            self.current_models[symbol] = result['ensemble']['model']
                        else:
                            logger.warning(f"Training failed for {symbol} (no result returned)")
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
            
            # 4. Calculate overall metrics
            duration = time.time() - start_time
            successful_symbols = [s for s, r in overall_results.items() if r]
            
            # Handle case where no models were trained successfully
            if not successful_symbols:
                avg_metrics = {
                    'accuracy': None,
                    'f1': None,
                    'precision': None,
                    'recall': None,
                    'num_symbols': 0,
                    'failed_symbols': len(symbols)
                }
            else:
                avg_metrics = {
                    'accuracy': np.nanmean([r['ensemble']['metrics']['accuracy'] 
                                         for r in overall_results.values() if r]),
                    'f1': np.nanmean([r['ensemble']['metrics']['f1'] 
                                     for r in overall_results.values() if r]),
                    'precision': np.nanmean([r['ensemble']['metrics']['precision'] 
                                           for r in overall_results.values() if r]),
                    'recall': np.nanmean([r['ensemble']['metrics']['recall'] 
                                         for r in overall_results.values() if r]),
                    'num_symbols': len(successful_symbols),
                    'failed_symbols': len(symbols) - len(successful_symbols)
                }
            
            # 5. Save training metadata
            config_dict = {k: v for k, v in vars(Config).items() if not k.startswith('__')}
            self.db.execute_query(
                """
                INSERT INTO training_metadata 
                (symbols, model_types, avg_metrics, duration_seconds, parameters)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    list(overall_results.keys()),
                    ['lightgbm', 'xgboost', 'random_forest', 'ensemble'],
                    json.dumps(self.model_evaluator.replace_nan_with_none(avg_metrics)),
                    duration,
                    json.dumps(config_dict)
                )
            )
            
            # 6. Update status and metrics
            self.training_status = "Completed"
            self.training_metrics = avg_metrics
            
            # 7. Send completion notification
            if TELEGRAM_TOKEN and CHAT_ID:
                self.send_telegram_notification(
                    f"✅ *Training Completed*\n\n"
                    f"• Symbols: {len(successful_symbols)}/{len(symbols)}\n"
                    f"• Duration: {duration:.2f}s\n"
                    f"• Avg F1: {avg_metrics['f1']:.4f if avg_metrics['f1'] is not None else 'N/A'}\n"
                    f"• Avg Accuracy: {avg_metrics['accuracy']:.4f if avg_metrics['accuracy'] is not None else 'N/A'}"
                )
            
            logger.info(f"🏁 Training completed in {duration:.2f} seconds")
            logger.info(f"📊 Results: {avg_metrics}")
            return overall_results
            
        except Exception as e:
            self.training_status = "Failed"
            logger.error(f"❌ Pipeline failed: {e}")
            
            if TELEGRAM_TOKEN and CHAT_ID:
                self.send_telegram_notification(
                    f"❌ *Training Failed*\n\nError: {str(e)}"
                )
            raise
        finally:
            self.db.close()

    def send_telegram_notification(self, message: str):
        """Send notification via Telegram with error handling"""
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    'chat_id': CHAT_ID,
                    'text': message,
                    'parse_mode': 'Markdown'
                },
                timeout=Config.TELEGRAM_TIMEOUT
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

# ---------------------- Flask API ----------------------
app = Flask(__name__)

@app.route('/status', methods=['GET'])
def get_status():
    pipeline = TradingModelPipeline()
    return jsonify({
        'status': pipeline.training_status,
        'last_training_time': pipeline.last_training_time.isoformat() if pipeline.last_training_time else None,
        'metrics': pipeline.training_metrics,
        'symbols_trained': len(pipeline.current_models)
    })

@app.route('/predict/<symbol>', methods=['POST'])
def predict(symbol: str):
    pipeline = TradingModelPipeline()
    
    if symbol not in pipeline.current_models:
        return jsonify({'error': 'Model not found for symbol'}), 404
    
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400
        
        # التحقق من أن الميزات هي قاموس
        if not isinstance(data['features'], dict):
            return jsonify({'error': 'Features must be a dictionary'}), 400
            
        features = pd.DataFrame([data['features']])
        
        # التحقق من تطابق أسماء الميزات
        model = pipeline.current_models[symbol]
        expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
        
        if set(features.columns) != set(expected_features):
            return jsonify({
                'error': 'Feature mismatch',
                'expected': list(expected_features),
                'received': list(features.columns)
            }), 400
            
        # التنبؤ
        prediction = model.predict(features)
        probabilities = model.predict_proba(features).tolist()[0] if hasattr(model, 'predict_proba') else None
        
        # حفظ التنبؤ في قاعدة البيانات
        pipeline.db.execute_query(
            """
            INSERT INTO enhanced_signals 
            (symbol, timestamp, prediction, confidence, features, model_version)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                symbol,
                datetime.now(),
                int(prediction[0]),
                float(max(probabilities)) if probabilities else 0.0,
                json.dumps(data['features']),
                type(model).__name__
            )
        )
        
        return jsonify({
            'symbol': symbol,
            'prediction': int(prediction[0]),
            'confidence': float(max(probabilities)) if probabilities else 0.0,
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

def run_flask():
    """Run Flask API with production-ready settings"""
    from waitress import serve
    serve(app, host='0.0.0.0', port=Config.FLASK_PORT, threads=6)

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    try:
        # Initialize components
        logger.info("🚀 Starting Enhanced ML Trading System")
        
        # Validate Binance connection
        data_fetcher = BinanceDataFetcher()
        try:
            data_fetcher.client.ping()
            logger.info("✅ Binance connection validated")
        except Exception as e:
            logger.critical(f"❌ Failed to connect to Binance: {e}")
            exit(1)
        
        # Initialize database
        db = DatabaseManager()
        if not db._connect():
            exit(1)
        
        # Start Flask API in a separate thread
        flask_thread = Thread(target=run_flask, daemon=True)
        flask_thread.start()
        logger.info(f"🌐 Flask API running on port {Config.FLASK_PORT}")
        
        # Run main training pipeline
        pipeline = TradingModelPipeline()
        pipeline.run_pipeline()
        
        # Keep application running
        flask_thread.join()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down gracefully...")
    except Exception as e:
        logger.critical(f"❌ Fatal error: {e}")
        exit(1)
    finally:
        logger.info("👋 System shutdown completed")