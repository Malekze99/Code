import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple, Union

# Flask and threading libraries
from flask import Flask, request, Response
from threading import Thread

# Import LightGBM
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from lightgbm import early_stopping, log_evaluation

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer')

# ---------------------- Load Environment Variables ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
    TELEGRAM_TOKEN: Optional[str] = config('TELEGRAM_BOT_TOKEN', default=None)
    CHAT_ID: Optional[str] = config('TELEGRAM_CHAT_ID', default=None)
except Exception as e:
    logger.critical(f"❌ Failed to load essential environment variables: {e}")
    exit(1)

logger.info(f"Binance API Key: {'Available' if API_KEY else 'Not available'}")
logger.info(f"Database URL: {'Available' if DB_URL else 'Not available'}")
logger.info(f"Webhook URL: {WEBHOOK_URL if WEBHOOK_URL else 'Not specified'} (Flask will always run for Render)")
logger.info(f"Telegram Token: {'Available' if TELEGRAM_TOKEN else 'Not available'}")
logger.info(f"Telegram Chat ID: {'Available' if CHAT_ID else 'Not available'}")

# ---------------------- Constants and Global Variables Setup ----------------------
SIGNAL_GENERATION_TIMEFRAME: str = '5m'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 90
BASE_ML_MODEL_NAME: str = 'LightGBM_SMC_Scalping_V1'

# Indicator Parameters (copied from c4.py for consistency)
VOLUME_LOOKBACK_CANDLES: int = 3
RSI_PERIOD: int = 9
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 2

# SMC Parameters
LIQUIDITY_WINDOW: int = 20
SWING_WINDOW: int = 5
ORDER_BLOCK_VOLUME_MULTIPLIER: float = 1.5

# New Indicators Parameters
VELOCITY_LOOKBACK: int = 5
MOMENTUM_LOOKBACK: int = 14
ADX_LOOKBACK: int = 14
MA_SLOPE_LOOKBACK: int = 5
MA_SLOPE_PERIOD: int = 20

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None

# Training status variables
training_status: str = "Idle"
last_training_time: Optional[datetime] = None
last_training_metrics: Dict[str, Any] = {}
training_error: Optional[str] = None

# ---------------------- Binance Client Setup ----------------------
try:
    logger.info("ℹ️ [Binance] Initializing Binance client...")
    client = Client(API_KEY, API_SECRET)
    client.ping()
    server_time = client.get_server_time()
    logger.info(f"✅ [Binance] Binance client initialized. Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
except BinanceRequestException as req_err:
    logger.critical(f"❌ [Binance] Binance request error (network or request issue): {req_err}")
    exit(1)
except BinanceAPIException as api_err:
    logger.critical(f"❌ [Binance] Binance API error (invalid keys or server issue): {api_err}")
    exit(1)
except Exception as e:
    logger.critical(f"❌ [Binance] Unexpected failure in Binance client initialization: {e}")
    exit(1)

# ---------------------- Database Connection Setup ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """Initializes database connection and creates tables if they don't exist."""
    global conn, cur
    logger.info("[DB] Starting database initialization...")
    for attempt in range(retries):
        try:
            logger.info(f"[DB] Attempting to connect to database (Attempt {attempt + 1}/{retries})...")
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            cur = conn.cursor()
            logger.info("✅ [DB] Successfully connected to database.")

            # --- Create or update signals table (Modified schema) ---
            logger.info("[DB] Checking for/creating 'signals' table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    initial_target DOUBLE PRECISION NOT NULL,
                    current_target DOUBLE PRECISION NOT NULL,
                    r2_score DOUBLE PRECISION,
                    volume_15m DOUBLE PRECISION,
                    achieved_target BOOLEAN DEFAULT FALSE,
                    closing_price DOUBLE PRECISION,
                    closed_at TIMESTAMP,
                    sent_at TIMESTAMP DEFAULT NOW(),
                    entry_time TIMESTAMP DEFAULT NOW(),
                    time_to_target INTERVAL,
                    profit_percentage DOUBLE PRECISION,
                    strategy_name TEXT,
                    signal_details JSONB
                );""")
            conn.commit()
            logger.info("✅ [DB] 'signals' table exists or created.")

            # --- Create ml_models table (NEW) ---
            logger.info("[DB] Checking for/creating 'ml_models' table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL,
                    trained_at TIMESTAMP DEFAULT NOW(),
                    metrics JSONB
                );""")
            conn.commit()
            logger.info("✅ [DB] 'ml_models' table exists or created.")

            # --- Create market_dominance table (if it doesn't exist) ---
            logger.info("[DB] Checking for/creating 'market_dominance' table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_dominance (
                    id SERIAL PRIMARY KEY,
                    recorded_at TIMESTAMP DEFAULT NOW(),
                    btc_dominance DOUBLE PRECISION,
                    eth_dominance DOUBLE PRECISION
                );
            """)
            conn.commit()
            logger.info("✅ [DB] 'market_dominance' table exists or created.")

            logger.info("✅ [DB] Database initialized successfully.")
            return

        except OperationalError as op_err:
            logger.error(f"❌ [DB] Operational error during connection (Attempt {attempt + 1}): {op_err}")
            if conn: conn.rollback()
            if attempt == retries - 1:
                logger.critical("❌ [DB] All database connection attempts failed.")
                raise op_err
            time.sleep(delay)
        except Exception as e:
            logger.critical(f"❌ [DB] Unexpected failure during database initialization (Attempt {attempt + 1}): {e}", exc_info=True)
            if conn: conn.rollback()
            if attempt == retries - 1:
                logger.critical("❌ [DB] All database connection attempts failed.")
                raise e
            time.sleep(delay)

    logger.critical("❌ [DB] Failed to connect to the database after multiple attempts.")
    exit(1)

def check_db_connection() -> bool:
    """Checks database connection status and re-initializes if necessary."""
    global conn
    try:
        if conn is None or conn.closed != 0:
            logger.warning("⚠️ [DB] Connection closed or non-existent. Re-initializing...")
            init_db()
            return True
        else:
            with conn.cursor() as check_cur:
                check_cur.execute("SELECT 1;")
                check_cur.fetchone()
            return True
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [DB] Database connection lost ({e}). Re-initializing...")
        try:
            init_db()
            return True
        except Exception as recon_err:
            logger.error(f"❌ [DB] Failed to re-establish connection after loss: {recon_err}")
            return False
    except Exception as e:
        logger.error(f"❌ [DB] Unexpected error during connection check: {e}", exc_info=True)
        try:
            init_db()
            return True
        except Exception as recon_err:
            logger.error(f"❌ [DB] Failed to re-establish connection after unexpected error: {recon_err}")
            return False

def convert_np_values(obj: Any) -> Any:
    """Converts NumPy data types to native Python types for JSON and DB compatibility."""
    if isinstance(obj, dict):
        return {k: convert_np_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_values(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """
    Reads the list of currency symbols from a text file, then validates them
    as valid USDT pairs available for Spot trading on Binance.
    """
    raw_symbols: List[str] = []
    logger.info(f"ℹ️ [Data] Reading symbol list from file '{filename}'...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        if not os.path.exists(file_path):
            file_path = os.path.abspath(filename)
            if not os.path.exists(file_path):
                logger.error(f"❌ [Data] File '{filename}' not found in script directory or current directory.")
                return []
            else:
                logger.warning(f"⚠️ [Data] File '{filename}' not found in script directory. Using file in current directory: '{file_path}'")

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = [f"{line.strip().upper().replace('USDT', '')}USDT"
                           for line in f if line.strip() and not line.startswith('#')]
        raw_symbols = sorted(list(set(raw_symbols)))
        logger.info(f"ℹ️ [Data] Read {len(raw_symbols)} initial symbols from '{file_path}'.")

    except FileNotFoundError:
        logger.error(f"❌ [Data] File '{filename}' not found.")
        return []
    except Exception as e:
        logger.error(f"❌ [Data] Error reading file '{filename}': {e}", exc_info=True)
        return []

    if not raw_symbols:
        logger.warning("⚠️ [Data] Initial symbol list is empty.")
        return []

    if not client:
        logger.error("❌ [Data Validation] Binance client not initialized. Cannot validate symbols.")
        return raw_symbols

    try:
        logger.info("ℹ️ [Data Validation] Validating symbols and trading status from Binance API...")
        exchange_info = client.get_exchange_info()
        valid_trading_usdt_symbols = {
            s['symbol'] for s in exchange_info['symbols']
            if s.get('quoteAsset') == 'USDT' and
               s.get('status') == 'TRADING' and
               s.get('isSpotTradingAllowed') is True
        }
        logger.info(f"ℹ️ [Data Validation] Found {len(valid_trading_usdt_symbols)} valid USDT spot trading pairs on Binance.")
        validated_symbols = [symbol for symbol in raw_symbols if symbol in valid_trading_usdt_symbols]

        removed_count = len(raw_symbols) - len(validated_symbols)
        if removed_count > 0:
            removed_symbols = set(raw_symbols) - set(validated_symbols)
            logger.warning(f"⚠️ [Data Validation] Removed {removed_count} invalid or unavailable USDT trading symbols from list: {', '.join(removed_symbols)}")

        logger.info(f"✅ [Data Validation] Symbols validated. Using {len(validated_symbols)} valid symbols.")
        return validated_symbols

    except (BinanceAPIException, BinanceRequestException) as binance_err:
        logger.error(f"❌ [Data Validation] Binance API or network error during symbol validation: {binance_err}")
        logger.warning("⚠️ [Data Validation] Using initial list from file without Binance validation.")
        return raw_symbols
    except Exception as api_err:
        logger.error(f"❌ [Data Validation] Unexpected error during Binance symbol validation: {api_err}", exc_info=True)
        logger.warning("⚠️ [Data Validation] Using initial list from file without Binance validation.")
        return raw_symbols

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """
    Fetches historical candlestick data from Binance for a specified number of days.
    """
    if not client:
        logger.error("❌ [Data] Binance client not initialized for data fetching.")
        return None
    try:
        # Calculate the start date for the entire data range needed
        start_dt = datetime.utcnow() - timedelta(days=days + 1)
        start_str_overall = start_dt.strftime("%Y-%m-%d %H:%M:%S")

        logger.debug(f"ℹ️ [Data] Fetching {interval} data for {symbol} from {start_str_overall} to now...")

        # Call get_historical_klines for the entire period.
        klines = client.get_historical_klines(symbol, interval, start_str_overall)

        if not klines:
            logger.warning(f"⚠️ [Data] No historical data ({interval}) for {symbol} found for the requested period.")
            return None

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[numeric_cols]
        initial_len = len(df)
        df.dropna(subset=numeric_cols, inplace=True)

        if len(df) < initial_len:
            logger.debug(f"ℹ️ [Data] {symbol}: Dropped {initial_len - len(df)} rows due to NaN values in OHLCV data.")

        if df.empty:
            logger.warning(f"⚠️ [Data] DataFrame for {symbol} is empty after removing essential NaN values.")
            return None

        # Sort by index (timestamp) to ensure chronological order
        df.sort_index(inplace=True)

        logger.debug(f"✅ [Data] Fetched and processed {len(df)} historical candles ({interval}) for {symbol}.")
        return df

    except BinanceAPIException as api_err:
        logger.error(f"❌ [Data] Binance API error while fetching data for {symbol}: {api_err}")
        return None
    except BinanceRequestException as req_err:
        logger.error(f"❌ [Data] Request or network error while fetching data for {symbol}: {req_err}")
        return None
    except Exception as e:
        logger.error(f"❌ [Data] Unexpected error while fetching historical data for {symbol}: {e}", exc_info=True)
        return None

# ---------------------- Technical Indicator Functions ----------------------
def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)."""
    if series is None or series.isnull().all() or len(series) < span:
        return pd.Series(index=series.index if series is not None else None, dtype=float)
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Calculates Relative Strength Index (RSI)."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning("⚠️ [Indicator RSI] 'close' column missing or empty.")
        df['rsi'] = np.nan
        return df
    if len(df) < period:
        logger.warning(f"⚠️ [Indicator RSI] Insufficient data ({len(df)} < {period}) to calculate RSI.")
        df['rsi'] = np.nan
        return df

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi_series = 100 - (100 / (1 + rs))
    df['rsi'] = rsi_series.ffill().fillna(50)

    return df

def _calculate_btc_trend_feature(df_btc: pd.DataFrame) -> Optional[pd.Series]:
    """
    Calculates a numerical representation of Bitcoin's trend based on EMA20 and EMA50.
    Returns 1 for bullish, -1 for bearish, 0 for neutral.
    """
    logger.debug("ℹ️ [Indicators] Calculating Bitcoin trend for features...")
    min_data_for_ema = 50 + 5 # 50 for EMA50, 5 buffer

    if df_btc is None or df_btc.empty or len(df_btc) < min_data_for_ema:
        logger.warning(f"⚠️ [Indicators] Insufficient BTC/USDT data ({len(df_btc) if df_btc is not None else 0} < {min_data_for_ema}) to calculate Bitcoin trend for features.")
        return pd.Series(index=df_btc.index if df_btc is not None else None, data=0.0)

    df_btc_copy = df_btc.copy()
    df_btc_copy['close'] = pd.to_numeric(df_btc_copy['close'], errors='coerce')
    df_btc_copy.dropna(subset=['close'], inplace=True)

    if len(df_btc_copy) < min_data_for_ema:
        logger.warning(f"⚠️ [Indicators] Insufficient BTC/USDT data after removing NaN values to calculate trend.")
        return pd.Series(index=df_btc.index, data=0.0)

    ema20 = calculate_ema(df_btc_copy['close'], 20)
    ema50 = calculate_ema(df_btc_copy['close'], 50)

    ema_df = pd.DataFrame({'ema20': ema20, 'ema50': ema50, 'close': df_btc_copy['close']})
    ema_df.dropna(inplace=True)

    if ema_df.empty:
        logger.warning("⚠️ [Indicators] EMA DataFrame is empty after removing NaN values. Cannot calculate Bitcoin trend.")
        return pd.Series(index=df_btc.index, data=0.0)

    trend_series = pd.Series(index=ema_df.index, data=0.0)

    # Apply trend logic:
    trend_series[(ema_df['close'] > ema_df['ema20']) & (ema_df['ema20'] > ema_df['ema50'])] = 1.0
    trend_series[(ema_df['close'] < ema_df['ema20']) & (ema_df['ema20'] < ema_df['ema50'])] = -1.0

    final_trend_series = trend_series.reindex(df_btc.index).fillna(0.0)
    logger.debug(f"✅ [Indicators] Bitcoin trend feature calculated. Examples: {final_trend_series.tail().tolist()}")
    return final_trend_series

# ---------------------- New Indicator Functions ----------------------
def calculate_velocity(df: pd.DataFrame, lookback: int = VELOCITY_LOOKBACK) -> pd.DataFrame:
    """Calculate Price Velocity (Rate of Change derivative)."""
    df = df.copy()
    df['price_velocity'] = df['close'].diff(lookback) / lookback  # معدل التغير خلال `lookback` شموع
    return df

def calculate_momentum(df: pd.DataFrame, lookback: int = MOMENTUM_LOOKBACK) -> pd.DataFrame:
    """Calculate Momentum (Price Change over `lookback` periods)."""
    df = df.copy()
    df['momentum'] = df['close'] - df['close'].shift(lookback)
    return df

def calculate_adx(df: pd.DataFrame, lookback: int = ADX_LOOKBACK) -> pd.DataFrame:
    """Calculate ADX (Average Directional Index) to determine trend strength."""
    df = df.copy()
    
    # First calculate ATR (Average True Range) which is needed for ADX calculation
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=lookback).mean()
    
    # حساب +DI و -DI
    df['up_move'] = df['high'].diff()
    df['down_move'] = -df['low'].diff()
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # حساب المتوسطات المتحركة لـ +DI و -DI
    df['plus_di'] = (df['plus_dm'].rolling(window=lookback).mean() / df['atr'].replace(0, 1)) * 100
    df['minus_di'] = (df['minus_dm'].rolling(window=lookback).mean() / df['atr'].replace(0, 1)) * 100
    
    # حساب ADX
    df['dx'] = (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])) * 100
    df['adx'] = df['dx'].rolling(window=lookback).mean()
    
    # تحديد الاتجاه العام
    df['trend_direction'] = np.where(
        df['plus_di'] > df['minus_di'], 
        1,  # اتجاه صعودي
        np.where(df['minus_di'] > df['plus_di'], -1, 0)  # اتجاه هبوطي أو جانبي
    )
    
    return df

def calculate_ma_slope(df: pd.DataFrame, ma_period: int = MA_SLOPE_PERIOD, lookback: int = MA_SLOPE_LOOKBACK) -> pd.DataFrame:
    """Calculate the slope of Moving Average to determine trend direction."""
    df = df.copy()
    df['ma'] = df['close'].rolling(window=ma_period).mean()
    df['ma_slope'] = df['ma'].diff(lookback) / lookback  # ميل المتوسط المتحرك
    
    # تصنيف الاتجاه
    df['trend_direction'] = np.where(
        df['ma_slope'] > 0.001, 1,  # اتجاه صعودي
        np.where(df['ma_slope'] < -0.001, -1, 0)  # اتجاه هبوطي أو جانبي
    )
    
    return df

# ---------------------- SMC Indicator Functions ----------------------
def detect_liquidity_pools(df: pd.DataFrame, window: int = LIQUIDITY_WINDOW) -> pd.DataFrame:
    """
    Detect liquidity pools (areas of high trading activity)
    """
    if df is None or df.empty or len(df) < window:
        logger.warning(f"⚠️ [SMC] Insufficient data to detect liquidity pools (min {window} candles needed).")
        df['liquidity_pool_high'] = 0
        df['liquidity_pool_low'] = 0
        return df
        
    df = df.copy()
    df['high_range'] = df['high'].rolling(window).max()
    df['low_range'] = df['low'].rolling(window).min()
    
    # Identify liquidity pools as areas near recent highs/lows
    df['liquidity_pool_high'] = ((df['high'] >= df['high_range'] * 0.998) & 
                                (df['high'] == df['high'].rolling(3, center=True).max())).astype(int)
    df['liquidity_pool_low'] = ((df['low'] <= df['low_range'] * 1.002) & 
                               (df['low'] == df['low'].rolling(3, center=True).min())).astype(int)
    return df

def identify_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify bullish/bearish order blocks
    """
    df = df.copy()
    df['bullish_ob'] = 0
    df['bearish_ob'] = 0
    
    if len(df) < 3:
        logger.warning("⚠️ [SMC] Insufficient data to identify order blocks (min 3 candles needed).")
        return df
    
    # Bullish order block: large green candle after a downtrend
    volume_threshold = df['volume'].rolling(5).mean() * ORDER_BLOCK_VOLUME_MULTIPLIER
    
    for i in range(1, len(df)-1):
        prev_candle = df.iloc[i-1]
        current_candle = df.iloc[i]
        next_candle = df.iloc[i+1]
        
        # Bullish OB: down candle followed by large up candle
        if (prev_candle['close'] < prev_candle['open'] and
            current_candle['close'] > current_candle['open'] and
            current_candle['volume'] > volume_threshold.iloc[i] and
            current_candle['close'] > prev_candle['close'] and
            next_candle['close'] > current_candle['close']):
            df.loc[df.index[i], 'bullish_ob'] = 1
        
        # Bearish OB: up candle followed by large down candle
        if (prev_candle['close'] > prev_candle['open'] and
            current_candle['close'] < current_candle['open'] and
            current_candle['volume'] > volume_threshold.iloc[i] and
            current_candle['close'] < prev_candle['close'] and
            next_candle['close'] < current_candle['close']):
            df.loc[df.index[i], 'bearish_ob'] = 1
    
    return df

def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Fair Value Gaps (FVG)
    """
    df = df.copy()
    df['bullish_fvg'] = 0
    df['bearish_fvg'] = 0
    
    if len(df) < 2:
        logger.warning("⚠️ [SMC] Insufficient data to detect FVGs (min 2 candles needed).")
        return df
    
    for i in range(1, len(df)):
        prev_high = df.iloc[i-1]['high']
        prev_low = df.iloc[i-1]['low']
        current_high = df.iloc[i]['high']
        current_low = df.iloc[i]['low']
        
        # Bullish FVG: current low > previous high
        if current_low > prev_high:
            df.loc[df.index[i], 'bullish_fvg'] = 1
        
        # Bearish FVG: current high < previous low
        elif current_high < prev_low:
            df.loc[df.index[i], 'bearish_fvg'] = 1
    
    return df

def identify_bos_choch(df: pd.DataFrame, swing_window: int = SWING_WINDOW) -> pd.DataFrame:
    """
    Identify Break of Structure (BOS) and Change of Character (CHOCH)
    """
    df = df.copy()
    df['bullish_bos'] = 0
    df['bearish_bos'] = 0
    df['bullish_choch'] = 0
    df['bearish_choch'] = 0
    
    if len(df) < swing_window + 2:
        logger.warning(f"⚠️ [SMC] Insufficient data to identify BOS/CHOCH (min {swing_window+2} candles needed).")
        return df
    
    # Calculate swing highs and lows
    df['swing_high'] = df['high'].rolling(swing_window, center=True).max()
    df['swing_low'] = df['low'].rolling(swing_window, center=True).min()
    
    for i in range(swing_window, len(df)-1):
        current_high = df.iloc[i]['high']
        current_low = df.iloc[i]['low']
        prev_swing_high = df.iloc[i-swing_window]['swing_high']
        prev_swing_low = df.iloc[i-swing_window]['swing_low']
        
        # Bullish BOS: Break above previous swing high
        if current_high > prev_swing_high:
            df.loc[df.index[i], 'bullish_bos'] = 1
        
        # Bearish BOS: Break below previous swing low
        if current_low < prev_swing_low:
            df.loc[df.index[i], 'bearish_bos'] = 1
        
        # Bullish CHOCH: Failure to make new low followed by reversal
        if (df.iloc[i]['close'] > df.iloc[i]['open'] and
            current_low > prev_swing_low and
            df.iloc[i-1]['close'] < df.iloc[i-1]['open'] and
            df.iloc[i-1]['low'] < prev_swing_low):
            df.loc[df.index[i], 'bullish_choch'] = 1
        
        # Bearish CHOCH: Failure to make new high followed by reversal
        if (df.iloc[i]['close'] < df.iloc[i]['open'] and
            current_high < prev_swing_high and
            df.iloc[i-1]['close'] > df.iloc[i-1]['open'] and
            df.iloc[i-1]['high'] > prev_swing_high):
            df.loc[df.index[i], 'bearish_choch'] = 1
    
    return df

# ---------------------- Model Training and Saving Functions ----------------------
def prepare_data_for_ml(df: pd.DataFrame, symbol: str, target_period: int = 5, btc_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """
    Prepares data for machine learning model training with SMC features.
    """
    logger.info(f"ℹ️ [ML Prep] Preparing data for ML model for {symbol} with SMC features...")

    # Determine minimum data length required
    min_len_required = max(
        VOLUME_LOOKBACK_CANDLES, 
        RSI_PERIOD + RSI_MOMENTUM_LOOKBACK_CANDLES, 
        LIQUIDITY_WINDOW + 5,
        target_period + 10,
        ADX_LOOKBACK * 2,
        MA_SLOPE_PERIOD + MA_SLOPE_LOOKBACK
    )

    if len(df) < min_len_required:
        logger.warning(f"⚠️ [ML Prep] DataFrame for {symbol} is too short ({len(df)} < {min_len_required}) for data preparation.")
        return None

    df_calc = df.copy()

    # Calculate required features
    df_calc['volume_15m_avg'] = df_calc['volume'].rolling(window=VOLUME_LOOKBACK_CANDLES, min_periods=1).mean()
    logger.debug(f"ℹ️ [ML Prep] Calculated 15-min average volume for {symbol}.")

    # Calculate RSI and momentum
    df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
    
    # Add bullish RSI momentum indicator
    df_calc['rsi_momentum_bullish'] = 0
    if len(df_calc) >= RSI_MOMENTUM_LOOKBACK_CANDLES + 1:
        for i in range(RSI_MOMENTUM_LOOKBACK_CANDLES, len(df_calc)):
            rsi_slice = df_calc['rsi'].iloc[i - RSI_MOMENTUM_LOOKBACK_CANDLES : i + 1]
            if not rsi_slice.isnull().any() and np.all(np.diff(rsi_slice) > 0) and rsi_slice.iloc[-1] > 50:
                df_calc.loc[df_calc.index[i], 'rsi_momentum_bullish'] = 1
    logger.debug(f"ℹ️ [ML Prep] Calculated bullish RSI momentum indicator for {symbol}.")

    # Calculate SMC indicators
    df_calc = detect_liquidity_pools(df_calc)
    df_calc = identify_order_blocks(df_calc)
    df_calc = detect_fvg(df_calc)
    df_calc = identify_bos_choch(df_calc)
    logger.info(f"ℹ️ [ML Prep] Calculated SMC indicators for {symbol}")

    # Calculate new indicators
    df_calc = calculate_velocity(df_calc)
    df_calc = calculate_momentum(df_calc)
    df_calc = calculate_adx(df_calc)
    # df_calc = calculate_ma_slope(df_calc)  # Uncomment if you prefer MA Slope over ADX
    logger.info(f"ℹ️ [ML Prep] Calculated new indicators (Velocity, Momentum, ADX) for {symbol}")

    # --- Fetch and calculate BTC trend feature ---
    local_btc_df = btc_df
    if local_btc_df is None:
        logger.debug(f"ℹ️ [ML Prep] BTC data not provided, fetching for {symbol}...")
        local_btc_df = fetch_historical_data("BTCUSDT", interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)

    btc_trend_series = None
    if local_btc_df is not None and not local_btc_df.empty:
        btc_trend_series = _calculate_btc_trend_feature(local_btc_df)
        if btc_trend_series is not None:
            df_calc = df_calc.merge(btc_trend_series.rename('btc_trend_feature'),
                                    left_index=True, right_index=True, how='left')
            df_calc['btc_trend_feature'] = df_calc['btc_trend_feature'].fillna(0.0)
            logger.debug(f"ℹ️ [ML Prep] Merged Bitcoin trend feature for {symbol}.")
        else:
            logger.warning(f"⚠️ [ML Prep] Failed to calculate Bitcoin trend feature. Will use 0 as default for 'btc_trend_feature'.")
            df_calc['btc_trend_feature'] = 0.0
    else:
        logger.warning(f"⚠️ [ML Prep] Failed to fetch Bitcoin historical data. Will use 0 as default for 'btc_trend_feature'.")
        df_calc['btc_trend_feature'] = 0.0

    # Define feature columns
    feature_columns = [
        'volume_15m_avg',
        'rsi_momentum_bullish',
        'btc_trend_feature',
        'liquidity_pool_high', 'liquidity_pool_low',
        'bullish_ob', 'bearish_ob',
        'bullish_fvg', 'bearish_fvg',
        'bullish_bos', 'bearish_bos',
        'bullish_choch', 'bearish_choch',
        # New indicators
        'price_velocity',
        'momentum',
        'adx', 'plus_di', 'minus_di', 'trend_direction'
        # Uncomment if using MA Slope:
        # 'ma_slope', 'trend_direction'
    ]

    # Ensure all features exist
    for col in feature_columns:
        if col not in df_calc.columns:
            logger.warning(f"⚠️ [ML Prep] Missing feature column: {col}. Adding it as 0.")
            df_calc[col] = 0
        else:
            df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce').fillna(0)

    # Create target variable
    price_change_threshold = 0.005 # 0.5%
    df_calc['close'] = pd.to_numeric(df_calc['close'], errors='coerce')
    df_calc['future_max_close'] = df_calc['close'].shift(-target_period).rolling(window=target_period, min_periods=1).max()
    df_calc['target'] = ((df_calc['future_max_close'] / df_calc['close']) - 1 > price_change_threshold).astype(int)

    initial_len = len(df_calc)
    df_cleaned = df_calc.dropna(subset=feature_columns + ['target']).copy()
    dropped_count = initial_len - len(df_cleaned)

    if dropped_count > 0:
        logger.info(f"ℹ️ [ML Prep] For {symbol}: Dropped {dropped_count} rows due to NaN values after indicator and target calculation.")
    if df_cleaned.empty:
        logger.warning(f"⚠️ [ML Prep] DataFrame for {symbol} is empty after removing NaNs for ML preparation.")
        return None

    logger.info(f"✅ [ML Prep] Data prepared for {symbol} successfully. Number of rows: {len(df_cleaned)}")
    return df_cleaned[feature_columns + ['target']]

def train_and_evaluate_model(data: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains a LightGBM model with SMC features and evaluates its performance.
    """
    logger.info("ℹ️ [ML Train] Starting model training and evaluation with SMC features...")

    if data.empty:
        logger.error("❌ [ML Train] Empty DataFrame for training.")
        return None, {}

    X = data.drop('target', axis=1)
    y = data['target']

    if X.empty or y.empty:
        logger.error("❌ [ML Train] Empty features or targets for training.")
        return None, {}

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as ve:
        logger.warning(f"⚠️ [ML Train] Cannot use stratify due to single class in target: {ve}. Proceeding without stratify.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Create sample weights based on SMC signals
    sample_weights = np.ones(len(y_train))
    smc_signals = X_train[['bullish_ob', 'bearish_ob', 'bullish_fvg', 'bearish_fvg']].max(axis=1)
    sample_weights[smc_signals == 1] = 2.0  # Double weight for SMC signals
    
    # Train LightGBM model
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced', # Handle class imbalance
        verbosity=-1
    )
    
    model.fit(
        X_train_scaled_df, 
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test_scaled_df, y_test)],
        eval_metric='binary_logloss',
        callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=10)  # بديل لـ verbose
        ]
    )
    
    logger.info("✅ [ML Train] LightGBM model trained successfully with SMC features.")

    # Evaluate model
    y_pred = model.predict(X_test_scaled_df)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Feature importance
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'num_samples_trained': len(X_train),
        'num_samples_tested': len(X_test),
        'feature_names': X.columns.tolist(),
        'feature_importances': feature_importances.to_dict(orient='records'),
        'smc_features_used': ['bullish_ob', 'bearish_ob', 'bullish_fvg', 'bearish_fvg', 
                              'liquidity_pool_high', 'liquidity_pool_low'],
        'new_features_used': ['price_velocity', 'momentum', 'adx', 'trend_direction']
    }

    logger.info(f"📊 [ML Train] Model performance metrics (LightGBM with SMC):")
    logger.info(f"  - Accuracy: {accuracy:.4f}")
    logger.info(f"  - Precision: {precision:.4f}")
    logger.info(f"  - Recall: {recall:.4f}")
    logger.info(f"  - F1 Score: {f1:.4f}")
    
    # Log feature importance
    logger.info("📊 Feature Importances:")
    for i, row in feature_importances.iterrows():
        logger.info(f"  - {row['Feature']}: {row['Importance']}")

    return model, metrics

def save_ml_model_to_db(model: Any, model_name: str, metrics: Dict[str, Any]) -> bool:
    """
    Saves the trained model and its metadata (metrics) to the database.
    """
    logger.info(f"ℹ️ [DB Save] Checking database connection before saving...")
    if not check_db_connection() or not conn:
        logger.error("❌ [DB Save] Cannot save ML model due to database connection issue.")
        return False

    logger.info(f"ℹ️ [DB Save] Attempting to save ML model '{model_name}' to database...")
    try:
        # Serialize the model using pickle
        model_binary = pickle.dumps(model)
        logger.info(f"✅ [DB Save] Model serialized successfully. Data size: {len(model_binary)} bytes.")

        # Convert metrics to JSONB
        metrics_json = json.dumps(convert_np_values(metrics))
        logger.info(f"✅ [DB Save] Metrics converted to JSON successfully.")

        with conn.cursor() as db_cur:
            # Check if the model already exists (for update or insert)
            db_cur.execute("SELECT id FROM ml_models WHERE model_name = %s;", (model_name,))
            existing_model = db_cur.fetchone()

            if existing_model:
                logger.info(f"ℹ️ [DB Save] Model '{model_name}' already exists. Will update it.")
                update_query = sql.SQL("""
                    UPDATE ml_models
                    SET model_data = %s, trained_at = NOW(), metrics = %s
                    WHERE id = %s;
                """)
                db_cur.execute(update_query, (model_binary, metrics_json, existing_model['id']))
                logger.info(f"✅ [DB Save] Successfully updated ML model '{model_name}' in database.")
            else:
                logger.info(f"ℹ️ [DB Save] Model '{model_name}' does not exist. Will insert as a new model.")
                insert_query = sql.SQL("""
                    INSERT INTO ml_models (model_name, model_data, trained_at, metrics)
                    VALUES (%s, %s, NOW(), %s);
                """)
                db_cur.execute(insert_query, (model_name, model_binary, metrics_json))
                logger.info(f"✅ [DB Save] Successfully saved new ML model '{model_name}' to database.")
        conn.commit()
        logger.info(f"✅ [DB Save] Database commit executed successfully.")
        return True
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB Save] Database error while saving ML model: {db_err}", exc_info=True)
        if conn: conn.rollback()
        return False
    except pickle.PicklingError as pickle_err:
        logger.error(f"❌ [DB Save] Error pickling ML model: {pickle_err}", exc_info=True)
        if conn: conn.rollback()
        return False
    except Exception as e:
        logger.error(f"❌ [DB Save] Unexpected error while saving ML model: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

def cleanup_resources() -> None:
    """Closes used resources like the database connection."""
    global conn
    logger.info("ℹ️ [Cleanup] Closing resources...")
    if conn:
        try:
            conn.close()
            logger.info("✅ [DB] Database connection closed.")
        except Exception as close_err:
            logger.error(f"⚠️ [DB] Error closing database connection: {close_err}")
    logger.info("✅ [Cleanup] Resource cleanup complete.")

# ---------------------- Telegram Functions ----------------------
def send_telegram_message(target_chat_id: str, text: str, reply_markup: Optional[Dict] = None, parse_mode: str = 'Markdown', disable_web_page_preview: bool = True, timeout: int = 20) -> Optional[Dict]:
    """Sends a message via Telegram Bot API with improved error handling."""
    if not TELEGRAM_TOKEN or not target_chat_id:
        logger.warning("⚠️ [Telegram] Cannot send Telegram message: TELEGRAM_TOKEN or CHAT_ID not provided.")
        return None

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': str(target_chat_id),
        'text': text,
        'parse_mode': parse_mode,
        'disable_web_page_preview': disable_web_page_preview
    }
    if reply_markup:
        try:
            payload['reply_markup'] = json.dumps(convert_np_values(reply_markup))
        except (TypeError, ValueError) as json_err:
            logger.error(f"❌ [Telegram] Failed to convert reply_markup to JSON: {json_err} - Markup: {reply_markup}")
            return None

    logger.debug(f"ℹ️ [Telegram] Sending message to {target_chat_id}...")
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        logger.info(f"✅ [Telegram] Message successfully sent to {target_chat_id}.")
        return response.json()
    except requests.exceptions.Timeout:
        logger.error(f"❌ [Telegram] Failed to send message to {target_chat_id} (timeout).")
        return None
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"❌ [Telegram] Failed to send message to {target_chat_id} (HTTP error: {http_err.response.status_code}).")
        try:
            error_details = http_err.response.json()
            logger.error(f"❌ [Telegram] API error details: {error_details}")
        except json.JSONDecodeError:
            logger.error(f"❌ [Telegram] Could not decode error response: {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"❌ [Telegram] Failed to send message to {target_chat_id} (request error): {req_err}")
        return None
    except Exception as e:
        logger.error(f"❌ [Telegram] Unexpected error while sending message: {e}", exc_info=True)
        return None

# ---------------------- Flask Service ----------------------
app = Flask(__name__)

@app.route('/')
def home() -> Response:
    """Simple home page to show the bot is running."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    status_message = (
        f"🤖 *ML Trainer Service Status (SMC Strategy):*\n"
        f"- Current Time: {now}\n"
        f"- Training Status: *{training_status}*\n"
    )
    if last_training_time:
        status_message += f"- Last Training Time: {last_training_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    if last_training_metrics:
        status_message += f"- Last Training Metrics (Accuracy): {last_training_metrics.get('avg_accuracy', 'N/A'):.4f}\n"
        status_message += f"- Successful Models: {last_training_metrics.get('successful_models', 'N/A')}/{last_training_metrics.get('total_models_trained', 'N/A')}\n"
    if training_error:
        status_message += f"- Last Error: {training_error}\n"

    return Response(status_message, status=200, mimetype='text/plain')

@app.route('/favicon.ico')
def favicon() -> Response:
    """Handles favicon request to avoid 404 errors in logs."""
    return Response(status=204)

def run_flask_service() -> None:
    """Runs the Flask application."""
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"ℹ️ [Flask] Starting Flask application on {host}:{port}...")
    try:
        from waitress import serve
        logger.info("✅ [Flask] Using 'waitress' server.")
        serve(app, host=host, port=port, threads=6)
    except ImportError:
        logger.warning("⚠️ [Flask] 'waitress' not installed. Falling back to Flask development server (not recommended for production).")
        try:
            app.run(host=host, port=port)
        except Exception as flask_run_err:
            logger.critical(f"❌ [Flask] Failed to start development server: {flask_run_err}", exc_info=True)
    except Exception as serve_err:
        logger.critical(f"❌ [Flask] Failed to start server (waitress?): {serve_err}", exc_info=True)

# ---------------------- Main Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 Starting ML model training script with SMC strategy...")
    logger.info(f"Local Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | UTC Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    flask_thread: Optional[Thread] = None
    initial_training_start_time = datetime.now() # Track overall training duration

    # Pre-fetch BTC data once before the loop to optimize
    logger.info("ℹ️ [Main] Pre-fetching BTCUSDT historical data for all models...")
    global_btc_df = fetch_historical_data("BTCUSDT", interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if global_btc_df is None or global_btc_df.empty:
        logger.critical("❌ [Main] Critical: Failed to pre-fetch BTCUSDT data. Cannot proceed with training as BTC trend is a required feature.")
        training_status = "Failed: BTC data not available"
        if TELEGRAM_TOKEN and CHAT_ID:
            send_telegram_message(CHAT_ID,
                                  f"❌ *فشل بدء تدريب نموذج ML:*\n"
                                  f"لم يتمكن من جلب بيانات BTCUSDT التاريخية المطلوبة. توقف التدريب.",
                                  parse_mode='Markdown')
        exit(1)
    logger.info("✅ [Main] BTCUSDT historical data pre-fetched successfully.")

    try:
        # 1. Start Flask service in a separate thread first
        flask_thread = Thread(target=run_flask_service, daemon=False, name="FlaskServiceThread")
        flask_thread.start()
        logger.info("✅ [Main] Flask service started.")
        time.sleep(2) # Give some time for Flask to start

        # 2. Initialize the database
        init_db()

        # 3. Fetch list of symbols
        symbols = get_crypto_symbols()
        if not symbols:
            logger.critical("❌ [Main] No valid symbols for training. Please check 'crypto_list.txt'.")
            training_status = "Failed: No valid symbols"
            # Send Telegram notification for failure
            if TELEGRAM_TOKEN and CHAT_ID:
                send_telegram_message(CHAT_ID,
                                      f"❌ *ML Model Training Start Failed:*\n"
                                      f"No valid symbols for training. Please check `crypto_list.txt`.",
                                      parse_mode='Markdown')
            exit(1)

        training_status = "In Progress: Training Models with SMC"
        training_error = None # Reset error
        
        overall_metrics: Dict[str, Any] = {
            'total_models_trained': 0,
            'successful_models': 0,
            'failed_models': 0,
            'avg_accuracy': 0.0,
            'avg_precision': 0.0,
            'avg_recall': 0.0,
            'avg_f1_score': 0.0,
            'smc_features': ['bullish_ob', 'bearish_ob', 'bullish_fvg', 'bearish_fvg', 
                             'liquidity_pool_high', 'liquidity_pool_low'],
            'new_features': ['price_velocity', 'momentum', 'adx', 'trend_direction'],
            'details_per_symbol': {}
        }
        
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1_score = 0.0

        # Send Telegram notification for training start
        if TELEGRAM_TOKEN and CHAT_ID:
            send_telegram_message(CHAT_ID,
                                  f"🚀 *Starting ML Model Training with SMC Strategy:*\n"
                                  f"Training models for {len(symbols)} symbols.\n"
                                  f"Time: {initial_training_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                  f"Strategy: Smart Money Concept (SMC)\n"
                                  f"New Indicators: Velocity, Momentum, ADX",
                                  parse_mode='Markdown')

        # 4. Train a model for each symbol separately
        for symbol in symbols:
            current_model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
            overall_metrics['total_models_trained'] += 1
            logger.info(f"\n--- ⏳ [Main] Starting model training for {symbol} ({current_model_name}) with SMC features ---")
            
            try:
                # Fetch historical data for the current symbol
                df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
                if df_hist is None or df_hist.empty:
                    logger.warning(f"⚠️ [Main] Could not fetch sufficient data for {symbol}. Skipping model training for this symbol.")
                    overall_metrics['failed_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Failed: No data', 'error': 'No sufficient historical data'}
                    continue

                # Prepare data for the machine learning model, passing the pre-fetched BTC data
                df_processed = prepare_data_for_ml(df_hist, symbol, btc_df=global_btc_df)
                if df_processed is None or df_processed.empty:
                    logger.warning(f"⚠️ [Main] No data ready for training for {symbol} after SMC preprocessing. Skipping.")
                    overall_metrics['failed_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Failed: No processed data', 'error': 'No sufficient processed data'}
                    continue

                # Train and evaluate the model
                trained_model, model_metrics = train_and_evaluate_model(df_processed)

                if trained_model is None:
                    logger.error(f"❌ [Main] Model training failed for {symbol}. Cannot save.")
                    overall_metrics['failed_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Failed: Training failed', 'error': 'Model training returned None'}
                    continue

                # Save the model to the database
                if save_ml_model_to_db(trained_model, current_model_name, model_metrics):
                    logger.info(f"✅ [Main] Model '{current_model_name}' successfully saved to database.")
                    overall_metrics['successful_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Completed Successfully', 'metrics': model_metrics}
                    
                    total_accuracy += model_metrics.get('accuracy', 0.0)
                    total_precision += model_metrics.get('precision', 0.0)
                    total_recall += model_metrics.get('recall', 0.0)
                    total_f1_score += model_metrics.get('f1_score', 0.0)
                else:
                    logger.error(f"❌ [Main] Failed to save model '{current_model_name}' to database.")
                    overall_metrics['failed_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Completed with Errors: Model save failed', 'error': 'Failed to save model to DB'}

            except Exception as e:
                logger.critical(f"❌ [Main] A fatal error occurred during model training for {symbol}: {e}", exc_info=True)
                overall_metrics['failed_models'] += 1
                overall_metrics['details_per_symbol'][symbol] = {'status': 'Failed: Unhandled exception', 'error': str(e)}
            
            logger.info(f"--- ✅ [Main] Model training finished for {symbol} ---")
            time.sleep(1) # Small delay between model training

        # Update overall training status
        if overall_metrics['successful_models'] > 0:
            overall_metrics['avg_accuracy'] = total_accuracy / overall_metrics['successful_models']
            overall_metrics['avg_precision'] = total_precision / overall_metrics['successful_models']
            overall_metrics['avg_recall'] = total_recall / overall_metrics['successful_models']
            overall_metrics['avg_f1_score'] = total_f1_score / overall_metrics['successful_models']

        if overall_metrics['successful_models'] == overall_metrics['total_models_trained']:
            training_status = "Completed Successfully (All Models Trained)"
        elif overall_metrics['successful_models'] > 0:
            training_status = "Completed with Errors (Some Models Failed)"
        else:
            training_status = "Failed (No Models Trained Successfully)"
        
        last_training_time = datetime.now()
        last_training_metrics = overall_metrics

        # Calculate total training duration
        training_duration = last_training_time - initial_training_start_time
        training_duration_str = str(training_duration).split('.')[0] # Remove microseconds

        # Send Telegram notification for training completion/failure
        if TELEGRAM_TOKEN and CHAT_ID:
            if training_status == "Completed Successfully (All Models Trained)":
                message_title = "✅ *ML Model Training with SMC Completed Successfully!*"
            elif training_status == "Completed with Errors (Some Models Failed)":
                message_title = "⚠️ *ML Model Training with SMC Completed with Errors!*"
            else:
                message_title = "❌ *ML Model Training with SMC Failed!*"
            
            telegram_message = (
                f"{message_title}\n"
                f"————————————————\n"
                f"📊 *Summary:*\n"
                f"- Total Models Trained: {overall_metrics['total_models_trained']}\n"
                f"- Successful Models: {overall_metrics['successful_models']}\n"
                f"- Failed Models: {overall_metrics['failed_models']}\n"
                f"- Avg Accuracy: {overall_metrics['avg_accuracy']:.4f}\n"
                f"- Avg Precision: {overall_metrics['avg_precision']:.4f}\n"
                f"- Avg Recall: {overall_metrics['avg_recall']:.4f}\n"
                f"- Avg F1 Score: {overall_metrics['avg_f1_score']:.4f}\n"
                f"————————————————\n"
                f"⏱️ *Total Training Duration:* {training_duration_str}\n"
                f"⏰ *Completion Time:* {last_training_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"📈 *SMC Features Used:* {', '.join(overall_metrics['smc_features'])}\n"
                f"🆕 *New Features Added:* {', '.join(overall_metrics['new_features'])}"
            )
            if training_error:
                telegram_message += f"\n\n🚨 *General Error:* {training_error}"
            
            send_telegram_message(CHAT_ID, telegram_message, parse_mode='Markdown')

        # Wait for Flask thread to finish (which keeps the program running)
        if flask_thread:
            flask_thread.join()

    except Exception as e:
        logger.critical(f"❌ [Main] A fatal error occurred during the main training script execution: {e}", exc_info=True)
        training_status = "Failed: Unhandled exception in main loop"
        training_error = str(e)
        # Send Telegram notification for critical unhandled error
        if TELEGRAM_TOKEN and CHAT_ID:
            error_message = (
                f"🚨 *Critical Error in ML Model Training Script with SMC:*\n"
                f"An unexpected error occurred and stopped the script.\n"
                f"Details: `{e}`\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            send_telegram_message(CHAT_ID, error_message, parse_mode='Markdown')
    finally:
        logger.info("🛑 [Main] Shutting down training script...")
        cleanup_resources()
        logger.info("👋 [Main] ML model training script stopped.")
