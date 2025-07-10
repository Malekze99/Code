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
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from lightgbm import early_stopping, log_evaluation

# New imports for advanced features and optimization

#import ta # Technical Analysis library
import ta
import inspect # لإلقاء نظرة داخل الكائنات
import sys

print(f"DEBUG: ta module path: {ta.__file__}")
print(f"DEBUG: sys.path: {sys.path}")
#print(f"DEBUG: ta version: {ta.__version__}")

# التحقق مما إذا كانت true_range موجودة في ta.volatility
if hasattr(ta.volatility, 'true_range'):
    print("DEBUG: 'true_range' function IS found in ta.volatility.")
else:
    print("DEBUG: 'true_range' function IS NOT found in ta.volatility.")
    # إذا لم يتم العثور عليها، اطبع محتويات ta.volatility
    print("DEBUG: Contents of ta.volatility module:")
    for name, obj in inspect.getmembers(ta.volatility):
        if not name.startswith('_'): # تجاهل السمات الداخلية
            print(f"  - {name}")

# حاول استيرادها مباشرة
try:
    from ta.volatility import true_range
    print("DEBUG: Successfully imported true_range directly.")
except ImportError as e:
    print(f"DEBUG: Failed to import true_range directly: {e}")

# (استمر الكود الخاص بك هنا...)


##########

from imblearn.over_sampling import SMOTE # For handling class imbalance
import optuna # For hyperparameter optimization

# Suppress warnings from Optuna if not needed in logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


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
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 120 # Increased lookback for more data
BASE_ML_MODEL_NAME: str = 'LightGBM_SMC_Scalping_V2' # Updated model name for new version

# Indicator Parameters (copied from c4.py for consistency)
VOLUME_LOOKBACK_CANDLES: int = 3
RSI_PERIOD: int = 9
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 2

# SMC Parameters
LIQUIDITY_WINDOW: int = 20
SWING_WINDOW: int = 5
ORDER_BLOCK_VOLUME_MULTIPLIER: float = 1.5

# New: Lagged Features Parameters
LAG_PERIODS: List[int] = [1, 2, 3, 5] # Lags for various features

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
        start_dt = datetime.utcnow() - timedelta(days=days + 1)
        start_str_overall = start_dt.strftime("%Y-%m-%d %H:%M:%S")

        logger.debug(f"ℹ️ [Data] Fetching {interval} data for {symbol} from {start_str_overall} to now...")

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

    # Avoid division by zero for rs
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    
    # Fill any NaNs created by initial periods or division by zero, typically with 50 (neutral)
    df['rsi'] = rsi_series.ffill().fillna(50) 
    df['rsi'].replace([np.inf, -np.inf], 50, inplace=True) # Handle cases where avg_loss is zero

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
    trend_series[(ema_df['close'] > ema_df['ema20']) & (ema_df['ema20'] > ema_df['ema50'])] = 1.0 # Bullish
    trend_series[(ema_df['close'] < ema_df['ema20']) & (ema_df['ema20'] < ema_df['ema50'])] = -1.0 # Bearish

    final_trend_series = trend_series.reindex(df_btc.index).fillna(0.0)
    logger.debug(f"✅ [Indicators] Bitcoin trend feature calculated. Examples: {final_trend_series.tail().tolist()}")
    return final_trend_series

# ---------------------- SMC Indicator Functions ----------------------
def detect_liquidity_pools(df: pd.DataFrame, window: int = LIQUIDITY_WINDOW) -> pd.DataFrame:
    """
    Detect liquidity pools (areas of high trading activity)
    """
    df = df.copy()
    if len(df) < window:
        logger.warning(f"⚠️ [SMC] Insufficient data ({len(df)} < {window}) to detect liquidity pools.")
        df['liquidity_pool_high'] = 0
        df['liquidity_pool_low'] = 0
        return df
        
    df['high_range'] = df['high'].rolling(window, min_periods=1).max()
    df['low_range'] = df['low'].rolling(window, min_periods=1).min()
    
    # Identify liquidity pools as areas near recent highs/lows
    # Using a small percentage tolerance to account for slight variations
    df['liquidity_pool_high'] = ((df['high'] >= df['high_range'] * 0.999) & 
                                (df['high'] == df['high'].rolling(3, center=True, min_periods=1).max())).astype(int)
    df['liquidity_pool_low'] = ((df['low'] <= df['low_range'] * 1.001) & 
                               (df['low'] == df['low'].rolling(3, center=True, min_periods=1).min())).astype(int)
    
    # New: Add lagged liquidity
    for lag in LAG_PERIODS:
        df[f'liquidity_pool_high_lag{lag}'] = df['liquidity_pool_high'].shift(lag).fillna(0).astype(int)
        df[f'liquidity_pool_low_lag{lag}'] = df['liquidity_pool_low'].shift(lag).fillna(0).astype(int)

    return df

def identify_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify bullish/bearish order blocks with strength.
    An order block is typically the last down/up candle before a strong move up/down.
    """
    df = df.copy()
    df['bullish_ob'] = 0
    df['bearish_ob'] = 0
    df['bullish_ob_strength'] = 0.0 # New
    df['bearish_ob_strength'] = 0.0 # New
    
    if len(df) < 3:
        logger.warning("⚠️ [SMC] Insufficient data ({len(df)} < 3) to identify order blocks.")
        return df
    
    # Calculate rolling volume threshold
    volume_threshold_series = df['volume'].rolling(window=5, min_periods=1).mean() * ORDER_BLOCK_VOLUME_MULTIPLIER
    
    for i in range(1, len(df) - 1):
        prev_candle = df.iloc[i-1]
        current_candle = df.iloc[i]
        next_candle = df.iloc[i+1]
        
        # Bullish OB: down candle followed by large up candle (engulfing/strong move)
        # and current candle volume is higher than threshold
        is_prev_down = prev_candle['close'] < prev_candle['open']
        is_current_up = current_candle['close'] > current_candle['open']
        is_strong_volume = current_candle['volume'] > volume_threshold_series.iloc[i]
        is_engulfing_or_strong_move = (current_candle['close'] > prev_candle['high']) # Engulfs previous high
        
        if is_prev_down and is_current_up and is_strong_volume and is_engulfing_or_strong_move:
            df.loc[df.index[i], 'bullish_ob'] = 1
            df.loc[df.index[i], 'bullish_ob_strength'] = (current_candle['close'] - current_candle['open']) / current_candle['open'] * current_candle['volume']
        
        # Bearish OB: up candle followed by large down candle (engulfing/strong move)
        is_prev_up = prev_candle['close'] > prev_candle['open']
        is_current_down = current_candle['close'] < current_candle['open']
        is_engulfing_or_strong_move_bear = (current_candle['close'] < prev_candle['low']) # Engulfs previous low
        
        if is_prev_up and is_current_down and is_strong_volume and is_engulfing_or_strong_move_bear:
            df.loc[df.index[i], 'bearish_ob'] = 1
            df.loc[df.index[i], 'bearish_ob_strength'] = (current_candle['open'] - current_candle['close']) / current_candle['open'] * current_candle['volume']

    # New: Add lagged order blocks
    for lag in LAG_PERIODS:
        df[f'bullish_ob_lag{lag}'] = df['bullish_ob'].shift(lag).fillna(0).astype(int)
        df[f'bearish_ob_lag{lag}'] = df['bearish_ob'].shift(lag).fillna(0).astype(int)
        df[f'bullish_ob_strength_lag{lag}'] = df['bullish_ob_strength'].shift(lag).fillna(0.0)
        df[f'bearish_ob_strength_lag{lag}'] = df['bearish_ob_strength'].shift(lag).fillna(0.0)

    return df

def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Fair Value Gaps (FVG) with size.
    A bullish FVG occurs when current_low > prev_high.
    A bearish FVG occurs when current_high < prev_low.
    """
    df = df.copy()
    df['bullish_fvg'] = 0
    df['bearish_fvg'] = 0
    df['bullish_fvg_size'] = 0.0 # New
    df['bearish_fvg_size'] = 0.0 # New
    
    if len(df) < 3: # Need at least 3 candles to define FVG (i-1, i, i+1)
        logger.warning("⚠️ [SMC] Insufficient data ({len(df)} < 3) to detect FVGs.")
        return df
    
    for i in range(1, len(df) - 1): # Iterate from second candle to second last
        candle_i_minus_1 = df.iloc[i-1]
        candle_i = df.iloc[i]
        candle_i_plus_1 = df.iloc[i+1]
        
        # Bullish FVG: current_low (candle_i) > previous_high (candle_i_minus_1) AND next_high (candle_i_plus_1) < current_low (candle_i)
        # This is a common interpretation (between candle_i-1 high and candle_i+1 low)
        if (candle_i_plus_1['low'] > candle_i_minus_1['high']): # This is the more common FVG definition for bullish
            df.loc[df.index[i], 'bullish_fvg'] = 1
            df.loc[df.index[i], 'bullish_fvg_size'] = candle_i_plus_1['low'] - candle_i_minus_1['high']
        
        # Bearish FVG: current_high (candle_i) < previous_low (candle_i_minus_1) AND next_low (candle_i_plus_1) > current_high (candle_i)
        if (candle_i_plus_1['high'] < candle_i_minus_1['low']): # This is the more common FVG definition for bearish
            df.loc[df.index[i], 'bearish_fvg'] = 1
            df.loc[df.index[i], 'bearish_fvg_size'] = candle_i_minus_1['low'] - candle_i_plus_1['high']

    # New: Add lagged FVGs
    for lag in LAG_PERIODS:
        df[f'bullish_fvg_lag{lag}'] = df['bullish_fvg'].shift(lag).fillna(0).astype(int)
        df[f'bearish_fvg_lag{lag}'] = df['bearish_fvg'].shift(lag).fillna(0).astype(int)
        df[f'bullish_fvg_size_lag{lag}'] = df['bullish_fvg_size'].shift(lag).fillna(0.0)
        df[f'bearish_fvg_size_lag{lag}'] = df['bearish_fvg_size'].shift(lag).fillna(0.0)
    
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
    
    if len(df) < swing_window * 2 + 1: # Need enough data for two swing points
        logger.warning(f"⚠️ [SMC] Insufficient data ({len(df)} < {swing_window*2+1}) to identify BOS/CHOCH.")
        return df
    
    # Calculate swing highs and lows (using a simple rolling window for now)
    # A swing high is a candle whose high is the highest in the window around it
    # A swing low is a candle whose low is the lowest in the window around it
    df['is_swing_high'] = (df['high'] == df['high'].rolling(swing_window * 2 + 1, center=True, min_periods=1).max()).astype(int)
    df['is_swing_low'] = (df['low'] == df['low'].rolling(swing_window * 2 + 1, center=True, min_periods=1).min()).astype(int)

    # Get the actual swing high/low values for detected swing points, NaN otherwise
    df['actual_swing_high'] = df['high'].where(df['is_swing_high'] == 1)
    df['actual_swing_low'] = df['low'].where(df['is_swing_low'] == 1)

    # Fill NaNs for swing points using ffill/bfill to propagate the last known swing point
    df['last_swing_high'] = df['actual_swing_high'].ffill().bfill()
    df['last_swing_low'] = df['actual_swing_low'].ffill().bfill()

    # Track market structure (simple trend tracking based on highs/lows)
    df['market_structure'] = 0 # 1: Bullish, -1: Bearish, 0: Ranging/Undetermined

    # Identify BOS/CHOCH based on breaking confirmed swing points
    for i in range(1, len(df)):
        current_close = df.iloc[i]['close']
        prev_close = df.iloc[i-1]['close']
        
        # Bullish BOS: Break above a confirmed previous swing high
        # Look for a break above the *last identified* swing high
        if current_close > df.iloc[i]['last_swing_high'] and df.iloc[i]['last_swing_high'] > df.iloc[i]['last_swing_low']:
            df.loc[df.index[i], 'bullish_bos'] = 1
            df.loc[df.index[i], 'market_structure'] = 1 # Confirm bullish structure
        
        # Bearish BOS: Break below a confirmed previous swing low
        if current_close < df.iloc[i]['last_swing_low'] and df.iloc[i]['last_swing_low'] < df.iloc[i]['last_swing_high']:
            df.loc[df.index[i], 'bearish_bos'] = 1
            df.loc[df.index[i], 'market_structure'] = -1 # Confirm bearish structure

        # CHOCH (Change of Character): Reversal in market structure
        # Bullish CHOCH: Price breaks higher than a previous swing high after a bearish structure
        if (df.loc[df.index[i-1], 'market_structure'] == -1 and 
            current_close > df.loc[df.index[i], 'last_swing_high'] and
            df.loc[df.index[i], 'bullish_bos'] == 0): # Not a BOS, but a change from bearish to potentially bullish
            df.loc[df.index[i], 'bullish_choch'] = 1
            df.loc[df.index[i], 'market_structure'] = 1

        # Bearish CHOCH: Price breaks lower than a previous swing low after a bullish structure
        if (df.loc[df.index[i-1], 'market_structure'] == 1 and 
            current_close < df.loc[df.index[i], 'last_swing_low'] and
            df.loc[df.index[i], 'bearish_bos'] == 0): # Not a BOS, but a change from bullish to potentially bearish
            df.loc[df.index[i], 'bearish_choch'] = 1
            df.loc[df.index[i], 'market_structure'] = -1
    
    # New: Add lagged BOS/CHOCH
    for lag in LAG_PERIODS:
        df[f'bullish_bos_lag{lag}'] = df['bullish_bos'].shift(lag).fillna(0).astype(int)
        df[f'bearish_bos_lag{lag}'] = df['bearish_bos'].shift(lag).fillna(0).astype(int)
        df[f'bullish_choch_lag{lag}'] = df['bullish_choch'].shift(lag).fillna(0).astype(int)
        df[f'bearish_choch_lag{lag}'] = df['bearish_choch'].shift(lag).fillna(0).astype(int)
        df[f'market_structure_lag{lag}'] = df['market_structure'].shift(lag).fillna(0)

    # Drop temporary swing point calculation columns
    df.drop(columns=['is_swing_high', 'is_swing_low', 'actual_swing_high', 'actual_swing_low', 
                     'last_swing_high', 'last_swing_low'], errors='ignore', inplace=True)
    
    return df

# ---------------------- Model Training and Saving Functions ----------------------
def prepare_data_for_ml(df: pd.DataFrame, symbol: str, target_period: int = 5, btc_df: Optional[pd.DataFrame] = None, eth_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """
    Prepares data for machine learning model training with extensive features.
    """
    logger.info(f"ℹ️ [ML Prep] Preparing data for ML model for {symbol} with extensive features...")

    # Determine minimum data length required (increased for more features and lags)
    min_len_required = max(
        VOLUME_LOOKBACK_CANDLES,
        RSI_PERIOD + RSI_MOMENTUM_LOOKBACK_CANDLES,
        LIQUIDITY_WINDOW + max(LAG_PERIODS) + 10, # Adjusted for new lags
        SWING_WINDOW * 2 + 1 + max(LAG_PERIODS) + 10, # Adjusted for new lags
        target_period + 10,
        20 + max(LAG_PERIODS) + 5 # For ATR, OBV, MFI etc.
    )

    if len(df) < min_len_required:
        logger.warning(f"⚠️ [ML Prep] DataFrame for {symbol} is too short ({len(df)} < {min_len_required}) for data preparation.")
        return None

    df_calc = df.copy()

    # --- Core Features ---
    df_calc['volume_15m_avg'] = df_calc['volume'].rolling(window=VOLUME_LOOKBACK_CANDLES, min_periods=1).mean()
    df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
    
    # Bullish RSI momentum indicator
    df_calc['rsi_momentum_bullish'] = 0
    if len(df_calc) >= RSI_MOMENTUM_LOOKBACK_CANDLES + 1:
        # Check for consistent increase in RSI over lookback period and above 50
        is_increasing_rsi = (df_calc['rsi'].diff(periods=1) > 0).rolling(window=RSI_MOMENTUM_LOOKBACK_CANDLES, min_periods=1).sum() == RSI_MOMENTUM_LOOKBACK_CANDLES
        df_calc['rsi_momentum_bullish'] = (is_increasing_rsi & (df_calc['rsi'] > 50)).astype(int)

    # Bearish RSI momentum indicator (New)
    df_calc['rsi_momentum_bearish'] = 0
    if len(df_calc) >= RSI_MOMENTUM_LOOKBACK_CANDLES + 1:
        # Check for consistent decrease in RSI over lookback period and below 50
        is_decreasing_rsi = (df_calc['rsi'].diff(periods=1) < 0).rolling(window=RSI_MOMENTUM_LOOKBACK_CANDLES, min_periods=1).sum() == RSI_MOMENTUM_LOOKBACK_CANDLES
        df_calc['rsi_momentum_bearish'] = (is_decreasing_rsi & (df_calc['rsi'] < 50)).astype(int)
    logger.debug(f"ℹ️ [ML Prep] Calculated RSI momentum indicators for {symbol}.")


    # --- Volatility & Momentum Features ---
    df_calc['true_range'] = ta.volatility.true_range(high=df_calc['high'], low=df_calc['low'], close=df_calc['close'])
    df_calc['atr'] = ta.volatility.average_true_range(high=df_calc['high'], low=df_calc['low'], close=df_calc['close'], window=14)
    df_calc['atr_20'] = ta.volatility.average_true_range(high=df_calc['high'], low=df_calc['low'], close=df_calc['close'], window=20)
    df_calc['price_change_ratio'] = (df_calc['close'] - df_calc['open']) / df_calc['open']
    df_calc['high_low_range_ratio'] = (df_calc['high'] - df_calc['low']) / df_calc['open']
    logger.debug(f"ℹ️ [ML Prep] Calculated volatility features for {symbol}.")


    # --- Smart Volume Features ---
    df_calc['obv'] = ta.volume.on_balance_volume(close=df_calc['close'], volume=df_calc['volume'])
    df_calc['mfi'] = ta.volume.money_flow_index(high=df_calc['high'], low=df_calc['low'], close=df_calc['close'], volume=df_calc['volume'], window=14)
    df_calc['volume_ratio_20'] = df_calc['volume'] / df_calc['volume'].rolling(window=20, min_periods=1).mean()
    logger.debug(f"ℹ️ [ML Prep] Calculated smart volume features for {symbol}.")


    # --- SMC Indicators ---
    df_calc = detect_liquidity_pools(df_calc)
    df_calc = identify_order_blocks(df_calc)
    df_calc = detect_fvg(df_calc)
    df_calc = identify_bos_choch(df_calc)
    logger.info(f"ℹ️ [ML Prep] Calculated SMC indicators for {symbol}")

    # --- Market Context Features (BTC & ETH Dominance/Trend) ---
    def _get_and_process_dominance(asset_symbol: str, global_df: Optional[pd.DataFrame]) -> pd.Series:
        local_df = global_df
        if local_df is None:
            logger.debug(f"ℹ️ [ML Prep] {asset_symbol} data not provided, fetching...")
            local_df = fetch_historical_data(asset_symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
        
        trend_series = pd.Series(index=df_calc.index, data=0.0) # Default to 0.0
        if local_df is not None and not local_df.empty:
            raw_trend = _calculate_btc_trend_feature(local_df) # This function is generic enough for any asset
            if raw_trend is not None:
                trend_series = raw_trend.reindex(df_calc.index, method='nearest').fillna(0.0)
            else:
                logger.warning(f"⚠️ [ML Prep] Failed to calculate {asset_symbol} trend feature.")
        else:
            logger.warning(f"⚠️ [ML Prep] Failed to fetch {asset_symbol} historical data.")
        return trend_series

    df_calc['btc_trend_feature'] = _get_and_process_dominance("BTCUSDT", btc_df)
    df_calc['eth_trend_feature'] = _get_and_process_dominance("ETHUSDT", eth_df) # Assuming ETHUSDT for ETH dominance trend

    # New: Rolling Correlation with BTC (as a proxy for market sentiment)
    # Ensure BTC close series is aligned to df_calc index for correlation
    if btc_df is not None and not btc_df.empty:
        btc_close_aligned = btc_df['close'].reindex(df_calc.index, method='nearest').fillna(method='ffill').fillna(method='bfill')
        if not btc_close_aligned.isnull().all() and len(df_calc) > 20: # Ensure enough data for rolling correlation
            df_calc['rolling_corr_btc'] = df_calc['close'].rolling(window=20).corr(btc_close_aligned)
            df_calc['rolling_corr_btc'].fillna(0, inplace=True) # Fill NaNs (e.g., initial periods) with 0
        else:
            logger.warning(f"⚠️ [ML Prep] Not enough data or issues with BTC close for rolling correlation for {symbol}. Setting to 0.")
            df_calc['rolling_corr_btc'] = 0.0
    else:
        df_calc['rolling_corr_btc'] = 0.0
    logger.debug(f"ℹ️ [ML Prep] Calculated market context features for {symbol}.")


    # --- Time-Based Features (Cyclical) ---
    df_calc['hour_sin'] = np.sin(2 * np.pi * df_calc.index.hour / 24)
    df_calc['hour_cos'] = np.cos(2 * np.pi * df_calc.index.hour / 24)
    df_calc['day_of_week_sin'] = np.sin(2 * np.pi * df_calc.index.dayofweek / 7)
    df_calc['day_of_week_cos'] = np.cos(2 * np.pi * df_calc.index.dayofweek / 7)
    logger.debug(f"ℹ️ [ML Prep] Calculated cyclical time features for {symbol}.")


    # --- Add Lagged Features for a broad set of initial features ---
    # Base features to lag
    features_to_lag = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'volume_15m_avg', 'atr', 'obv', 'mfi',
        'bullish_ob', 'bearish_ob', 'bullish_fvg', 'bearish_fvg',
        'bullish_bos', 'bearish_bos', 'bullish_choch', 'bearish_choch',
        'liquidity_pool_high', 'liquidity_pool_low',
        'btc_trend_feature', 'eth_trend_feature'
    ]
    
    for feature in features_to_lag:
        for lag in LAG_PERIODS:
            # Check if feature exists before trying to lag it
            if feature in df_calc.columns:
                df_calc[f'{feature}_lag{lag}'] = df_calc[feature].shift(lag)
            else:
                logger.warning(f"⚠️ [ML Prep] Feature '{feature}' not found in df_calc for lagging. This might indicate an earlier issue.")

    logger.info(f"ℹ️ [ML Prep] Added lagged features for {symbol}.")


    # Define final list of feature columns
    # We include all features calculated, ensuring they exist and handle NaNs later
    feature_columns = [col for col in df_calc.columns if col not in ['target', 'future_max_close', 'true_range', 'is_swing_high', 'is_swing_low', 'actual_swing_high', 'actual_swing_low', 'last_swing_high', 'last_swing_low']]
    
    # Ensure all features are numeric and fill NaNs
    for col in feature_columns:
        if col not in df_calc.columns:
            # This should ideally not happen if features_to_lag is properly handled
            logger.warning(f"⚠️ [ML Prep] Final feature column '{col}' is missing after all calculations. Adding as 0.")
            df_calc[col] = 0
        else:
            df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce').fillna(0)


    # Create target variable (Dynamic Threshold)
    # Target: Price increase by at least X% OR X * ATR within target_period
    # Let's use a dynamic threshold based on ATR, if ATR is available.
    # Otherwise, fall back to fixed percentage.
    
    # Calculate a volatility-adjusted threshold
    # Example: 0.5% or 0.5 * ATR whichever is larger/more appropriate for a meaningful move
    # Ensure ATR is calculated and numeric before using
    df_calc['atr_for_target'] = pd.to_numeric(df_calc['atr'], errors='coerce').fillna(0.0001) # Avoid division by zero
    
    # Dynamic target threshold: 0.5% or 0.5 * ATR (if ATR is significant)
    # Using 0.005 as a base, or 0.5 * ATR, capped to prevent extreme values
    # We define "meaningful move" as at least 0.5% or half of daily ATR, whichever is higher
    # This means a smaller move is considered significant in low volatility, larger in high volatility
    base_percentage_target = 0.005 # 0.5%
    df_calc['dynamic_target_threshold'] = np.maximum(base_percentage_target, 0.5 * df_calc['atr_for_target'] / df_calc['close'])
    
    # Future max close in the target period
    df_calc['future_max_close'] = df_calc['close'].shift(-target_period).rolling(window=target_period, min_periods=1).max()

    # Target: 1 if future max close exceeds current close by dynamic threshold
    df_calc['target'] = ((df_calc['future_max_close'] / df_calc['close']) - 1 > df_calc['dynamic_target_threshold']).astype(int)

    initial_len = len(df_calc)
    df_cleaned = df_calc.dropna(subset=feature_columns + ['target']).copy()
    dropped_count = initial_len - len(df_cleaned)

    if dropped_count > 0:
        logger.info(f"ℹ️ [ML Prep] For {symbol}: Dropped {dropped_count} rows due to NaN values after indicator, lagged, and target calculation.")
    if df_cleaned.empty:
        logger.warning(f"⚠️ [ML Prep] DataFrame for {symbol} is empty after removing NaNs for ML preparation. Skipping.")
        return None

    # Final check: ensure 'target' column has both 0s and 1s for stratification
    if df_cleaned['target'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] Target variable for {symbol} has only one class ({df_cleaned['target'].iloc[0]}). Cannot train a classifier. Skipping.")
        return None

    logger.info(f"✅ [ML Prep] Data prepared for {symbol} successfully. Number of rows: {len(df_cleaned)}, features: {len(feature_columns)}")
    return df_cleaned[feature_columns + ['target']]


def objective(trial: optuna.Trial, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, sample_weight_train: np.ndarray) -> float:
    """
    Objective function for Optuna to optimize LightGBM hyperparameters.
    Optimizes for F1-score on the test set.
    """
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 128),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced', # Keep balanced as a baseline for handling imbalance
        'verbose': -1, # Suppress verbose output during trials
    }

    model = lgb.LGBMClassifier(**lgb_params)
    
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_test, y_test)],
        eval_metric='binary_logloss',
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)] # Use verbose=False to suppress early stopping messages
    )
    
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    return f1


def train_and_evaluate_model(data: pd.DataFrame, symbol: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains a LightGBM model with extensive features, performs hyperparameter tuning,
    handles class imbalance, and evaluates its performance using TimeSeriesSplit.
    """
    logger.info(f"ℹ️ [ML Train] Starting model training and evaluation for {symbol}...")

    if data.empty:
        logger.error("❌ [ML Train] Empty DataFrame for training.")
        return None, {}

    X_full = data.drop('target', axis=1)
    y_full = data['target']

    if X_full.empty or y_full.empty:
        logger.error("❌ [ML Train] Empty features or targets for training.")
        return None, {}

    # Initialize scaler outside of CV loop
    scaler = StandardScaler()
    
    # Store metrics for each fold
    fold_metrics = []
    trained_models = [] # Store models to pick the best one or average

    # Use TimeSeriesSplit for robust validation
    # Adjust n_splits based on data size, e.g., 5 or more
    tscv = TimeSeriesSplit(n_splits=5) 
    
    # For Optuna optimization, we'll pick one fold or use a simplified approach
    # For simplicity and to integrate Optuna with TimeSeriesSplit, we can perform
    # Optuna tuning on the last split (most recent data for test set)
    # Or run Optuna across all folds and average results (more complex, but better)
    # For this implementation, we will use a simpler approach: fit scaler and SMOTE
    # on train data of each fold, then train and evaluate.
    # The hyperparameter tuning with Optuna will be done once on a representative split or full data
    # (or you can integrate it per fold, which is more robust but much slower).
    # For robust production, tuning *per fold* is ideal, but here we do it once before CV loop for speed.

    # --- Hyperparameter Tuning using Optuna ---
    # For simplicity, let's take the last fold's data for tuning
    # In a full production system, you might do this on a dedicated validation set,
    # or iterate Optuna for each fold and average/select best.
    
    logger.info(f"ℹ️ [ML Train] Performing hyperparameter tuning for {symbol} using Optuna (on the last time series split). This may take a while...")

    X_train_tune, X_test_tune, y_train_tune, y_test_tune = None, None, None, None
    for train_idx, test_idx in tscv.split(X_full):
        X_train_tune, X_test_tune = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train_tune, y_test_tune = y_full.iloc[train_idx], y_full.iloc[test_idx]

    if X_train_tune is None or X_test_tune is None or y_train_tune is None or y_test_tune is None:
        logger.error(f"❌ [ML Train] Not enough data to perform TimeSeriesSplit for {symbol}.")
        return None, {}

    # Scale tuning data
    X_train_tune_scaled = scaler.fit_transform(X_train_tune)
    X_test_tune_scaled = scaler.transform(X_test_tune)
    
    X_train_tune_scaled_df = pd.DataFrame(X_train_tune_scaled, columns=X_train_tune.columns, index=X_train_tune.index)
    X_test_tune_scaled_df = pd.DataFrame(X_test_tune_scaled, columns=X_test_tune.columns, index=X_test_tune.index)

    # Apply SMOTE to the training tuning data
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, len(y_train_tune[y_train_tune == 1]) - 1)) # Ensure k_neighbors is valid
        if smote.k_neighbors < 1: # Fallback if not enough minority samples
             logger.warning(f"⚠️ [ML Train] Not enough minority samples for SMOTE on {symbol}. Skipping SMOTE for tuning.")
             X_train_smote, y_train_smote = X_train_tune_scaled_df, y_train_tune
        else:
            X_train_smote, y_train_smote = smote.fit_resample(X_train_tune_scaled_df, y_train_tune)
            logger.info(f"ℹ️ [ML Train] SMOTE applied for tuning data: Original {len(y_train_tune)} samples, SMOTE {len(y_train_smote)} samples.")
    except Exception as e:
        logger.warning(f"⚠️ [ML Train] Failed to apply SMOTE for tuning data for {symbol}: {e}. Proceeding without SMOTE for tuning.")
        X_train_smote, y_train_smote = X_train_tune_scaled_df, y_train_tune

    # Create sample weights for tuning data
    sample_weights_tune = np.ones(len(y_train_smote))
    # Ensure SMC features exist in X_train_smote before trying to select them
    smc_features_present_tune = [f for f in ['bullish_ob', 'bearish_ob', 'bullish_fvg', 'bearish_fvg'] if f in X_train_smote.columns]
    if smc_features_present_tune:
        smc_signals_tune = X_train_smote[smc_features_present_tune].max(axis=1)
        sample_weights_tune[smc_signals_tune == 1] = 2.0 # Double weight for SMC signals


    # --- Optuna Study ---
    try:
        study = optuna.create_study(direction='maximize', study_name=f"lgbm_tuning_{symbol}", load_if_exists=True)
        # Pass sample_weights_tune here
        study.optimize(lambda trial: objective(trial, X_train_smote, X_test_tune_scaled_df, y_train_smote, y_test_tune, sample_weights_tune), n_trials=50, show_progress_bar=True)
        best_params = study.best_params_
        logger.info(f"✅ [ML Train] Optuna Best hyperparameters for {symbol}: {best_params}")
    except Exception as e:
        logger.error(f"❌ [ML Train] Optuna tuning failed for {symbol}: {e}. Falling back to default/pre-defined parameters.", exc_info=True)
        # Fallback to good default parameters if Optuna fails
        best_params = {
            'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31,
            'max_depth': -1, 'min_child_samples': 20, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1
        }

    # --- Train and Evaluate with TimeSeriesSplit and Best Params ---
    for fold_num, (train_index, test_index) in enumerate(tscv.split(X_full), 1):
        logger.info(f"ℹ️ [ML Train] Training Fold {fold_num} for {symbol}...")
        X_train, X_test = X_full.iloc[train_index], X_full.iloc[test_index]
        y_train, y_test = y_full.iloc[train_index], y_full.iloc[test_index]

        # Scale data for the current fold
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Apply SMOTE for the current fold's training data
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, len(y_train[y_train == 1]) - 1))
            if smote.k_neighbors < 1: # Fallback if not enough minority samples
                 logger.warning(f"⚠️ [ML Train] Not enough minority samples for SMOTE on {symbol} in fold {fold_num}. Skipping SMOTE.")
                 X_train_resampled, y_train_resampled = X_train_scaled_df, y_train
            else:
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled_df, y_train)
                logger.info(f"ℹ️ [ML Train] SMOTE applied for fold {fold_num}: Original {len(y_train)} samples, Resampled {len(y_train_resampled)} samples.")
        except Exception as e:
            logger.warning(f"⚠️ [ML Train] Failed to apply SMOTE for {symbol} in fold {fold_num}: {e}. Proceeding without SMOTE.")
            X_train_resampled, y_train_resampled = X_train_scaled_df, y_train

        # Create sample weights based on SMC signals for the current fold
        sample_weights = np.ones(len(y_train_resampled))
        smc_features_present = [f for f in ['bullish_ob', 'bearish_ob', 'bullish_fvg', 'bearish_fvg'] if f in X_train_resampled.columns]
        if smc_features_present:
            smc_signals = X_train_resampled[smc_features_present].max(axis=1)
            sample_weights[smc_signals == 1] = 2.0 # Double weight for SMC signals


        # Train LightGBM model with best_params
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced', # Still keep this as a general guideline
            **best_params # Apply parameters from Optuna
        )
        
        try:
            model.fit(
                X_train_resampled,
                y_train_resampled,
                sample_weight=sample_weights,
                eval_set=[(X_test_scaled_df, y_test)],
                eval_metric='binary_logloss',
                callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=10)]
            )
            
            y_pred = model.predict(X_test_scaled_df)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            fold_metrics.append({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'fold_num': fold_num
            })
            trained_models.append(model) # Store the model from this fold

            logger.info(f"📊 [ML Train] Fold {fold_num} Metrics for {symbol}: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")

        except ValueError as ve:
            logger.error(f"❌ [ML Train] Error during model fit for {symbol} (Fold {fold_num}): {ve}. This often means single class in target/resampled data.", exc_info=True)
            # If a fold fails, store null metrics and continue
            fold_metrics.append({'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'fold_num': fold_num, 'error': str(ve)})
        except Exception as e:
            logger.error(f"❌ [ML Train] Unexpected error during model training for {symbol} (Fold {fold_num}): {e}", exc_info=True)
            fold_metrics.append({'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'fold_num': fold_num, 'error': str(e)})


    if not trained_models:
        logger.error(f"❌ [ML Train] No models were successfully trained for {symbol} across all folds.")
        return None, {}

    # Aggregate metrics across all successful folds
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics if 'error' not in m])
    avg_precision = np.mean([m['precision'] for m in fold_metrics if 'error' not in m])
    avg_recall = np.mean([m['recall'] for m in fold_metrics if 'error' not in m])
    avg_f1_score = np.mean([m['f1_score'] for m in fold_metrics if 'error' not in m])

    # For saving, you can choose to save the model from the last fold, or retrain one final model on all data
    # Retraining on all data with best params for final model:
    final_model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        **best_params
    )
    
    # Scale full data
    X_full_scaled = scaler.fit_transform(X_full)
    X_full_scaled_df = pd.DataFrame(X_full_scaled, columns=X_full.columns, index=X_full.index)

    # Apply SMOTE to full data before final training
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, len(y_full[y_full == 1]) - 1))
        if smote.k_neighbors < 1:
            logger.warning(f"⚠️ [ML Train] Not enough minority samples for SMOTE on full data for {symbol}. Skipping SMOTE for final training.")
            X_full_resampled, y_full_resampled = X_full_scaled_df, y_full
        else:
            X_full_resampled, y_full_resampled = smote.fit_resample(X_full_scaled_df, y_full)
            logger.info(f"ℹ️ [ML Train] SMOTE applied for final training: Original {len(y_full)} samples, Resampled {len(y_full_resampled)} samples.")
    except Exception as e:
        logger.warning(f"⚠️ [ML Train] Failed to apply SMOTE for final training for {symbol}: {e}. Proceeding without SMOTE.")
        X_full_resampled, y_full_resampled = X_full_scaled_df, y_full

    # Sample weights for final training
    final_sample_weights = np.ones(len(y_full_resampled))
    smc_features_present_final = [f for f in ['bullish_ob', 'bearish_ob', 'bullish_fvg', 'bearish_fvg'] if f in X_full_resampled.columns]
    if smc_features_present_final:
        smc_signals_final = X_full_resampled[smc_features_present_final].max(axis=1)
        final_sample_weights[smc_signals_final == 1] = 2.0

    final_model.fit(X_full_resampled, y_full_resampled, sample_weight=final_sample_weights)
    
    logger.info(f"✅ [ML Train] LightGBM final model trained successfully for {symbol} with best hyperparameters and SMOTE.")

    # Feature importance from the final model
    feature_importances = pd.DataFrame({
        'Feature': X_full.columns,
        'Importance': final_model.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    metrics = {
        'avg_accuracy': avg_accuracy,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1_score': avg_f1_score,
        'num_samples_full_dataset': len(X_full),
        'num_folds_evaluated': tscv.n_splits,
        'best_hyperparameters': best_params,
        'feature_names': X_full.columns.tolist(),
        'feature_importances': feature_importances.to_dict(orient='records'),
        'smc_features_used': ['bullish_ob', 'bearish_ob', 'bullish_fvg', 'bearish_fvg', 
                              'liquidity_pool_high', 'liquidity_pool_low'], # Keep basic SMC list
        'all_metrics_per_fold': fold_metrics
    }

    logger.info(f"📊 [ML Train] Aggregated Model performance metrics for {symbol} (LightGBM with SMC & Adv. Features):")
    logger.info(f"  - Avg Accuracy: {avg_accuracy:.4f}")
    logger.info(f"  - Avg Precision: {avg_precision:.4f}")
    logger.info(f"  - Avg Recall: {avg_recall:.4f}")
    logger.info(f"  - Avg F1 Score: {avg_f1_score:.4f}")
    
    # Log top 10 feature importance
    logger.info("📊 Top 10 Feature Importances:")
    for i, row in feature_importances.head(10).iterrows():
        logger.info(f"  - {row['Feature']}: {row['Importance']}")

    return final_model, metrics

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
        f"🤖 *ML Trainer Service Status (SMC Strategy V2):*\n"
        f"- Current Time: {now}\n"
        f"- Training Status: *{training_status}*\n"
    )
    if last_training_time:
        status_message += f"- Last Training Time: {last_training_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    if last_training_metrics:
        status_message += f"- Last Training Metrics (Avg F1): {last_training_metrics.get('avg_f1_score', 'N/A'):.4f}\n" # Changed to F1
        status_message += f"- Last Training Metrics (Avg Precision): {last_training_metrics.get('avg_precision', 'N/A'):.4f}\n" # Added Precision
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
    logger.info("🚀 Starting ML model training script with SMC strategy V2 (Advanced Features & Optimization)...")
    logger.info(f"Local Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | UTC Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    flask_thread: Optional[Thread] = None
    initial_training_start_time = datetime.now() # Track overall training duration

    # Pre-fetch BTC and ETH data once before the loop to optimize
    logger.info("ℹ️ [Main] Pre-fetching BTCUSDT and ETHUSDT historical data for all models...")
    global_btc_df = fetch_historical_data("BTCUSDT", interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
    global_eth_df = fetch_historical_data("ETHUSDT", interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING) # Fetch ETH data too
    
    if global_btc_df is None or global_btc_df.empty:
        logger.critical("❌ [Main] Critical: Failed to pre-fetch BTCUSDT data. Cannot proceed with training as BTC trend is a required feature.")
        training_status = "Failed: BTC data not available"
        if TELEGRAM_TOKEN and CHAT_ID:
            send_telegram_message(CHAT_ID,
                                  f"❌ *فشل بدء تدريب نموذج ML:*\n"
                                  f"لم يتمكن من جلب بيانات BTCUSDT التاريخية المطلوبة. توقف التدريب.",
                                  parse_mode='Markdown')
        exit(1)
    if global_eth_df is None or global_eth_df.empty:
         logger.warning("⚠️ [Main] Failed to pre-fetch ETHUSDT data. ETH trend feature will be 0.0.")

    logger.info("✅ [Main] BTCUSDT and ETHUSDT historical data pre-fetched successfully (if available).")

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

        training_status = "In Progress: Training Models with SMC V2"
        training_error = None # Reset error
        
        overall_metrics: Dict[str, Any] = {
            'total_models_trained': 0,
            'successful_models': 0,
            'failed_models': 0,
            'avg_accuracy': 0.0,
            'avg_precision': 0.0,
            'avg_recall': 0.0,
            'avg_f1_score': 0.0,
            'advanced_features_used': True, # Indicate advanced features
            'smc_features': ['liquidity_pool_high', 'liquidity_pool_low', 'bullish_ob', 'bearish_ob', 
                             'bullish_ob_strength', 'bearish_ob_strength',
                             'bullish_fvg', 'bearish_fvg', 'bullish_fvg_size', 'bearish_fvg_size',
                             'bullish_bos', 'bearish_bos', 'bullish_choch', 'bearish_choch',
                             'market_structure'], # Updated list
            'details_per_symbol': {}
        }
        
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1_score = 0.0
        successful_model_count = 0 # To calculate accurate averages

        # Send Telegram notification for training start
        if TELEGRAM_TOKEN and CHAT_ID:
            send_telegram_message(CHAT_ID,
                                  f"🚀 *Starting ML Model Training with SMC Strategy V2:*\n"
                                  f"Training models for {len(symbols)} symbols.\n"
                                  f"Time: {initial_training_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                  f"Strategy: Smart Money Concept (SMC) + Advanced Features",
                                  parse_mode='Markdown')

        # 4. Train a model for each symbol separately
        for symbol in symbols:
            current_model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
            overall_metrics['total_models_trained'] += 1
            logger.info(f"\n--- ⏳ [Main] Starting model training for {symbol} ({current_model_name}) with Advanced Features ---")
            
            try:
                # Fetch historical data for the current symbol
                df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
                if df_hist is None or df_hist.empty:
                    logger.warning(f"⚠️ [Main] Could not fetch sufficient data for {symbol}. Skipping model training for this symbol.")
                    overall_metrics['failed_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Failed: No data', 'error': 'No sufficient historical data'}
                    continue

                # Prepare data for the machine learning model, passing the pre-fetched BTC and ETH data
                df_processed = prepare_data_for_ml(df_hist, symbol, btc_df=global_btc_df, eth_df=global_eth_df)
                if df_processed is None or df_processed.empty:
                    logger.warning(f"⚠️ [Main] No data ready for training for {symbol} after advanced preprocessing. Skipping.")
                    overall_metrics['failed_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Failed: No processed data', 'error': 'No sufficient processed data or target imbalance'}
                    continue

                # Train and evaluate the model
                trained_model, model_metrics = train_and_evaluate_model(df_processed, symbol)

                if trained_model is None:
                    logger.error(f"❌ [Main] Model training failed for {symbol}. Cannot save.")
                    overall_metrics['failed_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Failed: Training returned None', 'error': 'Model training failed in train_and_evaluate_model'}
                    continue

                # Save the model to the database
                if save_ml_model_to_db(trained_model, current_model_name, model_metrics):
                    logger.info(f"✅ [Main] Model '{current_model_name}' successfully saved to database.")
                    overall_metrics['successful_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Completed Successfully', 'metrics': model_metrics}
                    
                    total_accuracy += model_metrics.get('avg_accuracy', 0.0)
                    total_precision += model_metrics.get('avg_precision', 0.0)
                    total_recall += model_metrics.get('avg_recall', 0.0)
                    total_f1_score += model_metrics.get('avg_f1_score', 0.0)
                    successful_model_count += 1
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
        if successful_model_count > 0:
            overall_metrics['avg_accuracy'] = total_accuracy / successful_model_count
            overall_metrics['avg_precision'] = total_precision / successful_model_count
            overall_metrics['avg_recall'] = total_recall / successful_model_count
            overall_metrics['avg_f1_score'] = total_f1_score / successful_model_count

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
                message_title = "✅ *ML Model Training with SMC V2 Completed Successfully!*"
            elif training_status == "Completed with Errors (Some Models Failed)":
                message_title = "⚠️ *ML Model Training with SMC V2 Completed with Errors!*"
            else:
                message_title = "❌ *ML Model Training with SMC V2 Failed!*"
            
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
                f"✨ *Advanced Features Enabled:*{' نعم' if overall_metrics['advanced_features_used'] else ' لا'}"
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
                f"🚨 *Critical Error in ML Model Training Script with SMC V2:*\n"
                f"An unexpected error occurred and stopped the script.\n"
                f"Details: `{e}`\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            send_telegram_message(CHAT_ID, error_message, parse_mode='Markdown')
    finally:
        logger.info("🛑 [Main] Shutting down training script...")
        cleanup_resources()
        logger.info("👋 [Main] ML model training script stopped.")

