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
import math
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException, BinanceRequestException
from flask import Flask, request, Response, render_template, redirect, url_for, session
from threading import Thread
from datetime import datetime, timedelta, timezone
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_elliott_fib.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBot')

# ---------------------- تحميل المتغيرات البيئية ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
    DASHBOARD_USER: str = config('DASHBOARD_USER', default='admin')
    DASHBOARD_PASS: str = config('DASHBOARD_PASS', default='admin123')
    ACCOUNT_BALANCE: float = float(config('ACCOUNT_BALANCE', default=1000.0))
except Exception as e:
     logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
     exit(1)

logger.info(f"Binance API Key: {'Available' if API_KEY else 'Not available'}")
logger.info(f"Telegram Token: {TELEGRAM_TOKEN[:10]}...{'*' * (len(TELEGRAM_TOKEN)-10)}")
logger.info(f"Telegram Chat ID: {CHAT_ID}")
logger.info(f"Database URL: {'Available' if DB_URL else 'Not available'}")
logger.info(f"Webhook URL: {WEBHOOK_URL if WEBHOOK_URL else 'Not specified'} (Flask will always run for Render)")
logger.info(f"Account Balance: ${ACCOUNT_BALANCE:,.2f}")

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
TRADE_VALUE: float = 10.0
MAX_OPEN_TRADES: int = 20
SIGNAL_GENERATION_TIMEFRAME: str = '15m'  # Updated to 15 minutes to match ML model training
SCALP_TIMEFRAME: str = '1m'  # Timeframe for final scalping signals
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 3
SIGNAL_TRACKING_LOOKBACK_DAYS: int = 1
RISK_PER_TRADE: float = 0.01  # 1% من رأس المال

# Indicator Parameters
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
VOLUME_LOOKBACK_CANDLES: int = 3
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 2
EMA9_PERIOD: int = 9
EMA21_PERIOD: int = 21
VOLUME_SPIKE_THRESHOLD: float = 2.0  # Threshold for volume spike (relative_volume > 2)

MIN_PROFIT_MARGIN_PCT: float = 1.0
MIN_VOLUME_15M_USDT: float = 50000.0

# إعدادات جديدة: وقف الخسارة وجني الأرباح
STOP_LOSS_PCT: float = 3  # نسبة وقف الخسارة الثابتة (0.5%)
PROFIT_TARGET_PCT: float = 1.0  # نسبة جني الأرباح (1%)
TRAILING_STOP_PCT: float = 0.3  # نسبة التوقف المتابع بعد جني الأرباح

ENTRY_ATR_PERIOD: int = ATR_PERIOD
ENTRY_ATR_MULTIPLIER: float = 1.5

TARGET_APPROACH_THRESHOLD_PCT: float = 0.005

BINANCE_FEE_RATE: float = 0.001

BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V5'

# إعدادات التداول السريع في الوقت الحقيقي
REAL_TIME_SCALPING_ENABLED: bool = True
TOP_SYMBOLS_COUNT: int = 20
SCALPING_UPDATE_INTERVAL: int = 300  # 5 دقائق بالثواني
PARALLEL_PROCESSING_WORKERS: int = 10
USE_TA_LIB: bool = False  # سيتم تعيينه بعد محاولة استيراد TA-Lib

# محاولة تحميل TA-Lib إذا كانت متوفرة
try:
    import talib
    logger.info("✅ TA-Lib loaded successfully")
    USE_TA_LIB = True
except ImportError:
    logger.warning("⚠️ TA-Lib not installed. Using fallback calculations")
    USE_TA_LIB = False

# Feature columns expected by the ML model
FEATURE_COLUMNS = [
    'rsi', 'macd_hist', 'atr', 'relative_volume', 'hour_of_day',
    'price_vs_ema50', 'price_vs_ema200', 'btc_correlation',
    'stoch_rsi_k', 'stoch_rsi_d', 'macd_cross', 'market_condition',
    'bb_width'
]

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None
ticker_data: Dict[str, float] = {}
ml_models: Dict[str, Any] = {}
btc_cache = {
    'data': None,
    'last_updated': 0
}

# ---------------------- Binance Client Setup ----------------------
try:
    logger.info("ℹ️ [Binance] تهيئة عميل Binance...")
    client = Client(API_KEY, API_SECRET)
    client.ping()
    server_time = client.get_server_time()
    logger.info(f"✅ [Binance] تم تهيئة عميل Binance. وقت الخادم: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
except BinanceRequestException as req_err:
     logger.critical(f"❌ [Binance] خطأ في طلب Binance (مشكلة في الشبكة أو الطلب): {req_err}")
     exit(1)
except BinanceAPIException as api_err:
     logger.critical(f"❌ [Binance] خطأ في واجهة برمجة تطبيقات Binance (مفاتيح غير صالحة أو مشكلة في الخادم): {api_err}")
     exit(1)
except Exception as e:
    logger.critical(f"❌ [Binance] فشل غير متوقع في تهيئة عميل Binance: {e}")
    exit(1)

# ---------------------- تنفيذ الصفقات الحقيقية ----------------------
def execute_buy_order(symbol: str, quantity: float) -> Optional[Dict]:
    """تنفيذ أمر شراء حقيقي"""
    try:
        order = client.create_order(
            symbol=symbol,
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_MARKET,
            quantity=quantity
        )
        logger.info(f"✅ [Trade] تم تنفيذ أمر شراء لـ {symbol}: {order}")
        return order
    except BinanceAPIException as e:
        logger.error(f"❌ [Trade] خطأ في تنفيذ أمر الشراء لـ {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [Trade] خطأ غير متوقع في أمر الشراء لـ {symbol}: {e}")
        return None

def execute_sell_order(symbol: str, quantity: float) -> Optional[Dict]:
    """تنفيذ أمر بيع حقيقي"""
    try:
        order = client.create_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_MARKET,
            quantity=quantity
        )
        logger.info(f"✅ [Trade] تم تنفيذ أمر بيع لـ {symbol}: {order}")
        return order
    except BinanceAPIException as e:
        logger.error(f"❌ [Trade] خطأ في تنفيذ أمر البيع لـ {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [Trade] خطأ غير متوقع في أمر البيع لـ {symbol}: {e}")
        return None

# ---------------------- Additional Indicator Functions ----------------------
def get_fear_greed_index() -> str:
    """Fetches the Fear & Greed Index from alternative.me and translates classification to Arabic."""
    classification_translation_ar = {
        "Extreme Fear": "خوف شديد", "Fear": "خوف", "Neutral": "محايد",
        "Greed": "جشع", "Extreme Greed": "جشع شديد",
    }
    url = "https://api.alternative.me/fng/"
    logger.debug(f"ℹ️ [Indicators] جلب مؤشر الخوف والجشع من {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        value = int(data["data"][0]["value"])
        classification_en = data["data"][0]["value_classification"]
        classification_ar = classification_translation_ar.get(classification_en, classification_en)
        logger.debug(f"✅ [Indicators] مؤشر الخوف والجشع: {value} ({classification_ar})")
        return f"{value} ({classification_ar})"
    except requests.exceptions.RequestException as e:
         logger.error(f"❌ [Indicators] خطأ في الشبكة أثناء جلب مؤشر الخوف والجشع: {e}")
         return "N/A (خطأ في الشبكة)"
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"❌ [Indicators] خطأ في تنسيق البيانات لمؤشر الخوف والجشع: {e}")
        return "N/A (خطأ في البيانات)"
    except Exception as e:
        logger.error(f"❌ [Indicators] خطأ غير متوقع أثناء جلب مؤشر الخوف والجشع: {e}", exc_info=True)
        return "N/A (خطأ غير معروف)"

def get_btc_data() -> Optional[pd.DataFrame]:
    """Get BTC data with caching"""
    if btc_cache['data'] is not None and time.time() - btc_cache['last_updated'] < 300:
        return btc_cache['data']
    
    btc_df = fetch_historical_data("BTCUSDT", interval='15m', days=3)
    if btc_df is not None:
        btc_cache['data'] = btc_df
        btc_cache['last_updated'] = time.time()
    return btc_df

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """Fetches historical candlestick data from Binance."""
    if not client:
        logger.error(f"❌ [Data] عميل Binance غير مهيأ لجلب البيانات لـ {symbol}.")
        return None
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days + 1)
        start_str_overall = start_dt.strftime("%Y-%m-%d %H:%M:%S")

        logger.debug(f"ℹ️ [Data] جلب بيانات {interval} لـ {symbol} من {start_str_overall} حتى الآن...")

        klines = client.get_historical_klines(symbol, interval, start_str_overall)

        if not klines:
            logger.warning(f"⚠️ [Data] لا توجد بيانات تاريخية ({interval}) لـ {symbol} للفترة المطلوبة.")
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
            logger.debug(f"ℹ️ [Data] {symbol}: تم إسقاط {initial_len - len(df)} صفًا بسبب قيم NaN في بيانات OHLCV.")

        if df.empty:
            logger.warning(f"⚠️ [Data] DataFrame لـ {symbol} فارغ بعد إزالة قيم NaN الأساسية.")
            return None

        df.sort_index(inplace=True)

        logger.debug(f"✅ [Data] تم جلب ومعالجة {len(df)} شمعة تاريخية ({interval}) لـ {symbol}.")
        return df

    except BinanceAPIException as api_err:
         logger.error(f"❌ [Data] خطأ في Binance API أثناء جلب البيانات لـ {symbol}: {api_err}")
         return None
    except BinanceRequestException as req_err:
         logger.error(f"❌ [Data] خطأ في الطلب أو الشبكة أثناء جلب البيانات لـ {symbol}: {req_err}")
         return None
    except Exception as e:
        logger.error(f"❌ [Data] خطأ غير متوقع أثناء جلب البيانات التاريخية لـ {symbol}: {e}", exc_info=True)
        return None

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)."""
    if series is None or series.isnull().all() or len(series) < span:
        return pd.Series(index=series.index if series is not None else None, dtype=float)
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Calculates Relative Strength Index (RSI)."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning("⚠️ [Indicator RSI] عمود 'close' مفقود أو فارغ.")
        df['rsi'] = np.nan
        return df
    if len(df) < period:
        logger.warning(f"⚠️ [Indicator RSI] بيانات غير كافية ({len(df)} < {period}) لحساب RSI.")
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

def calculate_atr_indicator(df: pd.DataFrame, period: int = ENTRY_ATR_PERIOD) -> pd.DataFrame:
    """Calculates Average True Range (ATR)."""
    df = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator ATR] أعمدة 'high', 'low', 'close' مفقودة أو فارغة.")
        df['atr'] = np.nan
        return df
    if len(df) < period + 1:
        logger.warning(f"⚠️ [Indicator ATR] بيانات غير كافية ({len(df)} < {period + 1}) لحساب ATR.")
        df['atr'] = np.nan
        return df

    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()

    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)

    df['atr'] = tr.ewm(span=period, adjust=False).mean()
    return df

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculates Volume Weighted Average Price (VWAP)."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    vwap = (tp * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Calculates On-Balance Volume (OBV)."""
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv

def _calculate_btc_trend_feature(df_btc: pd.DataFrame) -> Optional[pd.Series]:
    """Calculates a numerical representation of Bitcoin's trend."""
    logger.debug("ℹ️ [Indicators] حساب اتجاه البيتكوين للميزات...")
    min_data_for_ema = 50 + 5

    if df_btc is not None and not df_btc.empty and len(df_btc) >= min_data_for_ema:
        df_btc_copy = df_btc.copy()
        df_btc_copy['close'] = pd.to_numeric(df_btc_copy['close'], errors='coerce')
        df_btc_copy.dropna(subset=['close'], inplace=True)

        if len(df_btc_copy) >= min_data_for_ema:
            ema20 = calculate_ema(df_btc_copy['close'], 20)
            ema50 = calculate_ema(df_btc_copy['close'], 50)

            ema_df = pd.DataFrame({'ema20': ema20, 'ema50': ema50, 'close': df_btc_copy['close']})
            ema_df.dropna(inplace=True)

            if not ema_df.empty:
                trend_series = pd.Series(index=ema_df.index, data=0.0)

                trend_series[(ema_df['close'] > ema_df['ema20']) & (ema_df['ema20'] > ema_df['ema50'])] = 1.0
                trend_series[(ema_df['close'] < ema_df['ema20']) & (ema_df['ema20'] < ema_df['ema50'])] = -1.0

                final_trend_series = trend_series.reindex(df_btc.index).fillna(0.0)
                logger.debug(f"✅ [Indicators] تم حساب ميزة اتجاه البيتكوين. أمثلة: {final_trend_series.tail().tolist()}")
                return final_trend_series

    logger.warning(f"⚠️ [Indicators] بيانات BTC/USDT غير كافية ({len(df_btc) if df_btc is not None else 0} < {min_data_for_ema}) لحساب اتجاه البيتكوين للميزات.")
    return pd.Series(index=df_btc.index if df_btc is not None else None, data=0.0)

# ---------------------- Database Connection Setup ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """Initializes database connection and creates tables if they don't exist."""
    global conn, cur
    logger.info("[DB] بدء تهيئة قاعدة البيانات...")
    for attempt in range(retries):
        try:
            logger.info(f"[DB] محاولة الاتصال بقاعدة البيانات (المحاولة {attempt + 1}/{retries})...")
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            cur = conn.cursor()
            logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")

            # --- Create or update signals table ---
            logger.info("[DB] التحقق من/إنشاء جدول 'signals'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    initial_target DOUBLE PRECISION NOT NULL,
                    current_target DOUBLE PRECISION NOT NULL,
                    stop_loss_price DOUBLE PRECISION,  -- عمود جديد: سعر وقف الخسارة
                    trailing_activated BOOLEAN DEFAULT FALSE,  -- عمود جديد: هل تم تفعيل التوقف المتابع؟
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
                    signal_details JSONB,
                    order_id BIGINT,  -- جديد
                    executed_quantity DOUBLE PRECISION,  -- جديد
                    buy_executed_at TIMESTAMP,  -- جديد
                    sell_executed_at TIMESTAMP  -- جديد
                );""")
            
            # إضافة الأعمدة إذا لم تكن موجودة
            cur.execute("""
                DO $$ 
                BEGIN
                    BEGIN
                        ALTER TABLE signals ADD COLUMN stop_loss_price DOUBLE PRECISION;
                    EXCEPTION
                        WHEN duplicate_column THEN 
                        -- لا تفعل شيئاً، العمود موجود بالفعل
                    END;
                    
                    BEGIN
                        ALTER TABLE signals ADD COLUMN trailing_activated BOOLEAN DEFAULT FALSE;
                    EXCEPTION
                        WHEN duplicate_column THEN 
                        -- لا تفعل شيئاً، العمود موجود بالفعل
                    END;
                    
                    BEGIN
                        ALTER TABLE signals ADD COLUMN order_id BIGINT;
                    EXCEPTION
                        WHEN duplicate_column THEN 
                        -- لا تفعل شيئاً، العمود موجود بالفعل
                    END;
                    
                    BEGIN
                        ALTER TABLE signals ADD COLUMN executed_quantity DOUBLE PRECISION;
                    EXCEPTION
                        WHEN duplicate_column THEN 
                        -- لا تفعل شيئاً، العمود موجود بالفعل
                    END;
                    
                    BEGIN
                        ALTER TABLE signals ADD COLUMN buy_executed_at TIMESTAMP;
                    EXCEPTION
                        WHEN duplicate_column THEN 
                        -- لا تفعل شيئاً، العمود موجود بالفعل
                    END;
                    
                    BEGIN
                        ALTER TABLE signals ADD COLUMN sell_executed_at TIMESTAMP;
                    EXCEPTION
                        WHEN duplicate_column THEN 
                        -- لا تفعل شيئاً، العمود موجود بالفعل
                    END;
                END $$;
            """)
            
            conn.commit()
            logger.info("✅ [DB] جدول 'signals' موجود أو تم إنشاؤه مع تحديث الأعمدة.")

            # --- Create ml_models table ---
            logger.info("[DB] التحقق من/إنشاء جدول 'ml_models'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL,
                    trained_at TIMESTAMP DEFAULT NOW(),
                    metrics JSONB
                );""")
            conn.commit()
            logger.info("✅ [DB] جدول 'ml_models' موجود أو تم إنشاؤه.")

            # --- Create market_dominance table ---
            logger.info("[DB] التحقق من/إنشاء جدول 'market_dominance'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_dominance (
                    id SERIAL PRIMARY KEY,
                    recorded_at TIMESTAMP DEFAULT NOW(),
                    btc_dominance DOUBLE PRECISION,
                    eth_dominance DOUBLE PRECISION
                );
            """)
            conn.commit()
            logger.info("✅ [DB] جدول 'market_dominance' موجود أو تم إنشاؤه.")

            # --- Create bot_settings table ---
            logger.info("[DB] التحقق من/إنشاء جدول 'bot_settings'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_settings (
                    id SERIAL PRIMARY KEY,
                    setting_name TEXT NOT NULL UNIQUE,
                    setting_value TEXT NOT NULL
                );
            """)
            conn.commit()
            logger.info("✅ [DB] جدول 'bot_settings' موجود أو تم إنشاؤه.")

            # --- Insert initial settings ---
            logger.info("[DB] إدراج/تحديث الإعدادات الأولية...")
            cur.execute("""
                INSERT INTO bot_settings (setting_name, setting_value)
                VALUES 
                    ('TRADE_VALUE', %s),
                    ('MAX_OPEN_TRADES', %s),
                    ('STOP_LOSS_PCT', %s),
                    ('PROFIT_TARGET_PCT', %s),
                    ('TRAILING_STOP_PCT', %s),
                    ('RISK_PER_TRADE', %s),
                    ('ACCOUNT_BALANCE', %s)
                ON CONFLICT (setting_name) DO UPDATE 
                SET setting_value = EXCLUDED.setting_value;
            """, (str(TRADE_VALUE), str(MAX_OPEN_TRADES), str(STOP_LOSS_PCT), 
                 str(PROFIT_TARGET_PCT), str(TRAILING_STOP_PCT), str(RISK_PER_TRADE), 
                 str(ACCOUNT_BALANCE)))
            conn.commit()
            logger.info("✅ [DB] تم إنشاء/تحديث إعدادات البوت")

            logger.info("✅ [DB] تم تهيئة قاعدة البيانات بنجاح.")
            return

        except OperationalError as op_err:
            logger.error(f"❌ [DB] خطأ تشغيلي في الاتصال (المحاولة {attempt + 1}): {op_err}")
            if conn: 
                conn.rollback()
                conn.close()
                conn = None
            if attempt == retries - 1:
                 logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
                 raise op_err
            time.sleep(delay)
        except Exception as e:
            logger.critical(f"❌ [DB] فشل غير متوقع في تهيئة قاعدة البيانات (المحاولة {attempt + 1}): {e}", exc_info=True)
            if conn: 
                conn.rollback()
                conn.close()
                conn = None
            if attempt == retries - 1:
                 logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
                 raise e
            time.sleep(delay)

    logger.critical("❌ [DB] فشل الاتصال بقاعدة البيانات بعد عدة محاولات.")
    exit(1)

def check_db_connection() -> bool:
    """Checks database connection status and re-initializes if necessary."""
    global conn
    try:
        if conn is None or conn.closed != 0:
            logger.warning("⚠️ [DB] الاتصال مغلق أو غير موجود. إعادة التهيئة...")
            init_db()
            return True
        else:
             with conn.cursor() as check_cur:
                  check_cur.execute("SELECT 1;")
                  check_cur.fetchone()
             return True
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [DB] فقدان الاتصال بقاعدة البيانات ({e}). إعادة التهيئة...")
        try:
             init_db()
             return True
        except Exception as recon_err:
            logger.error(f"❌ [DB] فشل محاولة إعادة الاتصال بعد فقدان الاتصال: {recon_err}")
            return False
    except Exception as e:
        logger.error(f"❌ [DB] خطأ غير متوقع أثناء التحقق من الاتصال: {e}", exc_info=True)
        try:
            init_db()
            return True
        except Exception as recon_err:
             logger.error(f"❌ [DB] فشل محاولة إعادة الاتصال بعد خطأ غير متوقع: {recon_err}")
             return False
    exit(1)

def load_ml_model_from_db(symbol: str) -> Optional[Any]:
    """Loads the latest trained ML model bundle for a specific symbol from the database."""
    global ml_models
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"

    if model_name in ml_models:
        logger.debug(f"ℹ️ [ML Model] النموذج '{model_name}' موجود بالفعل في الذاكرة.")
        return ml_models[model_name]

    if not check_db_connection() or not conn:
        logger.error(f"❌ [ML Model] لا يمكن تحميل نموذج ML لـ {symbol} بسبب مشكلة في اتصال قاعدة البيانات.")
        return None

    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (model_name,))
            result = db_cur.fetchone()
            if result and result['model_data']:
                model_bundle = pickle.loads(result['model_data'])
                ml_models[model_name] = model_bundle
                logger.info(f"✅ [ML Model] تم تحميل نموذج ML '{model_name}' من قاعدة البيانات بنجاح.")
                return model_bundle
            else:
                logger.warning(f"⚠️ [ML Model] لم يتم العثور على نموذج ML باسم '{model_name}' في قاعدة البيانات. يرجى تدريب النموذج أولاً.")
                return None
    except psycopg2.Error as db_err:
        logger.error(f"❌ [ML Model] خطأ في قاعدة البيانات أثناء تحميل نموذج ML لـ {symbol}: {db_err}", exc_info=True)
        return None
    except pickle.UnpicklingError as unpickle_err:
        logger.error(f"❌ [ML Model] خطأ في فك تسلسل نموذج ML لـ {symbol}: {unpickle_err}. قد يكون النموذج تالفًا أو تم حفظه بإصدار مختلف.", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"❌ [ML Model] خطأ غير متوقع أثناء تحميل نموذج ML لـ {symbol}: {e}", exc_info=True)
        return None

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

# ---------------------- WebSocket Management for Ticker Prices ----------------------
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    """Handles incoming WebSocket messages for mini-ticker prices."""
    global ticker_data
    try:
        if isinstance(msg, list):
            for ticker_item in msg:
                symbol = ticker_item.get('s')
                price_str = ticker_item.get('c')
                if symbol and 'USDT' in symbol and price_str:
                    try:
                        ticker_data[symbol] = float(price_str)
                    except ValueError:
                         logger.warning(f"⚠️ [WS] قيمة سعر غير صالحة للرمز {symbol}: '{price_str}'")
        elif isinstance(msg, dict):
             if msg.get('e') == 'error':
                 logger.error(f"❌ [WS] رسالة خطأ من WebSocket: {msg.get('m', 'لا توجد تفاصيل خطأ')}")
             elif msg.get('stream') and msg.get('data'):
                 for ticker_item in msg.get('data', []):
                    symbol = ticker_item.get('s')
                    price_str = ticker_item.get('c')
                    if symbol and 'USDT' in symbol and price_str:
                        try:
                            ticker_data[symbol] = float(price_str)
                        except ValueError:
                             logger.warning(f"⚠️ [WS] قيمة سعر غير صالحة للرمز {symbol} في البث المجمع: '{price_str}'")
        else:
             logger.warning(f"⚠️ [WS] تم استلام رسالة WebSocket بتنسيق غير متوقع: {type(msg)}")

    except Exception as e:
        logger.error(f"❌ [WS] خطأ في معالجة رسالة التيكر: {e}", exc_info=True)

def run_ticker_socket_manager() -> None:
    """Runs and manages the WebSocket connection for mini-ticker."""
    while True:
        try:
            logger.info("ℹ️ [WS] بدء إدارة WebSocket لأسعار التيكر...")
            twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            twm.start()

            stream_name = twm.start_miniticker_socket(callback=handle_ticker_message)
            logger.info(f"✅ [WS] تم بدء بث WebSocket: {stream_name}")

            twm.join()
            logger.warning("⚠️ [WS] توقفت إدارة WebSocket. إعادة التشغيل...")

        except Exception as e:
            logger.error(f"❌ [WS] خطأ فادح في إدارة WebSocket: {e}. إعادة التشغيل في 15 ثانية...", exc_info=True)

        time.sleep(15)

# ---------------------- Other Helper Functions (Volume) ----------------------
def fetch_recent_volume(symbol: str, interval: str = SIGNAL_GENERATION_TIMEFRAME, num_candles: int = VOLUME_LOOKBACK_CANDLES) -> float:
    """Fetches the trading volume in USDT for the last `num_candles` of the specified `interval`."""
    if not client:
         logger.error(f"❌ [Data Volume] عميل Binance غير مهيأ لجلب الحجم لـ {symbol}.")
         return 0.0
    try:
        logger.debug(f"ℹ️ [Data Volume] جلب حجم آخر {num_candles} شمعات {interval} لـ {symbol}...")
        klines = client.get_klines(symbol=symbol, interval=interval, limit=num_candles)
        if not klines or len(klines) < num_candles:
             logger.warning(f"⚠️ [Data Volume] بيانات {interval} غير كافية (أقل من {num_candles} شمعة) لـ {symbol}.")
             return 0.0

        volume_usdt = sum(float(k[7]) for k in klines if len(k) > 7 and k[7])
        logger.debug(f"✅ [Data Volume] سيولة آخر {num_candles} شمعات {interval} لـ {symbol}: {volume_usdt:,.2f} USDT")
        return volume_usdt
    except (BinanceAPIException, BinanceRequestException) as binance_err:
         logger.error(f"❌ [Data Volume] خطأ في Binance API أو الشبكة أثناء جلب الحجم لـ {symbol}: {binance_err}")
         return 0.0
    except Exception as e:
        logger.error(f"❌ [Data Volume] خطأ غير متوقع أثناء جلب الحجم لـ {symbol}: {e}", exc_info=True)
        return 0.0

# ---------------------- حساب الكمية للصفقات ----------------------
def calculate_quantity(symbol: str, trade_value: float, entry_price: float, stop_loss_price: float) -> float:
    """حساب الكمية بناءً على قيمة الصفقة والحد الأدنى والمخاطرة"""
    try:
        # حساب المخاطرة لكل وحدة
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit <= 0:
            logger.warning(f"⚠️ [Trade] مخاطرة غير صالحة للوحدة لـ {symbol}: {risk_per_unit}")
            return 0.0

        # حساب الكمية بناءً على المخاطرة الإجمالية
        risk_amount = ACCOUNT_BALANCE * RISK_PER_TRADE
        quantity_by_risk = risk_amount / risk_per_unit

        # حساب الكمية بناءً على قيمة الصفقة
        quantity_by_value = trade_value / entry_price

        # نأخذ الكمية الأصغر لضمان عدم تجاوز المخاطرة
        quantity = min(quantity_by_risk, quantity_by_value)

        symbol_info = client.get_symbol_info(symbol)
        if not symbol_info:
            logger.warning(f"⚠️ [Trade] لا يمكن الحصول على معلومات الرمز {symbol}")
            return 0.0
        
        # إيجاد فلتر LOT_SIZE
        lot_size_filter = next(
            (f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'),
            None
        )
        
        if lot_size_filter:
            min_qty = float(lot_size_filter['minQty'])
            step_size = float(lot_size_filter['stepSize'])
            
            # التقريب حسب stepSize
            precision = int(round(-math.log(step_size, 10)))
            quantity = round(quantity, precision)
            
            # التأكد من الحد الأدنى
            if quantity < min_qty:
                logger.warning(f"⚠️ [Trade] الكمية {quantity} أقل من الحد الأدنى {min_qty} لـ {symbol}")
                quantity = min_qty
                
            return quantity
        else:
            logger.warning(f"⚠️ [Trade] لا يوجد فلتر LOT_SIZE للرمز {symbol}")
            return 0.0
            
    except Exception as e:
        logger.error(f"❌ [Trade] خطأ في حساب الكمية لـ {symbol}: {e}")
        return 0.0

# ---------------------- مراقبة الرصيد ----------------------
def check_balance(asset: str) -> float:
    """التحقق من الرصيد المتاح"""
    try:
        balance = client.get_asset_balance(asset=asset)
        return float(balance['free'])
    except Exception as e:
        logger.error(f"❌ [Balance] خطأ في جلب رصيد {asset}: {e}")
        return 0.0

# ---------------------- Reading and Validating Symbols List ----------------------
def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """
    Reads the list of currency symbols from a text file, then validates them
    as valid USDT pairs available for Spot trading on Binance.
    """
    raw_symbols: List[str] = []
    logger.info(f"ℹ️ [Data] قراءة قائمة الرموز من الملف '{filename}'...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        if not os.path.exists(file_path):
            file_path = os.path.abspath(filename)
            if not os.path.exists(file_path):
                 logger.error(f"❌ [Data] الملف '{filename}' غير موجود في دليل السكربت أو الدليل الحالي.")
                 return []
            else:
                 logger.warning(f"⚠️ [Data] الملف '{filename}' غير موجود في دليل السكربت. استخدام الملف في الدليل الحالي: '{file_path}'")

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = [f"{line.strip().upper().replace('USDT', '')}USDT"
                           for line in f if line.strip() and not line.startswith('#')]
        raw_symbols = sorted(list(set(raw_symbols)))
        logger.info(f"ℹ️ [Data] تم قراءة {len(raw_symbols)} رمزًا مبدئيًا من '{file_path}'.")

    except FileNotFoundError:
         logger.error(f"❌ [Data] الملف '{filename}' غير موجود.")
         return []
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في قراءة الملف '{filename}': {e}", exc_info=True)
        return []

    if not raw_symbols:
         logger.warning("⚠️ [Data] قائمة الرموز الأولية فارغة.")
         return []

    if not client:
        logger.error("❌ [Data Validation] عميل Binance غير مهيأ. لا يمكن التحقق من الرموز.")
        return raw_symbols

    try:
        logger.info("ℹ️ [Data Validation] التحقق من الرموز وحالة التداول من Binance API...")
        exchange_info = client.get_exchange_info()
        valid_trading_usdt_symbols = {
            s['symbol'] for s in exchange_info['symbols']
            if s.get('quoteAsset') == 'USDT' and
               s.get('status') == 'TRADING' and
               s.get('isSpotTradingAllowed') is True
        }
        logger.info(f"ℹ️ [Data Validation] تم العثور على {len(valid_trading_usdt_symbols)} زوج تداول USDT صالح في Spot على Binance.")
        validated_symbols = [symbol for symbol in raw_symbols if symbol in valid_trading_usdt_symbols]

        removed_count = len(raw_symbols) - len(validated_symbols)
        if removed_count > 0:
            removed_symbols = set(raw_symbols) - set(validated_symbols)
            logger.warning(f"⚠️ [Data Validation] تم إزالة {removed_count} رمز تداول USDT غير صالح أو غير متاح من القائمة: {', '.join(removed_symbols)}")

        logger.info(f"✅ [Data Validation] تم التحقق من الرموز. استخدام {len(validated_symbols)} رمزًا صالحًا.")
        return validated_symbols

    except (BinanceAPIException, BinanceRequestException) as binance_err:
         logger.error(f"❌ [Data Validation] خطأ في Binance API أو الشبكة أثناء التحقق من الرموز: {binance_err}")
         logger.warning("⚠️ [Data Validation] استخدام القائمة الأولية من الملف بدون التحقق من Binance.")
         return raw_symbols
    except Exception as api_err:
         logger.error(f"❌ [Data Validation] خطأ غير متوقع أثناء التحقق من رموز Binance: {api_err}", exc_info=True)
         logger.warning("⚠️ [Data Validation] استخدام القائمة الأولية من الملف بدون التحقق من Binance.")
         return raw_symbols

# ---------------------- Comprehensive Performance Report Generation Function ----------------------
def generate_performance_report() -> str:
    """Generates a comprehensive performance report from the database in Arabic, including recent closed trades and USD profit/loss."""
    logger.info("ℹ️ [Report] إنشاء تقرير الأداء...")
    if not check_db_connection() or not conn:
        logger.error("❌ [Report] لا يمكن إنشاء التقرير، مشكلة في اتصال قاعدة البيانات.")
        return "❌ لا يمكن إنشاء التقرير، مشكلة في اتصال قاعدة البيانات."
    try:
        with conn.cursor() as report_cur:
            report_cur.execute("SELECT id, symbol, entry_price, entry_time FROM signals WHERE achieved_target = FALSE ORDER BY entry_time DESC;")
            open_signals = report_cur.fetchall()
            open_signals_count = len(open_signals)

            report_cur.execute("""
                SELECT
                    COUNT(*) AS total_closed,
                    COUNT(*) FILTER (WHERE profit_percentage > 0) AS winning_signals,
                    COUNT(*) FILTER (WHERE profit_percentage <= 0) AS losing_signals,
                    COALESCE(SUM(profit_percentage) FILTER (WHERE profit_percentage > 0), 0) AS gross_profit_pct_sum,
                    COALESCE(SUM(profit_percentage) FILTER (WHERE profit_percentage <= 0), 0) AS gross_loss_pct_sum,
                    COALESCE(AVG(profit_percentage) FILTER (WHERE profit_percentage > 0), 0) AS avg_win_pct,
                    COALESCE(AVG(profit_percentage) FILTER (WHERE profit_percentage <= 0), 0) AS avg_loss_pct
                FROM signals
                WHERE achieved_target = TRUE;
            """)
            closed_stats = report_cur.fetchone() or {}

            total_closed = closed_stats.get('total_closed', 0)
            winning_signals = closed_stats.get('winning_signals', 0)
            losing_signals = closed_stats.get('losing_signals', 0)
            gross_profit_pct_sum = closed_stats.get('gross_profit_pct_sum', 0.0)
            gross_loss_pct_sum = closed_stats.get('gross_loss_pct_sum', 0.0)
            avg_win_pct = closed_stats.get('avg_win_pct', 0.0)
            avg_loss_pct = closed_stats.get('avg_loss_pct', 0.0)

            gross_profit_usd = (gross_profit_pct_sum / 100.0) * TRADE_VALUE
            gross_loss_usd = (gross_loss_pct_sum / 100.0) * TRADE_VALUE

            total_fees_usd = total_closed * (TRADE_VALUE * BINANCE_FEE_RATE + (TRADE_VALUE * (1 + (avg_win_pct / 100.0 if avg_win_pct > 0 else 0))) * BINANCE_FEE_RATE)

            net_profit_usd = gross_profit_usd + gross_loss_usd - total_fees_usd
            net_profit_pct = (net_profit_usd / (total_closed * TRADE_VALUE)) * 100 if total_closed * TRADE_VALUE > 0 else 0.0

            win_rate = (winning_signals / total_closed) * 100 if total_closed > 0 else 0.0
            profit_factor = float('inf') if gross_loss_pct_sum == 0 else (gross_profit_pct_sum / abs(gross_loss_pct_sum))

        report = (
            f"📊 *تقرير الأداء الشامل:*\n"
            f"_(افتراض حجم الصفقة: ${TRADE_VALUE:,.2f} ورسوم Binance: {BINANCE_FEE_RATE*100:.2f}% لكل صفقة)_ \n"
            f"——————————————\n"
            f"📈 الإشارات المفتوحة حالياً: *{open_signals_count}*\n"
        )

        if open_signals:
            report += "  • التفاصيل:\n"
            for signal in open_signals:
                safe_symbol = str(signal['symbol']).replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
                entry_time_str = signal['entry_time'].strftime('%Y-%m-%d %H:%M') if signal['entry_time'] else 'N/A'
                report += f"    - `{safe_symbol}` (دخول: ${signal['entry_price']:.8g} | فتح: {entry_time_str})\n"
        else:
            report += "  • لا توجد إشارات مفتوحة حالياً.\n"

        report += (
            f"——————————————\n"
            f"📉 *إحصائيات الإشارات المغلقة:*\n"
            f"  • إجمالي الإشارات المغلقة: *{total_closed}*\n"
            f"  ✅ إشارات رابحة: *{winning_signals}* ({win_rate:.2f}%)\n"
            f"  ❌ إشارات خاسرة: *{losing_signals}*\n"
            f"——————————————\n"
            f"💰 *الربحية الإجمالية:*\n"
            f"  • إجمالي الربح الإجمالي: *{gross_profit_pct_sum:+.2f}%* (≈ *${gross_profit_usd:+.2f}*)\n"
            f"  • إجمالي الخسارة الإجمالية: *{gross_loss_pct_sum:+.2f}%* (≈ *${gross_loss_usd:+.2f}*)\n"
            f"  • إجمالي الرسوم المتوقعة: *${total_fees_usd:,.2f}*\n"
            f"  • *الربح الصافي:* *{net_profit_pct:+.2f}%* (≈ *${net_profit_usd:+.2f}*)\n"
            f"  • متوسط الصفقة الرابحة: *{avg_win_pct:+.2f}%*\n"
            f"  • متوسط الصفقة الخاسرة: *{avg_loss_pct:+.2f}%*\n"
            f"  • عامل الربح: *{'∞' if profit_factor == float('inf') else f'{profit_factor:.2f}'}*\n"
            f"——————————————\n"
            f"🕰️ _التقرير محدث حتى: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        )

        logger.info("✅ [Report] تم إنشاء تقرير الأداء بنجاح.")
        return report

    except psycopg2.Error as db_err:
        logger.error(f"❌ [Report] خطأ في قاعدة البيانات أثناء إنشاء تقرير الأداء: {db_err}")
        if conn: conn.rollback()
        return "❌ خطأ في قاعدة البيانات أثناء إنشاء تقرير الأداء."
    except Exception as e:
        logger.error(f"❌ [Report] خطأ غير متوقع أثناء إنشاء تقرير الأداء: {e}", exc_info=True)
        return "❌ حدث خطأ غير متوقع أثناء إنشاء تقرير الأداء."

# ---------------------- Trading Strategy (Enhanced with Scalping Conditions) -------------------
class ScalpingTradingStrategy:
    """Encapsulates the trading strategy logic with ML prediction and scalp conditions."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ml_bundle = load_ml_model_from_db(symbol)
        if self.ml_bundle is None:
            logger.warning(f"⚠️ [Strategy {self.symbol}] لم يتم تحميل نموذج تعلم الآلة لـ {symbol}. لن تتمكن الإستراتيجية من توليد إشارات.")
        else:
            logger.info(f"✅ [Strategy {self.symbol}] تم تحميل حزمة نموذج ML بنجاح.")

    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates all required indicators including new ones."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] حساب المؤشرات لنموذج ML...")
        min_len_required = max(RSI_PERIOD, RSI_MOMENTUM_LOOKBACK_CANDLES, VOLUME_LOOKBACK_CANDLES, 55) + 5

        if len(df) < min_len_required:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame قصير جدًا ({len(df)} < {min_len_required}) لحساب مؤشرات ML.")
            return None

        try:
            df_calc = df.copy()
            df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
            df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)

            # إضافة مؤشرات جديدة للتداول السريع
            df_calc['ema9'] = calculate_ema(df_calc['close'], EMA9_PERIOD)
            df_calc['ema21'] = calculate_ema(df_calc['close'], EMA21_PERIOD)
            df_calc['vwap'] = calculate_vwap(df_calc)
            df_calc['obv'] = calculate_obv(df_calc)
            df_calc['relative_volume'] = df_calc['volume'] / df_calc['volume'].rolling(window=30, min_periods=1).mean()
            df_calc['vol_spike'] = (df_calc['relative_volume'] > VOLUME_SPIKE_THRESHOLD).astype(int)

            # إضافة ميزات جديدة
            df_calc['volume_15m_avg'] = df_calc['volume'].rolling(window=VOLUME_LOOKBACK_CANDLES, min_periods=1).mean()
            df_calc['rsi_momentum_bullish'] = 0
            if len(df_calc) >= RSI_MOMENTUM_LOOKBACK_CANDLES + 1:
                for i in range(RSI_MOMENTUM_LOOKBACK_CANDLES, len(df_calc)):
                    rsi_slice = df_calc['rsi'].iloc[i - RSI_MOMENTUM_LOOKBACK_CANDLES : i + 1]
                    if not rsi_slice.isnull().any() and np.all(np.diff(rsi_slice) > 0) and rsi_slice.iloc[-1] > 50:
                        df_calc.loc[df_calc.index[i], 'rsi_momentum_bullish'] = 1

            # اتجاه البيتكوين
            btc_df = get_btc_data()
            btc_trend_series = None
            if btc_df is not None and not btc_df.empty:
                btc_trend_series = _calculate_btc_trend_feature(btc_df)
                if btc_trend_series is not None:
                    df_calc = df_calc.merge(btc_trend_series.rename('btc_trend_feature'),
                                            left_index=True, right_index=True, how='left')
                    df_calc['btc_trend_feature'] = df_calc['btc_trend_feature'].fillna(0.0)
                    logger.debug(f"ℹ️ [Strategy {self.symbol}] تم دمج ميزة اتجاه البيتكوين.")
                else:
                    logger.warning(f"⚠️ [Strategy {self.symbol}] فشل حساب ميزة اتجاه البيتكوين. سيتم استخدام 0 كقيمة افتراضية لـ 'btc_trend_feature'.")
                    df_calc['btc_trend_feature'] = 0.0
            else:
                logger.warning(f"⚠️ [Strategy {self.symbol}] فشل جلب البيانات التاريخية للبيتكوين. سيتم استخدام 0 كقيمة افتراضية لـ 'btc_trend_feature'.")
                df_calc['btc_trend_feature'] = 0.0

            # حساب الميزات الإضافية
            df_calc = self.calculate_features(df_calc)

            # التأكد من وجود جميع أعمدة الميزات
            for col in FEATURE_COLUMNS:
                if col not in df_calc.columns:
                    logger.warning(f"⚠️ [Strategy {self.symbol}] عمود الميزة المفقود لنموذج ML: {col}")
                    df_calc[col] = np.nan
                else:
                    df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

            initial_len = len(df_calc)
            all_required_cols = list(set(FEATURE_COLUMNS + [
                'open', 'high', 'low', 'close', 'volume', 'atr'
            ]))
            df_cleaned = df_calc.dropna(subset=all_required_cols).copy()
            dropped_count = initial_len - len(df_cleaned)

            if dropped_count > 0:
                 logger.debug(f"ℹ️ [Strategy {self.symbol}] تم إسقاط {dropped_count} صفًا بسبب قيم NaN في المؤشرات.")
            if df_cleaned.empty:
                logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame فارغ بعد إزالة قيم NaN للمؤشرات.")
                return None

            latest = df_cleaned.iloc[-1]
            logger.debug(f"✅ [Strategy {self.symbol}] تم حساب مؤشرات ML. أحدث حجم 15 دقيقة: {latest.get('volume_15m_avg', np.nan):.2f}, RSI Momentum: {latest.get('rsi_momentum_bullish', np.nan)}, BTC Trend: {latest.get('btc_trend_feature', np.nan)}, ATR: {latest.get('atr', np.nan):.4f}")
            return df_cleaned

        except KeyError as ke:
             logger.error(f"❌ [Strategy {self.symbol}] خطأ: لم يتم العثور على عمود مطلوب أثناء حساب المؤشر: {ke}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"❌ [Strategy {self.symbol}] خطأ غير متوقع أثناء حساب المؤشر: {e}", exc_info=True)
            return None

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates all the technical indicators required for the ML model."""
        df_calc = df.copy()

        if 'atr' not in df_calc.columns:
            high_low = df_calc['high'] - df_calc['low']
            high_close_prev = (df_calc['high'] - df_calc['close'].shift(1)).abs()
            low_close_prev = (df_calc['low'] - df_calc['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
            df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

        if 'rsi' not in df_calc.columns:
            delta = df_calc['close'].diff()
            gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
            loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
            df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))

        if 'macd_hist' not in df_calc.columns or 'macd_cross' not in df_calc.columns:
            ema_fast = df_calc['close'].ewm(span=12, adjust=False).mean()
            ema_slow = df_calc['close'].ewm(span=26, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            df_calc['macd_hist'] = macd_line - signal_line
            df_calc['macd_cross'] = 0
            df_calc.loc[(df_calc['macd_hist'].shift(1) < 0) & (df_calc['macd_hist'] >= 0), 'macd_cross'] = 1
            df_calc.loc[(df_calc['macd_hist'].shift(1) > 0) & (df_calc['macd_hist'] <= 0), 'macd_cross'] = -1

        if 'bb_width' not in df_calc.columns:
            sma = df_calc['close'].rolling(window=20).mean()
            std_dev = df_calc['close'].rolling(window=20).std()
            upper_band = sma + (std_dev * 2)
            lower_band = sma - (std_dev * 2)
            df_calc['bb_width'] = (upper_band - lower_band) / (sma + 1e-9)

        if 'stoch_rsi_k' not in df_calc.columns or 'stoch_rsi_d' not in df_calc.columns:
            rsi = df_calc['rsi']
            min_rsi = rsi.rolling(window=14).min()
            max_rsi = rsi.rolling(window=14).max()
            stoch_rsi_val = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
            df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(window=3).mean() * 100
            df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(window=3).mean()

        if 'relative_volume' not in df_calc.columns:
            df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)

        if 'market_condition' not in df_calc.columns:
            df_calc['market_condition'] = 0
            df_calc.loc[(df_calc['rsi'] > 70) | (df_calc['stoch_rsi_k'] > 80), 'market_condition'] = 1
            df_calc.loc[(df_calc['rsi'] < 30) | (df_calc['stoch_rsi_k'] < 20), 'market_condition'] = -1

        if 'price_vs_ema50' not in df_calc.columns:
            ema_fast_trend = df_calc['close'].ewm(span=50, adjust=False).mean()
            df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
        if 'price_vs_ema200' not in df_calc.columns:
            ema_slow_trend = df_calc['close'].ewm(span=200, adjust=False).mean()
            df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1

        if 'btc_correlation' not in df_calc.columns:
            btc_df = get_btc_data()
            if btc_df is None or btc_df.empty:
                logger.warning(f"⚠️ [Strategy {self.symbol}] لا توجد بيانات BTC لتحديد ميزة الارتباط. سيتم استخدام 0.")
                df_calc['btc_correlation'] = 0.0
            else:
                btc_df_for_corr = btc_df.copy()
                btc_df_for_corr['btc_returns'] = btc_df_for_corr['close'].pct_change()
                merged_df = df_calc.merge(btc_df_for_corr[['btc_returns']], left_index=True, right_index=True, how='left')
                df_calc['btc_returns'] = merged_df['btc_returns'].fillna(0.0)
                df_calc['returns'] = df_calc['close'].pct_change()
                df_calc['btc_correlation'] = df_calc['returns'].rolling(window=30).corr(df_calc['btc_returns']).fillna(0.0)

        if 'hour_of_day' not in df_calc.columns:
            df_calc['hour_of_day'] = df_calc.index.hour

        return df_calc

    def generate_scalp_signal(self, latest: pd.Series) -> int:
        """
        Generates scalp signal based on technical conditions.
        Returns 1 if all conditions are met, 0 otherwise.
        """
        try:
            # شروط إضافية
            rsi_condition = latest.get('rsi', 50) > 45
            macd_condition = latest.get('macd_hist', 0) > 0
            
            # التحقق من وجود جميع المؤشرات المطلوبة
            required_indicators = ['ema9', 'ema21', 'vwap', 'vol_spike']
            if any(indicator not in latest for indicator in required_indicators):
                logger.warning(f"⚠️ [Scalp Signal] مؤشرات مفقودة للرمز {self.symbol}")
                return 0

            # تطبيق شروط التداول السريع
            ema_condition = latest['ema9'] > latest['ema21']
            vwap_condition = latest['close'] > latest['vwap']
            volume_condition = latest['vol_spike'] == 1

            # إرجاع الإشارة إذا تحققت جميع الشروط
            if all([ema_condition, vwap_condition, volume_condition, rsi_condition, macd_condition]):
                logger.info(f"✅ [Scalp Signal] إشارة شراء قوية للرمز {self.symbol}")
                return 1
                
            return 0
        except Exception as e:
            logger.error(f"❌ [Scalp Signal] خطأ في توليد إشارة التداول: {e}")
            return 0

    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generates a buy signal using ML prediction and scalp conditions.
        """
        logger.debug(f"ℹ️ [Strategy {self.symbol}] إنشاء إشارة شراء (تعتمد على ML + Scalp)...")

        min_signal_data_len = max(VOLUME_LOOKBACK_CANDLES, RSI_MOMENTUM_LOOKBACK_CANDLES, 55) + 1
        if df_processed is None or df_processed.empty or len(df_processed) < min_signal_data_len:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame فارغ أو قصير جدًا (<{min_signal_data_len})، لا يمكن إنشاء إشارة.")
            return None

        required_cols_for_signal = list(set(FEATURE_COLUMNS + [
            'close', 'atr', 'ema9', 'ema21', 'vwap', 'vol_spike'
        ]))
        missing_cols = [col for col in required_cols_for_signal if col not in df_processed.columns]
        if missing_cols:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame يفتقد أعمدة مطلوبة للإشارة: {missing_cols}.")
            return None

        last_row = df_processed.iloc[-1]

        current_price = ticker_data.get(self.symbol)
        if current_price is None:
            logger.warning(f"⚠️ [Strategy {self.symbol}] السعر الحالي غير متاح من بيانات التيكر. لا يمكن إنشاء إشارة.")
            return None

        if last_row[FEATURE_COLUMNS].isnull().values.any() or pd.isna(last_row.get('atr')):
             logger.warning(f"⚠️ [Strategy {self.symbol}] البيانات التاريخية تحتوي على قيم NaN في أعمدة المؤشرات المطلوبة. لا يمكن إنشاء إشارة.")
             return None

        signal_details = {}

        # --- ML Model Prediction ---
        ml_prediction_result_text = "N/A (نموذج غير محمل)"
        ml_pred = 0
        ml_is_bullish = False

        if self.ml_bundle:
            try:
                # إصلاح: ترتيب الميزات حسب ترتيب التدريب
                if hasattr(self.ml_bundle['scaler'], 'feature_names_in_'):
                    feature_order = self.ml_bundle['scaler'].feature_names_in_
                    features_ordered = last_row[feature_order].values.reshape(1, -1)
                else:
                    features_ordered = last_row[FEATURE_COLUMNS].values.reshape(1, -1)
                
                scaled_features = self.ml_bundle['scaler'].transform(features_ordered)
                ml_pred = self.ml_bundle['model'].predict(scaled_features)[0]
                if ml_pred == 1:
                    ml_is_bullish = True
                    ml_prediction_result_text = 'صعودي ✅'
                elif ml_pred == -1:
                    ml_prediction_result_text = 'هابط قوي ❌'
                else:
                    ml_prediction_result_text = 'محايد ➖'
                logger.info(f"✨ [Strategy {self.symbol}] تنبؤ نموذج ML: {ml_prediction_result_text}.")
            except Exception as ml_err:
                logger.error(f"❌ [Strategy {self.symbol}] خطأ في تنبؤ نموذج ML: {ml_err}", exc_info=True)
                ml_prediction_result_text = "خطأ في التنبؤ"
        
        signal_details['ML_Prediction'] = ml_prediction_result_text
        signal_details['BTC_Trend_Feature_Value'] = last_row.get('btc_trend_feature', 0.0)

        # --- Scalp Signal Condition ---
        scalp_signal = self.generate_scalp_signal(last_row)
        signal_details['Scalp_Signal'] = 'نشط ✅' if scalp_signal else 'غير نشط ❌'
        
        if not ml_is_bullish or not scalp_signal:
            logger.info(f"ℹ️ [Strategy {self.symbol}] شروط التداول السريع غير متحققة (ML: {ml_is_bullish}, Scalp: {scalp_signal})")
            return None

        # --- Volume Check ---
        volume_recent = fetch_recent_volume(self.symbol, interval=SIGNAL_GENERATION_TIMEFRAME, num_candles=VOLUME_LOOKBACK_CANDLES)
        if volume_recent < MIN_VOLUME_15M_USDT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] السيولة ({volume_recent:,.0f} USDT) أقل من الحد الأدنى المطلوب ({MIN_VOLUME_15M_USDT:,.0f} USDT). تم رفض الإشارة.")
            signal_details['Volume_Check'] = f'فشل: سيولة غير كافية ({volume_recent:,.0f} USDT)'
            return None
        else:
            signal_details['Volume_Check'] = f'نجاح: سيولة كافية ({volume_recent:,.0f} USDT)'

        current_atr = last_row.get('atr')
        if pd.isna(current_atr) or current_atr <= 0:
             logger.warning(f"⚠️ [Strategy {self.symbol}] قيمة ATR غير صالحة ({current_atr}) لحساب الهدف. لا يمكن إنشاء إشارة.")
             return None

        target_multiplier = ENTRY_ATR_MULTIPLIER
        initial_target = current_price + (target_multiplier * current_atr)

        profit_margin_pct = ((initial_target / current_price) - 1) * 100 if current_price > 0 else 0
        if profit_margin_pct < MIN_PROFIT_MARGIN_PCT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] هامش الربح ({profit_margin_pct:.2f}%) أقل من الحد الأدنى المطلوب ({MIN_PROFIT_MARGIN_PCT:.2f}%). تم رفض الإشارة.")
            signal_details['Profit_Margin_Check'] = f'فشل: هامش ربح غير كافٍ ({profit_margin_pct:.2f}%)'
            return None
        else:
            signal_details['Profit_Margin_Check'] = f'نجاح: هامش ربح كافٍ ({profit_margin_pct:.2f}%)'

        # حساب وقف الخسارة
        stop_loss_price = current_price * (1 - STOP_LOSS_PCT/100.0)

        signal_output = {
            'symbol': self.symbol,
            'entry_price': float(f"{current_price:.8g}"),
            'initial_target': float(f"{initial_target:.8g}"),
            'current_target': float(f"{initial_target:.8g}"),
            'stop_loss_price': stop_loss_price,
            'r2_score': 1.0,
            'strategy_name': 'Scalping_ML_Enhanced',
            'signal_details': signal_details,
            'volume_15m': volume_recent,
            'trade_value': TRADE_VALUE,
            'total_possible_score': 1.0
        }

        logger.info(f"✅ [Strategy {self.symbol}] تم تأكيد إشارة الشراء (ML + Scalp). السعر: {current_price:.6f}, ATR: {current_atr:.6f}, الحجم: {volume_recent:,.0f}, تنبؤ ML: {ml_prediction_result_text}, وقف الخسارة: {stop_loss_price:.6f}")
        return signal_output

# ---------------------- Real-time Scalping Service ----------------------
class RealTimeScalpingService:
    def __init__(self):
        self.top_symbols: List[str] = []
        self.last_update_time: float = 0
        self.executor = ThreadPoolExecutor(max_workers=PARALLEL_PROCESSING_WORKERS)
        
    def update_top_symbols(self) -> None:
        """Updates the list of top symbols by trading volume"""
        if not client:
            logger.warning("⚠️ [Scalp] Binance client not available for volume check")
            return
            
        try:
            # جلب أحجام التداول لجميع الرموز
            tickers = client.get_ticker()
            usdt_tickers = [t for t in tickers if 'USDT' in t['symbol'] and not t['symbol'].endswith('UPUSDT') and not t['symbol'].endswith('DOWNUSDT')]
            
            # ترتيب الرموز حسب حجم التداول
            sorted_tickers = sorted(usdt_tickers, key=lambda x: float(x['quoteVolume']), reverse=True)
            self.top_symbols = [t['symbol'] for t in sorted_tickers[:TOP_SYMBOLS_COUNT]]
            
            logger.info(f"✅ [Scalp] Updated top {TOP_SYMBOLS_COUNT} symbols: {', '.join(self.top_symbols)}")
            self.last_update_time = time.time()
            
        except Exception as e:
            logger.error(f"❌ [Scalp] Failed to update top symbols: {e}")

    def calculate_indicators_realtime(self, symbol: str, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Calculates indicators using TA-Lib or fallback to manual calculations"""
        try:
            # استخدام TA-Lib إذا كان متاحاً
            if USE_TA_LIB:
                close_prices = df['close'].values
                high_prices = df['high'].values
                low_prices = df['low'].values
                volume_values = df['volume'].values
                
                # حساب المؤشرات
                rsi = talib.RSI(close_prices, timeperiod=RSI_PERIOD)[-1]
                atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=ATR_PERIOD)[-1]
                _, _, macd_hist = talib.MACD(close_prices)
                macd_hist = macd_hist[-1]
                ema9 = talib.EMA(close_prices, timeperiod=EMA9_PERIOD)[-1]
                ema21 = talib.EMA(close_prices, timeperiod=EMA21_PERIOD)[-1]
                
                # حساب VWAP
                typical_price = (high_prices + low_prices + close_prices) / 3
                vwap = np.sum(typical_price * volume_values) / np.sum(volume_values)
                
                # حساب الحجم النسبي
                avg_volume = np.mean(volume_values[-VOLUME_LOOKBACK_CANDLES:])
                relative_volume = volume_values[-1] / avg_volume if avg_volume > 0 else 1.0
                
                return {
                    'rsi': rsi,
                    'atr': atr,
                    'macd_hist': macd_hist,
                    'ema9': ema9,
                    'ema21': ema21,
                    'vwap': vwap,
                    'relative_volume': relative_volume,
                    'close': close_prices[-1]
                }
            else:
                # النسخة الإحتياطية اليدوية
                df = df.copy()
                df = calculate_rsi_indicator(df, RSI_PERIOD)
                df = calculate_atr_indicator(df, ENTRY_ATR_PERIOD)
                
                # حساب المتوسطات المتحركة
                df['ema9'] = calculate_ema(df['close'], EMA9_PERIOD)
                df['ema21'] = calculate_ema(df['close'], EMA21_PERIOD)
                
                # حساب VWAP
                df['vwap'] = calculate_vwap(df)
                
                # الحجم النسبي
                df['volume_ma'] = df['volume'].rolling(window=VOLUME_LOOKBACK_CANDLES).mean()
                df['relative_volume'] = df['volume'] / df['volume_ma']
                
                return df.iloc[-1][['rsi', 'atr', 'ema9', 'ema21', 'vwap', 'relative_volume', 'close']].to_dict()
                
        except Exception as e:
            logger.error(f"❌ [Scalp] Failed to calculate indicators for {symbol}: {e}")
            return None

    def process_symbol(self, symbol: str) -> None:
        """Process a single symbol for scalping opportunity"""
        try:
            # جلب البيانات الحديثة (آخر 30-50 شمعة)
            df = fetch_historical_data(symbol, interval='1m', days=0.5)
            if df is None or len(df) < 30:
                return
                
            # حساب المؤشرات
            indicators = self.calculate_indicators_realtime(symbol, df)
            if not indicators:
                return
                
            # التحقق من شروط التداول السريع
            ema_condition = indicators['ema9'] > indicators['ema21']
            vwap_condition = indicators['close'] > indicators['vwap']
            volume_condition = indicators['relative_volume'] > VOLUME_SPIKE_THRESHOLD
            
            # حساب هامش الربح المحتمل
            profit_margin = ((indicators['close'] + (ENTRY_ATR_MULTIPLIER * indicators['atr'])) / indicators['close'] - 1) * 100
            
            if all([ema_condition, vwap_condition, volume_condition]) and profit_margin >= 1.75:
                # إرسال إشارة التداول
                stop_loss_price = indicators['close'] * (1 - STOP_LOSS_PCT/100.0)
                
                signal_data = {
                    'symbol': symbol,
                    'entry_price': indicators['close'],
                    'initial_target': indicators['close'] + (ENTRY_ATR_MULTIPLIER * indicators['atr']),
                    'stop_loss_price': stop_loss_price,
                    'strategy_name': 'Realtime_Scalping',
                    'signal_details': {
                        'EMA_Condition': f"{indicators['ema9']:.6f} > {indicators['ema21']:.6f}",
                        'VWAP_Condition': f"{indicators['close']:.6f} > {indicators['vwap']:.6f}",
                        'Volume_Spike': f"{indicators['relative_volume']:.2f}x",
                        'Profit_Margin': f"{profit_margin:.2f}%",
                        'Stop_Loss': f"{stop_loss_price:.6f}"
                    }
                }
                
                # التحقق من وجود نفس الإشارة المفتوحة
                if not check_db_connection() or not conn:
                    return
                    
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 1 FROM signals 
                        WHERE symbol = %s 
                        AND achieved_target = FALSE
                        AND entry_time > NOW() - INTERVAL '1 hour'
                        LIMIT 1
                    """, (symbol,))
                    if not cur.fetchone():
                        if insert_signal_into_db(signal_data):
                            send_telegram_alert(signal_data, '1m')
                        else:
                            logger.error(f"❌ [Scalp] Failed to insert signal for {symbol}")
                
        except Exception as e:
            logger.error(f"❌ [Scalp] Error processing {symbol}: {e}")

    def run(self) -> None:
        """Main loop for real-time scalping service"""
        logger.info("🚀 Starting real-time scalping service")
        
        # التهيئة الأولية
        self.update_top_symbols()
        
        while REAL_TIME_SCALPING_ENABLED:
            try:
                # تحديث قائمة الرموز كل 5 دقائق
                if time.time() - self.last_update_time > SCALPING_UPDATE_INTERVAL:
                    self.update_top_symbols()
                
                # معالجة الرموز بشكل متوازي
                if self.top_symbols:
                    list(self.executor.map(self.process_symbol, self.top_symbols))
                
                # فاصل زمني قصير بين الدورات
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"❌ [Scalp] Critical error in scalping service: {e}")
                time.sleep(30)

# ---------------------- Telegram Functions ----------------------
def send_telegram_message(target_chat_id: str, text: str, reply_markup: Optional[Dict] = None, parse_mode: str = 'Markdown', disable_web_page_preview: bool = True, timeout: int = 20) -> Optional[Dict]:
    """Sends a message via Telegram Bot API with improved error handling."""
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
             logger.error(f"❌ [Telegram] فشل تحويل reply_markup إلى JSON: {json_err} - Markup: {reply_markup}")
             return None

    logger.debug(f"ℹ️ [Telegram] إرسال رسالة إلى {target_chat_id}...")
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        logger.info(f"✅ [Telegram] تم إرسال الرسالة بنجاح إلى {target_chat_id}.")
        return response.json()
    except requests.exceptions.Timeout:
         logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (مهلة).")
         return None
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (خطأ HTTP: {http_err.response.status_code}).")
        try:
            error_details = http_err.response.json()
            logger.error(f"❌ [Telegram] تفاصيل خطأ API: {error_details}")
        except json.JSONDecodeError:
            logger.error(f"❌ [Telegram] تعذر فك تشفير استجابة الخطأ: {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (خطأ في الطلب): {req_err}")
        return None
    except Exception as e:
         logger.error(f"❌ [Telegram] خطأ غير متوقع أثناء إرسال الرسالة: {e}", exc_info=True)
         return None

def send_telegram_alert(signal_data: Dict[str, Any], timeframe: str) -> None:
    """Formats and sends enhanced Telegram notifications for new signals in Arabic."""
    logger.debug(f"ℹ️ [Telegram Alert] تنسيق وإرسال تنبيه للإشارة: {signal_data.get('symbol', 'N/A')}")
    try:
        entry_price = float(signal_data['entry_price'])
        target_price = float(signal_data['initial_target'])
        stop_loss_price = float(signal_data['stop_loss_price'])
        symbol = signal_data['symbol']
        strategy_name = signal_data.get('strategy_name', 'N/A')
        volume_15m = signal_data.get('volume_15m', 0.0)
        trade_value_signal = signal_data.get('trade_value', TRADE_VALUE)
        signal_details = signal_data.get('signal_details', {})

        profit_pct = ((target_price / entry_price) - 1) * 100 if entry_price > 0 else 0

        entry_fee = trade_value_signal * BINANCE_FEE_RATE
        exit_value = trade_value_signal * (1 + profit_pct / 100.0)
        exit_fee = exit_value * BINANCE_FEE_RATE
        total_trade_fees = entry_fee + exit_fee

        profit_usdt_gross = trade_value_signal * (profit_pct / 100)
        profit_usdt_net = profit_usdt_gross - total_trade_fees

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')

        fear_greed = get_fear_greed_index()
        ml_prediction_status = signal_details.get('ML_Prediction', 'N/A')
        scalp_signal_status = signal_details.get('Scalp_Signal', 'N/A')
        btc_trend_feature_value = signal_details.get('BTC_Trend_Feature_Value', 0.0)
        btc_trend_display = "صعودي 📈" if btc_trend_feature_value == 1.0 else ("هبوطي 📉" if btc_trend_feature_value == -1.0 else "محايد 🔄")

        prediction_explanation = {
            'صعودي ✅': 'إشارة شراء قوية',
            'هابط قوي ❌': 'إشارة بيع قوية',
            'محايد ➖': 'لا توجد إشارة واضحة'
        }.get(ml_prediction_status, ml_prediction_status)

        # تحديد نوع الإشارة
        signal_type = "إشارة تداول جديدة (ML + Scalp)"
        if strategy_name == 'Realtime_Scalping':
            signal_type = "إشارة تداول سريع فوري ⚡"
            
        message = (
            f"💡 *{signal_type}* 💡\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **نوع الإشارة:** شراء (طويل)\n"
            f"🕰️ **الإطار الزمني:** {timeframe}\n"
            f"💧 **السيولة (آخر 45 دقيقة):** {volume_15m:,.0f} USDT\n"
            f"——————————————\n"
            f"➡️ **سعر الدخول المقترح:** `${entry_price:,.8g}`\n"
            f"🎯 **الهدف الأولي:** `${target_price:,.8g}`\n"
            f"🛑 **وقف الخسارة:** `${stop_loss_price:,.8g}`\n"
            f"💰 **الربح المتوقع (إجمالي):** ({profit_pct:+.2f}% / ≈ ${profit_usdt_gross:+.2f})\n"
            f"💸 **الرسوم المتوقعة:** ${total_trade_fees:,.2f}\n"
            f"📈 **الربح الصافي المتوقع:** ${profit_usdt_net:+.2f}\n"
            f"——————————————\n"
        )
        
        # إضافة تفاصيل إضافية للإشارات السريعة
        if strategy_name == 'Realtime_Scalping':
            message += (
                f"⚡ *شروط التداول السريع:*\n"
                f"  - EMA9 > EMA21: {signal_details.get('EMA_Condition', 'N/A')}\n"
                f"  - السعر > VWAP: {signal_details.get('VWAP_Condition', 'N/A')}\n"
                f"  - حجم تداول مرتفع: {signal_details.get('Volume_Spike', 'N/A')}\n"
                f"  - هامش ربح: {signal_details.get('Profit_Margin', 'N/A')}\n"
            )
        else:
            message += (
                f"🤖 *تنبؤ نموذج ML:* *{ml_prediction_status}* ({prediction_explanation})\n"
                f"⚡ *إشارة التداول السريع:* *{scalp_signal_status}*\n"
                f"✅ *الشروط الإضافية المحققة:*\n"
                f"  - فحص السيولة: {signal_details.get('Volume_Check', 'N/A')}\n"
                f"  - فحص هامش الربح: {signal_details.get('Profit_Margin_Check', 'N/A')}\n"
            )
            
        message += (
            f"——————————————\n"
            f"😨/🤑 **مؤشر الخوف والجشع:** {fear_greed}\n"
            f"₿ **اتجاه البيتكوين (ميزة ML):** {btc_trend_display}\n"
            f"——————————————\n"
            f"⏰ {timestamp_str}"
        )

        reply_markup = {
            "inline_keyboard": [
                [{"text": "📊 عرض تقرير الأداء", "callback_data": "get_report"}]
            ]
        }

        send_telegram_message(CHAT_ID, message, reply_markup=reply_markup, parse_mode='Markdown')

    except KeyError as ke:
        logger.error(f"❌ [Telegram Alert] بيانات الإشارة غير مكتملة للرمز {signal_data.get('symbol', 'N/A')}: مفتاح مفقود {ke}", exc_info=True)
    except Exception as e:
        logger.error(f"❌ [Telegram Alert] فشل إرسال تنبيه الإشارة للرمز {signal_data.get('symbol', 'N/A')}: {e}", exc_info=True)

def send_tracking_notification(details: Dict[str, Any]) -> None:
    """Formats and sends enhanced Telegram notifications for tracking events in Arabic."""
    symbol = details.get('symbol', 'N/A')
    signal_id = details.get('id', 'N/A')
    notification_type = details.get('type', 'unknown')
    message = ""
    safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
    closing_price = details.get('closing_price', 0.0)
    profit_pct = details.get('profit_pct', 0.0)
    current_price = details.get('current_price', 0.0)
    time_to_target = details.get('time_to_target', 'N/A')
    old_target = details.get('old_target', 0.0)
    new_target = details.get('new_target', 0.0)
    old_stop_loss = details.get('old_stop_loss', 0.0)
    new_stop_loss = details.get('new_stop_loss', 0.0)
    reason = details.get('reason', '')

    logger.debug(f"ℹ️ [Notification] تنسيق إشعار التتبع: ID={signal_id}, Type={notification_type}, Symbol={symbol}")

    if notification_type == 'target_hit':
        message = (
            f"✅ *تم الوصول إلى الهدف (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"🎯 **سعر الإغلاق (الهدف):** `${closing_price:,.8g}`\n"
            f"💰 **الربح المحقق:** {profit_pct:+.2f}%\n"
            f"⏱️ **الوقت المستغرق:** {time_to_target}"
        )
    elif notification_type == 'target_updated':
         message = (
             f"↗️ *تم تحديث الهدف (ID: {signal_id})*\n"
             f"——————————————\n"
             f"🪙 **الزوج:** `{safe_symbol}`\n"
             f"📈 **السعر الحالي:** `${current_price:,.8g}`\n"
             f"🎯 **الهدف السابق:** `${old_target:,.8g}`\n"
             f"🎯 **الهدف الجديد:** `${new_target:,.8g}`\n"
             f"ℹ️ *تم التحديد بناءً على استمرار الزخم الصعودي.*"
         )
    elif notification_type == 'stop_loss' or notification_type == 'take_profit':
        message = (
            f"🛑 *تم تنفيذ {reason} (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"💰 **الربح/الخسارة:** {profit_pct:+.2f}%\n"
            f"➡️ **سعر الدخول:** ${entry_price:.8g}\n"
            f"🛑 **سعر الإغلاق:** ${closing_price:.8g}\n"
            f"⏱️ **الوقت المستغرق:** {time_to_target}"
        )
    elif notification_type == 'trailing_activated':
        current_profit = details.get('current_profit', 0.0)
        new_stop_loss = details.get('new_stop_loss', 0.0)
        message = (
            f"🚀 *تم تفعيل التوقف المتابع (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **الربح الحالي:** {current_profit:.2f}%\n"
            f"🛑 **وقف الخسارة الجديد:** ${new_stop_loss:.8g}\n"
            f"ℹ️ *تم تفعيل نظام التوقف المتابع عند تحقيق {PROFIT_TARGET_PCT}% ربح.*"
        )
    elif notification_type == 'trailing_updated':
        old_stop_loss = details.get('old_stop_loss', 0.0)
        new_stop_loss = details.get('new_stop_loss', 0.0)
        message = (
            f"📈 *تم تحديث وقف الخسارة المتابع (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"🛑 **وقف الخسارة السابق:** ${old_stop_loss:.8g}\n"
            f"🛑 **وقف الخسارة الجديد:** ${new_stop_loss:.8g}\n"
            f"📈 **السعر الحالي:** ${current_price:.8g}"
        )
    else:
        logger.warning(f"⚠️ [Notification] نوع إشعار غير معروف: {notification_type} للتفاصيل: {details}")
        return

    if message:
        send_telegram_message(CHAT_ID, message, parse_mode='Markdown')

# ---------------------- Database Functions (Insert and Update) ----------------------
def insert_signal_into_db(signal: Dict[str, Any]) -> bool:
    """Inserts a new signal into the signals table with the weighted score and entry time."""
    if not check_db_connection() or not conn:
        logger.error(f"❌ [DB Insert] فشل إدراج الإشارة {signal.get('symbol', 'N/A')} بسبب مشكلة في اتصال قاعدة البيانات.")
        return False

    symbol = signal.get('symbol', 'N/A')
    logger.debug(f"ℹ️ [DB Insert] محاولة إدراج إشارة لـ {symbol}...")
    try:
        # حساب الكمية وتنفيذ الصفقة
        quantity = calculate_quantity(
            symbol, 
            TRADE_VALUE,
            float(signal['entry_price']),
            float(signal['stop_loss_price'])
        )
        
        if quantity <= 0:
            logger.error(f"❌ [Trade] كمية غير صالحة لـ {symbol}: {quantity}")
            return False

        # التحقق من الرصيد
        usdt_balance = check_balance('USDT')
        if usdt_balance < TRADE_VALUE:
            logger.error(f"❌ [Trade] الرصيد غير كافٍ: {usdt_balance} USDT < {TRADE_VALUE} USDT المطلوبة")
            return False

        # تنفيذ أمر الشراء
        buy_order = execute_buy_order(symbol, quantity)
        if not buy_order:
            logger.error(f"❌ [Trade] فشل تنفيذ أمر شراء لـ {symbol}")
            return False

        # استخراج تفاصيل التنفيذ
        order_id = buy_order['orderId']
        executed_qty = float(buy_order['executedQty'])
        fills = buy_order.get('fills', [])
        if fills:
            # حساب متوسط سعر الشراء
            total_quote = 0.0
            total_qty = 0.0
            for fill in fills:
                qty = float(fill['qty'])
                price = float(fill['price'])
                total_quote += qty * price
                total_qty += qty
            avg_price = total_quote / total_qty
        else:
            avg_price = float(signal['entry_price'])

        # تحديث إشارة الدخول بالسعر الحقيقي والكمية
        signal['entry_price'] = avg_price
        signal['executed_quantity'] = executed_qty
        signal['order_id'] = order_id

        signal_prepared = convert_np_values(signal)
        signal_details_json = json.dumps(signal_prepared.get('signal_details', {}))
        stop_loss_price = float(signal_prepared['stop_loss_price'])

        with conn.cursor() as cur_ins:
            insert_query = sql.SQL("""
                INSERT INTO signals
                 (symbol, entry_price, initial_target, current_target, stop_loss_price,
                 r2_score, strategy_name, signal_details, volume_15m, entry_time,
                 order_id, executed_quantity, buy_executed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, NOW());
            """)
            cur_ins.execute(insert_query, (
                signal_prepared['symbol'],
                signal_prepared['entry_price'],
                signal_prepared['initial_target'],
                signal_prepared['current_target'],
                stop_loss_price,
                signal_prepared.get('r2_score'),
                signal_prepared.get('strategy_name', 'unknown'),
                signal_details_json,
                signal_prepared.get('volume_15m'),
                signal_prepared.get('order_id'),
                signal_prepared.get('executed_quantity')
            ))
        conn.commit()
        logger.info(f"✅ [DB Insert] تم إدراج إشارة لـ {symbol} في قاعدة البيانات (وقف الخسارة: {stop_loss_price:.8g}).")
        return True
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB Insert] خطأ في قاعدة البيانات أثناء إدراج إشارة لـ {symbol}: {db_err}")
        if conn: conn.rollback()
        return False
    except (TypeError, ValueError) as convert_err:
         logger.error(f"❌ [DB Insert] خطأ في تحويل بيانات الإشارة قبل الإدراج لـ {symbol}: {convert_err} - بيانات الإشارة: {signal}")
         if conn: conn.rollback()
         return False
    except Exception as e:
        logger.error(f"❌ [DB Insert] خطأ غير متوقع أثناء إدراج إشارة لـ {symbol}: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# ---------------------- Open Signal Tracking Function ----------------------
def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> float:
    """تقريب الكمية حسب متطلبات LOT_SIZE على Binance"""
    try:
        symbol_info = client.get_symbol_info(symbol)
        if not symbol_info:
            return quantity
            
        # إيجاد فلتر LOT_SIZE
        lot_size_filter = next(
            (f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'),
            None
        )
        
        if lot_size_filter:
            min_qty = float(lot_size_filter['minQty'])
            step_size = float(lot_size_filter['stepSize'])
            
            # التقريب حسب stepSize
            precision = int(round(-math.log(step_size, 10)))
            adjusted_quantity = math.floor(quantity / step_size) * step_size
            
            # التأكد من الحد الأدنى
            if adjusted_quantity < min_qty:
                logger.warning(f"⚠️ [Quantity Adjust] {symbol}: الكمية {adjusted_quantity} أقل من الحد الأدنى {min_qty}")
                return 0.0
                
            logger.info(f"✅ [Quantity Adjust] {symbol}: {quantity:.6f} -> {adjusted_quantity:.6f} (step: {step_size})")
            return round(adjusted_quantity, precision)
        else:
            return quantity
            
    except Exception as e:
        logger.error(f"❌ [Quantity Adjust] خطأ في تقريب كمية {symbol}: {e}")
        return quantity

def check_sell_balance(symbol: str, quantity: float) -> bool:
    """التحقق من الرصيد الحقيقي قبل البيع مع هامش أمان"""
    try:
        # استخراج العملة الأساسية من الزوج (مثل BAT من BATUSDT)
        base_asset = symbol.replace('USDT', '')
        
        # جلب الرصيد الحقيقي
        balance = client.get_asset_balance(asset=base_asset)
        if balance:
            available_balance = float(balance['free'])
            
            # 🔧 إضافة هامش أمان (0.1%)
            margin = 0.001  # 0.1% هامش أمان
            adjusted_quantity = quantity * (1 - margin)
            
            logger.info(f"💰 [Balance Check] {symbol}: الرصيد المتاح: {available_balance:.6f} {base_asset}, الكمية المعدلة: {adjusted_quantity:.6f} (هامش: {margin*100}%)")
            
            if available_balance >= adjusted_quantity:
                return True
            else:
                logger.error(f"❌ [Balance Check] {symbol}: الرصيد غير كافٍ. المتاح: {available_balance:.6f}, المطلوب المعدل: {adjusted_quantity:.6f}")
                return False
        return False
    except Exception as e:
        logger.error(f"❌ [Balance Check] خطأ في التحقق من رصيد {symbol}: {e}")
        return False

def verify_trade_in_db(signal_id: int) -> None:
    """التحقق من حالة الصفقة في قاعدة البيانات"""
    try:
        with conn.cursor() as verify_cur:
            verify_cur.execute("""
                SELECT symbol, entry_price, executed_quantity, buy_executed_at, sell_executed_at
                FROM signals WHERE id = %s
            """, (signal_id,))
            trade = verify_cur.fetchone()
            if trade:
                logger.info(f"🔍 [Trade Verify] ID:{signal_id} - {trade['symbol']}: "
                           f"الدخول: {trade['entry_price']}, الكمية: {trade['executed_quantity']}, "
                           f"وقت الشراء: {trade['buy_executed_at']}, وقت البيع: {trade['sell_executed_at']}")
    except Exception as e:
        logger.error(f"❌ [Trade Verify] خطأ في التحقق من الصفقة {signal_id}: {e}")

def track_signals() -> None:
    """Tracks open signals and checks targets. Calculates time to target upon hit."""
    logger.info("ℹ️ [Tracker] بدء عملية تتبع الإشارات المفتوحة...")
    while True:
        active_signals_summary: List[str] = []
        processed_in_cycle = 0
        try:
            if not check_db_connection() or not conn:
                logger.warning("⚠️ [Tracker] تخطي دورة التتبع بسبب مشكلة في اتصال قاعدة البيانات.")
                time.sleep(15)
                continue

            with conn.cursor() as track_cur:
                 track_cur.execute("""
                    SELECT id, symbol, entry_price, initial_target, current_target, stop_loss_price, trailing_activated, entry_time, executed_quantity
                    FROM signals
                    WHERE achieved_target = FALSE;
                """)
                 open_signals: List[Dict] = track_cur.fetchall()

            if not open_signals:
                time.sleep(10)
                continue

            logger.debug(f"ℹ️ [Tracker] تتبع {len(open_signals)} إشارة مفتوحة...")

            for signal_row in open_signals:
                signal_id = signal_row['id']
                symbol = signal_row['symbol']
                processed_in_cycle += 1
                update_executed = False

                try:
                    entry_price = float(signal_row['entry_price'])
                    entry_time = signal_row['entry_time']
                    current_target = float(signal_row['current_target'])
                    stop_loss_price = float(signal_row['stop_loss_price'])
                    trailing_activated = signal_row['trailing_activated']
                    quantity = float(signal_row['executed_quantity']) if signal_row['executed_quantity'] else 0.0

                    current_price = ticker_data.get(symbol)
                    if current_price is None:
                         logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): السعر الحالي غير متاح في بيانات التيكر.")
                         continue

                    # 🔍 أضف هذا السطر للـ Debugging:
                    logger.info(f"🔍 [DEBUG] {symbol}: السعر={current_price:.6f}, الدخول={entry_price:.6f}, الربح={((current_price/entry_price)-1)*100:.2f}%, التوقف_مفعل={trailing_activated}, وقف_الخسارة={stop_loss_price:.6f}")

                    # التحقق من حالة الصفقة في DB
                    verify_trade_in_db(signal_id)

                    active_signals_summary.append(f"{symbol}({signal_id}): P={current_price:.4f} T={current_target:.4f} SL={stop_loss_price:.4f}")

                    update_query: Optional[sql.SQL] = None
                    update_params: Tuple = ()
                    log_message: Optional[str] = None
                    notification_details: Dict[str, Any] = {'symbol': symbol, 'id': signal_id, 'current_price': current_price}

                    # --- 1. التحقق من وقف الخسارة الأولي ---
                    if current_price <= stop_loss_price and not trailing_activated:
                        # تنفيذ أمر البيع عند وقف الخسارة الأولي
                        if quantity > 0:
                            # استخدام الرصيد الفعلي المتاح بدلاً من الكمية النظرية
                            base_asset = symbol.replace('USDT', '')
                            balance = client.get_asset_balance(asset=base_asset)
                            if balance:
                                actual_quantity = float(balance['free'])
                                # تقريب الكمية حسب متطلبات التداول
                                actual_quantity = adjust_quantity_to_lot_size(symbol, actual_quantity)
                                
                                if actual_quantity > 0:
                                    logger.info(f"🔄 [StopLoss Sell] {symbol}: بيع {actual_quantity:.6f} {base_asset} (الرصيد الفعلي)")
                                    sell_order = execute_sell_order(symbol, actual_quantity)
                                    if not sell_order:
                                        logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): فشل بيع عند وقف الخسارة.")
                                        continue
                                else:
                                    logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): كمية غير صالحة بعد التقريب: {actual_quantity}")
                                    continue
                            else:
                                logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): لا يمكن الحصول على الرصيد")
                                continue
                        else:
                            logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): كمية غير صالحة للبيع: {quantity}")
                            
                        profit_pct = ((current_price / entry_price) - 1) * 100
                        closed_at = datetime.now()
                        time_to_target_duration = closed_at - entry_time if entry_time else timedelta(0)
                        time_to_target_str = str(time_to_target_duration)

                        update_query = sql.SQL("""
                            UPDATE signals SET 
                                achieved_target = TRUE, 
                                closing_price = %s, 
                                closed_at = %s, 
                                profit_percentage = %s, 
                                time_to_target = %s,
                                sell_executed_at = NOW()
                            WHERE id = %s;
                        """)
                        update_params = (current_price, closed_at, profit_pct, time_to_target_duration, signal_id)
                        log_message = f"🛑 [Tracker] {symbol}(ID:{signal_id}): تم تنفيذ وقف الخسارة عند {current_price:.8g} (الربح: {profit_pct:+.2f}%, الوقت: {time_to_target_str})."
                        notification_details.update({
                            'type': 'stop_loss',
                            'closing_price': current_price,
                            'profit_pct': profit_pct,
                            'time_to_target': time_to_target_str,
                            'reason': 'وقف خسارة'
                        })
                        update_executed = True

                    # --- 2. التحقق من جني الأرباح لتفعيل التوقف المتابع ---
                    profit_pct_current = ((current_price / entry_price) - 1) * 100

                    if not update_executed and not trailing_activated and profit_pct_current >= PROFIT_TARGET_PCT:
                        # تفعيل التوقف المتابع
                        new_stop_loss = current_price * (1 - TRAILING_STOP_PCT/100.0)
                        
                        update_query = sql.SQL("""
                            UPDATE signals 
                            SET stop_loss_price = %s, 
                                trailing_activated = TRUE 
                            WHERE id = %s;
                        """)
                        update_params = (new_stop_loss, signal_id)
                        log_message = f"🚀 [Tracker] {symbol}(ID:{signal_id}): تم تفعيل التوقف المتابع عند {new_stop_loss:.6f} (الربح: {profit_pct_current:.2f}%)."
                        notification_details.update({
                            'type': 'trailing_activated',
                            'new_stop_loss': new_stop_loss,
                            'current_profit': profit_pct_current
                        })
                        update_executed = True
                        logger.info(f"✅ [TRAILING_ACTIVATED] {symbol}: التوقف المتابع مفعل - الدخول: {entry_price:.6f}, الحالي: {current_price:.6f}, الربح: {profit_pct_current:.2f}%, وقف الخسارة: {new_stop_loss:.6f}")

                    # --- 3. تحديث وقف الخسارة المتابع للأعلى ---
                    if not update_executed and trailing_activated:
                        # حساب وقف الخسارة الجديد بناءً على السعر الحالي
                        new_stop_loss_candidate = current_price * (1 - TRAILING_STOP_PCT/100.0)
                        
                        # فقط إذا كان وقف الخسارة الجديد أعلى من القديم
                        if new_stop_loss_candidate > stop_loss_price:
                            update_query = sql.SQL("""
                                UPDATE signals 
                                SET stop_loss_price = %s 
                                WHERE id = %s;
                            """)
                            update_params = (new_stop_loss_candidate, signal_id)
                            log_message = f"📈 [Tracker] {symbol}(ID:{signal_id}): تم تحديث وقف الخسارة المتابع من {stop_loss_price:.6f} إلى {new_stop_loss_candidate:.6f}."
                            notification_details.update({
                                'type': 'trailing_updated',
                                'old_stop_loss': stop_loss_price,
                                'new_stop_loss': new_stop_loss_candidate
                            })
                            update_executed = True
                            logger.info(f"🔼 [TRAILING_UPDATED] {symbol}: تم رفع وقف الخسارة إلى {new_stop_loss_candidate:.6f} (السعر: {current_price:.6f})")

                    # --- 4. البيع عند وقف الخسارة المتابع ---
                    if not update_executed and trailing_activated and current_price <= stop_loss_price:
                        # إضافة هامش صغير لتجنب مشاكل الدقة
                        if current_price <= stop_loss_price * 1.0001:  # هامش 0.01%
                            # البيع عند الارتداد
                            if quantity > 0:
                                logger.info(f"🎯 [TRAILING_SELL] {symbol}: تنفيذ بيع بالتوقف المتابع - الكمية: {quantity:.6f}, السعر: {current_price:.6f}, وقف الخسارة: {stop_loss_price:.6f}")
                                
                                # 🔧 استخدام الرصيد الفعلي مع هامش أمان
                                base_asset = symbol.replace('USDT', '')
                                balance = client.get_asset_balance(asset=base_asset)
                                if balance:
                                    actual_quantity = float(balance['free'])
                                    actual_quantity = adjust_quantity_to_lot_size(symbol, actual_quantity)
                                    
                                    if actual_quantity > 0:
                                        logger.info(f"🔄 [Sell Execution] {symbol}: بيع {actual_quantity:.6f} {base_asset} (الرصيد الفعلي)")
                                        sell_order = execute_sell_order(symbol, actual_quantity)
                                        if not sell_order:
                                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): فشل بيع عند وقف الخسارة المتابع.")
                                            continue
                                    else:
                                        logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): كمية غير صالحة بعد التقريب: {actual_quantity}")
                                        continue
                                else:
                                    logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): لا يمكن الحصول على الرصيد")
                                    continue
                            else:
                                logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): كمية غير صالحة للبيع: {quantity}")
                                
                            profit_pct = ((current_price / entry_price) - 1) * 100
                            closed_at = datetime.now()
                            time_to_target_duration = closed_at - entry_time if entry_time else timedelta(0)
                            time_to_target_str = str(time_to_target_duration)

                            update_query = sql.SQL("""
                                UPDATE signals SET 
                                    achieved_target = TRUE, 
                                    closing_price = %s, 
                                    closed_at = %s, 
                                    profit_percentage = %s, 
                                    time_to_target = %s,
                                    sell_executed_at = NOW()
                                WHERE id = %s;
                            """)
                            update_params = (current_price, closed_at, profit_pct, time_to_target_duration, signal_id)
                            log_message = f"🛑 [Tracker] {symbol}(ID:{signal_id}): تم البيع عند التوقف المتابع {current_price:.6f} (الربح: {profit_pct:+.2f}%, الوقت: {time_to_target_str})."
                            notification_details.update({
                                'type': 'trailing_stop',
                                'closing_price': current_price,
                                'profit_pct': profit_pct,
                                'time_to_target': time_to_target_str,
                                'reason': 'توقف متابع'
                            })
                            update_executed = True
                            logger.info(f"✅ [TRAILING_EXECUTED] {symbol}: تم البيع بنجاح - الربح: {profit_pct:+.2f}%")

                    if update_executed and update_query:
                        try:
                             with conn.cursor() as update_cur:
                                  update_cur.execute(update_query, update_params)
                             conn.commit()
                             if log_message: logger.info(log_message)
                             if notification_details.get('type'):
                                send_tracking_notification(notification_details)
                        except psycopg2.Error as db_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ في قاعدة البيانات أثناء التحديث: {db_err}")
                            if conn: conn.rollback()
                        except Exception as exec_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ غير متوقع أثناء تنفيذ التحديث/الإشعار: {exec_err}", exc_info=True)
                            if conn: conn.rollback()

                except (TypeError, ValueError) as convert_err:
                    logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ في تحويل قيم الإشارة الأولية: {convert_err}")
                    continue
                except Exception as inner_loop_err:
                     logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ غير متوقع أثناء معالجة الإشارة: {inner_loop_err}", exc_info=True)
                     continue

            if active_signals_summary:
                logger.debug(f"ℹ️ [Tracker] نهاية حالة الدورة ({processed_in_cycle} معالجة): {'; '.join(active_signals_summary)}")

            time.sleep(3)

        except psycopg2.Error as db_cycle_err:
             logger.error(f"❌ [Tracker] خطأ في قاعدة البيانات في دورة التتبع الرئيسية: {db_cycle_err}. محاولة إعادة الاتصال...")
             if conn: conn.rollback()
             time.sleep(30)
             check_db_connection()
        except Exception as cycle_err:
            logger.error(f"❌ [Tracker] خطأ غير متوقع في دورة تتبع الإشارة: {cycle_err}", exc_info=True)
            time.sleep(30)

def get_interval_minutes(interval: str) -> int:
    """Helper function to convert Binance interval string to minutes."""
    if interval.endswith('m'):
        return int(interval[:-1])
    elif interval.endswith('h'):
        return int(interval[:-1]) * 60
    elif interval.endswith('d'):
        return int(interval[:-1]) * 60 * 24
    return 0

# ---------------------- Flask Service (Optional for Webhook) ----------------------
app = Flask(__name__, template_folder='templates')  # استبدل 'my_templates' باسم المجلد الذي تريد
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))  # أضف هذا السطر
app.config['DEBUG'] = True  # أضف هذا السطر
@app.route('/')
def home() -> Response:
    """Simple home page to show the bot is running."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ws_alive = ws_thread.is_alive() if 'ws_thread' in globals() and ws_thread else False
    tracker_alive = tracker_thread.is_alive() if 'tracker_thread' in globals() and tracker_thread else False
    main_bot_alive = main_bot_thread.is_alive() if 'main_bot_thread' in globals() and main_bot_thread else False
    scalp_alive = scalp_thread.is_alive() if 'scalp_thread' in globals() and scalp_thread else False
    status = "running" if ws_alive and tracker_alive and main_bot_alive else "partially running"
    
    # إضافة حالة خدمة التداول السريع
    if REAL_TIME_SCALPING_ENABLED:
        status += f" | Scalping: {'running' if scalp_alive else 'stopped'}"
    
    return Response(f"📈 Crypto Signal Bot ({status}) - Last Check: {now}", status=200, mimetype='text/plain')

@app.route('/favicon.ico')
def favicon() -> Response:
    """Handles favicon request to avoid 404 errors in logs."""
    return Response(status=204)

@app.route('/webhook', methods=['POST'])
def webhook() -> Tuple[str, int]:
    """Handles incoming requests from Telegram (like button presses and commands)."""
    if not WEBHOOK_URL:
        logger.warning("⚠️ [Flask] تم استلام طلب webhook، ولكن WEBHOOK_URL غير مهيأ. تجاهل الطلب.")
        return "Webhook not configured", 200

    if not request.is_json:
        logger.warning("⚠️ [Flask] تم استلام طلب webhook غير JSON.")
        return "Invalid request format", 400

    try:
        data = request.get_json()
        logger.info(f"✅ [Flask] تم استلام بيانات webhook. حجم البيانات: {len(json.dumps(data))} بايت.")
        logger.debug(f"ℹ️ [Flask] بيانات webhook الكاملة: {json.dumps(data)}")

        if 'callback_query' in data:
            callback_query = data['callback_query']
            callback_id = callback_query['id']
            callback_data = callback_query.get('data')
            message_info = callback_query.get('message')

            logger.info(f"ℹ️ [Flask] تم استلام استعلام رد اتصال (Callback Query). ID: {callback_id}, البيانات: '{callback_data}'")

            if not message_info or not callback_data:
                 logger.warning(f"⚠️ [Flask] استعلام رد الاتصال (ID: {callback_id}) يفتقد الرسالة أو البيانات. تجاهل.")
                 try:
                     ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                     requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                 except Exception as ack_err:
                     logger.warning(f"⚠️ [Flask] فشل تأكيد استعلام رد الاتصال غير الصالح {callback_id}: {ack_err}")
                 return "OK", 200
            chat_id_callback = message_info.get('chat', {}).get('id')
            if not chat_id_callback:
                 logger.warning(f"⚠️ [Flask] استعلام رد الاتصال (ID: {callback_id}) يفتقد معرف الدردشة. تجاهل.")
                 try:
                     ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                     requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                 except Exception as ack_err:
                     logger.warning(f"⚠️ [Flask] فشل تأكيد استعلام رد الاتصال غير الصالح {callback_id}: {ack_err}")
                 return "OK", 200

            message_id = message_info['message_id']
            user_info = callback_query.get('from', {})
            user_id = user_info.get('id')
            username = user_info.get('username', 'N/A')

            logger.info(f"ℹ️ [Flask] معالجة استعلام رد الاتصال: البيانات='{callback_data}', المستخدم={username}({user_id}), الدردشة={chat_id_callback}")

            try:
                ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                logger.debug(f"✅ [Flask] تم تأكيد استعلام رد الاتصال {callback_id}.")
            except Exception as ack_err:
                 logger.warning(f"⚠️ [Flask] فشل تأكيد استعلام رد الاتصال {callback_id}: {ack_err}")

            if callback_data == "get_report":
                logger.info(f"ℹ️ [Flask] تم استلام طلب 'get_report' من الدردشة {chat_id_callback}. جاري إنشاء التقرير...")
                report_content = generate_performance_report()
                logger.info(f"✅ [Flask] تم إنشاء التقرير. طول التقرير: {len(report_content)} حرف.")
                report_thread = Thread(target=lambda: send_telegram_message(chat_id_callback, report_content, parse_mode='Markdown'))
                report_thread.start()
                logger.info(f"✅ [Flask] تم بدء خيط إرسال التقرير للدردشة {chat_id_callback}.")
            else:
                logger.warning(f"⚠️ [Flask] تم استلام بيانات رد اتصال غير معالجة: '{callback_data}'")

        elif 'message' in data:
            message_data = data['message']
            chat_info = message_data.get('chat')
            user_info = message_data.get('from', {})
            text_msg = message_data.get('text', '').strip()

            if not chat_info or not text_msg:
                 logger.debug("ℹ️ [Flask] تم استلام رسالة بدون معلومات الدردشة أو النص.")
                 return "OK", 200

            chat_id_msg = chat_info['id']
            user_id = user_info.get('id')
            username = user_info.get('username', 'N/A')

            logger.info(f"ℹ️ [Flask] تم استلام رسالة: النص='{text_msg}', المستخدم={username}({user_id}), الدردشة={chat_id_msg}")

            if text_msg.lower() == '/report':
                 report_thread = Thread(target=lambda: send_telegram_message(chat_id_msg, generate_performance_report(), parse_mode='Markdown'))
                 report_thread.start()
            elif text_msg.lower() == '/status':
                 status_thread = Thread(target=handle_status_command, args=(chat_id_msg,))
                 status_thread.start()

        else:
            logger.debug("ℹ️ [Flask] تم استلام بيانات webhook بدون 'callback_query' أو 'message'.")

        return "OK", 200
    except Exception as e:
         logger.error(f"❌ [Flask] خطأ في معالجة webhook: {e}", exc_info=True)
         return "Internal Server Error", 500

def handle_status_command(chat_id_msg: int) -> None:
    """Separate function to handle /status command to avoid blocking the Webhook."""
    logger.info(f"ℹ️ [Flask Status] معالجة أمر /status للدردشة {chat_id_msg}")
    status_msg = "⏳ جلب الحالة..."
    msg_sent = send_telegram_message(chat_id_msg, status_msg)
    if not (msg_sent and msg_sent.get('ok')):
         logger.error(f"❌ [Flask Status] فشل إرسال رسالة الحالة الأولية إلى {chat_id_msg}")
         return
    message_id_to_edit = msg_sent['result']['message_id'] if msg_sent and msg_sent.get('result') else None

    if message_id_to_edit is None:
        logger.error(f"❌ [Flask Status] فشل الحصول على message_id لتحديث الحالة في الدردشة {chat_id_msg}")
        return

    try:
        open_count = 0
        if check_db_connection() and conn:
            with conn.cursor() as status_cur:
                status_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                open_count = (status_cur.fetchone() or {}).get('count', 0)

        ws_status = 'نشط ✅' if 'ws_thread' in globals() and ws_thread and ws_thread.is_alive() else 'غير نشط ❌'
        tracker_status = 'نشط ✅' if 'tracker_thread' in globals() and tracker_thread and tracker_thread.is_alive() else 'غير نشط ❌'
        main_bot_alive = 'نشط ✅' if 'main_bot_thread' in globals() and main_bot_thread and main_bot_thread.is_alive() else 'غير نشط ❌'
        scalp_status = 'نشط ✅' if REAL_TIME_SCALPING_ENABLED and 'scalp_thread' in globals() and scalp_thread and scalp_thread.is_alive() else 'غير نشط ❌'
        
        final_status_msg = (
            f"🤖 *حالة البوت:*\n"
            f"- تتبع الأسعار (WS): {ws_status}\n"
            f"- تتبع الإشارات: {tracker_status}\n"
            f"- حلقة البوت الرئيسية: {main_bot_alive}\n"
            f"- خدمة التداول السريع: {scalp_status}\n"
            f"- الإشارات النشطة: *{open_count}* / {MAX_OPEN_TRADES}\n"
            f"- وقت الخادم الحالي: {datetime.now().strftime('%H:%M:%S')}"
        )
        edit_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
        edit_payload = {
            'chat_id': chat_id_msg,
             'message_id': message_id_to_edit,
            'text': final_status_msg,
            'parse_mode': 'Markdown'
        }
        response = requests.post(edit_url, json=edit_payload, timeout=10)
        response.raise_for_status()
        logger.info(f"✅ [Flask Status] تم تحديث الحالة للدردشة {chat_id_msg}")

    except Exception as status_err:
        logger.error(f"❌ [Flask Status] خطأ في جلب/تعديل تفاصيل الحالة للدردشة {chat_id_msg}: {status_err}", exc_info=True)
        send_telegram_message(chat_id_msg, "❌ حدث خطأ أثناء جلب تفاصيل الحالة.")

# ---------------------- Web Dashboard ----------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == DASHBOARD_USER and password == DASHBOARD_PASS:
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='بيانات الدخول غير صحيحة')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if not check_db_connection() or not conn:
        return "فشل الاتصال بقاعدة البيانات", 500
    
    try:
        # جلب الصفقات المفتوحة
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, symbol, entry_price, initial_target, current_target, stop_loss_price,
                       entry_time, volume_15m, signal_details, trailing_activated
                FROM signals 
                WHERE achieved_target = FALSE
                ORDER BY entry_time DESC
            """)
            open_trades = cur.fetchall()
            
            # جلب الصفقات المغلقة
            cur.execute("""
                SELECT id, symbol, entry_price, closing_price, profit_percentage,
                       time_to_target, closed_at, strategy_name
                FROM signals 
                WHERE achieved_target = TRUE
                ORDER BY closed_at DESC
                LIMIT 50
            """)
            closed_trades = cur.fetchall()
            
            # جلب إحصاءات الأداء
            cur.execute("""
                SELECT 
                    COUNT(*) AS total_signals,
                    COUNT(*) FILTER (WHERE achieved_target = TRUE) AS closed_signals,
                    COUNT(*) FILTER (WHERE achieved_target = FALSE) AS open_signals,
                    COALESCE(AVG(profit_percentage), 0) AS avg_profit,
                    COALESCE(SUM(profit_percentage), 0) AS total_profit
                FROM signals
            """)
            stats = cur.fetchone()
            
            # جلب إعدادات البوت
            cur.execute("""
                SELECT setting_name, setting_value 
                FROM bot_settings
            """)
            settings_rows = cur.fetchall()
            settings = {row['setting_name']: row['setting_value'] for row in settings_rows}
            
            # إضافة الإعدادات الجديدة إذا لم تكن موجودة
            if 'STOP_LOSS_PCT' not in settings:
                settings['STOP_LOSS_PCT'] = STOP_LOSS_PCT
            if 'PROFIT_TARGET_PCT' not in settings:
                settings['PROFIT_TARGET_PCT'] = PROFIT_TARGET_PCT
            if 'TRAILING_STOP_PCT' not in settings:
                settings['TRAILING_STOP_PCT'] = TRAILING_STOP_PCT

        # الحصول على الوقت الحالي
        current_time = datetime.now()
        
        # حساب الربح/الخسارة الحالية للصفقات المفتوحة
        for trade in open_trades:
            current_price = ticker_data.get(trade['symbol'])
            if current_price:
                # حساب نسبة الربح/الخسارة الحالية
                trade['current_profit_pct'] = ((current_price - float(trade['entry_price'])) / float(trade['entry_price'])) * 100
                
                # إضافة حالة الهدف والتتبع
                if trade['trailing_activated']:
                    trade['target_status'] = "ملغى (تتبع)"
                else:
                    trade['target_status'] = "نشط"
                
                trade['trailing_status'] = "مفعل" if trade['trailing_activated'] else "غير مفعل"
            else:
                trade['current_profit_pct'] = 0.0
                trade['target_status'] = "غير معروف"
                trade['trailing_status'] = "غير معروف"
        
        return render_template('dashboard.html', 
                              open_trades=open_trades,
                              closed_trades=closed_trades,
                              stats=stats,
                              settings=settings,
                              now=current_time)  # تمرير المتغير للقالب
        
    except psycopg2.Error as e:
        return f"خطأ في قاعدة البيانات: {e}", 500

@app.route('/update_settings', methods=['POST'])
def update_settings():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    trade_value = request.form.get('TRADE_VALUE')
    max_open_trades = request.form.get('MAX_OPEN_TRADES')
    stop_loss_pct = request.form.get('STOP_LOSS_PCT')
    profit_target_pct = request.form.get('PROFIT_TARGET_PCT')
    trailing_stop_pct = request.form.get('TRAILING_STOP_PCT')
    
    if not check_db_connection() or not conn:
        return "فشل الاتصال بقاعدة البيانات", 500
    
    try:
        with conn.cursor() as cur:
            # تحديث TRADE_VALUE و MAX_OPEN_TRADES كما كان
            cur.execute("""
                UPDATE bot_settings
                SET setting_value = %s
                WHERE setting_name = 'TRADE_VALUE'
            """, (trade_value,))
            
            cur.execute("""
                UPDATE bot_settings
                SET setting_value = %s
                WHERE setting_name = 'MAX_OPEN_TRADES'
            """, (max_open_trades,))
            
            # تحديث الإعدادات الجديدة
            # STOP_LOSS_PCT
            cur.execute("""
                INSERT INTO bot_settings (setting_name, setting_value)
                VALUES ('STOP_LOSS_PCT', %s)
                ON CONFLICT (setting_name) DO UPDATE 
                SET setting_value = EXCLUDED.setting_value;
            """, (stop_loss_pct,))
            
            # PROFIT_TARGET_PCT
            cur.execute("""
                INSERT INTO bot_settings (setting_name, setting_value)
                VALUES ('PROFIT_TARGET_PCT', %s)
                ON CONFLICT (setting_name) DO UPDATE 
                SET setting_value = EXCLUDED.setting_value;
            """, (profit_target_pct,))
            
            # TRAILING_STOP_PCT
            cur.execute("""
                INSERT INTO bot_settings (setting_name, setting_value)
                VALUES ('TRAILING_STOP_PCT', %s)
                ON CONFLICT (setting_name) DO UPDATE 
                SET setting_value = EXCLUDED.setting_value;
            """, (trailing_stop_pct,))
        
        conn.commit()
        
        # تحديث المتغيرات العاملة
        global TRADE_VALUE, MAX_OPEN_TRADES, STOP_LOSS_PCT, PROFIT_TARGET_PCT, TRAILING_STOP_PCT
        TRADE_VALUE = float(trade_value)
        MAX_OPEN_TRADES = int(max_open_trades)
        STOP_LOSS_PCT = float(stop_loss_pct)
        PROFIT_TARGET_PCT = float(profit_target_pct)
        TRAILING_STOP_PCT = float(trailing_stop_pct)
        
        return redirect(url_for('dashboard'))
        
    except psycopg2.Error as e:
        conn.rollback()
        return f"خطأ في تحديث الإعدادات: {e}", 500
    except Exception as e:
        logger.error(f"خطأ في تحديث الإعدادات: {e}")
        return f"خطأ داخلي: {e}", 500

@app.route('/close_trade/<int:trade_id>', methods=['POST'])
def close_trade(trade_id):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if not check_db_connection() or not conn:
        return "فشل الاتصال بقاعدة البيانات", 500
    
    try:
        # جلب بيانات الصفقة
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, symbol, entry_price, current_target, entry_time, executed_quantity
                FROM signals 
                WHERE id = %s
            """, (trade_id,))
            trade = cur.fetchone()
            if not trade:
                return "الصفقة غير موجودة", 404
            
            symbol = trade['symbol']
            current_price = ticker_data.get(symbol)
            if not current_price:
                return "لا يوجد سعر حالي لهذا الزوج", 400
            
            # حساب نسبة الربح/الخسارة
            entry_price = float(trade['entry_price'])
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            
            # تنفيذ أمر البيع
            quantity = float(trade['executed_quantity'])
            if quantity > 0:
                sell_order = execute_sell_order(symbol, quantity)
                if not sell_order:
                    return "فشل في تنفيذ أمر البيع", 500
            
            # تحديث الصفقة في قاعدة البيانات
            cur.execute("""
                UPDATE signals
                SET achieved_target = TRUE,
                    closing_price = %s,
                    closed_at = NOW(),
                    profit_percentage = %s,
                    sell_executed_at = NOW()
                WHERE id = %s
            """, (current_price, profit_pct, trade_id))
            
            conn.commit()
            
            # إرسال إشعار الإغلاق اليدوي
            message = (
                f"🔒 *تم الإغلاق يدويًا*\n"
                f"——————————————\n"
                f"🪙 **الزوج:** `{symbol}`\n"
                f"💰 **الربح/الخسارة:** {profit_pct:+.2f}%\n"
                f"➡️ **سعر الدخول:** ${entry_price:.8g}\n"
                f"🏁 **سعر الإغلاق:** ${current_price:.8g}\n"
                f"⏰ **وقت الإغلاق:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            send_telegram_message(CHAT_ID, message, parse_mode='Markdown')
            
        return redirect(url_for('dashboard'))
        
    except psycopg2.Error as e:
        conn.rollback()
        return f"خطأ في قاعدة البيانات: {e}", 500
    except Exception as e:
        logger.error(f"خطأ في إغلاق الصفقة يدويًا: {e}")
        return f"خطأ داخلي: {e}", 500

# ---------------------- Flask Run Function ----------------------
def run_flask() -> None:
    """Runs the Flask application."""
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 [Flask] Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', port=port, use_reloader=False)

# ---------------------- Main Loop and Check Function ----------------------
def main_loop() -> None:
    """Main loop to scan pairs and generate signals."""
    symbols_to_scan = get_crypto_symbols()
    if not symbols_to_scan:
        logger.critical("❌ [Main] لا توجد رموز صالحة تم تحميلها أو التحقق منها. لا يمكن المتابمة.")
        return

    logger.info(f"✅ [Main] تم تحميل {len(symbols_to_scan)} رمزًا صالحًا للمسح.")
    last_full_scan_time = time.time()

    while True:
        try:
            scan_start_time = time.time()
            logger.info("+" + "-"*60 + "+")
            logger.info(f"🔄 [Main] بدء دورة مسح السوق - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("+" + "-"*60 + "+")

            if not check_db_connection() or not conn:
                logger.error("❌ [Main] تخطي دورة المسح بسبب فشل اتصال قاعدة البيانات.")
                time.sleep(60)
                continue

            open_count = 0
            try:
                 with conn.cursor() as cur_check:
                    cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                    open_count = (cur_check.fetchone() or {}).get('count', 0)
            except psycopg2.Error as db_err:
                 logger.error(f"❌ [Main] خطأ في قاعدة البيانات أثناء التحقق من عدد الإشارات المفتوحة: {db_err}. تخطي الدورة.")
                 if conn: conn.rollback()
                 time.sleep(60)
                 continue

            logger.info(f"ℹ️ [Main] الإشارات المفتوحة حالياً: {open_count} / {MAX_OPEN_TRADES}")
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"⚠️ [Main] تم الوصول إلى الحد الأقصى لعدد الإشارات المفتوحة. انتظار...")
                time.sleep(get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME) * 60)
                continue

            processed_in_loop = 0
            signals_generated_in_loop = 0
            slots_available = MAX_OPEN_TRADES - open_count

            for symbol in symbols_to_scan:
                 if slots_available <= 0:
                      logger.info(f"ℹ️ [Main] تم الوصول إلى الحد الأقصى ({MAX_OPEN_TRADES}) أثناء المسح. إيقاف مسح الرموز لهذه الدورة.")
                      break

                 processed_in_loop += 1
                 logger.debug(f"🔍 [Main] مسح {symbol} ({processed_in_loop}/{len(symbols_to_scan)})...")

                 try:
                    with conn.cursor() as symbol_cur:
                        symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND achieved_target = FALSE LIMIT 1;", (symbol,))
                        if symbol_cur.fetchone():
                            continue

                    df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist is None or df_hist.empty:
                        continue

                    strategy = ScalpingTradingStrategy(symbol)
                    if strategy.ml_bundle is None:
                        logger.warning(f"⚠️ [Main] تخطي {symbol} لأن نموذج ML الخاص به لم يتم تحميله بنجاح.")
                        continue

                    df_indicators = strategy.populate_indicators(df_hist)
                    if df_indicators is None:
                        continue

                    potential_signal = strategy.generate_buy_signal(df_indicators)

                    if potential_signal:
                        logger.info(f"✨ [Main] تم العثور على إشارة محتملة لـ {symbol}! التحقق النهائي والإدراج...")
                        with conn.cursor() as final_check_cur:
                             final_check_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                             final_open_count = (final_check_cur.fetchone() or {}).get('count', 0)

                             if final_open_count < MAX_OPEN_TRADES:
                                 if insert_signal_into_db(potential_signal):
                                     send_telegram_alert(potential_signal, SCALP_TIMEFRAME)
                                     signals_generated_in_loop += 1
                                     slots_available -= 1
                                     time.sleep(2)
                                 else:
                                     logger.error(f"❌ [Main] فشل إدراج الإشارة لـ {symbol} في قاعدة البيانات.")
                             else:
                                 logger.warning(f"⚠️ [Main] تم الوصول إلى الحد الأقصى ({final_open_count}) قبل إدراج الإشارة لـ {symbol}. تم تجاهل الإشارة.")
                                 break

                 except psycopg2.Error as db_loop_err:
                      logger.error(f"❌ [Main] خطأ في قاعدة البيانات أثناء معالجة الرمز {symbol}: {db_loop_err}. الانتقال إلى التالي...")
                      if conn: conn.rollback()
                      continue
                 except Exception as symbol_proc_err:
                      logger.error(f"❌ [Main] خطأ عام في معالجة الرمز {symbol}: {symbol_proc_err}", exc_info=True)
                      continue

                 time.sleep(0.1)

            scan_duration = time.time() - scan_start_time
            logger.info(f"🏁 [Main] انتهت دورة المسح. الإشارات التي تم إنشاؤها: {signals_generated_in_loop}. مدة المسح: {scan_duration:.2f} ثانية.")
            frame_minutes = get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME)
            wait_time = max(frame_minutes * 60, 120 - scan_duration)
            logger.info(f"⏳ [Main] انتظار {wait_time:.1f} ثانية للدورة التالية...")
            time.sleep(wait_time)

        except KeyboardInterrupt:
             logger.info("🛑 [Main] تم طلب الإيقاف (KeyboardInterrupt). إيقاف التشغيل...")
             break
        except psycopg2.Error as db_main_err:
             logger.error(f"❌ [Main] خطأ فادح في قاعدة البيانات في الحلقة الرئيسية: {db_main_err}. محاولة إعادة الاتصال...")
             if conn: conn.rollback()
             time.sleep(60)
             try:
                 init_db()
             except Exception as recon_err:
                 logger.critical(f"❌ [Main] فشل إعادة الاتصال بقاعدة البيانات: {recon_err}. خروج...")
                 break
        except Exception as main_err:
            logger.error(f"❌ [Main] خطأ غير متوقع في الحلقة الرئيسية: {main_err}", exc_info=True)
            logger.info("ℹ️ [Main] انتظار 120 ثانية قبل إعادة المحاولة...")
            time.sleep(120)

def cleanup_resources() -> None:
    """Closes used resources like the database connection."""
    global conn
    logger.info("ℹ️ [Cleanup] إغلاق الموارد...")
    if conn:
        try:
            conn.close()
            logger.info("✅ [DB] تم إغلاق اتصال قاعدة البيانات.")
        except Exception as close_err:
            logger.error(f"⚠️ [DB] خطأ في إغلاق اتصال قاعدة البيانات: {close_err}")
    logger.info("✅ [Cleanup] اكتمل تنظيف الموارد.")


# ---------------------- Main Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء بوت إشارات التداول...")
    logger.info(f"الوقت المحلي: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | وقت UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    ws_thread: Optional[Thread] = None
    tracker_thread: Optional[Thread] = None
    flask_thread: Optional[Thread] = None
    main_bot_thread: Optional[Thread] = None
    scalp_thread: Optional[Thread] = None

    # بدء Flask أولاً
    flask_thread = Thread(target=run_flask, daemon=False, name="FlaskThread")
    flask_thread.start()
    logger.info("✅ [Main] تم بدء خادم Flask.")
    
    # انتظار 3 ثوان للتأكد من فتح البورت
    logger.info("⏳ [Main] انتظار 3 ثوانٍ لبدء خادم Flask...")
    time.sleep(3)

    try:
        init_db()

        ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
        ws_thread.start()
        logger.info("✅ [Main] تم بدء مؤشر WebSocket.")
        logger.info("ℹ️ [Main] انتظار 5 ثوانٍ لتهيئة WebSocket...")
        time.sleep(5)
        if not ticker_data:
             logger.warning("⚠️ [Main] لم يتم استلام بيانات أولية من WebSocket بعد 5 ثوانٍ.")
        else:
             logger.info(f"✅ [Main] تم استلام بيانات أولية من WebSocket لـ {len(ticker_data)} رمزًا.")

        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main] تم بدء مؤشر الإشارة.")

        main_bot_thread = Thread(target=main_loop, daemon=True, name="MainBotLoopThread")
        main_bot_thread.start()
        logger.info("✅ [Main] تم بدء حلقة البوت الرئيسية في خيط منفصل.")

        # بدء خدمة التداول السريع إذا كانت مفعلة
        if REAL_TIME_SCALPING_ENABLED:
            try:
                scalp_service = RealTimeScalpingService()
                scalp_thread = Thread(target=scalp_service.run, daemon=True, name="ScalpingService")
                scalp_thread.start()
                logger.info("✅ [Main] تم بدء خدمة التداول السريع")
            except Exception as e:
                logger.error(f"❌ [Main] فشل بدء خدمة التداول السريع: {e}")

        flask_thread.join()

    except Exception as startup_err:
        logger.critical(f"❌ [Main] حدث خطأ فادح أثناء بدء التشغيل أو في الحلقة الرئيسية: {startup_err}", exc_info=True)
    finally:
        logger.info("🛑 [Main] يتم إيقاف تشغيل البرنامج...")
        cleanup_resources()
        logger.info("👋 [Main] تم إيقاف بوت إشارات التداول.")
        os._exit(0)
