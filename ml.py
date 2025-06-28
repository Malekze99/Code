import time
import os
import json
import logging
import requests
import re
import math
import threading
import random
import numpy as np
import pandas as pd
import psycopg2
import pickle
import lightgbm as lgb
from collections import deque
from psycopg2.extras import RealDictCursor
from binance.client import Client
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from flask import Flask, jsonify
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from imblearn.over_sampling import SMOTE
import unittest
from requests.exceptions import RequestException

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer_final.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer_Final')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    TELEGRAM_TOKEN: Optional[str] = config('TELEGRAM_BOT_TOKEN', default=None)
    CHAT_ID: Optional[str] = config('TELEGRAM_CHAT_ID', default=None)
    EMERGENCY_MODE: bool = config('EMERGENCY_MODE', default=False, cast=bool)
    AUTO_RESOLVE_BAN: bool = config('AUTO_RESOLVE_BAN', default=False, cast=bool)
    PROXY_URL: Optional[str] = config('PROXY_URL', default=None)
    SERVER_ID: int = config('SERVER_ID', default=0, cast=int)
    TOTAL_SERVERS: int = config('TOTAL_SERVERS', default=1, cast=int)
    # إضافة متغيرات جديدة للتحكم في معدل الطلبات
    MAX_REQUESTS_PER_MINUTE: int = config('MAX_REQUESTS_PER_MINUTE', default=1000, cast=int)
    SAFETY_MARGIN: float = config('SAFETY_MARGIN', default=0.2, cast=float)
    MAX_CONSECUTIVE_WAITS: int = config('MAX_CONSECUTIVE_WAITS', default=5, cast=int)
    PROXY_POOL: List[str] = config('PROXY_POOL', default='', cast=lambda v: [s.strip() for s in v.split(',') if s.strip()])
except Exception as e:
     logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
     exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_Final'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 120
BTC_SYMBOL = 'BTCUSDT'

# --- Indicator & Feature Parameters ---
BBANDS_PERIOD: int = 20
RSI_PERIOD: int = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_CORR_PERIOD: int = 30
STOCH_RSI_PERIOD: int = 14
STOCH_K: int = 3
STOCH_D: int = 3
REL_VOL_PERIOD: int = 30
RSI_OVERBOUGHT: int = 70
RSI_OVERSOLD: int = 30
STOCH_RSI_OVERBOUGHT: int = 80
STOCH_RSI_OVERSOLD: int = 20

# Triple-Barrier Method Parameters
TP_ATR_MULTIPLIER: float = 2.0
SL_ATR_MULTIPLIER: float = 1.5
MAX_HOLD_PERIOD: int = 24

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
btc_data_cache: Optional[pd.DataFrame] = None
consecutive_blocks: int = 0  # تتبع الحظر المتتالي
last_block_time: Optional[float] = None  # وقت آخر حظر
active_proxy: Optional[str] = None  # البروكسي النشط حالياً

# ====================== نظام إدارة معدل الطلبات المتقدم ======================
class TokenBucket:
    """نظام إدارة معدل الطلبات باستخدام دلو الرموز"""
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # رموز في الثانية
        self.last_refill = time.time()

    def refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, tokens):
        self.refill()
        if tokens <= self.tokens:
            self.tokens -= tokens
            return True
        return False

class APIShieldSystem:
    """نظام حماية متقدم لإدارة طلبات API وتجنب الحظر"""
    def __init__(self):
        self.rate_limiter = TokenBucket(
            capacity=MAX_REQUESTS_PER_MINUTE,
            refill_rate=MAX_REQUESTS_PER_MINUTE / 60.0
        )
        self.consecutive_blocks = 0
        self.last_block_time = None
        self.request_history = deque(maxlen=MAX_REQUESTS_PER_MINUTE * 2)
        self.safety_margin = SAFETY_MARGIN
        self.proxy_pool = PROXY_POOL if PROXY_POOL else [None]
        self.current_proxy_idx = 0
        self.blacklisted_proxies = {}
        self.lock = threading.Lock()
    
    def get_next_proxy(self):
        """الحصول على الـ IP التالي للتناوب"""
        with self.lock:
            while True:
                self.current_proxy_idx = (self.current_proxy_idx + 1) % len(self.proxy_pool)
                proxy = self.proxy_pool[self.current_proxy_idx]
                
                # التحقق من انتهاء مدة الحظر
                if proxy in self.blacklisted_proxies:
                    if time.time() > self.blacklisted_proxies[proxy]:
                        del self.blacklisted_proxies[proxy]
                    else:
                        continue
                return proxy
    
    def report_blocked(self, proxy, block_duration):
        """تسجيل IP محظور مع مدة الحظر"""
        with self.lock:
            self.blacklisted_proxies[proxy] = time.time() + block_duration + 300  # +5 دقائق هامش أمان
            self.consecutive_blocks += 1
            self.last_block_time = time.time()
            global consecutive_blocks, last_block_time
            consecutive_blocks = self.consecutive_blocks
            last_block_time = self.last_block_time
    
    def calculate_wait_time(self, required_tokens):
        """حساب زمن الانتظار المطلوب بناءً على الطلبات السابقة"""
        available_tokens = self.rate_limiter.tokens - (self.rate_limiter.capacity * self.safety_margin)
        
        if required_tokens <= available_tokens:
            return 0
        
        deficit = required_tokens - available_tokens
        return deficit / self.rate_limiter.refill_rate
    
    def safe_request(self, func, *args, **kwargs):
        """إجراء طلب آمن مع مراعاة معدل الطلبات"""
        # التحقق من حالة الحظر المتتالي
        if self.consecutive_blocks >= 2:
            self.activate_smart_recovery()
            
        # حساب وزن الطلب
        weight = self.calculate_request_weight(func.__name__)
        
        # الانتظار الذكي إذا لزم الأمر
        wait_time = self.calculate_wait_time(weight)
        if wait_time > 0:
            logger.warning(f"⏳ [RateLimiter] الانتظار الذكي: {wait_time:.2f} ثانية")
            time.sleep(wait_time)
        
        # إجراء الطلب مع تناوب الـ IPs
        proxy = self.get_next_proxy()
        try:
            if proxy:
                kwargs["proxies"] = {"https": proxy}
            
            result = func(*args, **kwargs)
            
            # تحديث حالة معدل الطلبات
            self.rate_limiter.consume(weight)
            self.request_history.append((time.time(), weight))
            
            return result
        except Exception as e:
            error_str = str(e)
            if 'code=-1003' in error_str:
                ban_time = self.extract_ban_time(error_str)
                current_time = time.time() * 1000
                
                if ban_time > current_time:
                    wait_seconds = (ban_time - current_time) / 1000 + 10
                    logger.warning(f"⚠️ [Binance] IP محظور حتى {datetime.utcfromtimestamp(ban_time/1000)}. إعادة المحاولة بعد {wait_seconds:.1f} ثانية...")
                    self.report_blocked(proxy, wait_seconds)
                    time.sleep(wait_seconds)
                else:
                    logger.warning("⚠️ [Binance] تم رصد حظر منتهي. إعادة المحاولة فوراً.")
                    time.sleep(10)
            raise e
    
    def calculate_request_weight(self, endpoint):
        """حساب وزن الطلب بناءً على نوعه"""
        if 'klines' in endpoint:
            return 5
        elif 'exchange_info' in endpoint:
            return 10
        return 1
    
    def extract_ban_time(self, error_msg: str) -> float:
        """استخراج وقت انتهاء الحظر من رسالة الخطأ"""
        try:
            match = re.search(r'until (\d+)', error_msg)
            if match:
                return float(match.group(1))
        except:
            return (time.time() + 300) * 1000
        return (time.time() + 300) * 1000
    
    def activate_smart_recovery(self):
        """تفعيل نظام الاسترداد الذكي بعد حظر متتالي"""
        logger.warning("🛡️ تفعيل بروتوكول الاسترداد الذكي")
        
        # الانتظار التصاعدي
        base_wait = min(3600, 300 * (2 ** self.consecutive_blocks))
        randomized_wait = base_wait + random.randint(0, 600)
        logger.info(f"⏳ الاسترداد الذكي: الانتظار لمدة {randomized_wait//60} دقيقة")
        time.sleep(randomized_wait)
        
        # إعادة التعيين بعد الانتظار
        self.consecutive_blocks = 0
        global consecutive_blocks
        consecutive_blocks = 0

# تهيئة نظام الحماية
api_shield = APIShieldSystem()

# --- دوال الاتصال والتحقق ---
def init_db():
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY, model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL, trained_at TIMESTAMP DEFAULT NOW(), metrics JSONB );
                CREATE TABLE IF NOT EXISTS model_performance (
                    id SERIAL PRIMARY KEY,
                    model_id INT REFERENCES ml_models(id),
                    timestamp TIMESTAMP DEFAULT NOW(),
                    precision REAL,
                    recall REAL,
                    trade_outcome SMALLINT
                );
            """)
        conn.commit()
        logger.info("✅ [DB] تم تهيئة قاعدة البيانات بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [DB] فشل الاتصال بقاعدة البيانات: {e}")
        exit(1)

def get_binance_client():
    global client
    max_retries = 5
    
    for attempt in range(1, max_retries + 1):
        try:
            # إغلاق الاتصال السابق إذا كان موجوداً
            if client:
                try:
                    client.close_connection()
                except Exception as e:
                    logger.warning(f"⚠️ [Binance] Warning closing previous connection: {e}")
                client = None
            
            # إنشاء عميل جديد
            client_params = {}
            active_proxy = api_shield.get_next_proxy()
            if active_proxy:
                client_params["proxies"] = {"https": active_proxy}
            
            client = Client(API_KEY, API_SECRET, **client_params)
            client.ping()
            logger.info("✅ [Binance] تم الاتصال بنجاح.")
            return
        except Exception as e:
            logger.error(f"❌ [Binance] خطأ في المحاولة {attempt}/{max_retries}: {e}")
            time.sleep(10)
                
    logger.critical("❌ [Binance] فشل جميع محاولات الاتصال.")
    if AUTO_RESOLVE_BAN:
        rotate_ip_address()
    exit(1)

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client:
        logger.error("❌ [Validation] عميل Binance لم يتم تهيئته.")
        return []
    try:
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            symbols = {s.strip().upper() for s in f if s.strip() and not s.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in symbols}
        
        # استخدام نظام الحماية للطلب
        info = api_shield.safe_request(client.get_exchange_info)
        
        active = {s['symbol'] for s in info['symbols'] if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT'}
        validated = sorted(list(formatted.intersection(active)))
        
        if TOTAL_SERVERS > 1:
            validated = [s for i, s in enumerate(validated) if i % TOTAL_SERVERS == SERVER_ID]
        
        logger.info(f"✅ [Validation] تم العثور على {len(validated)} عملة صالحة للتداول.")
        return validated
    except FileNotFoundError:
        logger.error(f"❌ [Validation] ملف قائمة العملات '{filename}' غير موجود.")
        return []
    except Exception as e:
        logger.error(f"❌ [Validation] خطأ في التحقق من الرموز: {e}")
        return []

# --- دوال جلب ومعالجة البيانات ---
@retry(stop=stop_after_attempt(3), 
       wait=wait_exponential(multiplier=1, min=4, max=10),
       retry=retry_if_exception_type(RequestException))
def fetch_historical_data_retryable(symbol: str, interval: str, days: int) -> list:
    try:
        start_str = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        # استخدام نظام الحماية للطلب
        klines = api_shield.safe_request(
            client.get_historical_klines,
            symbol, 
            interval,
            start_str
        )
        return klines
    except Exception as e:
        raise e

@lru_cache(maxsize=32)
def cached_fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    try:
        klines = fetch_historical_data_retryable(symbol, interval, days)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[numeric_cols].dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات لـ {symbol}: {e}")
        return None

def fetch_and_cache_btc_data():
    global btc_data_cache
    logger.info("ℹ️ [BTC Data] جاري جلب بيانات البيتكوين وتخزينها...")
    btc_data_cache = cached_fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if btc_data_cache is None:
        logger.critical("❌ [BTC Data] فشل جلب بيانات البيتكوين.")
        exit(1)
    btc_data_cache['btc_returns'] = btc_data_cache['close'].pct_change().shift(1)

def vectorized_triple_barrier_labels(prices: pd.Series, atr: pd.Series) -> pd.Series:
    labels = pd.Series(0, index=prices.index)
    max_hold = MAX_HOLD_PERIOD
    
    upper_barriers = prices + (atr * TP_ATR_MULTIPLIER)
    lower_barriers = prices - (atr * SL_ATR_MULTIPLIER)
    
    price_matrix = pd.DataFrame({f't+{i}': prices.shift(-i) for i in range(1, max_hold+1)})
    
    upper_hits = (price_matrix > upper_barriers.values.reshape(-1, 1)).any(axis=1)
    lower_hits = (price_matrix < lower_barriers.values.reshape(-1, 1)).any(axis=1)
    
    labels[upper_hits] = 1
    labels[lower_hits] = -1
    return labels

def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()

    # ATR
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # RSI
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))

    # MACD
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = macd_line - signal_line
    df_calc['macd_cross'] = 0
    df_calc.loc[(df_calc['macd_hist'].shift(1) < 0) & (df_calc['macd_hist'] >= 0), 'macd_cross'] = 1
    df_calc.loc[(df_calc['macd_hist'].shift(1) > 0) & (df_calc['macd_hist'] <= 0), 'macd_cross'] = -1

    # Bollinger Bands
    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean()
    std_dev = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    df_calc['bb_width'] = (upper_band - lower_band) / (sma + 1e-9)

    # Stochastic RSI
    rsi = df_calc['rsi']
    min_rsi = rsi.rolling(window=STOCH_RSI_PERIOD).min()
    max_rsi = rsi.rolling(window=STOCH_RSI_PERIOD).max()
    stoch_rsi_val = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(window=STOCH_K).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(window=STOCH_D).mean()

    # Relative Volume
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)

    # Market Condition
    df_calc['market_condition'] = 0
    df_calc.loc[(df_calc['rsi'] > RSI_OVERBOUGHT) | (df_calc['stoch_rsi_k'] > STOCH_RSI_OVERBOUGHT), 'market_condition'] = 1
    df_calc.loc[(df_calc['rsi'] < RSI_OVERSOLD) | (df_calc['stoch_rsi_k'] < STOCH_RSI_OVERSOLD), 'market_condition'] = -1

    # VWAP
    df_calc['typical_price'] = (df_calc['high'] + df_calc['low'] + df_calc['close']) / 3
    df_calc['vwap'] = (df_calc['typical_price'] * df_calc['volume']).cumsum() / df_calc['volume'].cumsum()

    # Volume Spike
    median_vol = df_calc['volume'].rolling(window=50).median()
    df_calc['volume_spike'] = df_calc['volume'] / (median_vol + 1e-9)

    # EMA Trends
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, method='fft').mean()
    ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, method='fft').mean()
    df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
    df_calc['returns'] = df_calc['close'].pct_change()
    
    # BTC Correlation
    merged_df = pd.merge(
        df_calc, 
        btc_df[['btc_returns']], 
        left_index=True, 
        right_index=True, 
        how='left'
    ).fillna(0)
    merged_df['btc_returns_shifted'] = merged_df['btc_returns'].shift(1)
    df_calc['btc_correlation'] = (
        merged_df['returns']
        .rolling(window=BTC_CORR_PERIOD)
        .corr(merged_df['btc_returns_shifted'])
    )
    
    df_calc['hour_of_day'] = df_calc.index.hour
    
    return df_calc

def prepare_data_for_ml(df: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep] Preparing data for {symbol}...")
    df_featured = calculate_features(df, btc_df)
    df_featured['target'] = vectorized_triple_barrier_labels(df_featured['close'], df_featured['atr'])
    
    feature_columns = [
        'rsi', 'macd_hist', 'atr', 'relative_volume', 'hour_of_day',
        'price_vs_ema50', 'price_vs_ema200', 'btc_correlation',
        'stoch_rsi_k', 'stoch_rsi_d', 'macd_cross', 'market_condition',
        'bb_width', 'vwap', 'volume_spike'
    ]
    
    df_cleaned = df_featured.dropna(subset=feature_columns + ['target']).copy()
    if df_cleaned.empty or df_cleaned['target'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] Data for {symbol} has less than 2 classes. Skipping.")
        return None
    
    logger.info(f"📊 [ML Prep] Target distribution for {symbol}:\n{df_cleaned['target'].value_counts(normalize=True)}")
    X = df_cleaned[feature_columns]
    y = df_cleaned['target']
    
    # موازنة الفئات باستخدام SMOTE
    if y.nunique() > 1:
        smote = SMOTE(sampling_strategy={1: 1000, -1: 1000}, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res, feature_columns
    
    return X, y, feature_columns

def train_with_walk_forward_validation(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    logger.info("ℹ️ [ML Train] Starting training with Walk-Forward Validation...")
    tscv = TimeSeriesSplit(n_splits=10)
    final_model, final_scaler = None, None

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
        
        model = lgb.LGBMClassifier(
            objective='multiclass', num_class=3, random_state=42, n_estimators=300,
            learning_rate=0.05, class_weight='balanced', n_jobs=-1)
        
        model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)],
                  eval_metric='multi_logloss', callbacks=[lgb.early_stopping(30, verbose=False)])
        
        y_pred = model.predict(X_test_scaled)
        metrics = evaluate_model(y_test, y_pred)
        logger.info(f"--- Fold {i+1}: Accuracy: {metrics['accuracy']:.4f}, "
                    f"F1(1): {metrics['f1_class_1']:.4f}, "
                    f"F1(-1): {metrics['f1_class_-1']:.4f}, "
                    f"MCC: {metrics['mcc']:.4f}")
        
        final_model, final_scaler = model, scaler

    if not final_model or not final_scaler:
        logger.error("❌ [ML Train] Training failed, no model was created.")
        return None, None, None

    all_preds = []
    all_true = []
    for _, test_index in tscv.split(X):
        X_test_final = X.iloc[test_index]
        y_test_final = y.iloc[test_index]
        X_test_final_scaled = pd.DataFrame(final_scaler.transform(X_test_final), columns=X.columns, index=X_test_final.index)
        all_preds.extend(final_model.predict(X_test_final_scaled))
        all_true.extend(y_test_final)

    model_metrics = evaluate_model(all_true, all_preds)
    metrics_log_str = ', '.join([f"{k}: {v:.4f}" for k, v in model_metrics.items()])
    logger.info(f"📊 [ML Train] Average Walk-Forward Performance: {metrics_log_str}")
    return final_model, final_scaler, model_metrics

def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_class_1': f1_score(y_true, y_pred, labels=[1], average='macro'),
        'f1_class_-1': f1_score(y_true, y_pred, labels=[-1], average='macro'),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'precision_class_1': report['1']['precision'],
        'recall_class_1': report['1']['recall'],
        'num_samples_trained': len(y_true),
    }

def save_ml_model_to_db(model_bundle: Dict[str, Any], model_name: str, metrics: Dict[str, Any]):
    logger.info(f"ℹ️ [DB Save] Saving model bundle '{model_name}'...")
    try:
        model_binary = pickle.dumps(model_bundle)
        metrics_json = json.dumps(metrics)
        with conn.cursor() as db_cur:
            db_cur.execute("""
                INSERT INTO ml_models (model_name, model_data, trained_at, metrics) 
                VALUES (%s, %s, NOW(), %s) ON CONFLICT (model_name) DO UPDATE SET 
                model_data = EXCLUDED.model_data, trained_at = NOW(), metrics = EXCLUDED.metrics;
            """, (model_name, model_binary, metrics_json))
        conn.commit()
        logger.info(f"✅ [DB Save] Model bundle '{model_name}' saved successfully.")
    except Exception as e:
        logger.error(f"❌ [DB Save] Error saving model bundle: {e}")
        conn.rollback()

def clean_old_models(keep_last: int = 3):
    logger.info(f"ℹ️ [DB Clean] Cleaning old models, keeping last {keep_last} versions...")
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM ml_models 
                WHERE id NOT IN (
                    SELECT id FROM ml_models 
                    ORDER BY trained_at DESC 
                    LIMIT %s
                )
            """, (keep_last,))
        conn.commit()
        logger.info(f"✅ [DB Clean] Old models cleaned successfully.")
    except Exception as e:
        logger.error(f"❌ [DB Clean] Error cleaning old models: {e}")

# ====================== دوال التدريب الرئيسية ======================
def train_symbol(symbol: str) -> Tuple[str, bool]:
    logger.info(f"\n--- ⏳ [Training] Starting model training for {symbol} ---")
    try:
        df_hist = cached_fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
        if df_hist is None or df_hist.empty:
            logger.warning(f"⚠️ [Training] No data for {symbol}, skipping.")
            return symbol, False
        
        prepared_data = prepare_data_for_ml(df_hist, btc_data_cache, symbol)
        if prepared_data is None:
            return symbol, False
            
        X, y, feature_names = prepared_data
        training_result = train_with_walk_forward_validation(X, y)
        
        if not all(training_result):
            return symbol, False
            
        final_model, final_scaler, model_metrics = training_result
        
        # معايير الحفظ المحسنة
        if (final_model and final_scaler and 
            model_metrics.get('f1_class_1', 0) > 0.45 and 
            model_metrics.get('mcc', 0) > 0.2):
            
            model_bundle = {'model': final_model, 'scaler': final_scaler, 'feature_names': feature_names}
            model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
            save_ml_model_to_db(model_bundle, model_name, model_metrics)
            return symbol, True
        else:
            logger.warning(f"⚠️ [Training] Model for {symbol} doesn't meet quality standards. Discarding.")
            return symbol, False
            
    except Exception as e:
        logger.critical(f"❌ [Training] A fatal error occurred for {symbol}: {e}", exc_info=True)
        return symbol, False

def train_batch(batch: List[str]) -> Tuple[int, int]:
    """تدريب دفعة من الرموز مع التحكم الديناميكي"""
    batch_success = 0
    batch_fail = 0
    
    # حساب العدد الأمثل للخيوط
    max_workers = min(4, len(batch))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # استخدام قائمة بدلاً من قاموس لتجنب مشاكل التعريف
        futures = []
        for symbol in batch:
            futures.append(executor.submit(train_symbol, symbol))
        
        for future in futures:
            try:
                symbol, success = future.result()
                if success:
                    batch_success += 1
                else:
                    batch_fail += 1
            except Exception as e:
                logger.error(f"❌ [Training Batch] خطأ في تدريب الرمز: {e}")
                batch_fail += 1
    
    return batch_success, batch_fail

def dynamic_batch_size(total_symbols: int, current_usage: float) -> int:
    """تحديد حجم الدفعة بشكل ديناميكي بناءً على حمل النظام"""
    max_batch = 20
    min_batch = 5
    
    if current_usage > 0.8 * MAX_REQUESTS_PER_MINUTE:
        return max(min_batch, int(total_symbols * 0.1))
    elif current_usage > 0.6 * MAX_REQUESTS_PER_MINUTE:
        return max(min_batch, int(total_symbols * 0.2))
    else:
        return min(max_batch, int(total_symbols * 0.3))

def run_training_job():
    if EMERGENCY_MODE:
        logger.critical("🆘 تم تفعيل وضع الطوارئ - استخدام البيانات المخزنة مسبقاً")
        send_telegram_message("🆘 *وضع الطوارئ*: تم تفعيل وضع الطوارئ. سيتم استخدام البيانات المخزنة فقط.")
        return
    
    logger.info(f"🚀 بدء تدريب النموذج ({BASE_ML_MODEL_NAME})...")
    init_db()
    
    try:
        get_binance_client()
        fetch_and_cache_btc_data()
    except Exception as e:
        logger.critical(f"❌ فشل التهيئة: {e}")
        send_telegram_message("⛔ *فشل حرج*: تعذر الاتصال بـ Binance API")
        return

    symbols_to_train = get_validated_symbols(filename='crypto_list.txt')
    
    if not symbols_to_train:
        logger.critical("❌ [Main] No valid symbols found. Exiting.")
        return
        
    send_telegram_message(f"🚀 *{BASE_ML_MODEL_NAME} Training Started*\nWill train models for {len(symbols_to_train)} symbols.")
    
    successful_models = 0
    failed_models = 0
    batch_idx = 0
    total_symbols = len(symbols_to_train)
    
    while symbols_to_train:
        # حساب حمل الطلبات الحالي
        current_usage = len(api_shield.request_history)
        usage_ratio = current_usage / MAX_REQUESTS_PER_MINUTE
        
        # تحديد حجم الدفعة ديناميكياً
        batch_size = dynamic_batch_size(len(symbols_to_train), usage_ratio)
        batch = symbols_to_train[:batch_size]
        symbols_to_train = symbols_to_train[batch_size:]
        
        logger.info(f"🔁 معالجة الدفعة {batch_idx+1}: {len(batch)} عملات (الحمل الحالي: {usage_ratio:.1%})")
        
        # تدريب الدفعة
        batch_success, batch_fail = train_batch(batch)
        successful_models += batch_success
        failed_models += batch_fail
        
        logger.info(f"✅ الدفعة {batch_idx+1} النتائج: نجاح ({batch_success}), فشل ({batch_fail})")
        batch_idx += 1
        
        # تأخير بين الدفعات
        if symbols_to_train:
            # حساب وقت الانتظار الديناميكي
            batch_delay = max(30, min(300, 60 * usage_ratio))
            logger.info(f"⏸️ استراحة {batch_delay} ثانية بين الدفعات...")
            time.sleep(batch_delay)
        
        # التحقق من الحظر المتتالي
        if consecutive_blocks >= MAX_CONSECUTIVE_WAITS:
            logger.critical(f"🛑 تخطي الحد الأقصى لعمليات الانتظار المتتالية ({consecutive_blocks})!")
            send_telegram_message(f"🛑 *إنذار نظام*: تخطي الحد الأقصى لعمليات الانتظار المتتالية ({consecutive_blocks})")
            api_shield.activate_smart_recovery()
    
    # تنظيف النماذج القديمة
    clean_old_models(keep_last=3)
    
    # إرسال تقرير النتائج
    completion_message = (f"✅ *{BASE_ML_MODEL_NAME} Training Finished*\n"
                        f"- Successfully trained: {successful_models} models\n"
                        f"- Failed/Discarded: {failed_models} models\n"
                        f"- Total symbols: {total_symbols}")
    send_telegram_message(completion_message)
    logger.info(completion_message)

    if conn: 
        conn.close()
    logger.info("👋 [Main] ML training job finished.")

# ====================== دوال إضافية للتحكم في النظام ======================
def rotate_ip_address():
    """دوران عنوان IP (يعتمد على البيئة)"""
    logger.warning("🔄 محاولة تغيير عنوان IP...")
    send_telegram_message("🔄 *دوران IP*: تم تشغيل آلية تغيير IP")
    # هنا يمكن تنفيذ أمر تغيير IP حسب البيئة
    # مثال: requests.get('http://ip-rotation-service/rotate')
    
    # تحديث البروكسي في نظام الحماية
    global active_proxy
    active_proxy = api_shield.get_next_proxy()
    
    # إعادة تهيئة عميل Binance
    get_binance_client()

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: 
        requests.post(url, json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e: 
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

# اختبارات تلقائية
class TestTradingSystem(unittest.TestCase):
    def test_triple_barrier(self):
        prices = pd.Series([100, 102, 105, 103, 107, 110, 108])
        atr = pd.Series([1.0, 1.1, 1.2, 1.0, 1.3, 1.1, 1.0])
        labels = vectorized_triple_barrier_labels(prices, atr)
        self.assertEqual(labels.tolist(), [1, 0, 1, 0, 1, 0, 0])
    
    def test_feature_engineering(self):
        df = pd.DataFrame({
            'open': [100, 101, 102, 101, 103],
            'high': [105, 104, 106, 103, 107],
            'low': [95, 98, 100, 99, 101],
            'close': [102, 103, 105, 101, 106],
            'volume': [1000, 1200, 800, 1500, 900]
        }, index=pd.date_range('2023-01-01', periods=5, freq='15T'))
        
        btc_df = pd.DataFrame({'btc_returns': [0.01, -0.02, 0.03, -0.01, 0.02]}, index=df.index)
        
        featured_df = calculate_features(df, btc_df)
        self.assertIn('vwap', featured_df.columns)
        self.assertIn('volume_spike', featured_df.columns)
        self.assertAlmostEqual(featured_df['vwap'].iloc[2], 102.333, places=1)

# تطبيق Flask للحفاظ على الخدمة نشطة
app = Flask(__name__)

@app.route('/')
def health_check():
    return "ML Trainer Final service is running and healthy.", 200

@app.route('/status')
def status():
    try:
        # التحقق من اتصال قاعدة البيانات
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        
        # التحقق من اتصال Binance
        try:
            if client:
                client.ping()
                binance_status = "connected"
            else:
                binance_status = "not initialized"
        except:
            binance_status = "disconnected"
        
        # إضافة معلومات نظام الحماية
        current_usage = len(api_shield.request_history)
        usage_percent = (current_usage / MAX_REQUESTS_PER_MINUTE) * 100
        
        return jsonify({
            "status": "healthy",
            "binance": binance_status,
            "emergency_mode": EMERGENCY_MODE,
            "server_id": SERVER_ID,
            "total_servers": TOTAL_SERVERS,
            "api_requests": current_usage,
            "api_usage_percent": f"{usage_percent:.1f}%",
            "consecutive_blocks": consecutive_blocks,
            "active_proxy": active_proxy
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    # تشغيل الاختبارات إذا تم الطلب
    if os.getenv("RUN_TESTS"):
        unittest.main(argv=[''], exit=False)
    
    # بدء عملية التدريب في خيط منفصل
    training_thread = Thread(target=run_training_job)
    training_thread.daemon = True
    training_thread.start()
    
    # بدء خادم ويب للحفاظ على الخدمة نشطة
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)
