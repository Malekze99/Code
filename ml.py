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
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from binance.client import Client
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from flask import Flask
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
from imblearn.over_sampling import SMOTE
import unittest

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer_v6.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer_V6')

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
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V6'
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
        logger.critical(f"❌ [DB] فشل الاتصال بقاعدة البيانات: {e}"); exit(1)

def get_binance_client():
    global client
    try:
        client = Client(API_KEY, API_SECRET)
        client.ping()
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [Binance] فشل تهيئة عميل Binance: {e}"); exit(1)

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
        info = client.get_exchange_info()
        active = {s['symbol'] for s in info['symbols'] if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [Validation] تم العثور على {len(validated)} عملة صالحة للتداول.")
        return validated
    except FileNotFoundError:
        logger.error(f"❌ [Validation] ملف قائمة العملات '{filename}' غير موجود.")
        return []
    except Exception as e:
        logger.error(f"❌ [Validation] خطأ في التحقق من الرموز: {e}"); return []

# --- دوال جلب ومعالجة البيانات (مع التحسينات) ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_historical_data_retryable(symbol: str, interval: str, days: int) -> list:
    start_str = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    return client.get_historical_klines(symbol, interval, start_str)

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
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات لـ {symbol}: {e}"); return None

def fetch_and_cache_btc_data():
    global btc_data_cache
    logger.info("ℹ️ [BTC Data] جاري جلب بيانات البيتكوين وتخزينها...")
    btc_data_cache = cached_fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if btc_data_cache is None:
        logger.critical("❌ [BTC Data] فشل جلب بيانات البيتكوين."); exit(1)
    btc_data_cache['btc_returns'] = btc_data_cache['close'].pct_change().shift(1)  # منع تسرب البيانات

# طريقة Triple-Barrier المتجهة
def vectorized_triple_barrier_labels(prices: pd.Series, atr: pd.Series) -> pd.Series:
    labels = pd.Series(0, index=prices.index)
    max_hold = MAX_HOLD_PERIOD
    
    # إنشاء حواجز ديناميكية
    upper_barriers = prices + (atr * TP_ATR_MULTIPLIER)
    lower_barriers = prices - (atr * SL_ATR_MULTIPLIER)
    
    # إنشاء مصفوفة أسعار مستقبلية
    price_matrix = pd.DataFrame({f't+{i}': prices.shift(-i) for i in range(1, max_hold+1)})
    
    # تحديد نقاط الاختراق
    upper_hits = (price_matrix > upper_barriers.values.reshape(-1, 1)).any(axis=1)
    lower_hits = (price_matrix < lower_barriers.values.reshape(-1, 1)).any(axis=1)
    
    # تعيين التسميات
    labels[upper_hits] = 1
    labels[lower_hits] = -1
    return labels

# حساب المؤشرات الفنية مع الميزات الجديدة
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

    # MACD and MACD Cross
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = macd_line - signal_line
    df_calc['macd_cross'] = 0
    df_calc.loc[(df_calc['macd_hist'].shift(1) < 0) & (df_calc['macd_hist'] >= 0), 'macd_cross'] = 1
    df_calc.loc[(df_calc['macd_hist'].shift(1) > 0) & (df_calc['macd_hist'] <= 0), 'macd_cross'] = -1

    # Bollinger Bands Width
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

    # Overbought/Oversold Filter
    df_calc['market_condition'] = 0  # 0 for Neutral
    df_calc.loc[(df_calc['rsi'] > RSI_OVERBOUGHT) | (df_calc['stoch_rsi_k'] > STOCH_RSI_OVERBOUGHT), 'market_condition'] = 1  # 1 for Overbought
    df_calc.loc[(df_calc['rsi'] < RSI_OVERSOLD) | (df_calc['stoch_rsi_k'] < STOCH_RSI_OVERSOLD), 'market_condition'] = -1  # -1 for Oversold

    # VWAP (ميزة جديدة)
    df_calc['typical_price'] = (df_calc['high'] + df_calc['low'] + df_calc['close']) / 3
    df_calc['vwap'] = (df_calc['typical_price'] * df_calc['volume']).cumsum() / df_calc['volume'].cumsum()

    # Volume Spike (ميزة جديدة)
    median_vol = df_calc['volume'].rolling(window=50).median()
    df_calc['volume_spike'] = df_calc['volume'] / (median_vol + 1e-9)

    # Other existing features
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, method='fft').mean()  # تحسين الأداء
    ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, method='fft').mean()  # تحسين الأداء
    df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
    df_calc['returns'] = df_calc['close'].pct_change()
    
    # BTC Correlation مع منع تسرب البيانات
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

# تقييم متقدم للنموذج
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

def prepare_data_for_ml(df: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep] Preparing data for {symbol}...")
    df_featured = calculate_features(df, btc_df)
    df_featured['target'] = vectorized_triple_barrier_labels(df_featured['close'], df_featured['atr'])
    
    feature_columns = [
        'rsi', 'macd_hist', 'atr', 'relative_volume', 'hour_of_day',
        'price_vs_ema50', 'price_vs_ema200', 'btc_correlation',
        'stoch_rsi_k', 'stoch_rsi_d', 'macd_cross', 'market_condition',
        'bb_width', 'vwap', 'volume_spike'  # إضافة الميزات الجديدة
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
        logger.error(f"❌ [DB Save] Error saving model bundle: {e}"); conn.rollback()

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

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def train_symbol(symbol: str) -> Tuple[str, bool]:
    """تدريب نموذج لعملة معينة وإرجاع حالة النجاح"""
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

def run_training_job():
    logger.info(f"🚀 Starting ADVANCED ML model training job ({BASE_ML_MODEL_NAME})...")
    init_db()
    get_binance_client()
    fetch_and_cache_btc_data()
    symbols_to_train = get_validated_symbols(filename='crypto_list.txt')
    
    if not symbols_to_train:
        logger.critical("❌ [Main] No valid symbols found. Exiting.")
        return
        
    send_telegram_message(f"🚀 *{BASE_ML_MODEL_NAME} Training Started*\nWill train models for {len(symbols_to_train)} symbols.")
    
    # المعالجة المتوازية
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        results = list(tqdm(
            executor.map(train_symbol, symbols_to_train),
            total=len(symbols_to_train),
            desc="Training Models"
        ))
    
    # حساب النتائج
    successful_models = sum(1 for _, success in results if success)
    failed_models = len(symbols_to_train) - successful_models
    
    # تنظيف النماذج القديمة
    clean_old_models(keep_last=3)
    
    # إرسال تقرير النتائج
    completion_message = (f"✅ *{BASE_ML_MODEL_NAME} Training Finished*\n"
                        f"- Successfully trained: {successful_models} models\n"
                        f"- Failed/Discarded: {failed_models} models\n"
                        f"- Total symbols: {len(symbols_to_train)}")
    send_telegram_message(completion_message)
    logger.info(completion_message)

    if conn: conn.close()
    logger.info("👋 [Main] ML training job finished.")

# نظام التراجع (للتنفيذ في نظام التداول)
def get_fallback_strategy(symbol: str) -> int:
    """إستراتيجية احتياطية تعتمد على تقاطع المتوسطات"""
    df = cached_fetch_historical_data(symbol, '15m', 3)
    if df is None or df.empty: 
        return 0
    
    ema_short = df['close'].ewm(span=12, method='fft').mean()
    ema_long = df['close'].ewm(span=26, method='fft').mean()
    return 1 if ema_short.iloc[-1] > ema_long.iloc[-1] else -1

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
    return "ML Trainer V6 service is running and healthy.", 200

if __name__ == "__main__":
    # تشغيل الاختبارات إذا تم الطلب
    if os.getenv("RUN_TESTS"):
        unittest.main(argv=[''], exit=False)
    
    # بدء عملية التدريب في خيط منفصل
    training_thread = Thread(target=run_training_job)
    training_thread.daemon = True
    training_thread.start()
    
    # بدء خادم ويب للحفاظ على الخدمة نشطة
    port = int(os.environ.get("PORT", 10001))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)
