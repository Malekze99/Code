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
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from flask import Flask
from threading import Thread
from scipy.stats import entropy

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_scalp_trainer_v7.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLScalpTrainer_V7')

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

# ---------------------- إعداد ثوابت السكالب ----------------------
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalper_V7'
SIGNAL_GENERATION_TIMEFRAME: str = '5m'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 90
BTC_SYMBOL = 'BTCUSDT'

# --- معلمات السكالب ---
SCALP_TP = 0.015  # هدف ربح 1.5%
SCALP_SL = 0.007   # وقف خسارة 0.7%
SCALP_MAX_HOLD = 8 # أقصى مدة احتفاظ (8 بارات = 40 دقيقة)

# --- معالم المؤشرات ---
BBANDS_PERIOD: int = 20
RSI_PERIOD: int = 9
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 8, 17, 6
ATR_PERIOD: int = 10
EMA_SLOW_PERIOD: int = 100
EMA_FAST_PERIOD: int = 20
BTC_CORR_PERIOD: int = 20
STOCH_RSI_PERIOD: int = 10
STOCH_K: int = 3
STOCH_D: int = 3
REL_VOL_PERIOD: int = 20
RSI_OVERBOUGHT: int = 75
RSI_OVERSOLD: int = 25
STOCH_RSI_OVERBOUGHT: int = 85
STOCH_RSI_OVERSOLD: int = 15

# متغيرات عامة
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
btc_data_cache: Optional[pd.DataFrame] = None

# ---------------------- دوال المساعدة الأساسية ----------------------
def init_db():
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY, model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL, trained_at TIMESTAMP DEFAULT NOW(), metrics JSONB );
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

# ---------------------- دوال جلب ومعالجة البيانات ----------------------
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    try:
        start_str = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
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
    btc_data_cache = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if btc_data_cache is None:
        logger.critical("❌ [BTC Data] فشل جلب بيانات البيتكوين."); exit(1)
    btc_data_cache['btc_returns'] = btc_data_cache['close'].pct_change()

# ---------------------- مؤشرات السكالب المتقدمة ----------------------
def calculate_price_velocity(close_prices, period=3):
    """حساب سرعة تحرك السعر"""
    returns = close_prices.pct_change()
    return returns.rolling(period).mean() * 100

def calculate_buying_pressure(high, low, close):
    """قياس ضغط المشترين"""
    bp = (close - low) / (high - low + 1e-9)
    return bp.replace([np.inf, -np.inf], 0).clip(0, 1) * 100

def calculate_reversal_signal(open, high, low, close):
    """كشف إشارات الانعكاس الفورية"""
    # شموع المطرقة (Hammer) لإشارة الشراء
    hammer = (close > open) & ((close - low) > 1.5 * (high - close)) & ((open - low) > (high - open))
    # شموع الرجل المشنوق (Shooting Star) لإشارة البيع
    shooting_star = (close < open) & ((high - open) > 1.5 * (open - low)) & ((high - close) > (close - low))
    return np.where(hammer, 1, np.where(shooting_star, -1, 0))

def calculate_fair_value(high, low, volume):
    """تقدير القيمة العادلة الآنية"""
    typical_price = (high + low) / 2
    return (typical_price * volume).rolling(5).mean() / volume.rolling(5).mean()

def calculate_liquidity(close, volume):
    """قياس السيولة الفورية"""
    return (close.diff() / volume.replace(0, 1e-9)).rolling(3).mean()

def calculate_adx(high, low, close, window=10):
    """حساب ADX مختصر للسكالب"""
    up = high.diff()
    down = -low.diff()
    
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift()), np.abs(low - close.shift())))
    
    atr = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window).mean()
    return adx

# ---------------------- دوال معالجة البيانات ----------------------
def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()

    # ATR
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # RSI مختصر
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / (loss.replace(0, 1e-9))))

    # MACD سريع
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
    df_calc['bb_width'] = (std_dev * 2) / (sma + 1e-9)

    # Stochastic RSI
    rsi = df_calc['rsi']
    min_rsi = rsi.rolling(window=STOCH_RSI_PERIOD).min()
    max_rsi = rsi.rolling(window=STOCH_RSI_PERIOD).max()
    stoch_rsi_val = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(window=STOCH_K).mean() * 100

    # Relative Volume
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)

    # Overbought/Oversold Filter
    df_calc['market_condition'] = 0  # Neutral
    df_calc.loc[(df_calc['rsi'] > RSI_OVERBOUGHT) | (df_calc['stoch_rsi_k'] > STOCH_RSI_OVERBOUGHT), 'market_condition'] = 1  # Overbought
    df_calc.loc[(df_calc['rsi'] < RSI_OVERSOLD) | (df_calc['stoch_rsi_k'] < STOCH_RSI_OVERSOLD), 'market_condition'] = -1  # Oversold

    # مؤشرات الاتجاه
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema20'] = (df_calc['close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema100'] = (df_calc['close'] / ema_slow_trend) - 1
    
    # مؤشرات الربحية
    df_calc['returns'] = df_calc['close'].pct_change()
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    df_calc['minute_of_hour'] = df_calc.index.minute  # دقيقة الساعة
    
    # مؤشرات السكالب الجديدة
    df_calc['price_velocity'] = calculate_price_velocity(df_calc['close'])
    df_calc['buying_pressure'] = calculate_buying_pressure(df_calc['high'], df_calc['low'], df_calc['close'])
    df_calc['reversal_signal'] = calculate_reversal_signal(df_calc['open'], df_calc['high'], df_calc['low'], df_calc['close'])
    df_calc['fair_value'] = calculate_fair_value(df_calc['high'], df_calc['low'], df_calc['volume'])
    df_calc['liquidity'] = calculate_liquidity(df_calc['close'], df_calc['volume'])
    df_calc['adx'] = calculate_adx(df_calc['high'], df_calc['low'], df_calc['close'])
    
    # تفاعلات متقدمة
    df_calc['velocity_pressure'] = df_calc['price_velocity'] * df_calc['buying_pressure']
    df_calc['reversal_volume'] = df_calc['reversal_signal'] * df_calc['relative_volume']
    
    # التقلب السريع
    df_calc['volatility'] = df_calc['close'].pct_change().rolling(5).std()
    
    return df_calc

def get_scalping_labels(close_prices: pd.Series) -> pd.Series:
    """وضع علامات متخصصة للتداول السكالب"""
    labels = pd.Series(0, index=close_prices.index)
    for i in tqdm(range(len(close_prices) - SCALP_MAX_HOLD), desc="Scalp Labeling", leave=False):
        entry = close_prices.iloc[i]
        tp = entry * (1 + SCALP_TP)
        sl = entry * (1 - SCALP_SL)
        
        future_prices = close_prices.iloc[i+1:i+SCALP_MAX_HOLD+1]
        if any(future_prices >= tp):
            labels.iloc[i] = 1
        elif any(future_prices <= sl):
            labels.iloc[i] = -1
    return labels

def remove_outliers(df, columns, threshold=3):
    """إزالة القيم المتطرفة باستخدام تقنية Z-Score القوية"""
    for col in columns:
        median = df[col].median()
        mad = np.abs(df[col] - median).median()
        if mad == 0:  # تجنب القسمة على صفر
            continue
        z_score = 0.6745 * (df[col] - median) / mad
        df = df[np.abs(z_score) < threshold]
    return df

def prepare_data_for_ml(df: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep] Preparing data for {symbol}...")
    df_featured = calculate_features(df, btc_df)
    
    # وضع العلامات للسكالب
    df_featured['target'] = get_scalping_labels(df_featured['close'])
    
    feature_columns = [
        # المؤشرات الأساسية
        'rsi', 'macd_hist', 'atr', 'relative_volume', 'minute_of_hour',
        'price_vs_ema20', 'price_vs_ema100', 'btc_correlation',
        'stoch_rsi_k', 'macd_cross', 'market_condition', 'bb_width',
        
        # مؤشرات السكالب الجديدة
        'price_velocity', 'buying_pressure', 'reversal_signal',
        'fair_value', 'liquidity', 'adx', 'volatility',
        
        # التفاعلات المتقدمة
        'velocity_pressure', 'reversal_volume'
    ]
    
    df_cleaned = df_featured.dropna(subset=feature_columns + ['target']).copy()
    
    # إزالة القيم المتطرفة
    df_cleaned = remove_outliers(df_cleaned, feature_columns)
    
    if df_cleaned.empty or df_cleaned['target'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] Data for {symbol} has less than 2 classes. Skipping.")
        return None
        
    logger.info(f"📊 [ML Prep] Target distribution for {symbol}:\n{df_cleaned['target'].value_counts(normalize=True)}")
    
    X = df_cleaned[feature_columns]
    y = df_cleaned['target']
    
    return X, y, feature_columns

# ---------------------- دوال التدريب المتقدمة ----------------------
def train_enhanced_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    """تدريب متقدم للسكالب باستخدام البحث العشوائي"""
    logger.info("ℹ️ [ML Train] Starting advanced scalping model training...")
    
    # تقسيم البيانات مع التحقق من التسرب الزمني
    tscv = TimeSeriesSplit(n_splits=5)
    
    # معايرة المعلمات باستخدام البحث العشوائي
    param_dist = {
        'learning_rate': [0.05, 0.1, 0.15],
        'n_estimators': [300, 500, 700],
        'max_depth': [3, 5],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 0.1],
        'min_child_samples': [10, 20],
        'num_leaves': [31, 63],
        'class_weight': ['balanced', None]
    }
    
    best_model = None
    best_scaler = None
    best_score = -np.inf
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # المعايرة
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # البحث العشوائي عن أفضل معلمات
        model = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42)
        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=25, cv=3, scoring='f1_weighted', n_jobs=-1
        )
        random_search.fit(X_train_scaled, y_train)
        
        # تقييم النموذج
        val_preds = random_search.best_estimator_.predict(X_val_scaled)
        score = f1_score(y_val, val_preds, average='weighted')
        
        logger.info(f"🔍 [Fold {fold+1}] Best params: {random_search.best_params_}, Score: {score:.4f}")
        
        if score > best_score:
            best_model = random_search.best_estimator_
            best_scaler = scaler
            best_score = score
    
    if not best_model or not best_scaler:
        logger.error("❌ [ML Train] Training failed, no model was created.")
        return None, None, None
    
    # تدريب النموذج النهائي على كامل البيانات
    X_full_scaled = best_scaler.transform(X)
    best_model.fit(X_full_scaled, y)
    
    # حساب المقاييس النهائية
    y_pred = best_model.predict(X_full_scaled)
    final_report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    
    avg_metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'f1_weighted': f1_score(y, y_pred, average='weighted'),
        'precision_1': precision_score(y, y_pred, labels=[1], average='binary'),
        'recall_1': recall_score(y, y_pred, labels=[1], average='binary'),
        'precision_-1': precision_score(y, y_pred, labels=[-1], average='binary'),
        'recall_-1': recall_score(y, y_pred, labels=[-1], average='binary'),
        'num_samples_trained': len(X),
        'best_params': random_search.best_params_
    }
    
    metrics_log_str = ', '.join([f"{k}: {v:.4f}" for k, v in avg_metrics.items() if isinstance(v, float)])
    logger.info(f"📊 [ML Train] Final Model Performance: {metrics_log_str}")
    
    return best_model, best_scaler, avg_metrics

def detect_data_drift(X_ref: pd.DataFrame, X_current: pd.DataFrame) -> float:
    """كشف انجراح البيانات باستخدام تقنية Kullback-Leibler"""
    kl_divergences = []
    for col in X_ref.columns:
        try:
            # إنشاء سلال متوافقة للبيانات المرجعية والجارية
            combined = pd.concat([X_ref[col], X_current[col]])
            bins = np.histogram_bin_edges(combined, bins=50)
            
            # حساب التوزيعات
            p, _ = np.histogram(X_ref[col], bins=bins, density=True)
            q, _ = np.histogram(X_current[col], bins=bins, density=True)
            
            # تجنب القيم الصفرية
            p = np.clip(p, 1e-10, None)
            q = np.clip(q, 1e-10, None)
            
            # حساب تباعد KL
            kl_divergences.append(entropy(p, q))
        except Exception as e:
            logger.warning(f"⚠️ [Drift] Error calculating KL for {col}: {e}")
    
    return np.mean(kl_divergences) if kl_divergences else 0.0

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

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

# ---------------------- نظام تنفيذ السكالب ----------------------
def scalp_alert_system(prediction, probabilities):
    """نظام تنبيهات ذكي للتداول السكالب"""
    if probabilities[1] > 0.7:  # ثقة عالية في الصفقة
        if prediction == 1:
            return "إشارة شراء قوية"
        elif prediction == -1:
            return "إشارة بيع قوية"
    elif probabilities[0] > 0.6:  # ثقة عالية في الحياد
        return "الانتظار أفضل"
    return "لا إشارة واضحة"

def execute_scalp_trade(signal, symbol, price, model_name):
    """تنفيذ صفقات السكالب التلقائية"""
    tp_price = price * (1 + SCALP_TP)
    sl_price = price * (1 - SCALP_SL)
    
    if signal == "إشارة شراء قوية":
        # تنفيذ أمر شراء مع وقف الخسارة وجني الربح
        msg = f"📈 [Trade] شراء {symbol} عند {price:.4f} | TP: {tp_price:.4f} | SL: {sl_price:.4f}"
        logger.info(msg)
        send_telegram_message(f"*{model_name}*\n{msg}")
    elif signal == "إشارة بيع قوية":
        # تنفيذ أمر بيع مع وقف الخسارة وجني الربح
        msg = f"📉 [Trade] بيع {symbol} عند {price:.4f} | TP: {tp_price:.4f} | SL: {sl_price:.4f}"
        logger.info(msg)
        send_telegram_message(f"*{model_name}*\n{msg}")

# ---------------------- الوظيفة الرئيسية ----------------------
def run_training_job():
    logger.info(f"🚀 Starting SCALPING ML model training job ({BASE_ML_MODEL_NAME})...")
    init_db()
    get_binance_client()
    fetch_and_cache_btc_data()
    symbols_to_train = get_validated_symbols(filename='crypto_list.txt')
    if not symbols_to_train:
        logger.critical("❌ [Main] No valid symbols found. Exiting.")
        return
        
    send_telegram_message(f"🚀 *{BASE_ML_MODEL_NAME} Training Started*\nWill train models for {len(symbols_to_train)} symbols.")
    
    successful_models, failed_models = 0, 0
    reference_models = {}  # لتخزين نماذج مرجعية لكشف الانجراح
    
    for symbol in symbols_to_train:
        logger.info(f"\n--- ⏳ [Main] Starting model training for {symbol} ---")
        try:
            df_hist = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
            if df_hist is None or df_hist.empty:
                logger.warning(f"⚠️ [Main] No data for {symbol}, skipping."); failed_models += 1; continue
            
            prepared_data = prepare_data_for_ml(df_hist, btc_data_cache, symbol)
            if prepared_data is None:
                failed_models += 1; continue
            X, y, feature_names = prepared_data
            
            # كشف انجراح البيانات (إذا كان لدينا نموذج سابق)
            model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
            if symbol in reference_models:
                drift_score = detect_data_drift(reference_models[symbol], X)
                logger.info(f"📈 [Drift] Data drift score for {symbol}: {drift_score:.4f}")
                if drift_score > 0.25:
                    send_telegram_message(f"⚠️ *Data Drift Alert*: {symbol} (Score: {drift_score:.4f}")
            
            # تدريب النموذج
            training_result = train_enhanced_model(X, y)
            if not all(training_result):
                 failed_models += 1; continue
            final_model, final_scaler, model_metrics = training_result
            
            # حفظ النموذج إذا كان أداءه جيداً
            if final_model and final_scaler and model_metrics.get('precision_1', 0) > 0.35:
                model_bundle = {
                    'model': final_model, 
                    'scaler': final_scaler, 
                    'feature_names': feature_names,
                    'last_trained': datetime.utcnow().isoformat()
                }
                save_ml_model_to_db(model_bundle, model_name, model_metrics)
                successful_models += 1
                reference_models[symbol] = X  # حفظ كمرجع لكشف الانجراح المستقبلي
                
                # اختبار النظام على آخر نقطة بيانات
                last_point = X.iloc[[-1]]
                scaled_data = final_scaler.transform(last_point)
                prediction = final_model.predict(scaled_data)[0]
                probabilities = final_model.predict_proba(scaled_data)[0]
                
                signal = scalp_alert_system(prediction, probabilities)
                current_price = df_hist['close'].iloc[-1]
                execute_scalp_trade(signal, symbol, current_price, model_name)
                
            else:
                logger.warning(f"⚠️ [Main] Model for {symbol} is not useful. Discarding."); failed_models += 1
        except Exception as e:
            logger.critical(f"❌ [Main] A fatal error occurred for {symbol}: {e}", exc_info=True); failed_models += 1
        time.sleep(1)

    completion_message = (f"✅ *{BASE_ML_MODEL_NAME} Training Finished*\n"
                        f"- Successfully trained: {successful_models} models\n"
                        f"- Failed/Discarded: {failed_models} models\n"
                        f"- Total symbols: {len(symbols_to_train)}")
    send_telegram_message(completion_message)
    logger.info(completion_message)

    if conn: conn.close()
    logger.info("👋 [Main] ML training job finished.")

# ---------------------- خادم الويب للصيانة ----------------------
app = Flask(__name__)

@app.route('/')
def health_check():
    """Endpoint for Render health checks."""
    return "ML Scalper service is running and healthy.", 200

if __name__ == "__main__":
    training_thread = Thread(target=run_training_job)
    training_thread.daemon = True
    training_thread.start()
    
    port = int(os.environ.get("PORT", 10001))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)
