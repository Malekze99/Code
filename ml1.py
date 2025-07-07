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
import pandas_ta as ta  # <-- تم الإضافة
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
from imblearn.over_sampling import ADASYN

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer_v5_improved.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer_V5_Improved')

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
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V5_Improved'
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
        # تعديل طفيف ليكون أكثر قوة إذا تم تشغيله من مسارات مختلفة
        script_dir = os.path.dirname(os.path.abspath(__file__))
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

# --- دوال جلب ومعالجة البيانات ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    try:
        start_str = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: 
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
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

# --- دوال حساب المؤشرات (تم التحديث) ---
def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    نسخة محسنة من دالة حساب الميزات باستخدام مكتبة pandas-ta لضمان الدقة والكفاءة.
    """
    df_calc = df.copy(deep=True)
    
    # --- 1. حساب المؤشرات الفنية باستخدام pandas-ta ---
    df_calc.ta.atr(length=ATR_PERIOD, append=True)
    df_calc.ta.rsi(length=RSI_PERIOD, append=True)
    df_calc.ta.macd(fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL, append=True)
    df_calc.ta.bbands(length=BBANDS_PERIOD, append=True)
    df_calc.ta.stochrsi(length=STOCH_RSI_PERIOD, rsi_length=RSI_PERIOD, k=STOCH_K, d=STOCH_D, append=True)
    df_calc.ta.adx(length=ADX_PERIOD, append=True)
    df_calc.ta.mfi(length=14, append=True)
    df_calc.ta.cci(length=20, append=True)
    df_calc.ta.vwap(append=True)

    # --- 2. إعادة تسمية الأعمدة لتناسب باقي الكود ---
    df_calc.rename(columns={
        f'ATR_{ATR_PERIOD}': 'atr',
        f'RSI_{RSI_PERIOD}': 'rsi',
        f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}': 'macd_hist',
        f'STOCHRSIk_{STOCH_RSI_PERIOD}_{RSI_PERIOD}_{STOCH_K}_{STOCH_D}': 'stoch_rsi_k',
        f'STOCHRSId_{STOCH_RSI_PERIOD}_{RSI_PERIOD}_{STOCH_K}_{STOCH_D}': 'stoch_rsi_d',
        f'ADX_{ADX_PERIOD}': 'adx',
        f'DMP_{ADX_PERIOD}': 'adx_pos',
        f'DMN_{ADX_PERIOD}': 'adx_neg',
        'MFI_14': 'mfi',
        'CCI_20_0.015': 'cci',
        f'VWAP_D': 'vwap' # قد يختلف الاسم حسب pandas-ta
    }, inplace=True, errors='ignore')

    # --- 3. حساب الميزات المخصصة الأخرى ---
    
    # MACD Cross
    df_calc['macd_cross'] = 0
    df_calc.loc[(df_calc['macd_hist'].shift(1) < 0) & (df_calc['macd_hist'] >= 0), 'macd_cross'] = 1
    df_calc.loc[(df_calc['macd_hist'].shift(1) > 0) & (df_calc['macd_hist'] <= 0), 'macd_cross'] = -1
    
    # Bollinger Bands Width
    bb_upper_col = f'BBU_{BBANDS_PERIOD}_2.0'
    bb_lower_col = f'BBL_{BBANDS_PERIOD}_2.0'
    bb_middle_col = f'BBM_{BBANDS_PERIOD}_2.0'
    if all(col in df_calc.columns for col in [bb_upper_col, bb_lower_col, bb_middle_col]):
        df_calc['bb_width'] = (df_calc[bb_upper_col] - df_calc[bb_lower_col]) / (df_calc[bb_middle_col] + 1e-9)
    else:
        df_calc['bb_width'] = 0 # Fallback

    # Relative Volume
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)

    # Market Condition Filter
    df_calc['market_condition'] = 0  # 0 for Neutral
    df_calc.loc[(df_calc['rsi'] > RSI_OVERBOUGHT) | (df_calc['stoch_rsi_k'] > STOCH_RSI_OVERBOUGHT), 'market_condition'] = 1  # 1 for Overbought
    df_calc.loc[(df_calc['rsi'] < RSI_OVERSOLD) | (df_calc['stoch_rsi_k'] < STOCH_RSI_OVERSOLD), 'market_condition'] = -1  # -1 for Oversold

    # Other Features
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
    df_calc['returns'] = df_calc['close'].pct_change()
    
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    df_calc['hour_of_day'] = df_calc.index.hour
    df_calc['volatility'] = df_calc['close'].pct_change().rolling(14).std()

    # Interaction Features
    df_calc['rsi_macd_interaction'] = df_calc['rsi'] * df_calc['macd_hist']
    df_calc['atr_volume_interaction'] = df_calc['atr'] * df_calc['relative_volume']
    
    return df_calc

def get_enhanced_labels(prices: pd.Series, atr: pd.Series, volatility: pd.Series) -> pd.Series:
    """وضع علامات متقدمة باستخدام تقنية Triple-Barrier المعززة"""
    prices = pd.Series(prices)
    atr = pd.Series(atr)
    volatility = pd.Series(volatility)
    
    labels = pd.Series(0, index=prices.index)
    volatility_factor = 1 + (volatility.rolling(14).std() * 0.5)
    volatility_factor = volatility_factor.fillna(1)
    
    # تحذير: هذه الدالة بطيئة بسبب الحلقات. التحسين باستخدام vectorization ممكن ولكنه معقد.
    for i in tqdm(range(len(prices) - MAX_HOLD_PERIOD), desc="Enhanced Labeling", leave=False):
        entry_price = prices.iloc[i]
        current_atr = atr.iloc[i]
        vol_factor = volatility_factor.iloc[i] if not pd.isna(volatility_factor.iloc[i]) else 1
        
        if pd.isna(current_atr) or current_atr == 0: 
            continue
            
        tp_level = entry_price + (current_atr * TP_ATR_MULTIPLIER * vol_factor)
        sl_level = entry_price - (current_atr * SL_ATR_MULTIPLIER * vol_factor)
        
        best_gain = -np.inf
        worst_loss = np.inf
        
        for j in range(1, MAX_HOLD_PERIOD + 1):
            if i + j >= len(prices): 
                break
                
            current_price = prices.iloc[i + j]
            
            gain = (current_price - entry_price) / entry_price
            if gain > best_gain: best_gain = gain
                
            loss = (current_price - entry_price) / entry_price
            if loss < worst_loss: worst_loss = loss
                
            if current_price >= tp_level:
                labels.iloc[i] = 1; break
            if current_price <= sl_level:
                labels.iloc[i] = -1; break
        else:
            if best_gain > abs(worst_loss) and best_gain > 0.005:
                labels.iloc[i] = 1
            elif abs(worst_loss) > best_gain and abs(worst_loss) > 0.005:
                labels.iloc[i] = -1
                
    return labels

def remove_outliers(df, columns, threshold=3):
    """إزالة القيم المتطرفة باستخدام تقنية Z-Score القوية"""
    df_out = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df_out[col]):
            median = df_out[col].median()
            mad = np.abs(df_out[col] - median).median()
            if mad == 0: continue
            z_score = 0.6745 * (df_out[col] - median) / mad
            df_out = df_out[np.abs(z_score) < threshold]
    return df_out

def balance_classes(X, y):
    """موازنة الفئات باستخدام تقنية ADASYN المتقدمة"""
    try:
        ada = ADASYN(random_state=42, sampling_strategy='auto', n_neighbors=5)
        return ada.fit_resample(X, y)
    except Exception as e:
        logger.warning(f"⚠️ [ADASYN] فشل في موازنة الفئات: {e}. سيتم استخدام البيانات الأصلية.")
        return X, y

def prepare_data_for_ml(df: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep] Preparing data for {symbol}...")
    df_featured = calculate_features(df, btc_df)
    
    df_featured['target'] = get_enhanced_labels(df_featured['close'], df_featured['atr'], df_featured['volatility'])
    
    feature_columns = [
        'rsi', 'macd_hist', 'atr', 'relative_volume', 'hour_of_day',
        'price_vs_ema50', 'price_vs_ema200', 'btc_correlation',
        'stoch_rsi_k', 'stoch_rsi_d', 'macd_cross', 'market_condition',
        'bb_width', 'adx', 'adx_pos', 'adx_neg', 'mfi', 'vwap', 'cci', 
        'volatility', 'rsi_macd_interaction', 'atr_volume_interaction'
    ]
    
    df_cleaned = df_featured.dropna(subset=feature_columns + ['target']).copy()
    
    df_cleaned = remove_outliers(df_cleaned, feature_columns)
    
    if df_cleaned.empty or df_cleaned['target'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] Data for {symbol} has less than 2 classes after prep. Skipping.")
        return None
        
    logger.info(f"📊 [ML Prep] Target distribution for {symbol}:\n{df_cleaned['target'].value_counts(normalize=True)}")
    
    X = df_cleaned[feature_columns]
    y = df_cleaned['target']
    
    return X, y, feature_columns

# --- دوال التدريب المتقدمة (تم التحديث) ---
def train_enhanced_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    """
    تدريب متقدم باستخدام البحث العشوائي الآمن للسلاسل الزمنية وموازنة الفئات.
    """
    logger.info("ℹ️ [ML Train] Starting advanced model training...")
    
    X_bal, y_bal = balance_classes(X.values, y.values)
    
    # مقسم بيانات للسلاسل الزمنية للتقييم الخارجي
    tscv = TimeSeriesSplit(n_splits=5)
    
    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [300, 500, 700],
        'max_depth': [3, 5, 7], 'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9], 'reg_alpha': [0, 0.1, 0.5], 'reg_lambda': [0, 0.1, 0.5]
    }
    
    best_model = None
    best_scaler = None
    best_score = -np.inf
    best_params_overall = {}
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X_bal)):
        X_train, X_val = X_bal[train_index], X_bal[val_index]
        y_train, y_val = y_bal[train_index], y_bal[val_index]
        
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42, n_jobs=1)
        
        # --- الإصلاح الجوهري: استخدام TimeSeriesSplit داخل RandomizedSearchCV ---
        # هذا يمنع تسرب البيانات أثناء البحث عن أفضل المعلمات.
        inner_cv_splitter = TimeSeriesSplit(n_splits=3)
        random_search = RandomizedSearchCV(
            estimator=model, param_distributions=param_dist, n_iter=15, 
            cv=inner_cv_splitter, scoring='f1_weighted', n_jobs=-1, random_state=42
        )
        random_search.fit(X_train_scaled, y_train)
        
        val_preds = random_search.best_estimator_.predict(X_val_scaled)
        score = f1_score(y_val, val_preds, average='weighted')
        
        logger.info(f"🔍 [Fold {fold+1}] Best params: {random_search.best_params_}, Score: {score:.4f}")
        
        if score > best_score:
            best_model = random_search.best_estimator_
            best_scaler = scaler
            best_score = score
            best_params_overall = random_search.best_params_
    
    if not best_model or not best_scaler:
        logger.error("❌ [ML Train] Training failed, no model was created.")
        return None, None, None
    
    logger.info(f"✅ [ML Train] Best model selected with F1-score: {best_score:.4f}")
    
    # تدريب النموذج النهائي على كامل البيانات الموزونة باستخدام أفضل المعلمات
    X_full_scaled = best_scaler.transform(X_bal)
    best_model.fit(X_full_scaled, y_bal)
    
    y_pred = best_model.predict(X_full_scaled)
    final_report = classification_report(y_bal, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        'accuracy': accuracy_score(y_bal, y_pred),
        'f1_weighted': f1_score(y_bal, y_pred, average='weighted'),
        'precision_1': precision_score(y_bal, y_pred, labels=[1], average='binary', zero_division=0),
        'recall_1': recall_score(y_bal, y_pred, labels=[1], average='binary', zero_division=0),
        'precision_-1': precision_score(y_bal, y_pred, labels=[-1], average='binary', zero_division=0),
        'recall_-1': recall_score(y_bal, y_pred, labels=[-1], average='binary', zero_division=0),
        'num_samples_trained': len(X_bal),
        'best_params': best_params_overall
    }
    
    metrics_log_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, float)])
    logger.info(f"📊 [ML Train] Final Model Performance: {metrics_log_str}")
    
    return best_model, best_scaler, metrics

def detect_data_drift(X_ref: pd.DataFrame, X_current: pd.DataFrame) -> float:
    """كشف انجراف البيانات باستخدام تقنية Kullback-Leibler"""
    kl_divergences = []
    for col in X_ref.columns:
        try:
            combined = pd.concat([X_ref[col], X_current[col]])
            bins = np.histogram_bin_edges(combined, bins=50)
            p, _ = np.histogram(X_ref[col], bins=bins, density=True)
            q, _ = np.histogram(X_current[col], bins=bins, density=True)
            p = np.clip(p, 1e-10, None)
            q = np.clip(q, 1e-10, None)
            kl_divergences.append(entropy(p, q))
        except Exception as e:
            logger.warning(f"⚠️ [Drift] Error calculating KL for {col}: {e}")
    
    return np.mean(kl_divergences) if kl_divergences else 0.0

def save_ml_model_to_db(model_bundle: Dict[str, Any], model_name: str, metrics: Dict[str, Any]):
    logger.info(f"ℹ️ [DB Save] Saving model bundle '{model_name}'...")
    try:
        # استخدام الطريقة الموصى بها لحفظ نموذج LightGBM
        model_string = model_bundle['model'].model_to_string()
        bundle_to_save = {
            'model_string': model_string,
            'scaler': model_bundle['scaler'],
            'feature_names': model_bundle['feature_names'],
            'last_trained': model_bundle['last_trained']
        }
        model_binary = pickle.dumps(bundle_to_save)
        metrics_json = json.dumps(metrics, default=str) # Handle non-serializable types
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

def run_training_job():
    logger.info(f"🚀 Starting IMPROVED ML model training job ({BASE_ML_MODEL_NAME})...")
    init_db()
    get_binance_client()
    fetch_and_cache_btc_data()
    symbols_to_train = get_validated_symbols(filename='crypto_list.txt')
    if not symbols_to_train:
        logger.critical("❌ [Main] No valid symbols found. Exiting.")
        return
        
    send_telegram_message(f"🚀 *{BASE_ML_MODEL_NAME} Training Started*\nWill train models for {len(symbols_to_train)} symbols.")
    
    successful_models, failed_models = 0, 0
    reference_data = {}
    
    for symbol in symbols_to_train:
        logger.info(f"\n--- ⏳ [Main] Starting model training for {symbol} ---")
        try:
            df_hist = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
            if df_hist is None or len(df_hist) < 200: # Ensure enough data
                logger.warning(f"⚠️ [Main] Not enough data for {symbol}, skipping."); failed_models += 1; continue
            
            prepared_data = prepare_data_for_ml(df_hist, btc_data_cache, symbol)
            if prepared_data is None:
                failed_models += 1; continue
            X, y, feature_names = prepared_data
            
            model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
            if symbol in reference_data:
                drift_score = detect_data_drift(reference_data[symbol], X)
                logger.info(f"📈 [Drift] Data drift score for {symbol}: {drift_score:.4f}")
                if drift_score > 0.25:
                    send_telegram_message(f"⚠️ *Data Drift Alert*: {symbol} (Score: {drift_score:.4f})")
            
            training_result = train_enhanced_model(X, y)
            if not all(training_result):
                 failed_models += 1; continue
            final_model, final_scaler, model_metrics = training_result
            
            if final_model and final_scaler and model_metrics.get('precision_1', 0) > 0.35 and model_metrics.get('precision_-1', 0) > 0.35:
                model_bundle = {
                    'model': final_model, 'scaler': final_scaler, 
                    'feature_names': feature_names, 'last_trained': datetime.utcnow().isoformat()
                }
                save_ml_model_to_db(model_bundle, model_name, model_metrics)
                successful_models += 1
                reference_data[symbol] = X
            else:
                logger.warning(f"⚠️ [Main] Model for {symbol} did not meet performance criteria. Discarding."); failed_models += 1
        except Exception as e:
            logger.critical(f"❌ [Main] A fatal error occurred for {symbol}: {e}", exc_info=True); failed_models += 1
        time.sleep(1)

    completion_message = (f"✅ *{BASE_ML_MODEL_NAME} Training Finished*\n"
                        f"- Successfully trained: {successful_models} models\n"
                        f"- Failed/Discarded: {failed_models} models\n"
                        f"- Total symbols: {len(symbols_to_train)}")
    send_telegram_message(completion_message)
    logger.info(completion_message)

    if conn: 
        conn.close()
        logger.info("🔌 [DB] تم إغلاق اتصال قاعدة البيانات")

app = Flask(__name__)

@app.route('/')
def health_check():
    """Endpoint for Render health checks."""
    return "ML Trainer service is running and healthy.", 200

if __name__ == "__main__":
    training_thread = Thread(target=run_training_job)
    training_thread.daemon = True
    training_thread.start()
    
    # تم التعليق على السطر التالي للسماح لخادم فلاسك بالعمل فوراً
    # بينما تتم عملية التدريب في الخلفية. هذا ضروري لخدمات مثل Render.
    # training_thread.join()
    
    port = int(os.environ.get("PORT", 10001))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)

