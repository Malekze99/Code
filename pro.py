import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import warnings
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple, Union
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from flask import Flask, request, Response
from threading import Thread
import matplotlib.pyplot as plt

# تعطيل تحذيرات غير ضرورية
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer_enhanced.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EnhancedMLTrainer')

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

# ---------------------- Constants and Global Variables ----------------------
SIGNAL_GENERATION_TIMEFRAME: str = '5m'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 90
BASE_ML_MODEL_NAME: str = 'Enhanced_LightGBM_SMC_V2'

# Indicator Parameters
VOLUME_LOOKBACK_CANDLES: int = 3
RSI_PERIOD: int = 14  # Changed from 9 to 14 for better stability
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

# ---------------------- Enhanced Database Functions ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """Enhanced database initialization with additional tables for results tracking"""
    global conn, cur
    logger.info("[DB] Starting enhanced database initialization...")
    
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            cur = conn.cursor()
            
            # Create tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_signals (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    signal_strength DOUBLE PRECISION,
                    confidence_score DOUBLE PRECISION,
                    features_json JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS enhanced_model_results (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    accuracy DOUBLE PRECISION,
                    precision DOUBLE PRECISION,
                    recall DOUBLE PRECISION,
                    f1_score DOUBLE PRECISION,
                    auc_score DOUBLE PRECISION,
                    training_duration INTERVAL,
                    feature_importances JSONB,
                    trained_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS enhanced_training_logs (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT,
                    status TEXT,
                    metrics JSONB,
                    error_message TEXT,
                    log_time TIMESTAMP DEFAULT NOW()
                );
            """)
            conn.commit()
            logger.info("✅ [DB] Enhanced database schema initialized successfully.")
            return
            
        except Exception as e:
            logger.error(f"❌ [DB] Initialization attempt {attempt+1} failed: {str(e)}")
            if conn: conn.rollback()
            if attempt == retries - 1:
                raise
            time.sleep(delay)

# ---------------------- Enhanced Data Processing ----------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced data cleaning with outlier removal and normalization"""
    df = df.copy()
    
    # Remove outliers using robust statistical methods
    for col in ['close', 'volume', 'high', 'low']:
        q1 = df[col].quantile(0.05)
        q3 = df[col].quantile(0.95)
        df = df[(df[col] > q1) & (df[col] < q3)]
    
    # Advanced normalization
    df['volume'] = np.log1p(df['volume'])
    df['price_change'] = df['close'].pct_change()
    
    return df.dropna()

def calculate_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators with improved methods"""
    df = df.copy()
    
    # Enhanced RSI calculation
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Enhanced ADX with EMA smoothing
    df = calculate_adx(df)
    
    # Momentum and Velocity
    df['momentum'] = df['close'].pct_change(MOMENTUM_LOOKBACK)
    df['price_velocity'] = df['close'].diff(VELOCITY_LOOKBACK) / VELOCITY_LOOKBACK
    
    # SMC Indicators
    df = detect_liquidity_pools(df)
    df = identify_order_blocks(df)
    df = detect_fvg(df)
    df = identify_bos_choch(df)
    
    # New interactive features
    df['vol_price_interaction'] = df['volume'] * df['price_velocity']
    df['rsi_adx_combo'] = (df['rsi'] / 50) * (df['adx'] / 20)
    
    # Time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    return df

# ---------------------- Enhanced Indicator Functions ----------------------
def calculate_adx(df: pd.DataFrame, lookback: int = ADX_LOOKBACK) -> pd.DataFrame:
    """Enhanced ADX calculation with EMA smoothing"""
    df = df.copy()
    
    # Calculate True Range
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=lookback, adjust=False).mean()
    
    # Directional Movement
    up = df['high'].diff()
    down = -df['low'].diff()
    df['plus_dm'] = np.where((up > down) & (up > 0), up, 0)
    df['minus_dm'] = np.where((down > up) & (down > 0), down, 0)
    
    # Smoothed Directional Indicators
    df['plus_di'] = (df['plus_dm'].ewm(span=lookback, adjust=False).mean() / df['atr']) * 100
    df['minus_di'] = (df['minus_dm'].ewm(span=lookback, adjust=False).mean() / df['atr']) * 100
    
    # ADX Calculation
    dx = (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])) * 100
    df['adx'] = dx.ewm(span=lookback, adjust=False).mean()
    
    # Enhanced Trend Direction
    conditions = [
        (df['plus_di'] > df['minus_di']) & (df['adx'] > 25),
        (df['minus_di'] > df['plus_di']) & (df['adx'] > 25),
    ]
    choices = [1, -1]
    df['trend_direction'] = np.select(conditions, choices, default=0)
    
    return df

# ---------------------- Enhanced Model Training ----------------------
def train_enhanced_model(X_train, y_train, X_test, y_test):
    """Enhanced model training with hyperparameter tuning and ensemble"""
    
    # Base models for ensemble
    lgbm = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=7,
        min_child_samples=50,
        class_weight='balanced',
        verbosity=-1,
        n_jobs=-1
    )
    
    xgb = XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        n_jobs=-1
    )
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[('lgbm', lgbm), ('xgb', xgb), ('rf', rf)],
        voting='soft',
        n_jobs=-1
    )
    
    # Train with SMOTE
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    # Fit model
    ensemble.fit(X_res, y_res)
    
    return ensemble

# ---------------------- Main Training Pipeline ----------------------
def enhanced_training_pipeline():
    global training_status, last_training_time, last_training_metrics
    
    try:
        # Initialize
        init_db()
        client = Client(API_KEY, API_SECRET)
        symbols = get_crypto_symbols()
        
        training_status = "Enhanced Training Started"
        start_time = datetime.now()
        
        overall_metrics = {
            'total_models': len(symbols),
            'successful_models': 0,
            'failed_models': 0,
            'avg_accuracy': 0,
            'avg_auc': 0,
            'details': {}
        }
        
        # Pre-fetch BTC data
        btc_data = fetch_historical_data("BTCUSDT", SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
        
        for symbol in symbols:
            try:
                # Fetch and prepare data
                data = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
                if data is None or len(data) < 100:
                    raise ValueError("Insufficient data")
                
                data = clean_data(data)
                data = calculate_enhanced_indicators(data)
                
                # Prepare features and target
                features = [
                    'rsi', 'momentum', 'price_velocity', 'adx', 'trend_direction',
                    'liquidity_pool_high', 'liquidity_pool_low',
                    'bullish_ob', 'bearish_ob', 'bullish_fvg', 'bearish_fvg',
                    'vol_price_interaction', 'rsi_adx_combo', 'hour', 'day_of_week'
                ]
                
                data['target'] = (data['close'].shift(-5) > data['close'] * 1.005).astype(int)
                data = data.dropna(subset=features + ['target'])
                
                if len(data) < 100:
                    raise ValueError("Not enough data after processing")
                
                # Split data
                X = data[features]
                y = data['target']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = train_enhanced_model(X_train_scaled, y_train, X_test_scaled, y_test)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'auc': roc_auc_score(y_test, y_proba),
                    'feature_importances': dict(zip(features, model.named_estimators_['lgbm'].feature_importances_))
                }
                
                # Save results
                save_model_results(symbol, model, metrics)
                overall_metrics['successful_models'] += 1
                overall_metrics['avg_accuracy'] += metrics['accuracy']
                overall_metrics['avg_auc'] += metrics['auc']
                overall_metrics['details'][symbol] = metrics
                
            except Exception as e:
                logger.error(f"❌ Failed to train model for {symbol}: {str(e)}")
                overall_metrics['failed_models'] += 1
                overall_metrics['details'][symbol] = {'error': str(e)}
        
        # Calculate averages
        if overall_metrics['successful_models'] > 0:
            overall_metrics['avg_accuracy'] /= overall_metrics['successful_models']
            overall_metrics['avg_auc'] /= overall_metrics['successful_models']
        
        # Finalize
        training_status = "Enhanced Training Completed"
        last_training_time = datetime.now()
        last_training_metrics = overall_metrics
        
        # Send summary
        send_summary_report(overall_metrics, start_time)
        
    except Exception as e:
        training_status = "Enhanced Training Failed"
        training_error = str(e)
        logger.critical(f"❌ Critical error in training pipeline: {str(e)}")
        
    finally:
        cleanup_resources()

# ---------------------- Helper Functions ----------------------
def save_model_results(symbol: str, model: Any, metrics: dict):
    """Save model and results to database"""
    if not check_db_connection():
        return False
    
    try:
        model_binary = pickle.dumps(model)
        metrics_json = json.dumps(metrics)
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO enhanced_model_results 
                (model_name, accuracy, precision, recall, f1_score, auc_score, feature_importances)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (f"{BASE_ML_MODEL_NAME}_{symbol}", 
                 metrics['accuracy'], metrics['precision'], 
                 metrics['recall'], metrics['f1'], 
                 metrics['auc'], metrics['feature_importances']))
            
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save results for {symbol}: {str(e)}")
        if conn: conn.rollback()
        return False

def send_summary_report(metrics: dict, start_time: datetime):
    """Send training summary report"""
    duration = datetime.now() - start_time
    duration_str = str(duration).split('.')[0]
    
    message = (
        f"🚀 *Enhanced Model Training Complete*\n\n"
        f"📊 *Summary Metrics*\n"
        f"- Successful Models: {metrics['successful_models']}/{metrics['total_models']}\n"
        f"- Avg Accuracy: {metrics['avg_accuracy']:.4f}\n"
        f"- Avg AUC: {metrics['avg_auc']:.4f}\n"
        f"- Training Duration: {duration_str}\n\n"
        f"🔍 *Top Performing Models*\n"
    )
    
    # Add top 5 models by AUC
    top_models = sorted(
        [(k, v) for k, v in metrics['details'].items() if 'auc' in v],
        key=lambda x: x[1]['auc'],
        reverse=True
    )[:5]
    
    for symbol, m in top_models:
        message += f"- {symbol}: AUC={m['auc']:.4f}, F1={m['f1']:.4f}\n"
    
    if TELEGRAM_TOKEN and CHAT_ID:
        send_telegram_message(CHAT_ID, message)

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    logger.info("🚀 Starting Enhanced ML Training Pipeline")
    enhanced_training_pipeline()
    logger.info("🏁 Enhanced Training Pipeline Completed")import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import warnings
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple, Union
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from flask import Flask, request, Response
from threading import Thread
import matplotlib.pyplot as plt

# تعطيل تحذيرات غير ضرورية
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer_enhanced.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EnhancedMLTrainer')

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

# ---------------------- Constants and Global Variables ----------------------
SIGNAL_GENERATION_TIMEFRAME: str = '5m'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 90
BASE_ML_MODEL_NAME: str = 'Enhanced_LightGBM_SMC_V2'

# Indicator Parameters
VOLUME_LOOKBACK_CANDLES: int = 3
RSI_PERIOD: int = 14  # Changed from 9 to 14 for better stability
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

# ---------------------- Enhanced Database Functions ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """Enhanced database initialization with additional tables for results tracking"""
    global conn, cur
    logger.info("[DB] Starting enhanced database initialization...")
    
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            cur = conn.cursor()
            
            # Create tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_signals (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    signal_strength DOUBLE PRECISION,
                    confidence_score DOUBLE PRECISION,
                    features_json JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS enhanced_model_results (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    accuracy DOUBLE PRECISION,
                    precision DOUBLE PRECISION,
                    recall DOUBLE PRECISION,
                    f1_score DOUBLE PRECISION,
                    auc_score DOUBLE PRECISION,
                    training_duration INTERVAL,
                    feature_importances JSONB,
                    trained_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS enhanced_training_logs (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT,
                    status TEXT,
                    metrics JSONB,
                    error_message TEXT,
                    log_time TIMESTAMP DEFAULT NOW()
                );
            """)
            conn.commit()
            logger.info("✅ [DB] Enhanced database schema initialized successfully.")
            return
            
        except Exception as e:
            logger.error(f"❌ [DB] Initialization attempt {attempt+1} failed: {str(e)}")
            if conn: conn.rollback()
            if attempt == retries - 1:
                raise
            time.sleep(delay)

# ---------------------- Enhanced Data Processing ----------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced data cleaning with outlier removal and normalization"""
    df = df.copy()
    
    # Remove outliers using robust statistical methods
    for col in ['close', 'volume', 'high', 'low']:
        q1 = df[col].quantile(0.05)
        q3 = df[col].quantile(0.95)
        df = df[(df[col] > q1) & (df[col] < q3)]
    
    # Advanced normalization
    df['volume'] = np.log1p(df['volume'])
    df['price_change'] = df['close'].pct_change()
    
    return df.dropna()

def calculate_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators with improved methods"""
    df = df.copy()
    
    # Enhanced RSI calculation
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Enhanced ADX with EMA smoothing
    df = calculate_adx(df)
    
    # Momentum and Velocity
    df['momentum'] = df['close'].pct_change(MOMENTUM_LOOKBACK)
    df['price_velocity'] = df['close'].diff(VELOCITY_LOOKBACK) / VELOCITY_LOOKBACK
    
    # SMC Indicators
    df = detect_liquidity_pools(df)
    df = identify_order_blocks(df)
    df = detect_fvg(df)
    df = identify_bos_choch(df)
    
    # New interactive features
    df['vol_price_interaction'] = df['volume'] * df['price_velocity']
    df['rsi_adx_combo'] = (df['rsi'] / 50) * (df['adx'] / 20)
    
    # Time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    return df

# ---------------------- Enhanced Indicator Functions ----------------------
def calculate_adx(df: pd.DataFrame, lookback: int = ADX_LOOKBACK) -> pd.DataFrame:
    """Enhanced ADX calculation with EMA smoothing"""
    df = df.copy()
    
    # Calculate True Range
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=lookback, adjust=False).mean()
    
    # Directional Movement
    up = df['high'].diff()
    down = -df['low'].diff()
    df['plus_dm'] = np.where((up > down) & (up > 0), up, 0)
    df['minus_dm'] = np.where((down > up) & (down > 0), down, 0)
    
    # Smoothed Directional Indicators
    df['plus_di'] = (df['plus_dm'].ewm(span=lookback, adjust=False).mean() / df['atr']) * 100
    df['minus_di'] = (df['minus_dm'].ewm(span=lookback, adjust=False).mean() / df['atr']) * 100
    
    # ADX Calculation
    dx = (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])) * 100
    df['adx'] = dx.ewm(span=lookback, adjust=False).mean()
    
    # Enhanced Trend Direction
    conditions = [
        (df['plus_di'] > df['minus_di']) & (df['adx'] > 25),
        (df['minus_di'] > df['plus_di']) & (df['adx'] > 25),
    ]
    choices = [1, -1]
    df['trend_direction'] = np.select(conditions, choices, default=0)
    
    return df

# ---------------------- Enhanced Model Training ----------------------
def train_enhanced_model(X_train, y_train, X_test, y_test):
    """Enhanced model training with hyperparameter tuning and ensemble"""
    
    # Base models for ensemble
    lgbm = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=7,
        min_child_samples=50,
        class_weight='balanced',
        verbosity=-1,
        n_jobs=-1
    )
    
    xgb = XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        n_jobs=-1
    )
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[('lgbm', lgbm), ('xgb', xgb), ('rf', rf)],
        voting='soft',
        n_jobs=-1
    )
    
    # Train with SMOTE
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    # Fit model
    ensemble.fit(X_res, y_res)
    
    return ensemble

# ---------------------- Main Training Pipeline ----------------------
def enhanced_training_pipeline():
    global training_status, last_training_time, last_training_metrics
    
    try:
        # Initialize
        init_db()
        client = Client(API_KEY, API_SECRET)
        symbols = get_crypto_symbols()
        
        training_status = "Enhanced Training Started"
        start_time = datetime.now()
        
        overall_metrics = {
            'total_models': len(symbols),
            'successful_models': 0,
            'failed_models': 0,
            'avg_accuracy': 0,
            'avg_auc': 0,
            'details': {}
        }
        
        # Pre-fetch BTC data
        btc_data = fetch_historical_data("BTCUSDT", SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
        
        for symbol in symbols:
            try:
                # Fetch and prepare data
                data = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
                if data is None or len(data) < 100:
                    raise ValueError("Insufficient data")
                
                data = clean_data(data)
                data = calculate_enhanced_indicators(data)
                
                # Prepare features and target
                features = [
                    'rsi', 'momentum', 'price_velocity', 'adx', 'trend_direction',
                    'liquidity_pool_high', 'liquidity_pool_low',
                    'bullish_ob', 'bearish_ob', 'bullish_fvg', 'bearish_fvg',
                    'vol_price_interaction', 'rsi_adx_combo', 'hour', 'day_of_week'
                ]
                
                data['target'] = (data['close'].shift(-5) > data['close'] * 1.005).astype(int)
                data = data.dropna(subset=features + ['target'])
                
                if len(data) < 100:
                    raise ValueError("Not enough data after processing")
                
                # Split data
                X = data[features]
                y = data['target']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = train_enhanced_model(X_train_scaled, y_train, X_test_scaled, y_test)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'auc': roc_auc_score(y_test, y_proba),
                    'feature_importances': dict(zip(features, model.named_estimators_['lgbm'].feature_importances_))
                }
                
                # Save results
                save_model_results(symbol, model, metrics)
                overall_metrics['successful_models'] += 1
                overall_metrics['avg_accuracy'] += metrics['accuracy']
                overall_metrics['avg_auc'] += metrics['auc']
                overall_metrics['details'][symbol] = metrics
                
            except Exception as e:
                logger.error(f"❌ Failed to train model for {symbol}: {str(e)}")
                overall_metrics['failed_models'] += 1
                overall_metrics['details'][symbol] = {'error': str(e)}
        
        # Calculate averages
        if overall_metrics['successful_models'] > 0:
            overall_metrics['avg_accuracy'] /= overall_metrics['successful_models']
            overall_metrics['avg_auc'] /= overall_metrics['successful_models']
        
        # Finalize
        training_status = "Enhanced Training Completed"
        last_training_time = datetime.now()
        last_training_metrics = overall_metrics
        
        # Send summary
        send_summary_report(overall_metrics, start_time)
        
    except Exception as e:
        training_status = "Enhanced Training Failed"
        training_error = str(e)
        logger.critical(f"❌ Critical error in training pipeline: {str(e)}")
        
    finally:
        cleanup_resources()

# ---------------------- Helper Functions ----------------------
def save_model_results(symbol: str, model: Any, metrics: dict):
    """Save model and results to database"""
    if not check_db_connection():
        return False
    
    try:
        model_binary = pickle.dumps(model)
        metrics_json = json.dumps(metrics)
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO enhanced_model_results 
                (model_name, accuracy, precision, recall, f1_score, auc_score, feature_importances)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (f"{BASE_ML_MODEL_NAME}_{symbol}", 
                 metrics['accuracy'], metrics['precision'], 
                 metrics['recall'], metrics['f1'], 
                 metrics['auc'], metrics['feature_importances']))
            
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save results for {symbol}: {str(e)}")
        if conn: conn.rollback()
        return False

def send_summary_report(metrics: dict, start_time: datetime):
    """Send training summary report"""
    duration = datetime.now() - start_time
    duration_str = str(duration).split('.')[0]
    
    message = (
        f"🚀 *Enhanced Model Training Complete*\n\n"
        f"📊 *Summary Metrics*\n"
        f"- Successful Models: {metrics['successful_models']}/{metrics['total_models']}\n"
        f"- Avg Accuracy: {metrics['avg_accuracy']:.4f}\n"
        f"- Avg AUC: {metrics['avg_auc']:.4f}\n"
        f"- Training Duration: {duration_str}\n\n"
        f"🔍 *Top Performing Models*\n"
    )
    
    # Add top 5 models by AUC
    top_models = sorted(
        [(k, v) for k, v in metrics['details'].items() if 'auc' in v],
        key=lambda x: x[1]['auc'],
        reverse=True
    )[:5]
    
    for symbol, m in top_models:
        message += f"- {symbol}: AUC={m['auc']:.4f}, F1={m['f1']:.4f}\n"
    
    if TELEGRAM_TOKEN and CHAT_ID:
        send_telegram_message(CHAT_ID, message)

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    logger.info("🚀 Starting Enhanced ML Training Pipeline")
    enhanced_training_pipeline()
    logger.info("🏁 Enhanced Training Pipeline Completed")