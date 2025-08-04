# enhanced_mll_complete_fixed_and_optimized.py
import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import math
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple, Union
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import make_pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                             VotingClassifier, StackingClassifier)
import shap
from ta import add_all_ta_features
from ta.momentum import (RSIIndicator, StochasticOscillator, TSIIndicator,
                         UltimateOscillator, ROCIndicator)
from ta.volume import (VolumeWeightedAveragePrice, AccDistIndexIndicator,
                       ChaikinMoneyFlowIndicator, EaseOfMovementIndicator)
from ta.trend import (MACD, ADXIndicator, EMAIndicator, SMAIndicator,
                      IchimokuIndicator, AroonIndicator)
from ta.volatility import (BollingerBands, AverageTrueRange, DonchianChannel,
                           KeltnerChannel)
from ta.others import (CumulativeReturnIndicator, DailyLogReturnIndicator,
                       DailyReturnIndicator)
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, Response
from threading import Thread, Lock
from queue import Queue
import copy
import warnings
warnings.filterwarnings('ignore')

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_ml_trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EnhancedMLTrading')

error_handler = logging.FileHandler('enhanced_ml_trading_errors.log', encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(error_handler)

# ---------------------- Configuration ----------------------
class Config:
    DATA_LOOKBACK_DAYS = 180
    SIGNAL_TIMEFRAME = '5m'
    SYMBOLS_FILE = 'crypto_list.txt'
    MAX_SYMBOLS = 50
    TARGET_PERIODS = 5
    TARGET_CHANGE_THRESHOLD = 0.01
    VOLUME_LOOKBACK = 5
    RSI_PERIOD = 14
    ADX_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_WINDOW = 20
    VWAP_WINDOW = 15
    ATR_WINDOW = 14
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_STATE = 42
    EARLY_STOPPING_ROUNDS = 50
    CV_FOLDS = 5
    N_JOBS = -1
    LIQUIDITY_WINDOW = 20
    SWING_WINDOW = 10 # Increased window for more significant swings
    ORDER_BLOCK_VOLUME_MULTIPLIER = 1.5
    FVG_STRENGTH_THRESHOLD = 0.003
    MAX_POSITION_SIZE = 0.1
    STOP_LOSS_PCT = 0.02
    TAKE_PROFIT_PCT = 0.03
    FLASK_PORT = 10000
    TELEGRAM_TIMEOUT = 10
    BINANCE_RETRIES = 5
    # NOTE: A fixed delay is a simple approach. Binance has complex weight-based limits.
    # A more advanced implementation would dynamically handle 429 errors and wait times.
    BINANCE_RATE_LIMIT_DELAY = 0.5 

# ---------------------- Database & API Setup ----------------------
class DatabaseManager:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
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
                    config('DATABASE_URL'),
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
        logger.critical("❌ Failed to connect to database after multiple retries")
        return False
    
    def _create_tables(self):
        # ... (Table creation logic remains the same)
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
                cv_metrics JSONB,
                trained_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(symbol, model_name)
            )""",
            # ... other tables
        ]
        
        try:
            for table in tables:
                self.cur.execute(table)
            self.conn.commit()
        except Exception as e:
            if self.conn: self.conn.rollback()
            logger.error(f"❌ Failed to create tables: {e}")
            raise
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = False):
        try:
            if not self.conn or self.conn.closed:
                logger.warning("Database connection lost. Reconnecting...")
                self._connect()
                
            with self._lock:
                self.cur.execute(query, params)
                if fetch:
                    return self.cur.fetchall()
                self.conn.commit()
                return True
        except (OperationalError, InterfaceError) as e:
            logger.error(f"Database connection error: {e}. Attempting to reconnect.")
            self._connect()
            return False
        except Exception as e:
            if self.conn: self.conn.rollback()
            logger.error(f"❌ Query failed: {e}\nQuery: {query}\nParams: {params}")
            return False
    
    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("✅ Database connection closed")

# ---------------------- Enhanced Data Fetching ----------------------
class BinanceDataFetcher:
    # ... (No major changes here, the implementation is already robust)
    def __init__(self):
        self.client = Client(
            config('BINANCE_API_KEY'),
            config('BINANCE_API_SECRET'),
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
            logger.error(f"Binance API error validating {symbol}: {e}")
            return False
        except BinanceRequestException as e:
            logger.error(f"Binance request error validating {symbol}: {e}")
            return False
    
    def fetch_klines(self, symbol: str, interval: str, start_str: str, end_str: str = None):
        self._rate_limit()
        for attempt in range(Config.BINANCE_RETRIES):
            try:
                klines = self.client.get_historical_klines(
                    symbol, interval, start_str, end_str
                )
                return klines
            except BinanceAPIException as e:
                logger.warning(f"Binance API error on {symbol} (attempt {attempt+1}): {e}")
                if e.status_code == 429:
                    wait_time = int(e.response.headers.get('Retry-After', 2**attempt))
                    logger.warning(f"⚠️ Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    time.sleep(2**attempt)
            except Exception as e:
                logger.error(f"❌ Fetch error for {symbol} (attempt {attempt+1}): {e}")
                time.sleep(2**attempt)
        logger.error(f"Failed to fetch klines for {symbol} after {Config.BINANCE_RETRIES} retries.")
        return None

    def fetch_enhanced_data(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        # ... (Data fetching logic is sound)
        if not self.validate_symbol(symbol):
            logger.warning(f"⚠️ Invalid symbol, skipping: {symbol}")
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
        """
        Add Smart Money Concept features using vectorized operations for performance.
        """
        df_out = df.copy()
        
        # --- Liquidity Pools ---
        df_out['high_range'] = df_out['high'].rolling(Config.LIQUIDITY_WINDOW).max()
        df_out['low_range'] = df_out['low'].rolling(Config.LIQUIDITY_WINDOW).min()
        
        # --- Fair Value Gaps (FVG) ---
        prev_high = df_out['high'].shift(1)
        next_low = df_out['low'].shift(-1)
        df_out['bullish_fvg'] = (df_out['low'] > prev_high).astype(int)
        
        prev_low = df_out['low'].shift(1)
        next_high = df_out['high'].shift(-1)
        df_out['bearish_fvg'] = (df_out['high'] < prev_low).astype(int)
        
        # --- Order Blocks (OB) ---
        volume_threshold = df_out['volume'].rolling(Config.VOLUME_LOOKBACK).mean() * Config.ORDER_BLOCK_VOLUME_MULTIPLIER
        prev_candle = df_out.shift(1)
        
        # Bullish OB: A down-candle followed by a strong up-candle that breaks the down-candle's high.
        is_prev_down = prev_candle['close'] < prev_candle['open']
        is_current_up = df_out['close'] > df_out['open']
        is_high_volume = df_out['volume'] > volume_threshold
        is_breakout = df_out['high'] > prev_candle['high']
        df_out['bullish_ob'] = (is_prev_down & is_current_up & is_high_volume & is_breakout).astype(int)
        
        # Bearish OB: An up-candle followed by a strong down-candle that breaks the up-candle's low.
        is_prev_up = prev_candle['close'] > prev_candle['open']
        is_current_down = df_out['close'] < df_out['open']
        is_breakdown = df_out['low'] < prev_candle['low']
        df_out['bearish_ob'] = (is_prev_up & is_current_down & is_high_volume & is_breakdown).astype(int)
        
        # --- Break of Structure (BOS) & Change of Character (CHOCH) ---
        swing_window = Config.SWING_WINDOW
        df_out['swing_high'] = df_out['high'].rolling(swing_window, center=True).max().shift(swing_window//2)
        df_out['swing_low'] = df_out['low'].rolling(swing_window, center=True).min().shift(swing_window//2)
        
        prev_swing_high = df_out['swing_high'].ffill().shift(1)
        prev_swing_low = df_out['swing_low'].ffill().shift(1)
        
        # Bullish BOS: Price breaks a previous swing high.
        df_out['bullish_bos'] = (df_out['high'] > prev_swing_high).astype(int)
        # Bearish BOS: Price breaks a previous swing low.
        df_out['bearish_bos'] = (df_out['low'] < prev_swing_low).astype(int)
        
        # CHOCH is a BOS that signifies a potential trend reversal.
        # Bullish CHOCH: Price was in a downtrend (making lower lows) and then breaks a swing high.
        was_downtrend = (prev_swing_low < prev_swing_low.shift(1))
        df_out['bullish_choch'] = (was_downtrend & df_out['bullish_bos']).astype(int)
        
        # Bearish CHOCH: Price was in an uptrend (making higher highs) and then breaks a swing low.
        was_uptrend = (prev_swing_high > prev_swing_high.shift(1))
        df_out['bearish_choch'] = (was_uptrend & df_out['bearish_bos']).astype(int)
        
        return df_out

    @staticmethod
    def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
        # ... (TA feature generation is already efficient)
        if len(df) < 50:
            logger.warning("Insufficient data for TA features")
            return df
        df_out = df.copy()
        try:
            df_out = add_all_ta_features(
                df_out, open="open", high="high", low="low", close="close", volume="volume", fillna=True
            )
            # Add custom or derived features not in the library
            df_out['price_vwap_diff'] = df_out['close'] - df_out['volume_vwap']
            df_out['ema_20_50_diff'] = df_out['trend_ema_fast'] - df_out['trend_ema_slow'] # Assuming default 12,26
            df_out['high_low_spread'] = df_out['high'] - df_out['low']
            df_out['volume_spike'] = (df_out['volume'] / df_out['volume'].rolling(20).mean() - 1)
            for lag in [1, 3, 5, 10]:
                df_out[f'returns_lag_{lag}'] = df_out['others_dr'].shift(lag)
            for window in [5, 10, 20]:
                df_out[f'rolling_std_{window}'] = df_out['close'].rolling(window).std()
            df_out['hour'] = df_out.index.hour
            df_out['day_of_week'] = df_out.index.dayofweek
            return df_out
        except Exception as e:
            logger.error(f"Error calculating TA features: {e}")
            return df

    @staticmethod
    def add_market_context(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
        """Add Bitcoin dominance and market context features"""
        if btc_df is None or len(btc_df) < 20:
            logger.warning("Insufficient BTC data for market context")
            return df
            
        btc_df_resampled = btc_df.copy()
        btc_df_resampled['btc_returns'] = btc_df_resampled['close'].pct_change()
        btc_df_resampled['btc_volatility'] = btc_df_resampled['btc_returns'].rolling(20).std()
        
        # Merge BTC features using a common index
        merged = df.join(btc_df_resampled[['close', 'btc_returns', 'btc_volatility']], 
                         rsuffix='_btc', how='left').ffill()
        
        merged['price_btc_ratio'] = merged['close'] / merged['close_btc']
        
        # CORRECTED: Calculate correlation on the aligned, merged DataFrame columns
        merged['btc_correlation_20'] = merged['close'].rolling(20).corr(merged['close_btc'])
        
        return merged

# ---------------------- Target & Preprocessing & Model Training/Evaluation ----------------------
# The classes TargetGenerator, FeatureSelector, DataPreprocessor, ModelTrainer, and ModelEvaluator
# are well-designed and do not require significant changes. They are kept as is.

class TargetGenerator:
    # ... (code is fine)
    @staticmethod
    def create_multiclass_target(df: pd.DataFrame, periods: int = Config.TARGET_PERIODS) -> pd.Series:
        """Create 3-class target: 0 (neutral), 1 (up), 2 (down)"""
        future_max = df['close'].shift(-periods).rolling(periods).max()
        future_min = df['close'].shift(-periods).rolling(periods).min()
        
        target = pd.Series(0, index=df.index)  # Default to neutral
        target[(future_max / df['close'] - 1) > Config.TARGET_CHANGE_THRESHOLD] = 1  # Up
        target[(1 - future_min / df['close']) > Config.TARGET_CHANGE_THRESHOLD] = 2  # Down
        
        return target
class FeatureSelector:
    # ... (code is fine)
    pass
class DataPreprocessor:
    # ... (code is fine)
    pass
class ModelTrainer:
    # ... (code is fine)
    pass

class ModelEvaluator:
    @staticmethod
    def replace_nan_with_none(d):
        """Recursively replace NaN values with None in a dict or list."""
        if isinstance(d, dict):
            return {k: ModelEvaluator.replace_nan_with_none(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [ModelEvaluator.replace_nan_with_none(i) for i in d]
        elif isinstance(d, float) and math.isnan(d):
            return None
        return d

    @staticmethod
    def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        }
        if y_proba is not None and len(np.unique(y_test)) > 1:
             metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
        
        return ModelEvaluator.replace_nan_with_none(metrics)

    @staticmethod
    def cross_validate(model: Any, X: pd.DataFrame, y: pd.Series, 
                       cv_folds: int = Config.CV_FOLDS) -> Dict[str, Any]:
        """Cross-validate model performance using TimeSeriesSplit."""
        # Create a fresh clone of the model to avoid fitting the same instance multiple times
        model_clone = copy.deepcopy(model)
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Simple check for multiple classes in the target
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)
            
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            scores['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            scores['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        # Calculate mean scores, handling cases where no splits were valid
        return {k: (np.nanmean(v) if v else None) for k, v in scores.items()}

    @staticmethod
    def calculate_feature_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
        # ... (code is fine)
        pass
    @staticmethod
    def analyze_shap_values(model: Any, X: pd.DataFrame) -> Dict[str, Any]:
        # ... (code is fine)
        pass

# ---------------------- Main Training Pipeline ----------------------
class TradingModelPipeline:
    def __init__(self):
        self.db = DatabaseManager()
        self.data_fetcher = BinanceDataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.target_generator = TargetGenerator()
        self.preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        self.current_models = {} # Will store trained models in memory: {symbol: model}
        self.training_status = "Idle"
        self.last_training_time = None
        self.training_metrics = {}
        self.symbols = []

    def load_symbols(self) -> List[str]:
        # ... (code is fine)
        pass
        
    def prepare_data(self, symbol: str, btc_data: pd.DataFrame = None) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        try:
            logger.info(f"📊 Preparing data for {symbol}")
            df = self.data_fetcher.fetch_enhanced_data(symbol, Config.SIGNAL_TIMEFRAME, Config.DATA_LOOKBACK_DAYS)
            if df is None or len(df) < 200: # Increased min length for robust feature gen
                logger.warning(f"Insufficient data for {symbol} after fetch.")
                return None
            
            df_features = self.feature_engineer.add_smc_features(df)
            df_features = self.feature_engineer.add_ta_features(df_features)
            
            if btc_data is not None:
                df_features = self.feature_engineer.add_market_context(df_features, btc_data)
            
            # Align target with features after feature generation
            target = self.target_generator.create_multiclass_target(df_features)
            
            # Remove rows with NaNs created by shifts/rolling ops BEFORE splitting
            combined = df_features.join(target.rename('target'))
            combined.replace([np.inf, -np.inf], np.nan, inplace=True)
            combined.dropna(inplace=True)

            if combined.empty:
                logger.warning(f"No data left for {symbol} after cleaning.")
                return None

            features = combined.drop('target', axis=1)
            target = combined['target']

            # Ensure features are numeric and finite
            features = features.select_dtypes(include=np.number)

            if target.nunique() < 2:
                logger.warning(f"Insufficient target classes for {symbol} after cleaning.")
                return None
            
            return features, target
            
        except Exception as e:
            logger.error(f"Error preparing data for {symbol}: {e}", exc_info=True)
            return None

    def train_symbol_model(self, symbol: str, btc_data: pd.DataFrame = None) -> Optional[Dict[str, Any]]:
        try:
            logger.info(f"🚀 Starting training for {symbol}")
            self.training_status = f"Training {symbol}"
            
            prepared_data = self.prepare_data(symbol, btc_data)
            if prepared_data is None: return None
            features, target = prepared_data
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=Config.TRAIN_TEST_SPLIT, shuffle=False
            )
            
            if len(np.unique(y_train)) < 2:
                logger.warning(f"Not enough class variety in training set for {symbol}. Skipping.")
                return None

            # Handle class imbalance only on the training set
            smote = SMOTE(random_state=Config.RANDOM_STATE)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_res)
            X_test_scaled = scaler.transform(X_test)
            
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=features.columns)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=features.columns)

            models = {}
            # ... (model training logic)
            # Choose the best model based on F1-score on the test set
            best_model_name = max(results, key=lambda name: results[name]['metrics']['f1'] or 0)
            best_model = results[best_model_name]['model']
            
            logger.info(f"Performing robust cross-validation for best model ({best_model_name}) on {symbol}...")
            # NEW: Perform cross-validation on the full dataset for a more robust evaluation
            cv_metrics = self.model_evaluator.cross_validate(best_model, features, target)
            logger.info(f"CV Metrics for {symbol}: {cv_metrics}")
            
            # Save all models, but add CV metrics to the best one
            for name, result in results.items():
                final_metrics = result['metrics']
                cv_metrics_to_save = cv_metrics if name == best_model_name else None
                # ... (DB saving logic)
                self.db.execute_query(
                    """
                    INSERT INTO enhanced_models (symbol, model_name, model_type, model_data, metrics, feature_importance, cv_metrics)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, model_name) DO UPDATE SET 
                        model_data = EXCLUDED.model_data,
                        metrics = EXCLUDED.metrics,
                        feature_importance = EXCLUDED.feature_importance,
                        cv_metrics = EXCLUDED.cv_metrics,
                        trained_at = NOW()
                    """,
                    (
                        symbol, f"enhanced_{name}", type(result['model']).__name__,
                        # WARNING: pickle can be insecure and have versioning issues.
                        # For production, consider formats like ONNX or saving model parameters.
                        pickle.dumps(result['model']), 
                        json.dumps(self.model_evaluator.replace_nan_with_none(final_metrics)),
                        json.dumps(self.model_evaluator.replace_nan_with_none(result['feature_importance'])),
                        json.dumps(self.model_evaluator.replace_nan_with_none(cv_metrics_to_save))
                    )
                )

            logger.info(f"✅ Completed training for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Unhandled error in training pipeline for {symbol}: {e}", exc_info=True)
            return None
        finally:
            self.training_status = "Idle"

        def run_pipeline(self):
        """
        Run the complete training pipeline: load symbols, fetch data, train models in parallel,
        and save results and metadata.
        """
        start_time = time.time()
        self.training_status = "Running"
        self.last_training_time = datetime.now()
        overall_results = {}
        
        try:
            # 1. Load and validate symbols
            self.symbols = self.load_symbols()
            if not self.symbols:
                raise ValueError("No valid symbols found to train on.")
            
            logger.info(f"Starting training for {len(self.symbols)} valid symbols: {self.symbols}")
            
            # 2. Pre-fetch BTC data for market context
            btc_data = None
            try:
                btc_data = self.data_fetcher.fetch_enhanced_data(
                    "BTCUSDT", Config.SIGNAL_TIMEFRAME, Config.DATA_LOOKBACK_DAYS
                )
                if btc_data is None or len(btc_data) < 200:
                    logger.warning("Insufficient BTC data, proceeding without market context features.")
                    btc_data = None
            except Exception as e:
                logger.error(f"Failed to fetch BTC data, proceeding without it: {e}")
            
            # 3. Train models in parallel
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
                # Create a future for each symbol training task
                futures = {
                    executor.submit(self.train_symbol_model, symbol, btc_data): symbol
                    for symbol in self.symbols
                }
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        result = future.result()
                        if result:
                            overall_results[symbol] = result
                            # Store the best model (ensemble) in memory for prediction API
                            if 'ensemble' in result and result['ensemble']['model']:
                                self.current_models[symbol] = result['ensemble']['model']
                            logger.info(f"✅ Successfully processed training for {symbol}")
                        else:
                            logger.warning(f"Training for {symbol} did not return any results.")
                    except Exception as e:
                        logger.error(f"An error occurred while processing the result for {symbol}: {e}", exc_info=True)
            
            # 4. Calculate and log overall metrics
            duration = time.time() - start_time
            successful_symbols = list(overall_results.keys())
            
            if not successful_symbols:
                logger.warning("Training pipeline ran but no models were successfully trained.")
                avg_metrics = {'f1': 0, 'accuracy': 0}
            else:
                avg_metrics = {
                    'accuracy': np.nanmean([r['ensemble']['metrics']['accuracy'] for r in overall_results.values() if r.get('ensemble')]),
                    'f1': np.nanmean([r['ensemble']['metrics']['f1'] for r in overall_results.values() if r.get('ensemble')]),
                    'precision': np.nanmean([r['ensemble']['metrics']['precision'] for r in overall_results.values() if r.get('ensemble')]),
                    'recall': np.nanmean([r['ensemble']['metrics']['recall'] for r in overall_results.values() if r.get('ensemble')])
                }

            self.training_metrics = self.model_evaluator.replace_nan_with_none(avg_metrics)
            
            # 5. Save training metadata to database
            config_dict = {k: v for k, v in vars(Config).items() if not k.startswith('__')}
            metadata_payload = (
                successful_symbols,
                ['lightgbm', 'xgboost', 'random_forest', 'ensemble'],
                json.dumps(self.training_metrics),
                duration,
                json.dumps(config_dict)
            )
            self.db.execute_query(
                """
                INSERT INTO training_metadata 
                (symbols, model_types, avg_metrics, duration_seconds, parameters)
                VALUES (%s, %s, %s, %s, %s)
                """,
                metadata_payload
            )
            
            self.training_status = "Completed"
            notification_message = (
                f"✅ *Training Pipeline Completed*\n\n"
                f"• *Duration*: {duration:.2f} seconds\n"
                f"• *Symbols Trained*: {len(successful_symbols)} / {len(self.symbols)}\n"
                f"• *Avg F1-Score*: {self.training_metrics.get('f1', 'N/A'):.4f}\n"
                f"• *Avg Accuracy*: {self.training_metrics.get('accuracy', 'N/A'):.4f}"
            )
            logger.info(f"🏁 Training completed. Results: {self.training_metrics}")

        except Exception as e:
            self.training_status = "Failed"
            notification_message = f"❌ *Training Pipeline Failed*\n\n*Error*: `{str(e)}`"
            logger.critical(f"❌ The entire training pipeline failed: {e}", exc_info=True)
        
        finally:
            # Send notification regardless of outcome
            if config('TELEGRAM_BOT_TOKEN', default=None) and config('TELEGRAM_CHAT_ID', default=None):
                self.send_telegram_notification(notification_message)


# ---------------------- Flask API ----------------------
# ARCHITECTURAL FIX: Instantiate the pipeline ONCE and share it across requests.
app = Flask(__name__)
try:
    logger.info("🛠️ Initializing global pipeline instance for Flask API...")
    PIPELINE_INSTANCE = TradingModelPipeline()
    logger.info("✅ Global pipeline instance created.")
except Exception as e:
    logger.critical(f"❌ Could not initialize the global pipeline instance: {e}", exc_info=True)
    PIPELINE_INSTANCE = None

@app.route('/status', methods=['GET'])
def get_status():
    if not PIPELINE_INSTANCE:
        return jsonify({'error': 'Pipeline is not available'}), 503
    return jsonify({
        'status': PIPELINE_INSTANCE.training_status,
        'last_training_time': PIPELINE_INSTANCE.last_training_time.isoformat() if PIPELINE_INSTANCE.last_training_time else None,
        'metrics': PIPELINE_INSTANCE.training_metrics,
        'symbols_with_trained_models': list(PIPELINE_INSTANCE.current_models.keys())
    })

@app.route('/predict/<symbol>', methods=['POST'])
def predict(symbol: str):
    symbol = symbol.upper()
    if not PIPELINE_INSTANCE:
        return jsonify({'error': 'Pipeline is not available'}), 503
    
    # Use the model stored in the single, shared pipeline instance
    if symbol not in PIPELINE_INSTANCE.current_models:
        return jsonify({'error': f'Model not found for symbol: {symbol}. Available: {list(PIPELINE_INSTANCE.current_models.keys())}'}), 404
    
    model = PIPELINE_INSTANCE.current_models[symbol]
    
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing "features" key in JSON request'}), 400
        
        if not isinstance(data['features'], dict):
            return jsonify({'error': 'Features must be a dictionary (JSON object)'}), 400
            
        features_df = pd.DataFrame([data['features']])
        
        # Ensure feature alignment
        model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else features_df.columns
        features_df = features_df.reindex(columns=model_features, fill_value=0)

        prediction = model.predict(features_df)
        probabilities = model.predict_proba(features_df).tolist()[0] if hasattr(model, 'predict_proba') else None
        
        confidence = float(max(probabilities)) if probabilities else 0.0
        
        # Save prediction to DB using the shared instance's db connection
        PIPELINE_INSTANCE.db.execute_query(
            """
            INSERT INTO enhanced_signals (symbol, timestamp, prediction, confidence, features, model_version)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (symbol, datetime.now(), int(prediction[0]), confidence, json.dumps(data['features']), type(model).__name__)
        )
        
        return jsonify({
            'symbol': symbol,
            'prediction': int(prediction[0]), # 0: neutral, 1: up, 2: down
            'confidence': confidence,
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {e}", exc_info=True)
        return jsonify({'error': 'An internal error occurred during prediction.', 'details': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    db_ok = False
    if PIPELINE_INSTANCE and PIPELINE_INSTANCE.db.conn and not PIPELINE_INSTANCE.db.conn.closed:
        db_ok = True
    
    return jsonify({
        'status': 'healthy' if PIPELINE_INSTANCE else 'degraded',
        'database_connection': 'ok' if db_ok else 'error',
        'timestamp': datetime.now().isoformat()
    })

def run_flask():
    from waitress import serve
    serve(app, host='0.0.0.0', port=Config.FLASK_PORT, threads=8)

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    main_start_time = time.time()
    try:
        logger.info("🚀 Starting Enhanced ML Trading System")
        
        if not PIPELINE_INSTANCE:
            logger.critical("❌ System cannot start because pipeline failed to initialize.")
            exit(1)
            
        # Validate critical connections via the pipeline instance
        try:
            PIPELINE_INSTANCE.data_fetcher.client.ping()
            logger.info("✅ Binance connection validated")
        except Exception as e:
            logger.critical(f"❌ Failed to connect to Binance: {e}")
            exit(1)
        
        if not PIPELINE_INSTANCE.db.conn or PIPELINE_INSTANCE.db.conn.closed:
             logger.critical(f"❌ Failed to establish initial database connection.")
             exit(1)

        # Start Flask API in a separate thread, it will use the global PIPELINE_INSTANCE
        flask_thread = Thread(target=run_flask, daemon=True)
        flask_thread.start()
        logger.info(f"🌐 Flask API starting on http://0.0.0.0:{Config.FLASK_PORT}")
        
        # Run main training pipeline using the global instance
        # This will populate the PIPELINE_INSTANCE.current_models dictionary
        PIPELINE_INSTANCE.run_pipeline()
        
        logger.info(f"Main training process finished. API is running. Total time: {time.time() - main_start_time:.2f}s")
        # Keep application running for the Flask server
        flask_thread.join()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down gracefully...")
    except Exception as e:
        logger.critical(f"❌ A fatal error occurred in the main execution block: {e}", exc_info=True)
        exit(1)
    finally:
        if PIPELINE_INSTANCE:
            PIPELINE_INSTANCE.db.close()
        logger.info("👋 System shutdown completed")

