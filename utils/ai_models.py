"""
Professional AI Models Module
Advanced machine learning models for market prediction and signal generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """AI prediction result structure"""
    symbol: str
    prediction: str  # BUY/SELL/HOLD
    confidence: float
    target_price: float
    probability: float
    timeframe: str
    model_scores: Dict[str, float]
    feature_importance: Dict[str, float]
    reasoning: str

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    win_rate: float
    avg_return: float

class FeatureEngineer:
    """Advanced feature engineering for financial data"""
    
    def __init__(self):
        """Initialize feature engineer"""
        self.technical_features = [
            'rsi', 'macd', 'bb_position', 'volume_ratio', 'price_momentum',
            'volatility', 'support_resistance_score', 'trend_strength'
        ]
        
        self.fundamental_features = [
            'pe_ratio', 'pb_ratio', 'debt_equity', 'roe', 'revenue_growth',
            'profit_margin', 'asset_turnover', 'current_ratio'
        ]
        
        self.market_features = [
            'market_momentum', 'sector_performance', 'fii_flow', 'dii_flow',
            'vix_level', 'correlation_with_nifty', 'relative_strength'
        ]
        
        self.news_features = [
            'news_sentiment', 'news_volume', 'news_impact_score',
            'earnings_surprise', 'analyst_revisions'
        ]
    
    def create_features(self, price_data: pd.DataFrame, technical_analysis: Dict,
                       fundamental_data: Dict, news_sentiment: Dict,
                       market_data: Dict) -> pd.DataFrame:
        """Create comprehensive feature set"""
        try:
            features_df = pd.DataFrame()
            
            # Technical features
            features_df = self._add_technical_features(features_df, price_data, technical_analysis)
            
            # Price-based features
            features_df = self._add_price_features(features_df, price_data)
            
            # Fundamental features
            features_df = self._add_fundamental_features(features_df, fundamental_data)
            
            # Market features
            features_df = self._add_market_features(features_df, market_data)
            
            # News sentiment features
            features_df = self._add_news_features(features_df, news_sentiment)
            
            # Interaction features
            features_df = self._add_interaction_features(features_df)
            
            # Clean and validate features
            features_df = self._clean_features(features_df)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return pd.DataFrame()
    
    def _add_technical_features(self, df: pd.DataFrame, price_data: pd.DataFrame,
                               technical_analysis: Dict) -> pd.DataFrame:
        """Add technical analysis features"""
        try:
            # RSI features
            if 'momentum_analysis' in technical_analysis:
                momentum = technical_analysis['momentum_analysis']
                df['rsi'] = momentum.get('rsi', 50)
                df['rsi_oversold'] = 1 if df['rsi'].iloc[-1] < 30 else 0
                df['rsi_overbought'] = 1 if df['rsi'].iloc[-1] > 70 else 0
            
            # MACD features
            if 'trend_analysis' in technical_analysis:
                trend = technical_analysis['trend_analysis']
                df['macd'] = trend.get('macd', 0)
                df['macd_signal'] = trend.get('macd_signal', 0)
                df['macd_histogram'] = trend.get('macd_histogram', 0)
                df['macd_bullish'] = 1 if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else 0
            
            # Bollinger Bands features
            if 'volatility_analysis' in technical_analysis:
                volatility = technical_analysis['volatility_analysis']
                df['bb_position'] = volatility.get('bb_position', 0.5)
                df['bb_squeeze'] = 1 if volatility.get('bb_width', 1) < 0.05 else 0
            
            # Volume features
            if 'volume_analysis' in technical_analysis:
                volume = technical_analysis['volume_analysis']
                df['volume_ratio'] = volume.get('volume_ratio', 1.0)
                df['high_volume'] = 1 if df['volume_ratio'].iloc[-1] > 1.5 else 0
            
            # Price momentum
            if len(price_data) >= 20:
                df['price_momentum_5'] = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[-6] - 1) * 100
                df['price_momentum_10'] = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[-11] - 1) * 100
                df['price_momentum_20'] = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[-21] - 1) * 100
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical features: {str(e)}")
            return df
    
    def _add_price_features(self, df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        try:
            if len(price_data) < 2:
                return df
            
            # Price changes
            df['price_change'] = price_data['Close'].pct_change().iloc[-1] * 100
            df['price_change_abs'] = abs(df['price_change'].iloc[-1])
            
            # Volatility measures
            if len(price_data) >= 20:
                returns = price_data['Close'].pct_change().dropna()
                df['volatility_20'] = returns.tail(20).std() * np.sqrt(252) * 100
                df['volatility_5'] = returns.tail(5).std() * np.sqrt(252) * 100
                df['vol_ratio'] = df['volatility_5'].iloc[-1] / df['volatility_20'].iloc[-1]
            
            # Price levels
            current_price = price_data['Close'].iloc[-1]
            if len(price_data) >= 50:
                high_52w = price_data['High'].tail(252).max() if len(price_data) >= 252 else price_data['High'].max()
                low_52w = price_data['Low'].tail(252).min() if len(price_data) >= 252 else price_data['Low'].min()
                df['price_position'] = (current_price - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5
            
            # Gap analysis
            if len(price_data) >= 2:
                prev_close = price_data['Close'].iloc[-2]
                current_open = price_data['Open'].iloc[-1] if 'Open' in price_data.columns else current_price
                df['gap_pct'] = ((current_open - prev_close) / prev_close) * 100
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding price features: {str(e)}")
            return df
    
    def _add_fundamental_features(self, df: pd.DataFrame, fundamental_data: Dict) -> pd.DataFrame:
        """Add fundamental analysis features"""
        try:
            # Financial ratios
            df['pe_ratio'] = fundamental_data.get('pe_ratio', 20)
            df['pb_ratio'] = fundamental_data.get('pb_ratio', 2)
            df['debt_equity'] = fundamental_data.get('debt_equity', 0.5)
            df['roe'] = fundamental_data.get('roe', 15)
            df['revenue_growth'] = fundamental_data.get('revenue_growth', 10)
            df['profit_margin'] = fundamental_data.get('profit_margin', 10)
            
            # Valuation indicators
            df['undervalued'] = 1 if df['pe_ratio'].iloc[-1] < 15 else 0
            df['high_growth'] = 1 if df['revenue_growth'].iloc[-1] > 20 else 0
            df['profitable'] = 1 if df['profit_margin'].iloc[-1] > 10 else 0
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding fundamental features: {str(e)}")
            return df
    
    def _add_market_features(self, df: pd.DataFrame, market_data: Dict) -> pd.DataFrame:
        """Add market-wide features"""
        try:
            # Market sentiment
            df['market_momentum'] = market_data.get('nifty_momentum', 0)
            df['market_bullish'] = 1 if df['market_momentum'].iloc[-1] > 0 else 0
            
            # FII/DII flows
            df['fii_flow'] = market_data.get('fii_net_flow', 0)
            df['dii_flow'] = market_data.get('dii_net_flow', 0)
            df['institutional_bullish'] = 1 if (df['fii_flow'].iloc[-1] + df['dii_flow'].iloc[-1]) > 1000 else 0
            
            # VIX level
            df['vix_level'] = market_data.get('vix', 20)
            df['low_volatility'] = 1 if df['vix_level'].iloc[-1] < 15 else 0
            df['high_volatility'] = 1 if df['vix_level'].iloc[-1] > 25 else 0
            
            # Sector performance
            df['sector_performance'] = market_data.get('sector_momentum', 0)
            df['sector_outperform'] = 1 if df['sector_performance'].iloc[-1] > 2 else 0
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding market features: {str(e)}")
            return df
    
    def _add_news_features(self, df: pd.DataFrame, news_sentiment: Dict) -> pd.DataFrame:
        """Add news sentiment features"""
        try:
            df['news_sentiment'] = news_sentiment.get('sentiment_score', 0)
            df['news_impact'] = news_sentiment.get('impact_score', 0)
            df['news_volume'] = news_sentiment.get('news_count', 0)
            
            # News indicators
            df['positive_news'] = 1 if df['news_sentiment'].iloc[-1] > 0.2 else 0
            df['negative_news'] = 1 if df['news_sentiment'].iloc[-1] < -0.2 else 0
            df['high_impact_news'] = 1 if df['news_impact'].iloc[-1] > 5 else 0
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding news features: {str(e)}")
            return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between different categories"""
        try:
            # Technical + Fundamental interactions
            if 'rsi' in df.columns and 'pe_ratio' in df.columns:
                df['rsi_pe_interaction'] = df['rsi'] * (1 / df['pe_ratio'])
            
            # Momentum + Volume interaction
            if 'price_momentum_5' in df.columns and 'volume_ratio' in df.columns:
                df['momentum_volume'] = df['price_momentum_5'] * df['volume_ratio']
            
            # News + Technical interaction
            if 'news_sentiment' in df.columns and 'rsi' in df.columns:
                df['news_rsi_interaction'] = df['news_sentiment'] * (50 - abs(df['rsi'] - 50))
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding interaction features: {str(e)}")
            return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        try:
            # Fill missing values
            df = df.fillna(0)
            
            # Remove infinite values
            df = df.replace([np.inf, -np.inf], 0)
            
            # Ensure all features are numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning features: {str(e)}")
            return df

class EnsemblePredictor:
    """Ensemble machine learning model for stock prediction"""
    
    def __init__(self):
        """Initialize ensemble predictor"""
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_history = []
        self.is_trained = False
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different ML models"""
        try:
            # Random Forest
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            # XGBoost
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            # Gradient Boosting
            self.models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            
            # Logistic Regression
            self.models['logistic_regression'] = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
            
            # SVM
            self.models['svm'] = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
            
            # Ensemble Voting Classifier
            self.models['ensemble'] = VotingClassifier(
                estimators=[
                    ('rf', self.models['random_forest']),
                    ('xgb', self.models['xgboost']),
                    ('gb', self.models['gradient_boosting'])
                ],
                voting='soft'
            )
            
            # Initialize scalers
            for model_name in self.models.keys():
                self.scalers[model_name] = StandardScaler()
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
    
    def train_models(self, features_df: pd.DataFrame, labels: pd.Series, 
                    validation_split: float = 0.2) -> Dict[str, ModelPerformance]:
        """Train all models with cross-validation"""
        try:
            # Prepare data
            X = features_df.values
            y = labels.values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            performance_results = {}
            
            for model_name, model in self.models.items():
                try:
                    logger.info(f"Training {model_name}...")
                    
                    # Scale features
                    scaler = self.scalers[model_name]
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)
                    
                    # Calculate performance metrics
                    performance = self._calculate_performance(y_test, y_pred, y_pred_proba)
                    performance.model_name = model_name
                    performance_results[model_name] = performance
                    
                    # Feature importance (for tree-based models)
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[model_name] = dict(zip(
                            features_df.columns, model.feature_importances_
                        ))
                    
                    logger.info(f"{model_name} training completed. Accuracy: {performance.accuracy:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    continue
            
            self.is_trained = True
            return performance_results
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            return {}
    
    def predict(self, features_df: pd.DataFrame, symbol: str) -> PredictionResult:
        """Make prediction using ensemble of models"""
        try:
            if not self.is_trained:
                logger.warning("Models not trained. Using default prediction.")
                return self._default_prediction(symbol)
            
            X = features_df.values
            predictions = {}
            probabilities = {}
            
            # Get predictions from all models
            for model_name, model in self.models.items():
                try:
                    scaler = self.scalers[model_name]
                    X_scaled = scaler.transform(X)
                    
                    # Predict
                    pred = model.predict(X_scaled)[0]
                    pred_proba = model.predict_proba(X_scaled)[0]
                    
                    predictions[model_name] = pred
                    probabilities[model_name] = max(pred_proba)
                    
                except Exception as e:
                    logger.error(f"Error predicting with {model_name}: {str(e)}")
                    continue
            
            # Ensemble prediction
            if predictions:
                # Weighted voting based on historical performance
                weights = self._get_model_weights()
                
                weighted_votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
                confidence_scores = {}
                
                for model_name, pred in predictions.items():
                    weight = weights.get(model_name, 1.0)
                    prob = probabilities.get(model_name, 0.5)
                    
                    weighted_votes[pred] += weight * prob
                    confidence_scores[model_name] = prob
                
                # Final prediction
                final_prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]
                final_confidence = max(weighted_votes.values()) / sum(weights.values()) * 100
                
                # Calculate target price (simplified)
                current_price = self._get_current_price(symbol)
                target_price = self._calculate_target_price(final_prediction, current_price, features_df)
                
                # Get feature importance for reasoning
                reasoning = self._generate_reasoning(final_prediction, features_df, confidence_scores)
                
                return PredictionResult(
                    symbol=symbol,
                    prediction=final_prediction,
                    confidence=min(final_confidence, 95.0),
                    target_price=target_price,
                    probability=final_confidence,
                    timeframe="5-10 days",
                    model_scores=confidence_scores,
                    feature_importance=self.feature_importance.get('ensemble', {}),
                    reasoning=reasoning
                )
            
            return self._default_prediction(symbol)
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return self._default_prediction(symbol)
    
    def _calculate_performance(self, y_true, y_pred, y_pred_proba) -> ModelPerformance:
        """Calculate comprehensive model performance metrics"""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Calculate financial metrics (simplified)
            # In practice, these would use actual returns data
            win_rate = accuracy  # Simplified
            avg_return = precision * 10  # Simplified percentage return
            sharpe_ratio = (avg_return - 5) / 15 if avg_return > 5 else 0  # Simplified Sharpe
            
            return ModelPerformance(
                model_name="",
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                avg_return=avg_return
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance: {str(e)}")
            return ModelPerformance("", 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.0)
    
    def _get_model_weights(self) -> Dict[str, float]:
        """Get model weights based on historical performance"""
        # Default equal weights
        default_weights = {model_name: 1.0 for model_name in self.models.keys()}
        
        # In production, this would use actual performance history
        # For now, give higher weight to ensemble and XGBoost
        default_weights['ensemble'] = 1.5
        default_weights['xgboost'] = 1.3
        default_weights['random_forest'] = 1.2
        
        return default_weights
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for target calculation"""
        # This would fetch from data source
        # For now, using default values
        default_prices = {
            'RELIANCE': 2750.0,
            'TCS': 3890.0,
            'INFY': 1678.0,
            'HDFCBANK': 1654.0,
            'ICICIBANK': 1145.0
        }
        return default_prices.get(symbol, 1000.0)
    
    def _calculate_target_price(self, prediction: str, current_price: float, 
                               features_df: pd.DataFrame) -> float:
        """Calculate target price based on prediction"""
        try:
            # Get momentum and volatility indicators
            momentum = features_df.get('price_momentum_5', pd.Series([0])).iloc[0]
            volatility = features_df.get('volatility_20', pd.Series([20])).iloc[0]
            
            # Calculate expected move based on volatility
            expected_move = (volatility / 100) * current_price * np.sqrt(10/365)  # 10-day move
            
            if prediction == 'BUY':
                # Target 2x expected move up, minimum 3%
                target_move = max(expected_move * 2, current_price * 0.03)
                return current_price + target_move
            elif prediction == 'SELL':
                # Target 2x expected move down, minimum 3%
                target_move = max(expected_move * 2, current_price * 0.03)
                return current_price - target_move
            else:  # HOLD
                return current_price
                
        except Exception as e:
            logger.error(f"Error calculating target price: {str(e)}")
            # Default target: 5% move
            if prediction == 'BUY':
                return current_price * 1.05
            elif prediction == 'SELL':
                return current_price * 0.95
            else:
                return current_price
    
    def _generate_reasoning(self, prediction: str, features_df: pd.DataFrame, 
                           confidence_scores: Dict[str, float]) -> str:
        """Generate human-readable reasoning for the prediction"""
        try:
            reasoning_parts = []
            
            # Technical reasoning
            if 'rsi' in features_df.columns:
                rsi = features_df['rsi'].iloc[0]
                if rsi < 30:
                    reasoning_parts.append("RSI indicates oversold conditions")
                elif rsi > 70:
                    reasoning_parts.append("RSI indicates overbought conditions")
            
            # Momentum reasoning
            if 'price_momentum_5' in features_df.columns:
                momentum = features_df['price_momentum_5'].iloc[0]
                if momentum > 3:
                    reasoning_parts.append("Strong positive momentum")
                elif momentum < -3:
                    reasoning_parts.append("Strong negative momentum")
            
            # Volume reasoning
            if 'volume_ratio' in features_df.columns:
                vol_ratio = features_df['volume_ratio'].iloc[0]
                if vol_ratio > 1.5:
                    reasoning_parts.append("Above-average volume supports move")
            
            # News reasoning
            if 'news_sentiment' in features_df.columns:
                news_sent = features_df['news_sentiment'].iloc[0]
                if news_sent > 0.3:
                    reasoning_parts.append("Positive news sentiment")
                elif news_sent < -0.3:
                    reasoning_parts.append("Negative news sentiment")
            
            # Model confidence
            avg_confidence = np.mean(list(confidence_scores.values()))
            if avg_confidence > 0.8:
                reasoning_parts.append("High model consensus")
            elif avg_confidence > 0.6:
                reasoning_parts.append("Moderate model consensus")
            
            if reasoning_parts:
                return ". ".join(reasoning_parts) + "."
            else:
                return f"Model consensus suggests {prediction.lower()} signal with {avg_confidence:.0%} confidence."
                
        except Exception as e:
            logger.error(f"Error generating reasoning: {str(e)}")
            return f"{prediction} signal generated by AI analysis."
    
    def _default_prediction(self, symbol: str) -> PredictionResult:
        """Generate default prediction when models aren't available"""
        return PredictionResult(
            symbol=symbol,
            prediction='HOLD',
            confidence=50.0,
            target_price=self._get_current_price(symbol),
            probability=50.0,
            timeframe="5-10 days",
            model_scores={'default': 0.5},
            feature_importance={},
            reasoning="Default prediction - insufficient training data."
        )
    
    def update_performance(self, symbol: str, prediction: str, actual_outcome: str, 
                          return_pct: float):
        """Update model performance based on actual outcomes"""
        try:
            performance_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'prediction': prediction,
                'actual_outcome': actual_outcome,
                'return_pct': return_pct,
                'correct': prediction == actual_outcome
            }
            
            self.performance_history.append(performance_record)
            
            # Keep only last 1000 records
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            logger.info(f"Updated performance: {symbol} - {prediction} vs {actual_outcome}")
            
        except Exception as e:
            logger.error(f"Error updating performance: {str(e)}")
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get overall performance summary"""
        try:
            if not self.performance_history:
                return {'accuracy': 0.5, 'avg_return': 0.0, 'total_predictions': 0}
            
            df = pd.DataFrame(self.performance_history)
            
            accuracy = df['correct'].mean()
            avg_return = df['return_pct'].mean()
            total_predictions = len(df)
            
            # Recent performance (last 50 predictions)
            recent_df = df.tail(50)
            recent_accuracy = recent_df['correct'].mean()
            recent_return = recent_df['return_pct'].mean()
            
            return {
                'accuracy': accuracy,
                'avg_return': avg_return,
                'total_predictions': total_predictions,
                'recent_accuracy': recent_accuracy,
                'recent_return': recent_return
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {'accuracy': 0.5, 'avg_return': 0.0, 'total_predictions': 0}

class AIPredictor:
    """Main AI Predictor class that combines all components"""
    
    def __init__(self):
        """Initialize AI predictor"""
        self.feature_engineer = FeatureEngineer()
        self.ensemble_predictor = EnsemblePredictor()
        self.models_dir = "models"
        
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load pre-trained models if available
        self._load_models()
    
    def generate_prediction(self, symbol: str, price_data: pd.DataFrame,
                           technical_analysis: Dict, fundamental_data: Dict,
                           news_sentiment: Dict, market_data: Dict) -> PredictionResult:
        """Generate comprehensive AI prediction"""
        try:
            # Create features
            features_df = self.feature_engineer.create_features(
                price_data, technical_analysis, fundamental_data,
                news_sentiment, market_data
            )
            
            if features_df.empty:
                logger.warning(f"No features created for {symbol}")
                return self._create_fallback_prediction(symbol, price_data, technical_analysis)
            
            # Generate prediction
            prediction = self.ensemble_predictor.predict(features_df, symbol)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {str(e)}")
            return self._create_fallback_prediction(symbol, price_data, technical_analysis)
    
    def _create_fallback_prediction(self, symbol: str, price_data: pd.DataFrame, 
                                   technical_analysis: Dict) -> PredictionResult:
        """Create fallback prediction based on simple technical analysis"""
        try:
            current_price = price_data['Close'].iloc[-1] if not price_data.empty else 1000.0
            
            # Simple technical score
            technical_score = technical_analysis.get('technical_score', 50.0)
            
            if technical_score > 65:
                prediction = 'BUY'
                target_price = current_price * 1.05
                confidence = min(technical_score, 85.0)
            elif technical_score < 35:
                prediction = 'SELL'
                target_price = current_price * 0.95
                confidence = min(100 - technical_score, 85.0)
            else:
                prediction = 'HOLD'
                target_price = current_price
                confidence = 50.0
            
            return PredictionResult(
                symbol=symbol,
                prediction=prediction,
                confidence=confidence,
                target_price=target_price,
                probability=confidence,
                timeframe="5-10 days",
                model_scores={'technical_fallback': confidence/100},
                feature_importance={},
                reasoning=f"Technical analysis suggests {prediction.lower()} with {confidence:.0f}% confidence."
            )
            
        except Exception as e:
            logger.error(f"Error creating fallback prediction: {str(e)}")
            return PredictionResult(
                symbol=symbol,
                prediction='HOLD',
                confidence=50.0,
                target_price=1000.0,
                probability=50.0,
                timeframe="5-10 days",
                model_scores={'default': 0.5},
                feature_importance={},
                reasoning="Default prediction due to analysis error."
            )
    
    def train_with_historical_data(self, historical_data: List[Dict]):
        """Train models with historical data"""
        try:
            logger.info("Training AI models with historical data...")
            
            # Process historical data
            features_list = []
            labels_list = []
            
            for data_point in historical_data:
                # Extract features and labels from historical data
                # This would be implemented based on available historical data format
                pass
            
            if features_list and labels_list:
                features_df = pd.DataFrame(features_list)
                labels = pd.Series(labels_list)
                
                # Train models
                performance = self.ensemble_predictor.train_models(features_df, labels)
                
                # Save models
                self._save_models()
                
                logger.info("AI models training completed")
                return performance
            else:
                logger.warning("No valid historical data for training")
                return {}
                
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return {}
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            models_path = os.path.join(self.models_dir, "ensemble_models.joblib")
            scalers_path = os.path.join(self.models_dir, "scalers.joblib")
            
            joblib.dump(self.ensemble_predictor.models, models_path)
            joblib.dump(self.ensemble_predictor.scalers, scalers_path)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            models_path = os.path.join(self.models_dir, "ensemble_models.joblib")
            scalers_path = os.path.join(self.models_dir, "scalers.joblib")
            
            if os.path.exists(models_path) and os.path.exists(scalers_path):
                self.ensemble_predictor.models = joblib.load(models_path)
                self.ensemble_predictor.scalers = joblib.load(scalers_path)
                self.ensemble_predictor.is_trained = True
                
                logger.info("Pre-trained models loaded successfully")
            else:
                logger.info("No pre-trained models found")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def get_model_performance(self) -> Dict[str, any]:
        """Get current model performance metrics"""
        return self.ensemble_predictor.get_performance_summary()

# Export main classes
__all__ = ['AIPredictor', 'PredictionResult', 'ModelPerformance', 'FeatureEngineer', 'EnsemblePredictor']
