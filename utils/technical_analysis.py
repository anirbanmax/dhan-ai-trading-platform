"""
Professional Technical Analysis Module
Comprehensive technical indicators and pattern recognition
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import talib
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TechnicalSignal:
    """Technical analysis signal structure"""
    indicator: str
    signal: str  # BUY/SELL/NEUTRAL
    strength: float  # 0-100
    value: float
    description: str

@dataclass
class SupportResistance:
    """Support and resistance levels"""
    level: float
    level_type: str  # SUPPORT/RESISTANCE
    strength: int  # Number of touches
    last_touch: str  # Date of last touch

class TechnicalAnalyzer:
    """Professional Technical Analysis Engine"""
    
    def __init__(self):
        """Initialize technical analyzer"""
        self.indicators = {
            'trend': ['SMA', 'EMA', 'MACD', 'PSAR'],
            'momentum': ['RSI', 'STOCH', 'WILLIAMS_R', 'ROC'],
            'volatility': ['BBANDS', 'ATR', 'KELTNER'],
            'volume': ['OBV', 'AD', 'CHAIKIN_MF', 'VWAP']
        }
    
    def analyze_stock(self, data: pd.DataFrame) -> Dict[str, any]:
        """Comprehensive technical analysis of stock data"""
        try:
            # Ensure we have enough data
            if len(data) < 50:
                logger.warning("Insufficient data for comprehensive analysis")
                return self._basic_analysis(data)
            
            analysis = {
                'trend_analysis': self._analyze_trend(data),
                'momentum_analysis': self._analyze_momentum(data),
                'volatility_analysis': self._analyze_volatility(data),
                'volume_analysis': self._analyze_volume(data),
                'support_resistance': self._find_support_resistance(data),
                'pattern_recognition': self._detect_patterns(data),
                'overall_signal': None,
                'technical_score': 0.0
            }
            
            # Calculate overall signal and score
            analysis['overall_signal'], analysis['technical_score'] = self._calculate_overall_signal(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {str(e)}")
            return self._basic_analysis(data)
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, any]:
        """Analyze trend indicators"""
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        
        try:
            # Simple Moving Averages
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            sma_200 = talib.SMA(close, timeperiod=200)
            
            # Exponential Moving Averages
            ema_12 = talib.EMA(close, timeperiod=12)
            ema_26 = talib.EMA(close, timeperiod=26)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            
            # Parabolic SAR
            sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            
            # ADX for trend strength
            adx = talib.ADX(high, low, close, timeperiod=14)
            
            current_price = close[-1]
            
            # Trend signals
            signals = []
            
            # Moving Average Analysis
            if current_price > sma_20[-1]:
                signals.append(TechnicalSignal('SMA_20', 'BUY', 60, sma_20[-1], 'Price above 20-day SMA'))
            else:
                signals.append(TechnicalSignal('SMA_20', 'SELL', 40, sma_20[-1], 'Price below 20-day SMA'))
            
            # Golden Cross / Death Cross
            if sma_20[-1] > sma_50[-1] and sma_20[-2] <= sma_50[-2]:
                signals.append(TechnicalSignal('GOLDEN_CROSS', 'BUY', 85, sma_20[-1], 'Golden Cross - SMA 20 crossed above SMA 50'))
            elif sma_20[-1] < sma_50[-1] and sma_20[-2] >= sma_50[-2]:
                signals.append(TechnicalSignal('DEATH_CROSS', 'SELL', 85, sma_20[-1], 'Death Cross - SMA 20 crossed below SMA 50'))
            
            # MACD Analysis
            if macd[-1] > macd_signal[-1] and macd[-2] <= macd_signal[-2]:
                signals.append(TechnicalSignal('MACD', 'BUY', 75, macd[-1], 'MACD bullish crossover'))
            elif macd[-1] < macd_signal[-1] and macd[-2] >= macd_signal[-2]:
                signals.append(TechnicalSignal('MACD', 'SELL', 75, macd[-1], 'MACD bearish crossover'))
            
            # Parabolic SAR
            if current_price > sar[-1]:
                signals.append(TechnicalSignal('PSAR', 'BUY', 70, sar[-1], 'Price above Parabolic SAR'))
            else:
                signals.append(TechnicalSignal('PSAR', 'SELL', 70, sar[-1], 'Price below Parabolic SAR'))
            
            return {
                'sma_20': sma_20[-1] if not np.isnan(sma_20[-1]) else None,
                'sma_50': sma_50[-1] if not np.isnan(sma_50[-1]) else None,
                'sma_200': sma_200[-1] if not np.isnan(sma_200[-1]) else None,
                'ema_12': ema_12[-1] if not np.isnan(ema_12[-1]) else None,
                'ema_26': ema_26[-1] if not np.isnan(ema_26[-1]) else None,
                'macd': macd[-1] if not np.isnan(macd[-1]) else None,
                'macd_signal': macd_signal[-1] if not np.isnan(macd_signal[-1]) else None,
                'macd_histogram': macd_hist[-1] if not np.isnan(macd_hist[-1]) else None,
                'sar': sar[-1] if not np.isnan(sar[-1]) else None,
                'adx': adx[-1] if not np.isnan(adx[-1]) else None,
                'signals': signals
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            return {'signals': []}
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, any]:
        """Analyze momentum indicators"""
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        
        try:
            # RSI
            rsi = talib.RSI(close, timeperiod=14)
            
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            
            # Williams %R
            willr = talib.WILLR(high, low, close, timeperiod=14)
            
            # Rate of Change
            roc = talib.ROC(close, timeperiod=10)
            
            # CCI
            cci = talib.CCI(high, low, close, timeperiod=14)
            
            signals = []
            
            # RSI Analysis
            current_rsi = rsi[-1]
            if current_rsi < 30:
                signals.append(TechnicalSignal('RSI', 'BUY', 80, current_rsi, 'RSI oversold - potential bounce'))
            elif current_rsi > 70:
                signals.append(TechnicalSignal('RSI', 'SELL', 80, current_rsi, 'RSI overbought - potential correction'))
            elif 40 <= current_rsi <= 60:
                signals.append(TechnicalSignal('RSI', 'NEUTRAL', 50, current_rsi, 'RSI in neutral zone'))
            
            # Stochastic Analysis
            if slowk[-1] < 20 and slowd[-1] < 20:
                signals.append(TechnicalSignal('STOCH', 'BUY', 75, slowk[-1], 'Stochastic oversold'))
            elif slowk[-1] > 80 and slowd[-1] > 80:
                signals.append(TechnicalSignal('STOCH', 'SELL', 75, slowk[-1], 'Stochastic overbought'))
            
            # Williams %R Analysis
            if willr[-1] < -80:
                signals.append(TechnicalSignal('WILLIAMS_R', 'BUY', 70, willr[-1], 'Williams %R oversold'))
            elif willr[-1] > -20:
                signals.append(TechnicalSignal('WILLIAMS_R', 'SELL', 70, willr[-1], 'Williams %R overbought'))
            
            return {
                'rsi': current_rsi if not np.isnan(current_rsi) else None,
                'stoch_k': slowk[-1] if not np.isnan(slowk[-1]) else None,
                'stoch_d': slowd[-1] if not np.isnan(slowd[-1]) else None,
                'williams_r': willr[-1] if not np.isnan(willr[-1]) else None,
                'roc': roc[-1] if not np.isnan(roc[-1]) else None,
                'cci': cci[-1] if not np.isnan(cci[-1]) else None,
                'signals': signals
            }
            
        except Exception as e:
            logger.error(f"Error in momentum analysis: {str(e)}")
            return {'signals': []}
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, any]:
        """Analyze volatility indicators"""
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        
        try:
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            
            # Average True Range
            atr = talib.ATR(high, low, close, timeperiod=14)
            
            # Keltner Channels
            kc_middle = talib.EMA(close, timeperiod=20)
            kc_upper = kc_middle + (2 * atr)
            kc_lower = kc_middle - (2 * atr)
            
            current_price = close[-1]
            
            signals = []
            
            # Bollinger Bands Analysis
            bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            
            if current_price <= bb_lower[-1]:
                signals.append(TechnicalSignal('BBANDS', 'BUY', 75, bb_position, 'Price at lower Bollinger Band'))
            elif current_price >= bb_upper[-1]:
                signals.append(TechnicalSignal('BBANDS', 'SELL', 75, bb_position, 'Price at upper Bollinger Band'))
            
            # Bollinger Band Squeeze
            bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
            bb_width_ma = np.mean([(bb_upper[i] - bb_lower[i]) / bb_middle[i] for i in range(-20, -1)])
            
            if bb_width < bb_width_ma * 0.8:
                signals.append(TechnicalSignal('BB_SQUEEZE', 'NEUTRAL', 60, bb_width, 'Bollinger Band squeeze - expect breakout'))
            
            return {
                'bb_upper': bb_upper[-1] if not np.isnan(bb_upper[-1]) else None,
                'bb_middle': bb_middle[-1] if not np.isnan(bb_middle[-1]) else None,
                'bb_lower': bb_lower[-1] if not np.isnan(bb_lower[-1]) else None,
                'bb_position': bb_position,
                'atr': atr[-1] if not np.isnan(atr[-1]) else None,
                'kc_upper': kc_upper[-1] if not np.isnan(kc_upper[-1]) else None,
                'kc_lower': kc_lower[-1] if not np.isnan(kc_lower[-1]) else None,
                'signals': signals
            }
            
        except Exception as e:
            logger.error(f"Error in volatility analysis: {str(e)}")
            return {'signals': []}
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, any]:
        """Analyze volume indicators"""
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        volume = data['Volume'].values
        
        try:
            # On Balance Volume
            obv = talib.OBV(close, volume)
            
            # Accumulation/Distribution Line
            ad = talib.AD(high, low, close, volume)
            
            # Chaikin Money Flow
            cmf = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
            
            # Volume Weighted Average Price (simplified)
            vwap = np.sum(close * volume) / np.sum(volume)
            
            # Volume analysis
            avg_volume = np.mean(volume[-20:])  # 20-day average volume
            current_volume = volume[-1]
            volume_ratio = current_volume / avg_volume
            
            signals = []
            
            # Volume Analysis
            if volume_ratio > 1.5:
                if close[-1] > close[-2]:
                    signals.append(TechnicalSignal('VOLUME', 'BUY', 70, volume_ratio, 'High volume with price increase'))
                else:
                    signals.append(TechnicalSignal('VOLUME', 'SELL', 70, volume_ratio, 'High volume with price decrease'))
            
            # OBV Analysis
            if len(obv) > 1:
                if obv[-1] > obv[-2] and close[-1] > close[-2]:
                    signals.append(TechnicalSignal('OBV', 'BUY', 65, obv[-1], 'OBV confirming price uptrend'))
                elif obv[-1] < obv[-2] and close[-1] < close[-2]:
                    signals.append(TechnicalSignal('OBV', 'SELL', 65, obv[-1], 'OBV confirming price downtrend'))
            
            # VWAP Analysis
            if close[-1] > vwap:
                signals.append(TechnicalSignal('VWAP', 'BUY', 60, vwap, 'Price above VWAP'))
            else:
                signals.append(TechnicalSignal('VWAP', 'SELL', 60, vwap, 'Price below VWAP'))
            
            return {
                'obv': obv[-1] if not np.isnan(obv[-1]) else None,
                'ad': ad[-1] if not np.isnan(ad[-1]) else None,
                'cmf': cmf[-1] if not np.isnan(cmf[-1]) else None,
                'vwap': vwap,
                'volume_ratio': volume_ratio,
                'avg_volume': avg_volume,
                'current_volume': current_volume,
                'signals': signals
            }
            
        except Exception as e:
            logger.error(f"Error in volume analysis: {str(e)}")
            return {'signals': []}
    
    def _find_support_resistance(self, data: pd.DataFrame) -> List[SupportResistance]:
        """Find support and resistance levels"""
        try:
            high = data['High'].values
            low = data['Low'].values
            close = data['Close'].values
            dates = data['Date']
            
            levels = []
            
            # Find pivot points
            for i in range(2, len(high) - 2):
                # Resistance levels (local maxima)
                if (high[i] > high[i-1] and high[i] > high[i-2] and 
                    high[i] > high[i+1] and high[i] > high[i+2]):
                    
                    # Count touches
                    touches = self._count_touches(high, high[i], tolerance=0.01)
                    if touches >= 2:
                        levels.append(SupportResistance(
                            level=high[i],
                            level_type='RESISTANCE',
                            strength=touches,
                            last_touch=dates.iloc[i].strftime('%Y-%m-%d')
                        ))
                
                # Support levels (local minima)
                if (low[i] < low[i-1] and low[i] < low[i-2] and 
                    low[i] < low[i+1] and low[i] < low[i+2]):
                    
                    # Count touches
                    touches = self._count_touches(low, low[i], tolerance=0.01)
                    if touches >= 2:
                        levels.append(SupportResistance(
                            level=low[i],
                            level_type='SUPPORT',
                            strength=touches,
                            last_touch=dates.iloc[i].strftime('%Y-%m-%d')
                        ))
            
            # Sort by strength and return top levels
            levels.sort(key=lambda x: x.strength, reverse=True)
            return levels[:10]  # Return top 10 levels
            
        except Exception as e:
            logger.error(f"Error finding support/resistance: {str(e)}")
            return []
    
    def _count_touches(self, price_array: np.ndarray, level: float, tolerance: float = 0.01) -> int:
        """Count how many times price touched a level"""
        return np.sum(np.abs(price_array - level) / level <= tolerance)
    
    def _detect_patterns(self, data: pd.DataFrame) -> List[Dict[str, any]]:
        """Detect candlestick patterns"""
        try:
            open_prices = data['Open'].values
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            
            patterns = []
            
            # Doji
            doji = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            if doji[-1] != 0:
                patterns.append({
                    'pattern': 'Doji',
                    'signal': 'REVERSAL',
                    'strength': 60,
                    'description': 'Indecision candle - potential reversal'
                })
            
            # Hammer
            hammer = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            if hammer[-1] != 0:
                patterns.append({
                    'pattern': 'Hammer',
                    'signal': 'BUY',
                    'strength': 75,
                    'description': 'Bullish reversal pattern'
                })
            
            # Shooting Star
            shooting_star = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
            if shooting_star[-1] != 0:
                patterns.append({
                    'pattern': 'Shooting Star',
                    'signal': 'SELL',
                    'strength': 75,
                    'description': 'Bearish reversal pattern'
                })
            
            # Engulfing patterns
            bullish_engulfing = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            if bullish_engulfing[-1] > 0:
                patterns.append({
                    'pattern': 'Bullish Engulfing',
                    'signal': 'BUY',
                    'strength': 80,
                    'description': 'Strong bullish reversal pattern'
                })
            elif bullish_engulfing[-1] < 0:
                patterns.append({
                    'pattern': 'Bearish Engulfing',
                    'signal': 'SELL',
                    'strength': 80,
                    'description': 'Strong bearish reversal pattern'
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            return []
    
    def _calculate_overall_signal(self, analysis: Dict[str, any]) -> Tuple[str, float]:
        """Calculate overall technical signal and score"""
        try:
            all_signals = []
            
            # Collect all signals
            for category in ['trend_analysis', 'momentum_analysis', 'volatility_analysis', 'volume_analysis']:
                if category in analysis and 'signals' in analysis[category]:
                    all_signals.extend(analysis[category]['signals'])
            
            if not all_signals:
                return 'NEUTRAL', 50.0
            
            # Calculate weighted scores
            buy_score = 0
            sell_score = 0
            total_weight = 0
            
            for signal in all_signals:
                weight = signal.strength / 100.0
                total_weight += weight
                
                if signal.signal == 'BUY':
                    buy_score += weight
                elif signal.signal == 'SELL':
                    sell_score += weight
            
            if total_weight == 0:
                return 'NEUTRAL', 50.0
            
            # Normalize scores
            buy_pct = (buy_score / total_weight) * 100
            sell_pct = (sell_score / total_weight) * 100
            
            # Determine overall signal
            if buy_pct > sell_pct + 15:
                overall_signal = 'BUY'
                technical_score = min(buy_pct, 95)
            elif sell_pct > buy_pct + 15:
                overall_signal = 'SELL'
                technical_score = max(100 - sell_pct, 5)
            else:
                overall_signal = 'NEUTRAL'
                technical_score = 50.0
            
            return overall_signal, technical_score
            
        except Exception as e:
            logger.error(f"Error calculating overall signal: {str(e)}")
            return 'NEUTRAL', 50.0
    
    def _basic_analysis(self, data: pd.DataFrame) -> Dict[str, any]:
        """Basic analysis for insufficient data"""
        if len(data) == 0:
            return {
                'trend_analysis': {'signals': []},
                'momentum_analysis': {'signals': []},
                'volatility_analysis': {'signals': []},
                'volume_analysis': {'signals': []},
                'support_resistance': [],
                'pattern_recognition': [],
                'overall_signal': 'NEUTRAL',
                'technical_score': 50.0
            }
        
        # Basic price analysis
        current_price = data['Close'].iloc[-1]
        previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        
        price_change = current_price - previous_price
        price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0
        
        if price_change_pct > 2:
            overall_signal = 'BUY'
            technical_score = 65.0
        elif price_change_pct < -2:
            overall_signal = 'SELL'
            technical_score = 35.0
        else:
            overall_signal = 'NEUTRAL'
            technical_score = 50.0
        
        return {
            'trend_analysis': {'signals': []},
            'momentum_analysis': {'signals': []},
            'volatility_analysis': {'signals': []},
            'volume_analysis': {'signals': []},
            'support_resistance': [],
            'pattern_recognition': [],
            'overall_signal': overall_signal,
            'technical_score': technical_score,
            'price_change_pct': price_change_pct
        }
    
    def get_quick_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get quick technical indicators for dashboard"""
        try:
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            volume = data['Volume'].values
            
            # Calculate key indicators
            indicators = {}
            
            if len(close) >= 14:
                indicators['rsi'] = talib.RSI(close, timeperiod=14)[-1]
            else:
                indicators['rsi'] = 50.0
            
            if len(close) >= 26:
                macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                indicators['macd'] = macd[-1]
                indicators['macd_signal'] = macd_signal[-1]
            else:
                indicators['macd'] = 0.0
                indicators['macd_signal'] = 0.0
            
            if len(close) >= 20:
                indicators['sma_20'] = talib.SMA(close, timeperiod=20)[-1]
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
                indicators['bb_upper'] = bb_upper[-1]
                indicators['bb_lower'] = bb_lower[-1]
            else:
                indicators['sma_20'] = close[-1]
                indicators['bb_upper'] = close[-1] * 1.02
                indicators['bb_lower'] = close[-1] * 0.98
            
            indicators['volume'] = volume[-1] if len(volume) > 0 else 0
            indicators['avg_volume'] = np.mean(volume[-20:]) if len(volume) >= 20 else volume[-1]
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating quick indicators: {str(e)}")
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'sma_20': data['Close'].iloc[-1] if not data.empty else 0,
                'bb_upper': 0,
                'bb_lower': 0,
                'volume': 0,
                'avg_volume': 0
            }

# Export main class
__all__ = ['TechnicalAnalyzer', 'TechnicalSignal', 'SupportResistance']
