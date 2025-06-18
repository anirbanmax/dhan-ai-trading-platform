"""
Professional Risk Management Module
Advanced risk assessment, position sizing, and portfolio protection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics structure"""
    var_1day: float  # 1-day Value at Risk
    var_5day: float  # 5-day Value at Risk
    expected_shortfall: float  # Expected Shortfall (CVaR)
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    volatility: float
    correlation_with_market: float

@dataclass
class PositionSize:
    """Position sizing recommendation"""
    symbol: str
    recommended_quantity: int
    recommended_value: float
    max_position_size: float
    risk_per_trade: float
    kelly_criterion: float
    position_weight: float
    reasoning: str

@dataclass
class RiskAlert:
    """Risk alert structure"""
    alert_type: str  # PORTFOLIO/POSITION/MARKET
    severity: str    # LOW/MEDIUM/HIGH/CRITICAL
    symbol: Optional[str]
    message: str
    recommendation: str
    timestamp: datetime

@dataclass
class StopLossRecommendation:
    """Stop loss recommendation"""
    symbol: str
    entry_price: float
    stop_loss_price: float
    stop_loss_pct: float
    atr_based_stop: float
    support_based_stop: float
    recommended_stop: float
    reasoning: str

class RiskManager:
    """Professional Risk Management System"""
    
    def __init__(self):
        """Initialize risk manager"""
        self.default_settings = {
            'max_portfolio_risk': 0.15,      # 15% maximum portfolio risk
            'max_position_risk': 0.02,       # 2% per position
            'max_single_position': 0.10,     # 10% max in single stock
            'max_sector_exposure': 0.25,     # 25% max in single sector
            'var_confidence': 0.95,          # 95% VaR confidence
            'min_diversification': 8,        # Minimum 8 positions
            'max_correlation': 0.7,          # Max correlation between positions
            'default_stop_loss': 0.05,       # 5% default stop loss
            'volatility_lookback': 20,       # 20 days for volatility calculation
            'drawdown_alert': 0.10           # Alert at 10% drawdown
        }
        
        self.portfolio_history = []
        self.risk_alerts = []
        
    def assess_portfolio_risk(self, portfolio: Dict[str, Dict], 
                             market_data: Dict[str, pd.DataFrame],
                             portfolio_value: float) -> Dict[str, any]:
        """Comprehensive portfolio risk assessment"""
        try:
            # Calculate individual position risks
            position_risks = {}
            total_value = 0
            
            for symbol, position in portfolio.items():
                if symbol in market_data and not market_data[symbol].empty:
                    risk_metrics = self._calculate_position_risk(
                        market_data[symbol], position, portfolio_value
                    )
                    position_risks[symbol] = risk_metrics
                    total_value += position['market_value']
            
            # Portfolio-level risk metrics
            portfolio_risk = self._calculate_portfolio_risk(position_risks, portfolio_value)
            
            # Diversification analysis
            diversification = self._analyze_diversification(portfolio, total_value)
            
            # Correlation analysis
            correlation_analysis = self._analyze_correlations(market_data, portfolio)
            
            # Risk alerts
            alerts = self._generate_risk_alerts(portfolio_risk, diversification, correlation_analysis)
            
            # Risk recommendations
            recommendations = self._generate_risk_recommendations(
                portfolio_risk, diversification, correlation_analysis
            )
            
            return {
                'portfolio_risk': portfolio_risk,
                'position_risks': position_risks,
                'diversification': diversification,
                'correlation_analysis': correlation_analysis,
                'risk_alerts': alerts,
                'recommendations': recommendations,
                'risk_score': self._calculate_overall_risk_score(portfolio_risk, diversification)
            }
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {str(e)}")
            return self._default_risk_assessment()
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss_price: float, portfolio_value: float,
                               volatility: float, confidence: float = 0.75) -> PositionSize:
        """Calculate optimal position size using multiple methods"""
        try:
            # Risk per trade (default 2% of portfolio)
            risk_per_trade = portfolio_value * self.default_settings['max_position_risk']
            
            # Fixed percentage method
            risk_amount = abs(entry_price - stop_loss_price)
            fixed_pct_quantity = int(risk_per_trade / risk_amount) if risk_amount > 0 else 0
            
            # ATR-based method
            atr_risk = volatility * entry_price * 2  # 2x ATR stop
            atr_quantity = int(risk_per_trade / atr_risk) if atr_risk > 0 else 0
            
            # Kelly Criterion method
            win_rate = confidence  # Use AI confidence as win rate proxy
            avg_win = 0.08  # Average 8% win
            avg_loss = 0.05  # Average 5% loss (stop loss)
            
            kelly_fraction = self._calculate_kelly_criterion(win_rate, avg_win, avg_loss)
            kelly_quantity = int((portfolio_value * kelly_fraction) / entry_price)
            
            # Volatility-adjusted sizing
            vol_adjustment = min(1.0, 0.20 / volatility)  # Reduce size if volatility > 20%
            vol_adjusted_quantity = int(fixed_pct_quantity * vol_adjustment)
            
            # Final recommendation (most conservative)
            recommended_quantity = min(
                fixed_pct_quantity,
                atr_quantity,
                kelly_quantity,
                vol_adjusted_quantity
            )
            
            # Position value checks
            position_value = recommended_quantity * entry_price
            max_position_value = portfolio_value * self.default_settings['max_single_position']
            
            if position_value > max_position_value:
                recommended_quantity = int(max_position_value / entry_price)
                position_value = recommended_quantity * entry_price
            
            # Position weight
            position_weight = position_value / portfolio_value
            
            # Reasoning
            reasoning = self._generate_sizing_reasoning(
                recommended_quantity, kelly_fraction, vol_adjustment, position_weight
            )
            
            return PositionSize(
                symbol=symbol,
                recommended_quantity=recommended_quantity,
                recommended_value=position_value,
                max_position_size=max_position_value,
                risk_per_trade=risk_per_trade,
                kelly_criterion=kelly_fraction,
                position_weight=position_weight,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return self._default_position_size(symbol, entry_price, portfolio_value)
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, 
                           price_data: pd.DataFrame, trade_direction: str = 'BUY') -> StopLossRecommendation:
        """Calculate optimal stop loss using multiple methods"""
        try:
            # ATR-based stop loss
            atr_stop = self._calculate_atr_stop_loss(price_data, entry_price, trade_direction)
            
            # Support/Resistance based stop loss
            support_resistance_stop = self._calculate_sr_stop_loss(
                price_data, entry_price, trade_direction
            )
            
            # Percentage-based stop loss
            pct_stop = self._calculate_percentage_stop_loss(entry_price, trade_direction)
            
            # Volatility-adjusted stop loss
            volatility_stop = self._calculate_volatility_stop_loss(
                price_data, entry_price, trade_direction
            )
            
            # Choose the most appropriate stop loss
            recommended_stop = self._select_optimal_stop_loss(
                atr_stop, support_resistance_stop, pct_stop, volatility_stop, trade_direction
            )
            
            stop_loss_pct = abs(recommended_stop - entry_price) / entry_price * 100
            
            reasoning = self._generate_stop_loss_reasoning(
                atr_stop, support_resistance_stop, recommended_stop, stop_loss_pct
            )
            
            return StopLossRecommendation(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss_price=recommended_stop,
                stop_loss_pct=stop_loss_pct,
                atr_based_stop=atr_stop,
                support_based_stop=support_resistance_stop,
                recommended_stop=recommended_stop,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return self._default_stop_loss(symbol, entry_price, trade_direction)
    
    def _calculate_position_risk(self, price_data: pd.DataFrame, position: Dict, 
                                portfolio_value: float) -> RiskMetrics:
        """Calculate risk metrics for individual position"""
        try:
            returns = price_data['Close'].pct_change().dropna()
            
            if len(returns) < 20:
                return self._default_risk_metrics()
            
            # Basic risk metrics
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            
            # VaR calculations
            var_95_1day = np.percentile(returns, 5) * position['market_value']
            var_95_5day = var_95_1day * np.sqrt(5)
            
            # Expected Shortfall (CVaR)
            tail_returns = returns[returns <= np.percentile(returns, 5)]
            expected_shortfall = tail_returns.mean() * position['market_value']
            
            # Maximum Drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # Sharpe Ratio (assuming 6.5% risk-free rate)
            excess_returns = returns - 0.065/252  # Daily risk-free rate
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
            
            # Sortino Ratio
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                sortino_ratio = excess_returns.mean() / negative_returns.std() * np.sqrt(252)
            else:
                sortino_ratio = sharpe_ratio
            
            # Beta (simplified - would need market data for accurate calculation)
            beta = 1.0  # Default beta
            
            # Correlation with market (simplified)
            correlation_with_market = 0.7  # Default correlation
            
            return RiskMetrics(
                var_1day=var_95_1day,
                var_5day=var_95_5day,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                beta=beta,
                volatility=volatility,
                correlation_with_market=correlation_with_market
            )
            
        except Exception as e:
            logger.error(f"Error calculating position risk: {str(e)}")
            return self._default_risk_metrics()
    
    def _calculate_portfolio_risk(self, position_risks: Dict[str, RiskMetrics], 
                                 portfolio_value: float) -> RiskMetrics:
        """Calculate portfolio-level risk metrics"""
        try:
            if not position_risks:
                return self._default_risk_metrics()
            
            # Aggregate VaR (simplified - assumes independence)
            total_var_1day = sum(risk.var_1day for risk in position_risks.values())
            total_var_5day = sum(risk.var_5day for risk in position_risks.values())
            
            # Weighted averages
            weights = [1/len(position_risks)] * len(position_risks)  # Equal weights for simplicity
            
            avg_volatility = np.average([risk.volatility for risk in position_risks.values()], weights=weights)
            avg_sharpe = np.average([risk.sharpe_ratio for risk in position_risks.values()], weights=weights)
            avg_sortino = np.average([risk.sortino_ratio for risk in position_risks.values()], weights=weights)
            avg_beta = np.average([risk.beta for risk in position_risks.values()], weights=weights)
            max_drawdown = min(risk.max_drawdown for risk in position_risks.values())
            
            # Expected shortfall
            total_expected_shortfall = sum(risk.expected_shortfall for risk in position_risks.values())
            
            return RiskMetrics(
                var_1day=total_var_1day,
                var_5day=total_var_5day,
                expected_shortfall=total_expected_shortfall,
                max_drawdown=max_drawdown,
                sharpe_ratio=avg_sharpe,
                sortino_ratio=avg_sortino,
                beta=avg_beta,
                volatility=avg_volatility,
                correlation_with_market=0.7  # Default
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {str(e)}")
            return self._default_risk_metrics()
    
    def _analyze_diversification(self, portfolio: Dict[str, Dict], total_value: float) -> Dict[str, any]:
        """Analyze portfolio diversification"""
        try:
            # Position concentration
            position_weights = {}
            for symbol, position in portfolio.items():
                weight = position['market_value'] / total_value
                position_weights[symbol] = weight
            
            # Concentration metrics
            max_position_weight = max(position_weights.values()) if position_weights else 0
            top_5_weight = sum(sorted(position_weights.values(), reverse=True)[:5])
            num_positions = len(portfolio)
            
            # Herfindahl-Hirschman Index (concentration measure)
            hhi = sum(weight**2 for weight in position_weights.values())
            
            # Diversification score (0-100)
            diversification_score = self._calculate_diversification_score(
                num_positions, max_position_weight, hhi
            )
            
            # Sector analysis (simplified - would need sector classification)
            sector_analysis = self._analyze_sector_diversification(portfolio)
            
            return {
                'num_positions': num_positions,
                'max_position_weight': max_position_weight,
                'top_5_weight': top_5_weight,
                'hhi': hhi,
                'diversification_score': diversification_score,
                'sector_analysis': sector_analysis,
                'is_well_diversified': diversification_score > 70
            }
            
        except Exception as e:
            logger.error(f"Error analyzing diversification: {str(e)}")
            return {'diversification_score': 50, 'is_well_diversified': False}
    
    def _analyze_correlations(self, market_data: Dict[str, pd.DataFrame], 
                             portfolio: Dict[str, Dict]) -> Dict[str, any]:
        """Analyze correlations between portfolio positions"""
        try:
            symbols = list(portfolio.keys())
            correlation_matrix = {}
            
            # Calculate pairwise correlations
            for i, symbol1 in enumerate(symbols):
                correlation_matrix[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    if symbol1 in market_data and symbol2 in market_data:
                        if not market_data[symbol1].empty and not market_data[symbol2].empty:
                            returns1 = market_data[symbol1]['Close'].pct_change().dropna()
                            returns2 = market_data[symbol2]['Close'].pct_change().dropna()
                            
                            # Align data
                            aligned_data = pd.concat([returns1, returns2], axis=1).dropna()
                            if len(aligned_data) > 10:
                                corr = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                                correlation_matrix[symbol1][symbol2] = corr
                            else:
                                correlation_matrix[symbol1][symbol2] = 0.5
                        else:
                            correlation_matrix[symbol1][symbol2] = 0.5
                    else:
                        correlation_matrix[symbol1][symbol2] = 0.5
            
            # Correlation statistics
            all_correlations = []
            for symbol1 in correlation_matrix:
                for symbol2, corr in correlation_matrix[symbol1].items():
                    if symbol1 != symbol2:
                        all_correlations.append(abs(corr))
            
            avg_correlation = np.mean(all_correlations) if all_correlations else 0.5
            max_correlation = max(all_correlations) if all_correlations else 0.5
            
            # High correlation pairs
            high_corr_pairs = []
            for symbol1 in correlation_matrix:
                for symbol2, corr in correlation_matrix[symbol1].items():
                    if symbol1 != symbol2 and abs(corr) > self.default_settings['max_correlation']:
                        high_corr_pairs.append((symbol1, symbol2, corr))
            
            return {
                'correlation_matrix': correlation_matrix,
                'avg_correlation': avg_correlation,
                'max_correlation': max_correlation,
                'high_correlation_pairs': high_corr_pairs,
                'correlation_risk': avg_correlation > 0.6
            }
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {str(e)}")
            return {'avg_correlation': 0.5, 'correlation_risk': False}
    
    def _generate_risk_alerts(self, portfolio_risk: RiskMetrics, diversification: Dict,
                             correlation_analysis: Dict) -> List[RiskAlert]:
        """Generate risk alerts based on analysis"""
        alerts = []
        
        try:
            # Portfolio VaR alert
            if abs(portfolio_risk.var_1day) > 50000:  # Alert if 1-day VaR > 50K
                alerts.append(RiskAlert(
                    alert_type="PORTFOLIO",
                    severity="HIGH",
                    symbol=None,
                    message=f"High portfolio VaR: ₹{abs(portfolio_risk.var_1day):,.0f}",
                    recommendation="Consider reducing position sizes or hedging",
                    timestamp=datetime.now()
                ))
            
            # Concentration alert
            if diversification.get('max_position_weight', 0) > 0.15:
                alerts.append(RiskAlert(
                    alert_type="PORTFOLIO",
                    severity="MEDIUM",
                    symbol=None,
                    message=f"High concentration: {diversification['max_position_weight']:.1%} in single position",
                    recommendation="Consider reducing largest position size",
                    timestamp=datetime.now()
                ))
            
            # Diversification alert
            if diversification.get('num_positions', 0) < 5:
                alerts.append(RiskAlert(
                    alert_type="PORTFOLIO",
                    severity="MEDIUM",
                    symbol=None,
                    message=f"Low diversification: Only {diversification['num_positions']} positions",
                    recommendation="Consider adding more positions for better diversification",
                    timestamp=datetime.now()
                ))
            
            # Correlation alert
            if correlation_analysis.get('avg_correlation', 0) > 0.7:
                alerts.append(RiskAlert(
                    alert_type="PORTFOLIO",
                    severity="MEDIUM",
                    symbol=None,
                    message=f"High average correlation: {correlation_analysis['avg_correlation']:.2f}",
                    recommendation="Consider diversifying into less correlated assets",
                    timestamp=datetime.now()
                ))
            
            # Volatility alert
            if portfolio_risk.volatility > 30:
                alerts.append(RiskAlert(
                    alert_type="PORTFOLIO",
                    severity="MEDIUM",
                    symbol=None,
                    message=f"High portfolio volatility: {portfolio_risk.volatility:.1f}%",
                    recommendation="Consider reducing volatile positions or hedging",
                    timestamp=datetime.now()
                ))
            
        except Exception as e:
            logger.error(f"Error generating risk alerts: {str(e)}")
        
        return alerts
    
    def _calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion for position sizing"""
        try:
            if avg_loss <= 0:
                return 0.0
            
            # Kelly formula: f* = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Cap Kelly fraction at 25% for safety
            return max(0, min(kelly_fraction, 0.25))
            
        except Exception as e:
            logger.error(f"Error calculating Kelly criterion: {str(e)}")
            return 0.05  # Conservative default
    
    def _calculate_atr_stop_loss(self, price_data: pd.DataFrame, entry_price: float, 
                                direction: str) -> float:
        """Calculate ATR-based stop loss"""
        try:
            # Calculate ATR
            high = price_data['High'].values
            low = price_data['Low'].values
            close = price_data['Close'].values
            
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = np.mean(true_range[-14:])  # 14-period ATR
            
            # ATR multiplier (typically 2-3)
            atr_multiplier = 2.0
            
            if direction.upper() == 'BUY':
                return entry_price - (atr * atr_multiplier)
            else:  # SELL
                return entry_price + (atr * atr_multiplier)
                
        except Exception as e:
            logger.error(f"Error calculating ATR stop loss: {str(e)}")
            return entry_price * (0.95 if direction.upper() == 'BUY' else 1.05)
    
    def _calculate_sr_stop_loss(self, price_data: pd.DataFrame, entry_price: float,
                               direction: str) -> float:
        """Calculate support/resistance based stop loss"""
        try:
            # Find recent support/resistance levels
            lows = price_data['Low'].rolling(window=5).min()
            highs = price_data['High'].rolling(window=5).max()
            
            if direction.upper() == 'BUY':
                # Find nearest support level below entry
                support_levels = lows[lows < entry_price].tail(10)
                if not support_levels.empty:
                    return support_levels.max() * 0.99  # Slightly below support
                else:
                    return entry_price * 0.95  # Default 5% stop
            else:  # SELL
                # Find nearest resistance level above entry
                resistance_levels = highs[highs > entry_price].tail(10)
                if not resistance_levels.empty:
                    return resistance_levels.min() * 1.01  # Slightly above resistance
                else:
                    return entry_price * 1.05  # Default 5% stop
                    
        except Exception as e:
            logger.error(f"Error calculating S/R stop loss: {str(e)}")
            return entry_price * (0.95 if direction.upper() == 'BUY' else 1.05)
    
    def _calculate_percentage_stop_loss(self, entry_price: float, direction: str) -> float:
        """Calculate percentage-based stop loss"""
        stop_pct = self.default_settings['default_stop_loss']
        
        if direction.upper() == 'BUY':
            return entry_price * (1 - stop_pct)
        else:  # SELL
            return entry_price * (1 + stop_pct)
    
    def _calculate_volatility_stop_loss(self, price_data: pd.DataFrame, entry_price: float,
                                       direction: str) -> float:
        """Calculate volatility-adjusted stop loss"""
        try:
            returns = price_data['Close'].pct_change().dropna()
            volatility = returns.std()
            
            # Adjust stop loss based on volatility
            # Higher volatility = wider stop loss
            vol_multiplier = max(1.0, min(3.0, volatility * 100))  # 1x to 3x multiplier
            base_stop = self.default_settings['default_stop_loss']
            adjusted_stop = base_stop * vol_multiplier
            
            if direction.upper() == 'BUY':
                return entry_price * (1 - adjusted_stop)
            else:  # SELL
                return entry_price * (1 + adjusted_stop)
                
        except Exception as e:
            logger.error(f"Error calculating volatility stop loss: {str(e)}")
            return self._calculate_percentage_stop_loss(entry_price, direction)
    
    def _select_optimal_stop_loss(self, atr_stop: float, sr_stop: float, 
                                 pct_stop: float, vol_stop: float, direction: str) -> float:
        """Select the most appropriate stop loss from different methods"""
        try:
            stops = [atr_stop, sr_stop, pct_stop, vol_stop]
            
            if direction.upper() == 'BUY':
                # For buy orders, use the highest (closest to entry) stop loss
                return max(stops)
            else:  # SELL
                # For sell orders, use the lowest (closest to entry) stop loss
                return min(stops)
                
        except Exception as e:
            logger.error(f"Error selecting optimal stop loss: {str(e)}")
            return pct_stop
    
    # Helper methods
    def _default_risk_metrics(self) -> RiskMetrics:
        """Return default risk metrics"""
        return RiskMetrics(
            var_1day=-1000.0,
            var_5day=-2236.0,
            expected_shortfall=-1500.0,
            max_drawdown=-5.0,
            sharpe_ratio=0.5,
            sortino_ratio=0.6,
            beta=1.0,
            volatility=20.0,
            correlation_with_market=0.7
        )
    
    def _default_position_size(self, symbol: str, entry_price: float, 
                              portfolio_value: float) -> PositionSize:
        """Return default position size"""
        max_position_value = portfolio_value * 0.05  # 5% default
        quantity = int(max_position_value / entry_price)
        
        return PositionSize(
            symbol=symbol,
            recommended_quantity=quantity,
            recommended_value=quantity * entry_price,
            max_position_size=max_position_value,
            risk_per_trade=portfolio_value * 0.02,
            kelly_criterion=0.05,
            position_weight=0.05,
            reasoning="Default conservative sizing due to calculation error"
        )
    
    def _default_stop_loss(self, symbol: str, entry_price: float, direction: str) -> StopLossRecommendation:
        """Return default stop loss"""
        stop_pct = 5.0
        if direction.upper() == 'BUY':
            stop_price = entry_price * 0.95
        else:
            stop_price = entry_price * 1.05
        
        return StopLossRecommendation(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss_price=stop_price,
            stop_loss_pct=stop_pct,
            atr_based_stop=stop_price,
            support_based_stop=stop_price,
            recommended_stop=stop_price,
            reasoning="Default 5% stop loss due to calculation error"
        )
    
    def _default_risk_assessment(self) -> Dict[str, any]:
        """Return default risk assessment"""
        return {
            'portfolio_risk': self._default_risk_metrics(),
            'position_risks': {},
            'diversification': {'diversification_score': 50, 'is_well_diversified': False},
            'correlation_analysis': {'avg_correlation': 0.5, 'correlation_risk': False},
            'risk_alerts': [],
            'recommendations': [],
            'risk_score': 50
        }
    
    def _calculate_diversification_score(self, num_positions: int, max_weight: float, hhi: float) -> float:
        """Calculate diversification score (0-100)"""
        try:
            # Position count score (0-40 points)
            count_score = min(40, num_positions * 5)  # 5 points per position, max 40
            
            # Concentration score (0-30 points)
            if max_weight > 0.2:
                concentration_score = 0
            elif max_weight > 0.15:
                concentration_score = 10
            elif max_weight > 0.1:
                concentration_score = 20
            else:
                concentration_score = 30
            
            # HHI score (0-30 points)
            if hhi > 0.2:
                hhi_score = 0
            elif hhi > 0.15:
                hhi_score = 10
            elif hhi > 0.1:
                hhi_score = 20
            else:
                hhi_score = 30
            
            return count_score + concentration_score + hhi_score
            
        except Exception as e:
            logger.error(f"Error calculating diversification score: {str(e)}")
            return 50.0
    
    def _analyze_sector_diversification(self, portfolio: Dict[str, Dict]) -> Dict[str, any]:
        """Analyze sector diversification (simplified)"""
        # In a real implementation, this would use actual sector classifications
        # For now, we'll simulate sector analysis
        
        sectors = ['Technology', 'Banking', 'Pharma', 'FMCG', 'Auto']
        sector_weights = {}
        
        # Simulate sector allocation
        symbols = list(portfolio.keys())
        for i, symbol in enumerate(symbols):
            sector = sectors[i % len(sectors)]
            if sector not in sector_weights:
                sector_weights[sector] = 0
            sector_weights[sector] += 1 / len(symbols)
        
        max_sector_weight = max(sector_weights.values()) if sector_weights else 0
        num_sectors = len(sector_weights)
        
        return {
            'sector_weights': sector_weights,
            'num_sectors': num_sectors,
            'max_sector_weight': max_sector_weight,
            'sector_concentration_risk': max_sector_weight > 0.4
        }
    
    def _calculate_overall_risk_score(self, portfolio_risk: RiskMetrics, diversification: Dict) -> float:
        """Calculate overall portfolio risk score (0-100, lower is better)"""
        try:
            risk_score = 0
            
            # Volatility component (0-30 points)
            vol_score = min(30, portfolio_risk.volatility)
            risk_score += vol_score
            
            # Diversification component (0-30 points)
            div_score = 30 - (diversification.get('diversification_score', 50) * 0.3)
            risk_score += div_score
            
            # VaR component (0-20 points)
            var_pct = abs(portfolio_risk.var_1day) / 100000  # Normalize to 100K portfolio
            var_score = min(20, var_pct * 20)
            risk_score += var_score
            
            # Sharpe ratio component (0-20 points, inverted)
            sharpe_score = max(0, 20 - portfolio_risk.sharpe_ratio * 10)
            risk_score += sharpe_score
            
            return min(100, risk_score)
            
        except Exception as e:
            logger.error(f"Error calculating overall risk score: {str(e)}")
            return 50.0
    
    def _generate_risk_recommendations(self, portfolio_risk: RiskMetrics, 
                                     diversification: Dict, correlation_analysis: Dict) -> List[str]:
        """Generate actionable risk management recommendations"""
        recommendations = []
        
        try:
            # Diversification recommendations
            if diversification.get('num_positions', 0) < 8:
                recommendations.append("Increase portfolio diversification by adding 3-5 more positions")
            
            if diversification.get('max_position_weight', 0) > 0.15:
                recommendations.append("Reduce position size of largest holding to below 15%")
            
            # Risk recommendations
            if portfolio_risk.volatility > 25:
                recommendations.append("Consider reducing volatile positions or adding defensive stocks")
            
            if portfolio_risk.sharpe_ratio < 0.5:
                recommendations.append("Improve risk-adjusted returns by rebalancing portfolio")
            
            # Correlation recommendations
            if correlation_analysis.get('avg_correlation', 0) > 0.6:
                recommendations.append("Reduce correlation risk by diversifying across sectors/asset classes")
            
            # Stop loss recommendations
            recommendations.append("Implement stop losses for all positions to limit downside risk")
            
            # Position sizing recommendations
            recommendations.append("Use position sizing based on Kelly criterion and volatility")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def _generate_sizing_reasoning(self, quantity: int, kelly_fraction: float, 
                                  vol_adjustment: float, position_weight: float) -> str:
        """Generate reasoning for position sizing"""
        try:
            reasons = []
            
            if kelly_fraction > 0.1:
                reasons.append("High Kelly criterion suggests larger position")
            elif kelly_fraction < 0.05:
                reasons.append("Low Kelly criterion suggests smaller position")
            
            if vol_adjustment < 0.8:
                reasons.append("Position reduced due to high volatility")
            
            if position_weight > 0.08:
                reasons.append("Position capped due to concentration limits")
            
            if not reasons:
                reasons.append("Standard risk-based position sizing applied")
            
            return ". ".join(reasons) + f". Position weight: {position_weight:.1%}"
            
        except Exception as e:
            logger.error(f"Error generating sizing reasoning: {str(e)}")
            return "Standard position sizing applied"
    
    def _generate_stop_loss_reasoning(self, atr_stop: float, sr_stop: float, 
                                     recommended_stop: float, stop_pct: float) -> str:
        """Generate reasoning for stop loss selection"""
        try:
            if abs(recommended_stop - atr_stop) < abs(recommended_stop - sr_stop):
                return f"ATR-based stop loss selected ({stop_pct:.1f}% risk) for volatility protection"
            else:
                return f"Support/resistance based stop loss selected ({stop_pct:.1f}% risk) for technical protection"
                
        except Exception as e:
            logger.error(f"Error generating stop loss reasoning: {str(e)}")
            return f"Standard {stop_pct:.1f}% stop loss applied"

# Export main classes
__all__ = [
    'RiskManager', 'RiskMetrics', 'PositionSize', 'RiskAlert', 'StopLossRecommendation'
]
