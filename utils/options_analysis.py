"""
Professional Options Analysis Module
Advanced options chain analysis, Greeks calculation, and strategy recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import scipy.stats as stats
from scipy.optimize import brentq
import logging
import math

logger = logging.getLogger(__name__)

@dataclass
class OptionData:
    """Option data structure"""
    strike: float
    option_type: str  # CALL/PUT
    ltp: float
    bid: float
    ask: float
    volume: int
    oi: int  # Open Interest
    iv: float  # Implied Volatility
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

@dataclass
class OptionsStrategy:
    """Options trading strategy"""
    name: str
    strategy_type: str
    legs: List[Dict]
    max_profit: float
    max_loss: float
    breakeven: List[float]
    probability: float
    risk_reward: float
    description: str

@dataclass
class OptionsSignal:
    """Options trading signal"""
    underlying: str
    strategy: str
    action: str  # BUY/SELL
    strike: float
    option_type: str
    expiry: str
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    max_risk: float
    max_reward: float
    probability: float
    reasoning: str

class OptionsAnalyzer:
    """Professional Options Analysis Engine"""
    
    def __init__(self):
        """Initialize options analyzer"""
        self.risk_free_rate = 0.065  # Current RBI repo rate
        self.strategies = [
            'long_call', 'long_put', 'covered_call', 'protective_put',
            'bull_call_spread', 'bear_put_spread', 'iron_condor', 
            'long_straddle', 'short_strangle', 'butterfly_spread'
        ]
    
    def analyze_options_chain(self, options_data: pd.DataFrame, spot_price: float, 
                            expiry_days: int = 30) -> Dict[str, any]:
        """Comprehensive options chain analysis"""
        try:
            # Calculate Greeks for all options
            options_with_greeks = self._calculate_greeks(options_data, spot_price, expiry_days)
            
            # PCR Analysis
            pcr_analysis = self._calculate_pcr(options_with_greeks)
            
            # Open Interest Analysis
            oi_analysis = self._analyze_open_interest(options_with_greeks)
            
            # Max Pain Analysis
            max_pain = self._calculate_max_pain(options_with_greeks)
            
            # Volatility Analysis
            iv_analysis = self._analyze_implied_volatility(options_with_greeks)
            
            # Options Flow Analysis
            flow_analysis = self._analyze_options_flow(options_with_greeks)
            
            # Strategy Recommendations
            strategies = self._recommend_strategies(options_with_greeks, spot_price, expiry_days)
            
            return {
                'options_data': options_with_greeks,
                'pcr_analysis': pcr_analysis,
                'oi_analysis': oi_analysis,
                'max_pain': max_pain,
                'iv_analysis': iv_analysis,
                'flow_analysis': flow_analysis,
                'recommended_strategies': strategies,
                'market_sentiment': self._determine_market_sentiment(pcr_analysis, oi_analysis, flow_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error in options chain analysis: {str(e)}")
            return self._basic_options_analysis(options_data, spot_price)
    
    def _calculate_greeks(self, options_data: pd.DataFrame, spot_price: float, 
                         expiry_days: int) -> pd.DataFrame:
        """Calculate options Greeks using Black-Scholes model"""
        try:
            options_with_greeks = options_data.copy()
            time_to_expiry = expiry_days / 365.0
            
            for idx, row in options_with_greeks.iterrows():
                strike = row['strike']
                option_type = row['option_type']
                iv = row['iv'] / 100.0  # Convert percentage to decimal
                
                # Calculate d1 and d2 for Black-Scholes
                d1 = (np.log(spot_price / strike) + (self.risk_free_rate + 0.5 * iv**2) * time_to_expiry) / (iv * np.sqrt(time_to_expiry))
                d2 = d1 - iv * np.sqrt(time_to_expiry)
                
                # Calculate Greeks
                if option_type == 'CALL':
                    delta = stats.norm.cdf(d1)
                    theta = (-spot_price * stats.norm.pdf(d1) * iv / (2 * np.sqrt(time_to_expiry)) 
                            - self.risk_free_rate * strike * np.exp(-self.risk_free_rate * time_to_expiry) * stats.norm.cdf(d2)) / 365
                else:  # PUT
                    delta = stats.norm.cdf(d1) - 1
                    theta = (-spot_price * stats.norm.pdf(d1) * iv / (2 * np.sqrt(time_to_expiry)) 
                            + self.risk_free_rate * strike * np.exp(-self.risk_free_rate * time_to_expiry) * stats.norm.cdf(-d2)) / 365
                
                gamma = stats.norm.pdf(d1) / (spot_price * iv * np.sqrt(time_to_expiry))
                vega = spot_price * stats.norm.pdf(d1) * np.sqrt(time_to_expiry) / 100
                
                options_with_greeks.at[idx, 'delta'] = delta
                options_with_greeks.at[idx, 'gamma'] = gamma
                options_with_greeks.at[idx, 'theta'] = theta
                options_with_greeks.at[idx, 'vega'] = vega
            
            return options_with_greeks
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {str(e)}")
            return options_data
    
    def _calculate_pcr(self, options_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Put-Call Ratio analysis"""
        try:
            calls = options_data[options_data['option_type'] == 'CALL']
            puts = options_data[options_data['option_type'] == 'PUT']
            
            # PCR by Volume
            call_volume = calls['volume'].sum()
            put_volume = puts['volume'].sum()
            pcr_volume = put_volume / call_volume if call_volume > 0 else 0
            
            # PCR by Open Interest
            call_oi = calls['oi'].sum()
            put_oi = puts['oi'].sum()
            pcr_oi = put_oi / call_oi if call_oi > 0 else 0
            
            # PCR interpretation
            if pcr_volume > 1.2:
                volume_sentiment = 'BEARISH'
            elif pcr_volume < 0.8:
                volume_sentiment = 'BULLISH'
            else:
                volume_sentiment = 'NEUTRAL'
            
            if pcr_oi > 1.2:
                oi_sentiment = 'BEARISH'
            elif pcr_oi < 0.8:
                oi_sentiment = 'BULLISH'
            else:
                oi_sentiment = 'NEUTRAL'
            
            return {
                'pcr_volume': pcr_volume,
                'pcr_oi': pcr_oi,
                'volume_sentiment': volume_sentiment,
                'oi_sentiment': oi_sentiment,
                'call_volume': call_volume,
                'put_volume': put_volume,
                'call_oi': call_oi,
                'put_oi': put_oi
            }
            
        except Exception as e:
            logger.error(f"Error calculating PCR: {str(e)}")
            return {'pcr_volume': 1.0, 'pcr_oi': 1.0, 'volume_sentiment': 'NEUTRAL', 'oi_sentiment': 'NEUTRAL'}
    
    def _analyze_open_interest(self, options_data: pd.DataFrame) -> Dict[str, any]:
        """Analyze Open Interest patterns"""
        try:
            # Group by strike price
            oi_by_strike = options_data.groupby(['strike', 'option_type'])['oi'].sum().unstack(fill_value=0)
            
            # Find max OI strikes
            call_max_oi_strike = oi_by_strike['CALL'].idxmax() if 'CALL' in oi_by_strike.columns else None
            put_max_oi_strike = oi_by_strike['PUT'].idxmax() if 'PUT' in oi_by_strike.columns else None
            
            # OI distribution analysis
            total_call_oi = oi_by_strike['CALL'].sum() if 'CALL' in oi_by_strike.columns else 0
            total_put_oi = oi_by_strike['PUT'].sum() if 'PUT' in oi_by_strike.columns else 0
            
            # Find concentration levels
            call_concentration = self._calculate_oi_concentration(oi_by_strike['CALL']) if 'CALL' in oi_by_strike.columns else 0
            put_concentration = self._calculate_oi_concentration(oi_by_strike['PUT']) if 'PUT' in oi_by_strike.columns else 0
            
            return {
                'call_max_oi_strike': call_max_oi_strike,
                'put_max_oi_strike': put_max_oi_strike,
                'total_call_oi': total_call_oi,
                'total_put_oi': total_put_oi,
                'call_concentration': call_concentration,
                'put_concentration': put_concentration,
                'oi_distribution': oi_by_strike.to_dict() if not oi_by_strike.empty else {}
            }
            
        except Exception as e:
            logger.error(f"Error analyzing OI: {str(e)}")
            return {}
    
    def _calculate_oi_concentration(self, oi_series: pd.Series) -> float:
        """Calculate OI concentration (what % of total OI is in top 3 strikes)"""
        if oi_series.empty:
            return 0
        
        top_3_oi = oi_series.nlargest(3).sum()
        total_oi = oi_series.sum()
        
        return (top_3_oi / total_oi * 100) if total_oi > 0 else 0
    
    def _calculate_max_pain(self, options_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Max Pain level"""
        try:
            strikes = sorted(options_data['strike'].unique())
            max_pain_values = []
            
            for strike in strikes:
                total_pain = 0
                
                for _, option in options_data.iterrows():
                    if option['option_type'] == 'CALL' and strike > option['strike']:
                        total_pain += (strike - option['strike']) * option['oi']
                    elif option['option_type'] == 'PUT' and strike < option['strike']:
                        total_pain += (option['strike'] - strike) * option['oi']
                
                max_pain_values.append(total_pain)
            
            max_pain_index = np.argmin(max_pain_values)
            max_pain_strike = strikes[max_pain_index]
            max_pain_value = max_pain_values[max_pain_index]
            
            return {
                'max_pain_strike': max_pain_strike,
                'max_pain_value': max_pain_value,
                'pain_distribution': dict(zip(strikes, max_pain_values))
            }
            
        except Exception as e:
            logger.error(f"Error calculating max pain: {str(e)}")
            return {'max_pain_strike': 0, 'max_pain_value': 0}
    
    def _analyze_implied_volatility(self, options_data: pd.DataFrame) -> Dict[str, any]:
        """Analyze implied volatility patterns"""
        try:
            calls = options_data[options_data['option_type'] == 'CALL']
            puts = options_data[options_data['option_type'] == 'PUT']
            
            # Average IV
            avg_call_iv = calls['iv'].mean()
            avg_put_iv = puts['iv'].mean()
            overall_avg_iv = options_data['iv'].mean()
            
            # IV Skew
            iv_skew = avg_put_iv - avg_call_iv
            
            # High IV vs Low IV options
            high_iv_threshold = overall_avg_iv + options_data['iv'].std()
            low_iv_threshold = overall_avg_iv - options_data['iv'].std()
            
            high_iv_options = options_data[options_data['iv'] > high_iv_threshold]
            low_iv_options = options_data[options_data['iv'] < low_iv_threshold]
            
            return {
                'avg_call_iv': avg_call_iv,
                'avg_put_iv': avg_put_iv,
                'overall_avg_iv': overall_avg_iv,
                'iv_skew': iv_skew,
                'high_iv_count': len(high_iv_options),
                'low_iv_count': len(low_iv_options),
                'iv_percentile': self._calculate_iv_percentile(overall_avg_iv)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing IV: {str(e)}")
            return {}
    
    def _calculate_iv_percentile(self, current_iv: float) -> float:
        """Calculate IV percentile (simplified)"""
        # This would typically use historical IV data
        # For now, using a simplified approach
        typical_low_iv = 15.0
        typical_high_iv = 40.0
        
        if current_iv <= typical_low_iv:
            return 10.0
        elif current_iv >= typical_high_iv:
            return 90.0
        else:
            return ((current_iv - typical_low_iv) / (typical_high_iv - typical_low_iv)) * 80 + 10
    
    def _analyze_options_flow(self, options_data: pd.DataFrame) -> Dict[str, any]:
        """Analyze options flow and activity"""
        try:
            # Volume analysis
            total_volume = options_data['volume'].sum()
            call_volume = options_data[options_data['option_type'] == 'CALL']['volume'].sum()
            put_volume = options_data[options_data['option_type'] == 'PUT']['volume'].sum()
            
            # High volume strikes
            volume_by_strike = options_data.groupby('strike')['volume'].sum().sort_values(ascending=False)
            top_volume_strikes = volume_by_strike.head(5).to_dict()
            
            # Unusual activity detection
            avg_volume = options_data['volume'].mean()
            unusual_volume_threshold = avg_volume * 3
            unusual_activity = options_data[options_data['volume'] > unusual_volume_threshold]
            
            return {
                'total_volume': total_volume,
                'call_volume': call_volume,
                'put_volume': put_volume,
                'call_put_volume_ratio': call_volume / put_volume if put_volume > 0 else 0,
                'top_volume_strikes': top_volume_strikes,
                'unusual_activity_count': len(unusual_activity),
                'unusual_activity': unusual_activity.to_dict('records') if not unusual_activity.empty else []
            }
            
        except Exception as e:
            logger.error(f"Error analyzing options flow: {str(e)}")
            return {}
    
    def _recommend_strategies(self, options_data: pd.DataFrame, spot_price: float, 
                            expiry_days: int) -> List[OptionsStrategy]:
        """Recommend options strategies based on analysis"""
        try:
            strategies = []
            
            # Get ATM options
            atm_strike = self._find_atm_strike(options_data, spot_price)
            atm_call = options_data[(options_data['strike'] == atm_strike) & 
                                  (options_data['option_type'] == 'CALL')].iloc[0] if not options_data.empty else None
            atm_put = options_data[(options_data['strike'] == atm_strike) & 
                                 (options_data['option_type'] == 'PUT')].iloc[0] if not options_data.empty else None
            
            if atm_call is not None and atm_put is not None:
                # Long Straddle Strategy
                straddle = self._analyze_long_straddle(atm_call, atm_put, spot_price, expiry_days)
                if straddle:
                    strategies.append(straddle)
                
                # Bull Call Spread
                bull_call_spread = self._analyze_bull_call_spread(options_data, spot_price, expiry_days)
                if bull_call_spread:
                    strategies.append(bull_call_spread)
                
                # Iron Condor
                iron_condor = self._analyze_iron_condor(options_data, spot_price, expiry_days)
                if iron_condor:
                    strategies.append(iron_condor)
            
            # Sort by probability of profit
            strategies.sort(key=lambda x: x.probability, reverse=True)
            
            return strategies[:5]  # Return top 5 strategies
            
        except Exception as e:
            logger.error(f"Error recommending strategies: {str(e)}")
            return []
    
    def _find_atm_strike(self, options_data: pd.DataFrame, spot_price: float) -> float:
        """Find At-The-Money strike price"""
        strikes = options_data['strike'].unique()
        return min(strikes, key=lambda x: abs(x - spot_price))
    
    def _analyze_long_straddle(self, call_option: pd.Series, put_option: pd.Series, 
                              spot_price: float, expiry_days: int) -> Optional[OptionsStrategy]:
        """Analyze long straddle strategy"""
        try:
            total_premium = call_option['ltp'] + put_option['ltp']
            strike = call_option['strike']
            
            # Breakeven points
            upper_breakeven = strike + total_premium
            lower_breakeven = strike - total_premium
            
            # Calculate probability of profit (simplified)
            # This would typically use historical volatility data
            prob_profit = self._estimate_breakeven_probability(spot_price, [upper_breakeven, lower_breakeven], expiry_days)
            
            return OptionsStrategy(
                name="Long Straddle",
                strategy_type="VOLATILITY",
                legs=[
                    {"action": "BUY", "option_type": "CALL", "strike": strike, "premium": call_option['ltp']},
                    {"action": "BUY", "option_type": "PUT", "strike": strike, "premium": put_option['ltp']}
                ],
                max_profit=float('inf'),  # Unlimited
                max_loss=total_premium,
                breakeven=[upper_breakeven, lower_breakeven],
                probability=prob_profit,
                risk_reward=float('inf'),  # Unlimited upside
                description=f"Profit if {call_option['strike']} moves beyond {upper_breakeven:.0f} or {lower_breakeven:.0f}"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing long straddle: {str(e)}")
            return None
    
    def _analyze_bull_call_spread(self, options_data: pd.DataFrame, spot_price: float, 
                                 expiry_days: int) -> Optional[OptionsStrategy]:
        """Analyze bull call spread strategy"""
        try:
            calls = options_data[options_data['option_type'] == 'CALL'].sort_values('strike')
            
            if len(calls) < 2:
                return None
            
            # Find suitable strikes (ITM and OTM)
            itm_calls = calls[calls['strike'] <= spot_price]
            otm_calls = calls[calls['strike'] > spot_price]
            
            if len(itm_calls) == 0 or len(otm_calls) == 0:
                return None
            
            buy_call = itm_calls.iloc[-1]  # Closest ITM
            sell_call = otm_calls.iloc[0]   # Closest OTM
            
            net_premium = buy_call['ltp'] - sell_call['ltp']
            max_profit = (sell_call['strike'] - buy_call['strike']) - net_premium
            breakeven = buy_call['strike'] + net_premium
            
            prob_profit = self._estimate_breakeven_probability(spot_price, [breakeven], expiry_days)
            
            return OptionsStrategy(
                name="Bull Call Spread",
                strategy_type="DIRECTIONAL",
                legs=[
                    {"action": "BUY", "option_type": "CALL", "strike": buy_call['strike'], "premium": buy_call['ltp']},
                    {"action": "SELL", "option_type": "CALL", "strike": sell_call['strike'], "premium": sell_call['ltp']}
                ],
                max_profit=max_profit,
                max_loss=net_premium,
                breakeven=[breakeven],
                probability=prob_profit,
                risk_reward=max_profit / net_premium if net_premium > 0 else 0,
                description=f"Profit if price rises above {breakeven:.0f} by expiry"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing bull call spread: {str(e)}")
            return None
    
    def _analyze_iron_condor(self, options_data: pd.DataFrame, spot_price: float, 
                            expiry_days: int) -> Optional[OptionsStrategy]:
        """Analyze iron condor strategy"""
        try:
            # This is a simplified iron condor analysis
            # In practice, would need more sophisticated strike selection
            
            strikes = sorted(options_data['strike'].unique())
            if len(strikes) < 4:
                return None
            
            # Select strikes around current price
            center_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot_price))
            
            if center_idx < 2 or center_idx >= len(strikes) - 2:
                return None
            
            put_buy_strike = strikes[center_idx - 2]
            put_sell_strike = strikes[center_idx - 1]
            call_sell_strike = strikes[center_idx + 1]
            call_buy_strike = strikes[center_idx + 2]
            
            # Get option premiums
            put_buy = options_data[(options_data['strike'] == put_buy_strike) & 
                                 (options_data['option_type'] == 'PUT')]
            put_sell = options_data[(options_data['strike'] == put_sell_strike) & 
                                  (options_data['option_type'] == 'PUT')]
            call_sell = options_data[(options_data['strike'] == call_sell_strike) & 
                                   (options_data['option_type'] == 'CALL')]
            call_buy = options_data[(options_data['strike'] == call_buy_strike) & 
                                  (options_data['option_type'] == 'CALL')]
            
            if any(df.empty for df in [put_buy, put_sell, call_sell, call_buy]):
                return None
            
            net_credit = (put_sell.iloc[0]['ltp'] + call_sell.iloc[0]['ltp'] - 
                         put_buy.iloc[0]['ltp'] - call_buy.iloc[0]['ltp'])
            
            max_profit = net_credit
            max_loss = min(put_sell_strike - put_buy_strike, call_buy_strike - call_sell_strike) - net_credit
            
            lower_breakeven = put_sell_strike - net_credit
            upper_breakeven = call_sell_strike + net_credit
            
            prob_profit = self._estimate_range_probability(spot_price, lower_breakeven, upper_breakeven, expiry_days)
            
            return OptionsStrategy(
                name="Iron Condor",
                strategy_type="RANGE_BOUND",
                legs=[
                    {"action": "BUY", "option_type": "PUT", "strike": put_buy_strike, "premium": put_buy.iloc[0]['ltp']},
                    {"action": "SELL", "option_type": "PUT", "strike": put_sell_strike, "premium": put_sell.iloc[0]['ltp']},
                    {"action": "SELL", "option_type": "CALL", "strike": call_sell_strike, "premium": call_sell.iloc[0]['ltp']},
                    {"action": "BUY", "option_type": "CALL", "strike": call_buy_strike, "premium": call_buy.iloc[0]['ltp']}
                ],
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven=[lower_breakeven, upper_breakeven],
                probability=prob_profit,
                risk_reward=max_profit / abs(max_loss) if max_loss != 0 else 0,
                description=f"Profit if price stays between {lower_breakeven:.0f} and {upper_breakeven:.0f}"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing iron condor: {str(e)}")
            return None
    
    def _estimate_breakeven_probability(self, spot_price: float, breakevens: List[float], 
                                      expiry_days: int) -> float:
        """Estimate probability of reaching breakeven (simplified)"""
        # Simplified probability calculation using normal distribution
        # In practice, would use more sophisticated models
        
        annual_volatility = 0.25  # 25% assumed volatility
        time_factor = np.sqrt(expiry_days / 365.0)
        
        total_prob = 0
        for breakeven in breakevens:
            # Calculate probability of reaching breakeven
            z_score = abs(np.log(breakeven / spot_price)) / (annual_volatility * time_factor)
            prob = 1 - stats.norm.cdf(z_score)
            total_prob += prob
        
        return min(total_prob * 100, 95.0)  # Cap at 95%
    
    def _estimate_range_probability(self, spot_price: float, lower_bound: float, 
                                   upper_bound: float, expiry_days: int) -> float:
        """Estimate probability of staying within range"""
        annual_volatility = 0.25
        time_factor = np.sqrt(expiry_days / 365.0)
        
        lower_z = np.log(lower_bound / spot_price) / (annual_volatility * time_factor)
        upper_z = np.log(upper_bound / spot_price) / (annual_volatility * time_factor)
        
        prob = stats.norm.cdf(upper_z) - stats.norm.cdf(lower_z)
        return prob * 100
    
    def _determine_market_sentiment(self, pcr_analysis: Dict, oi_analysis: Dict, 
                                   flow_analysis: Dict) -> str:
        """Determine overall market sentiment from options data"""
        try:
            sentiment_score = 0
            
            # PCR sentiment
            if pcr_analysis.get('volume_sentiment') == 'BULLISH':
                sentiment_score += 1
            elif pcr_analysis.get('volume_sentiment') == 'BEARISH':
                sentiment_score -= 1
            
            if pcr_analysis.get('oi_sentiment') == 'BULLISH':
                sentiment_score += 1
            elif pcr_analysis.get('oi_sentiment') == 'BEARISH':
                sentiment_score -= 1
            
            # Volume flow sentiment
            call_put_ratio = flow_analysis.get('call_put_volume_ratio', 1.0)
            if call_put_ratio > 1.2:
                sentiment_score += 1
            elif call_put_ratio < 0.8:
                sentiment_score -= 1
            
            # Determine overall sentiment
            if sentiment_score >= 2:
                return 'BULLISH'
            elif sentiment_score <= -2:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Error determining market sentiment: {str(e)}")
            return 'NEUTRAL'
    
    def _basic_options_analysis(self, options_data: pd.DataFrame, spot_price: float) -> Dict[str, any]:
        """Basic options analysis for error cases"""
        return {
            'options_data': options_data,
            'pcr_analysis': {'pcr_volume': 1.0, 'pcr_oi': 1.0, 'volume_sentiment': 'NEUTRAL', 'oi_sentiment': 'NEUTRAL'},
            'oi_analysis': {},
            'max_pain': {'max_pain_strike': spot_price, 'max_pain_value': 0},
            'iv_analysis': {'overall_avg_iv': 25.0},
            'flow_analysis': {'total_volume': 0},
            'recommended_strategies': [],
            'market_sentiment': 'NEUTRAL'
        }
    
    def generate_options_signals(self, underlying: str, options_analysis: Dict, 
                                spot_price: float, expiry_days: int) -> List[OptionsSignal]:
        """Generate specific options trading signals"""
        try:
            signals = []
            
            # Get recommended strategies
            strategies = options_analysis.get('recommended_strategies', [])
            
            for strategy in strategies[:3]:  # Top 3 strategies
                if strategy.probability > 60:  # Only high-probability strategies
                    
                    # Create signal based on strategy
                    if strategy.name == "Long Straddle":
                        signal = self._create_straddle_signal(underlying, strategy, spot_price, expiry_days)
                    elif strategy.name == "Bull Call Spread":
                        signal = self._create_bull_call_signal(underlying, strategy, spot_price, expiry_days)
                    elif strategy.name == "Iron Condor":
                        signal = self._create_iron_condor_signal(underlying, strategy, spot_price, expiry_days)
                    else:
                        continue
                    
                    if signal:
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating options signals: {str(e)}")
            return []
    
    def _create_straddle_signal(self, underlying: str, strategy: OptionsStrategy, 
                               spot_price: float, expiry_days: int) -> Optional[OptionsSignal]:
        """Create signal for straddle strategy"""
        try:
            call_leg = next(leg for leg in strategy.legs if leg['option_type'] == 'CALL')
            
            return OptionsSignal(
                underlying=underlying,
                strategy="Long Straddle",
                action="BUY",
                strike=call_leg['strike'],
                option_type="BOTH",
                expiry=f"{expiry_days} days",
                entry_price=sum(leg['premium'] for leg in strategy.legs),
                target_price=sum(leg['premium'] for leg in strategy.legs) * 2,
                stop_loss=sum(leg['premium'] for leg in strategy.legs) * 0.5,
                confidence=strategy.probability,
                max_risk=strategy.max_loss,
                max_reward=strategy.max_profit if strategy.max_profit != float('inf') else strategy.max_loss * 3,
                probability=strategy.probability,
                reasoning=f"High volatility expected. Profit if price moves beyond {strategy.breakeven[0]:.0f} or {strategy.breakeven[1]:.0f}"
            )
            
        except Exception as e:
            logger.error(f"Error creating straddle signal: {str(e)}")
            return None
    
    def _create_bull_call_signal(self, underlying: str, strategy: OptionsStrategy, 
                                spot_price: float, expiry_days: int) -> Optional[OptionsSignal]:
        """Create signal for bull call spread"""
        try:
            buy_leg = next(leg for leg in strategy.legs if leg['action'] == 'BUY')
            
            return OptionsSignal(
                underlying=underlying,
                strategy="Bull Call Spread",
                action="BUY",
                strike=buy_leg['strike'],
                option_type="CALL",
                expiry=f"{expiry_days} days",
                entry_price=strategy.max_loss,  # Net debit
                target_price=strategy.max_profit,
                stop_loss=strategy.max_loss * 0.5,
                confidence=strategy.probability,
                max_risk=strategy.max_loss,
                max_reward=strategy.max_profit,
                probability=strategy.probability,
                reasoning=f"Bullish outlook. Profit if price rises above {strategy.breakeven[0]:.0f}"
            )
            
        except Exception as e:
            logger.error(f"Error creating bull call signal: {str(e)}")
            return None
    
    def _create_iron_condor_signal(self, underlying: str, strategy: OptionsStrategy, 
                                  spot_price: float, expiry_days: int) -> Optional[OptionsSignal]:
        """Create signal for iron condor"""
        try:
            return OptionsSignal(
                underlying=underlying,
                strategy="Iron Condor",
                action="SELL",
                strike=spot_price,  # Center strike
                option_type="BOTH",
                expiry=f"{expiry_days} days",
                entry_price=strategy.max_profit,  # Net credit
                target_price=strategy.max_profit * 0.5,  # Take profit at 50%
                stop_loss=strategy.max_loss * 0.5,
                confidence=strategy.probability,
                max_risk=strategy.max_loss,
                max_reward=strategy.max_profit,
                probability=strategy.probability,
                reasoning=f"Range-bound market expected. Profit if price stays between {strategy.breakeven[0]:.0f} and {strategy.breakeven[1]:.0f}"
            )
            
        except Exception as e:
            logger.error(f"Error creating iron condor signal: {str(e)}")
            return None

# Export main classes
__all__ = ['OptionsAnalyzer', 'OptionData', 'OptionsStrategy', 'OptionsSignal']
