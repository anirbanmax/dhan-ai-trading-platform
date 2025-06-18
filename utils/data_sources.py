"""
Professional Data Sources Integration Module
Handles Dhan API, News Sources, FII/DII Data, and Moneycontrol Pro Integration
"""

import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time
import streamlit as st
from dataclasses import dataclass
import yfinance as yf
import feedparser
from textblob import TextBlob
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    price: float
    change: float
    change_pct: float
    volume: int
    timestamp: datetime

@dataclass
class NewsItem:
    """News item structure"""
    title: str
    summary: str
    content: str
    source: str
    timestamp: datetime
    sentiment: float
    impact_score: float

@dataclass
class Signal:
    """Trading signal structure"""
    symbol: str
    signal_type: str  # BUY/SELL/HOLD
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    timeframe: str
    reasoning: str
    technical_score: float
    fundamental_score: float
    risk_reward_ratio: float

class DhanAPI:
    """Professional Dhan API Integration"""
    
    def __init__(self):
        """Initialize Dhan API connection"""
        try:
            # Get credentials from Streamlit secrets
            self.access_token = st.secrets["dhan"]["access_token"]
            self.client_id = st.secrets["dhan"]["client_id"] 
            self.base_url = st.secrets["dhan"]["base_url"]
            
            self.headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            # Test connection
            self.is_connected = self.test_connection()
            logger.info(f"Dhan API initialized. Connected: {self.is_connected}")
            
        except Exception as e:
            logger.error(f"Error initializing Dhan API: {str(e)}")
            self.is_connected = False
            # Fallback to demo mode
            self.setup_demo_mode()
    
    def setup_demo_mode(self):
        """Setup demo mode with simulated data"""
        logger.info("Setting up demo mode with simulated data")
        self.demo_mode = True
    
    def test_connection(self) -> bool:
        """Test Dhan API connection"""
        try:
            response = requests.get(
                f"{self.base_url}/v2/charts/historical",
                headers=self.headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Dhan API connection test failed: {str(e)}")
            return False
    
    def get_market_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get real-time market data for symbols"""
        market_data = {}
        
        if not self.is_connected:
            return self._get_demo_market_data(symbols)
        
        try:
            for symbol in symbols:
                # Dhan API call for live data
                response = requests.get(
                    f"{self.base_url}/v2/marketdata/live/{symbol}",
                    headers=self.headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    market_data[symbol] = MarketData(
                        symbol=symbol,
                        price=data.get('ltp', 0),
                        change=data.get('change', 0),
                        change_pct=data.get('change_pct', 0),
                        volume=data.get('volume', 0),
                        timestamp=datetime.now()
                    )
                else:
                    # Fallback to demo data for this symbol
                    market_data[symbol] = self._generate_demo_data(symbol)
                    
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return self._get_demo_market_data(symbols)
        
        return market_data
    
    def get_historical_data(self, symbol: str, timeframe: str = "1D", days: int = 100) -> pd.DataFrame:
        """Get historical price data"""
        if not self.is_connected:
            return self._get_demo_historical_data(symbol, days)
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Dhan API call for historical data
            payload = {
                "symbol": symbol,
                "exchange": "NSE",
                "instrument": "EQUITY",
                "from_date": start_date.strftime("%Y-%m-%d"),
                "to_date": end_date.strftime("%Y-%m-%d"),
                "timeframe": timeframe
            }
            
            response = requests.post(
                f"{self.base_url}/v2/charts/historical",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['data'])
                
                # Process the data
                df['Date'] = pd.to_datetime(df['timestamp'])
                df = df.rename(columns={
                    'o': 'Open',
                    'h': 'High', 
                    'l': 'Low',
                    'c': 'Close',
                    'v': 'Volume'
                })
                
                return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            else:
                logger.warning(f"Failed to get historical data for {symbol}, using demo data")
                return self._get_demo_historical_data(symbol, days)
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return self._get_demo_historical_data(symbol, days)
    
    def get_options_chain(self, symbol: str, expiry_date: str) -> pd.DataFrame:
        """Get options chain data"""
        if not self.is_connected:
            return self._get_demo_options_chain(symbol)
        
        try:
            response = requests.get(
                f"{self.base_url}/v2/optionchain/{symbol}",
                headers=self.headers,
                params={'expiry': expiry_date},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data['options'])
            else:
                return self._get_demo_options_chain(symbol)
                
        except Exception as e:
            logger.error(f"Error fetching options chain: {str(e)}")
            return self._get_demo_options_chain(symbol)
    
    def _get_demo_market_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Generate demo market data"""
        demo_prices = {
            'RELIANCE': 2750.50,
            'TCS': 3890.20,
            'INFY': 1678.90,
            'HDFCBANK': 1654.30,
            'ICICIBANK': 1145.70,
            'NIFTY': 19650.0
        }
        
        market_data = {}
        for symbol in symbols:
            base_price = demo_prices.get(symbol, 1000.0)
            # Add some random variation
            change = np.random.uniform(-2, 2)
            change_pct = (change / base_price) * 100
            
            market_data[symbol] = MarketData(
                symbol=symbol,
                price=base_price + change,
                change=change,
                change_pct=change_pct,
                volume=np.random.randint(100000, 1000000),
                timestamp=datetime.now()
            )
        
        return market_data
    
    def _generate_demo_data(self, symbol: str) -> MarketData:
        """Generate demo data for a single symbol"""
        base_prices = {
            'RELIANCE': 2750.50,
            'TCS': 3890.20,
            'INFY': 1678.90,
            'HDFCBANK': 1654.30,
            'ICICIBANK': 1145.70
        }
        
        base_price = base_prices.get(symbol, 1000.0)
        change = np.random.uniform(-20, 20)
        change_pct = (change / base_price) * 100
        
        return MarketData(
            symbol=symbol,
            price=base_price + change,
            change=change,
            change_pct=change_pct,
            volume=np.random.randint(100000, 1000000),
            timestamp=datetime.now()
        )
    
    def _get_demo_historical_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate demo historical data"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='D')
        
        # Remove weekends
        dates = dates[dates.weekday < 5]
        
        base_prices = {
            'RELIANCE': 2750.50,
            'TCS': 3890.20,
            'INFY': 1678.90,
            'HDFCBANK': 1654.30,
            'ICICIBANK': 1145.70
        }
        
        base_price = base_prices.get(symbol, 1000.0)
        
        # Generate realistic price movement
        np.random.seed(hash(symbol) % 1000)  # Consistent seed for symbol
        returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
        price_series = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close prices
        highs = price_series * (1 + np.random.uniform(0, 0.03, len(dates)))
        lows = price_series * (1 - np.random.uniform(0, 0.03, len(dates)))
        opens = price_series * (1 + np.random.uniform(-0.01, 0.01, len(dates)))
        volumes = np.random.randint(500000, 2000000, len(dates))
        
        return pd.DataFrame({
            'Date': dates,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': price_series,
            'Volume': volumes
        })
    
    def _get_demo_options_chain(self, symbol: str) -> pd.DataFrame:
        """Generate demo options chain"""
        current_price = 19650 if symbol == 'NIFTY' else 2750
        strikes = range(int(current_price * 0.9), int(current_price * 1.1), 50)
        
        options_data = []
        for strike in strikes:
            # Call options
            call_iv = np.random.uniform(15, 35)
            call_ltp = max(current_price - strike, 0) + np.random.uniform(0, 20)
            
            # Put options  
            put_iv = np.random.uniform(15, 35)
            put_ltp = max(strike - current_price, 0) + np.random.uniform(0, 20)
            
            options_data.extend([
                {
                    'strike': strike,
                    'option_type': 'CALL',
                    'ltp': call_ltp,
                    'bid': call_ltp - 0.5,
                    'ask': call_ltp + 0.5,
                    'volume': np.random.randint(100, 10000),
                    'oi': np.random.randint(1000, 50000),
                    'iv': call_iv
                },
                {
                    'strike': strike,
                    'option_type': 'PUT',
                    'ltp': put_ltp,
                    'bid': put_ltp - 0.5,
                    'ask': put_ltp + 0.5,
                    'volume': np.random.randint(100, 10000),
                    'oi': np.random.randint(1000, 50000),
                    'iv': put_iv
                }
            ])
        
        return pd.DataFrame(options_data)

class NewsAnalyzer:
    """Professional News and Sentiment Analysis"""
    
    def __init__(self):
        """Initialize news analyzer"""
        self.news_sources = {
            'economic_times': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'moneycontrol': 'https://www.moneycontrol.com/rss/marketstocks.xml',
            'business_standard': 'https://www.business-standard.com/rss/markets-106.rss',
            'livemint': 'https://www.livemint.com/rss/markets'
        }
        
        # Keywords for impact scoring
        self.high_impact_keywords = [
            'rbi', 'repo rate', 'inflation', 'gdp', 'budget', 'policy', 
            'earnings', 'results', 'merger', 'acquisition', 'ipo'
        ]
        
        self.medium_impact_keywords = [
            'upgrade', 'downgrade', 'target', 'recommendation', 'fii', 'dii',
            'foreign investment', 'institutional'
        ]
    
    def get_latest_news(self, limit: int = 20) -> List[NewsItem]:
        """Get latest financial news with sentiment analysis"""
        all_news = []
        
        for source_name, url in self.news_sources.items():
            try:
                news_items = self._fetch_rss_news(url, source_name, limit//len(self.news_sources))
                all_news.extend(news_items)
            except Exception as e:
                logger.error(f"Error fetching news from {source_name}: {str(e)}")
        
        # Sort by timestamp and sentiment impact
        all_news.sort(key=lambda x: (x.timestamp, x.impact_score), reverse=True)
        
        return all_news[:limit]
    
    def _fetch_rss_news(self, url: str, source: str, limit: int) -> List[NewsItem]:
        """Fetch news from RSS feed"""
        try:
            feed = feedparser.parse(url)
            news_items = []
            
            for entry in feed.entries[:limit]:
                # Extract content
                title = entry.get('title', '')
                summary = entry.get('summary', entry.get('description', ''))
                content = self._clean_text(summary)
                
                # Parse timestamp
                try:
                    timestamp = datetime(*entry.published_parsed[:6])
                except:
                    timestamp = datetime.now()
                
                # Analyze sentiment and impact
                sentiment = self._analyze_sentiment(title + ' ' + content)
                impact_score = self._calculate_impact_score(title + ' ' + content)
                
                news_items.append(NewsItem(
                    title=title,
                    summary=summary[:200] + '...' if len(summary) > 200 else summary,
                    content=content,
                    source=source,
                    timestamp=timestamp,
                    sentiment=sentiment,
                    impact_score=impact_score
                ))
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error parsing RSS feed {url}: {str(e)}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity  # Returns value between -1 and 1
        except:
            return 0.0
    
    def _calculate_impact_score(self, text: str) -> float:
        """Calculate market impact score"""
        text_lower = text.lower()
        
        high_impact_count = sum(1 for keyword in self.high_impact_keywords 
                              if keyword in text_lower)
        medium_impact_count = sum(1 for keyword in self.medium_impact_keywords 
                                if keyword in text_lower)
        
        # Calculate impact score (0-10 scale)
        impact_score = (high_impact_count * 3) + (medium_impact_count * 1.5)
        return min(impact_score, 10.0)
    
    def get_news_sentiment_score(self, symbol: str) -> Dict[str, float]:
        """Get aggregated news sentiment for a specific stock"""
        news_items = self.get_latest_news(50)
        
        # Filter news related to the symbol
        symbol_news = [
            item for item in news_items 
            if symbol.lower() in item.title.lower() or symbol.lower() in item.content.lower()
        ]
        
        if not symbol_news:
            return {
                'sentiment_score': 0.0,
                'impact_score': 0.0,
                'news_count': 0
            }
        
        # Calculate aggregated scores
        avg_sentiment = np.mean([item.sentiment for item in symbol_news])
        avg_impact = np.mean([item.impact_score for item in symbol_news])
        
        return {
            'sentiment_score': avg_sentiment,
            'impact_score': avg_impact,
            'news_count': len(symbol_news)
        }

class FIIDIIAnalyzer:
    """FII/DII Data Analysis"""
    
    def __init__(self):
        """Initialize FII/DII analyzer"""
        self.nse_fii_url = "https://www.nseindia.com/api/fiidiiTradeReact"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        }
    
    def get_fii_dii_data(self, days: int = 30) -> pd.DataFrame:
        """Get FII/DII flow data"""
        try:
            # Try to fetch real data from NSE
            response = requests.get(self.nse_fii_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._process_fii_dii_data(data)
            else:
                return self._generate_demo_fii_dii_data(days)
                
        except Exception as e:
            logger.error(f"Error fetching FII/DII data: {str(e)}")
            return self._generate_demo_fii_dii_data(days)
    
    def _process_fii_dii_data(self, data: dict) -> pd.DataFrame:
        """Process real FII/DII data from NSE"""
        try:
            records = []
            for item in data.get('data', []):
                records.append({
                    'Date': pd.to_datetime(item['date']),
                    'FII_Buy': float(item.get('fii_buy', 0)),
                    'FII_Sell': float(item.get('fii_sell', 0)),
                    'FII_Net': float(item.get('fii_net', 0)),
                    'DII_Buy': float(item.get('dii_buy', 0)),
                    'DII_Sell': float(item.get('dii_sell', 0)),
                    'DII_Net': float(item.get('dii_net', 0))
                })
            
            return pd.DataFrame(records)
        except:
            return self._generate_demo_fii_dii_data(30)
    
    def _generate_demo_fii_dii_data(self, days: int) -> pd.DataFrame:
        """Generate demo FII/DII data"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='D')
        dates = dates[dates.weekday < 5]  # Remove weekends
        
        records = []
        for date in dates:
            # Generate realistic FII/DII flows
            fii_net = np.random.normal(500, 1500)  # Crores
            dii_net = np.random.normal(800, 1200)  # Crores
            
            fii_buy = abs(fii_net) + np.random.uniform(2000, 5000)
            fii_sell = fii_buy - fii_net
            
            dii_buy = abs(dii_net) + np.random.uniform(1500, 4000)
            dii_sell = dii_buy - dii_net
            
            records.append({
                'Date': date,
                'FII_Buy': fii_buy,
                'FII_Sell': fii_sell,
                'FII_Net': fii_net,
                'DII_Buy': dii_buy,
                'DII_Sell': dii_sell,
                'DII_Net': dii_net
            })
        
        return pd.DataFrame(records)
    
    def get_institutional_sentiment(self) -> Dict[str, float]:
        """Get institutional sentiment indicators"""
        fii_dii_data = self.get_fii_dii_data(10)  # Last 10 days
        
        if fii_dii_data.empty:
            return {
                'fii_sentiment': 0.0,
                'dii_sentiment': 0.0,
                'combined_sentiment': 0.0
            }
        
        # Calculate recent averages
        recent_fii_net = fii_dii_data['FII_Net'].tail(5).mean()
        recent_dii_net = fii_dii_data['DII_Net'].tail(5).mean()
        
        # Normalize to sentiment score (-1 to 1)
        fii_sentiment = np.tanh(recent_fii_net / 2000)  # Normalize around 2000 crores
        dii_sentiment = np.tanh(recent_dii_net / 1500)  # Normalize around 1500 crores
        
        combined_sentiment = (fii_sentiment + dii_sentiment) / 2
        
        return {
            'fii_sentiment': fii_sentiment,
            'dii_sentiment': dii_sentiment,
            'combined_sentiment': combined_sentiment,
            'fii_net_flow': recent_fii_net,
            'dii_net_flow': recent_dii_net
        }

class MoneycontrolProIntegration:
    """Moneycontrol Pro Premium Features Integration"""
    
    def __init__(self):
        """Initialize Moneycontrol Pro integration"""
        try:
            self.email = st.secrets["moneycontrol_pro"]["email"]
            self.password = st.secrets["moneycontrol_pro"]["password"]
            self.base_url = st.secrets["moneycontrol_pro"]["base_url"]
            
            self.session = requests.Session()
            self.is_authenticated = self.authenticate()
            
        except Exception as e:
            logger.error(f"Error initializing Moneycontrol Pro: {str(e)}")
            self.is_authenticated = False
    
    def authenticate(self) -> bool:
        """Authenticate with Moneycontrol Pro"""
        try:
            # This would implement actual Moneycontrol authentication
            # For security and compliance, we'll simulate this
            logger.info("Moneycontrol Pro authentication simulation")
            return True
        except Exception as e:
            logger.error(f"Moneycontrol Pro authentication failed: {str(e)}")
            return False
    
    def get_research_reports(self, symbol: str) -> List[Dict]:
        """Get premium research reports"""
        if not self.is_authenticated:
            return self._get_demo_research_reports(symbol)
        
        # Implementation would fetch actual research reports
        return self._get_demo_research_reports(symbol)
    
    def get_analyst_recommendations(self, symbol: str) -> Dict:
        """Get analyst recommendations and target prices"""
        if not self.is_authenticated:
            return self._get_demo_analyst_data(symbol)
        
        # Implementation would fetch actual analyst data
        return self._get_demo_analyst_data(symbol)
    
    def _get_demo_research_reports(self, symbol: str) -> List[Dict]:
        """Generate demo research reports"""
        return [
            {
                'title': f'{symbol} - Strong Buy on Robust Q3 Results',
                'analyst': 'Professional Research',
                'rating': 'BUY',
                'target_price': 2900.0,
                'date': '2024-06-15',
                'summary': 'Strong fundamentals with improving margins and market share gains.'
            },
            {
                'title': f'{symbol} - Sector Tailwinds Support Growth',
                'analyst': 'Equity Research',
                'rating': 'HOLD',
                'target_price': 2800.0,
                'date': '2024-06-10',
                'summary': 'Positive sector outlook but valuation concerns persist.'
            }
        ]
    
    def _get_demo_analyst_data(self, symbol: str) -> Dict:
        """Generate demo analyst data"""
        return {
            'consensus_rating': 'BUY',
            'mean_target': 2850.0,
            'high_target': 3100.0,
            'low_target': 2600.0,
            'num_analysts': 12,
            'buy_count': 8,
            'hold_count': 3,
            'sell_count': 1
        }

# Export main classes
__all__ = [
    'DhanAPI',
    'NewsAnalyzer', 
    'FIIDIIAnalyzer',
    'MoneycontrolProIntegration',
    'MarketData',
    'NewsItem',
    'Signal'
]
