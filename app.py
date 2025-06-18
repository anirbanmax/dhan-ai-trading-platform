import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from utils.data_sources import DhanAPI, NewsAnalyzer, FIIDIIAnalyzer
from utils.technical_analysis import TechnicalAnalyzer
from utils.options_analysis import OptionsAnalyzer
from utils.ai_models import AIPredictor
from utils.risk_management import RiskManager
from utils.ui_components import UIComponents

# Page configuration for mobile-first design
st.set_page_config(
    page_title="🚀 Dhan AI Trading",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Mobile-First CSS
PROFESSIONAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #F8FAFC;
}

/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
.stDeployButton {display:none;}
footer {visibility: hidden;}
.stDecoration {display:none;}

/* Professional header */
.main-header {
    background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
    padding: 1.5rem 1rem;
    border-radius: 0 0 20px 20px;
    color: white;
    text-align: center;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(30, 58, 138, 0.3);
}

.main-header h1 {
    margin: 0;
    font-size: 1.8rem;
    font-weight: 700;
}

.main-header p {
    margin: 0.5rem 0 0 0;
    opacity: 0.9;
    font-size: 1rem;
}

/* Professional metric cards */
.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-left: 4px solid #1E3A8A;
    margin-bottom: 1rem;
    transition: transform 0.2s ease;
}

.metric-card:active {
    transform: scale(0.98);
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0.5rem 0;
}

.metric-change {
    font-size: 0.9rem;
    font-weight: 600;
}

.positive { color: #10B981; }
.negative { color: #EF4444; }
.neutral { color: #6B7280; }

/* Signal cards */
.signal-card {
    background: white;
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    border: 1px solid #E5E7EB;
    position: relative;
    overflow: hidden;
}

.signal-card.buy::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, #10B981, #059669);
}

.signal-card.sell::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, #EF4444, #DC2626);
}

/* Professional buttons */
.pro-button {
    background: linear-gradient(135deg, #1E3A8A, #3B82F6);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 16px;
    min-height: 48px;
    width: 100%;
    margin: 8px 0;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(30, 58, 138, 0.3);
}

.pro-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(30, 58, 138, 0.4);
}

/* Chart containers */
.chart-container {
    background: white;
    border-radius: 16px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border: 1px solid #E5E7EB;
}

/* Mobile responsive grid */
.mobile-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

/* Bottom navigation */
.bottom-nav {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: white;
    border-top: 1px solid #E5E7EB;
    padding: 0.5rem;
    box-shadow: 0 -4px 12px rgba(0,0,0,0.1);
    z-index: 1000;
}

.nav-item {
    flex: 1;
    text-align: center;
    padding: 0.5rem;
    color: #6B7280;
    text-decoration: none;
    transition: color 0.2s;
}

.nav-item.active {
    color: #1E3A8A;
}

/* Loading animations */
.loading-spinner {
    border: 3px solid #F3F4F6;
    border-top: 3px solid #1E3A8A;
    border-radius: 50%;
    width: 30px;
    height: 30px;
    animation: spin 1s linear infinite;
    margin: 0 auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Mobile-specific adjustments */
@media (max-width: 768px) {
    .stSelectbox > div > div {
        font-size: 16px;
        min-height: 48px;
    }
    
    .stNumberInput > div > div > input {
        font-size: 16px;
        min-height: 48px;
    }
    
    .stButton > button {
        font-size: 16px;
        min-height: 48px;
        width: 100%;
    }
    
    .main-header h1 {
        font-size: 1.5rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
    }
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #F1F5F9;
}

::-webkit-scrollbar-thumb {
    background: #CBD5E1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #94A3B8;
}
</style>
"""

class DhanAITradingApp:
    def __init__(self):
        self.ui = UIComponents()
        self.setup_session_state()
        self.initialize_data_sources()
    
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Dashboard'
        
        if 'portfolio_value' not in st.session_state:
            st.session_state.portfolio_value = 1050000
        
        if 'daily_pnl' not in st.session_state:
            st.session_state.daily_pnl = 25000
        
        if 'signals_data' not in st.session_state:
            st.session_state.signals_data = []
        
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
    
    def initialize_data_sources(self):
        """Initialize all data source connections"""
        try:
            # Initialize Dhan API
            self.dhan_api = DhanAPI()
            
            # Initialize analyzers
            self.technical_analyzer = TechnicalAnalyzer()
            self.options_analyzer = OptionsAnalyzer()
            self.news_analyzer = NewsAnalyzer()
            self.fii_dii_analyzer = FIIDIIAnalyzer()
            self.ai_predictor = AIPredictor()
            self.risk_manager = RiskManager()
            
            # Check API connections
            self.api_status = self.check_api_connections()
            
        except Exception as e:
            st.error(f"Error initializing data sources: {str(e)}")
            self.api_status = {"dhan": False}
    
    def check_api_connections(self):
        """Check status of all API connections"""
        status = {}
        try:
            # Test Dhan API connection
            status['dhan'] = self.dhan_api.test_connection()
        except:
            status['dhan'] = False
        
        return status
    
    def render_header(self):
        """Render professional mobile header"""
        st.markdown("""
        <div class="main-header">
            <h1>🚀 DHAN AI TRADING</h1>
            <p>Professional AI-Powered Trading Platform</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_navigation(self):
        """Render mobile-friendly navigation"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        pages = {
            "📊 Dashboard": "Dashboard",
            "🎯 Signals": "Signals", 
            "💼 Portfolio": "Portfolio",
            "📈 Analysis": "Analysis",
            "⚙️ Settings": "Settings"
        }
        
        for i, (icon_label, page) in enumerate(pages.items()):
            col = [col1, col2, col3, col4, col5][i]
            if col.button(icon_label, key=f"nav_{page}", use_container_width=True):
                st.session_state.current_page = page
                st.experimental_rerun()
    
    def render_dashboard(self):
        """Render the main dashboard"""
        # Key metrics row
        col1, col2 = st.columns(2)
        
        with col1:
            pnl_pct = (st.session_state.daily_pnl / st.session_state.portfolio_value) * 100
            pnl_color = "positive" if st.session_state.daily_pnl > 0 else "negative"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; color: #374151; font-size: 0.9rem;">Portfolio Value</h3>
                <div class="metric-value">₹{st.session_state.portfolio_value:,.0f}</div>
                <div class="metric-change {pnl_color}">
                    Today: ₹{st.session_state.daily_pnl:+,.0f} ({pnl_pct:+.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Market overview (NIFTY)
            nifty_data = self.get_nifty_data()
            nifty_change_color = "positive" if nifty_data['change'] > 0 else "negative"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; color: #374151; font-size: 0.9rem;">NIFTY 50</h3>
                <div class="metric-value">{nifty_data['price']:,.0f}</div>
                <div class="metric-change {nifty_change_color}">
                    {nifty_data['change']:+.0f} ({nifty_data['change_pct']:+.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # AI Signals Section
        st.markdown("### 🎯 AI Signals")
        
        signals = self.get_latest_signals()
        if signals:
            for signal in signals[:3]:  # Show top 3 signals
                signal_type_class = signal['type'].lower()
                confidence_color = self.get_confidence_color(signal['confidence'])
                
                st.markdown(f"""
                <div class="signal-card {signal_type_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0; font-size: 1.1rem; color: #111827;">{signal['symbol']}</h4>
                            <p style="margin: 0.25rem 0; color: #6B7280;">₹{signal['price']:.2f}</p>
                        </div>
                        <div style="text-align: right;">
                            <span style="background: {confidence_color}; color: white; padding: 4px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                                {signal['type']} {signal['confidence']:.0f}%
                            </span>
                        </div>
                    </div>
                    <div style="margin-top: 0.75rem; font-size: 0.85rem; color: #4B5563;">
                        Target: ₹{signal['target']:.2f} | SL: ₹{signal['stop_loss']:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No active signals at the moment. Market analysis in progress...")
        
        # Market Overview Chart
        st.markdown("### 📈 Market Overview")
        chart_container = st.container()
        with chart_container:
            self.render_market_chart()
        
        # News & Events
        st.markdown("### 📰 Latest Market News")
        news_items = self.get_latest_news()
        
        for news in news_items[:3]:
            impact_color = self.get_impact_color(news['impact'])
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid {impact_color};">
                <h5 style="margin: 0 0 0.5rem 0; color: #111827;">{news['title']}</h5>
                <p style="margin: 0; color: #6B7280; font-size: 0.85rem;">{news['summary']}</p>
                <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #9CA3AF;">
                    {news['time']} | Impact: {news['impact']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_signals(self):
        """Render the signals page"""
        st.markdown("## 🎯 AI Trading Signals")
        
        # Filter tabs
        col1, col2, col3, col4 = st.columns(4)
        signal_filter = col1.selectbox("Filter", ["All", "BUY", "SELL", "OPTIONS"])
        timeframe = col2.selectbox("Timeframe", ["1D", "1W", "1M"])
        confidence_filter = col3.slider("Min Confidence", 0, 100, 70)
        
        if col4.button("🔄 Refresh Signals", use_container_width=True):
            self.refresh_signals()
        
        # Get filtered signals
        signals = self.get_filtered_signals(signal_filter, timeframe, confidence_filter)
        
        # Display signals
        for signal in signals:
            self.render_signal_card(signal)
    
    def render_signal_card(self, signal):
        """Render detailed signal card"""
        signal_type_color = "#10B981" if signal['type'] == 'BUY' else "#EF4444"
        confidence_color = self.get_confidence_color(signal['confidence'])
        
        st.markdown(f"""
        <div class="signal-card" style="border-left: 4px solid {signal_type_color};">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                <div>
                    <h3 style="margin: 0; color: #111827;">{signal['symbol']}</h3>
                    <p style="margin: 0.25rem 0; color: #6B7280; font-size: 0.9rem;">{signal['company_name']}</p>
                </div>
                <div style="text-align: right;">
                    <span style="background: {signal_type_color}; color: white; padding: 6px 12px; border-radius: 8px; font-weight: 600;">
                        {signal['type']}
                    </span>
                    <div style="margin-top: 0.5rem;">
                        <span style="background: {confidence_color}; color: white; padding: 4px 8px; border-radius: 6px; font-size: 0.85rem;">
                            {signal['confidence']:.0f}% Confidence
                        </span>
                    </div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1rem;">
                <div>
                    <h5 style="margin: 0; color: #6B7280; font-size: 0.8rem;">ENTRY PRICE</h5>
                    <p style="margin: 0.25rem 0; font-size: 1.1rem; font-weight: 600;">₹{signal['entry_price']:.2f}</p>
                </div>
                <div>
                    <h5 style="margin: 0; color: #6B7280; font-size: 0.8rem;">CURRENT PRICE</h5>
                    <p style="margin: 0.25rem 0; font-size: 1.1rem; font-weight: 600;">₹{signal['current_price']:.2f}</p>
                </div>
                <div>
                    <h5 style="margin: 0; color: #10B981; font-size: 0.8rem;">TARGET</h5>
                    <p style="margin: 0.25rem 0; font-size: 1.1rem; font-weight: 600; color: #10B981;">₹{signal['target']:.2f}</p>
                </div>
                <div>
                    <h5 style="margin: 0; color: #EF4444; font-size: 0.8rem;">STOP LOSS</h5>
                    <p style="margin: 0.25rem 0; font-size: 1.1rem; font-weight: 600; color: #EF4444;">₹{signal['stop_loss']:.2f}</p>
                </div>
            </div>
            
            <div style="background: #F8FAFC; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <h5 style="margin: 0 0 0.5rem 0; color: #374151;">📊 Technical Analysis</h5>
                <p style="margin: 0; color: #6B7280; font-size: 0.9rem;">{signal['technical_reason']}</p>
            </div>
            
            <div style="background: #F0F9FF; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <h5 style="margin: 0 0 0.5rem 0; color: #374151;">🧠 AI Analysis</h5>
                <p style="margin: 0; color: #6B7280; font-size: 0.9rem;">{signal['ai_reason']}</p>
            </div>
            
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 0.8rem; color: #9CA3AF;">
                    Risk-Reward: 1:{signal['risk_reward']:.1f} | Generated: {signal['generated_time']}
                </div>
                <button style="background: #1E3A8A; color: white; border: none; padding: 8px 16px; border-radius: 6px; font-size: 0.85rem;">
                    📱 Set Alert
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_portfolio(self):
        """Render portfolio analysis page"""
        st.markdown("## 💼 Portfolio Analysis")
        
        # Portfolio summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Value", f"₹{st.session_state.portfolio_value:,.0f}", 
                     delta=f"₹{st.session_state.daily_pnl:+,.0f}")
        
        with col2:
            total_invested = 950000
            returns = st.session_state.portfolio_value - total_invested
            returns_pct = (returns / total_invested) * 100
            st.metric("Total Returns", f"₹{returns:+,.0f}", delta=f"{returns_pct:+.1f}%")
        
        with col3:
            st.metric("Active Positions", "12", delta="2")
        
        # Holdings table
        st.markdown("### 📈 Current Holdings")
        holdings_data = self.get_portfolio_holdings()
        
        for holding in holdings_data:
            pnl_color = "positive" if holding['pnl'] > 0 else "negative"
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid {'#10B981' if holding['pnl'] > 0 else '#EF4444'};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0; color: #111827;">{holding['symbol']}</h4>
                        <p style="margin: 0.25rem 0; color: #6B7280; font-size: 0.85rem;">
                            {holding['quantity']} shares @ ₹{holding['avg_price']:.2f}
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <p style="margin: 0; font-size: 1.1rem; font-weight: 600;">₹{holding['current_value']:,.0f}</p>
                        <p style="margin: 0.25rem 0; color: {'#10B981' if holding['pnl'] > 0 else '#EF4444'}; font-weight: 600;">
                            ₹{holding['pnl']:+,.0f} ({holding['pnl_pct']:+.1f}%)
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_analysis(self):
        """Render technical analysis page"""
        st.markdown("## 📈 Technical Analysis")
        
        # Stock selector
        col1, col2 = st.columns([2, 1])
        symbol = col1.selectbox("Select Stock", ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"])
        timeframe = col2.selectbox("Timeframe", ["1D", "1W", "1M"])
        
        # Get stock data and render chart
        stock_data = self.get_stock_data(symbol, timeframe)
        if stock_data is not None:
            fig = self.create_technical_chart(stock_data, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators summary
            st.markdown("### 📊 Technical Indicators")
            indicators = self.get_technical_indicators(stock_data)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RSI", f"{indicators['rsi']:.1f}", 
                       delta="Oversold" if indicators['rsi'] < 30 else "Overbought" if indicators['rsi'] > 70 else "Neutral")
            col2.metric("MACD", f"{indicators['macd']:.2f}", 
                       delta="Bullish" if indicators['macd'] > 0 else "Bearish")
            col3.metric("SMA 20", f"₹{indicators['sma_20']:.2f}")
            col4.metric("Volume", f"{indicators['volume']:,.0f}")
    
    def render_settings(self):
        """Render settings page"""
        st.markdown("## ⚙️ Settings")
        
        # API Status
        st.markdown("### 🔌 API Connections")
        status_color = "#10B981" if self.api_status.get('dhan', False) else "#EF4444"
        status_text = "Connected" if self.api_status.get('dhan', False) else "Disconnected"
        
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; border-left: 4px solid {status_color};">
            <h4 style="margin: 0; color: #111827;">Dhan API</h4>
            <p style="margin: 0.25rem 0; color: {status_color}; font-weight: 600;">{status_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk Management Settings
        st.markdown("### ⚖️ Risk Management")
        
        col1, col2 = st.columns(2)
        max_position_size = col1.slider("Max Position Size (%)", 1, 10, 5)
        stop_loss_pct = col2.slider("Default Stop Loss (%)", 1, 10, 3)
        
        risk_reward_ratio = col1.slider("Min Risk-Reward Ratio", 1.0, 5.0, 2.0)
        max_daily_trades = col2.number_input("Max Daily Trades", 1, 20, 5)
        
        # AI Model Settings
        st.markdown("### 🧠 AI Model Configuration")
        
        confidence_threshold = st.slider("Signal Confidence Threshold (%)", 50, 95, 75)
        enable_auto_signals = st.checkbox("Enable Automatic Signal Generation", True)
        enable_news_analysis = st.checkbox("Enable News Sentiment Analysis", True)
        
        if st.button("💾 Save Settings", use_container_width=True):
            self.save_settings({
                'max_position_size': max_position_size,
                'stop_loss_pct': stop_loss_pct,
                'risk_reward_ratio': risk_reward_ratio,
                'max_daily_trades': max_daily_trades,
                'confidence_threshold': confidence_threshold,
                'enable_auto_signals': enable_auto_signals,
                'enable_news_analysis': enable_news_analysis
            })
            st.success("✅ Settings saved successfully!")
    
    # Helper methods
    def get_nifty_data(self):
        """Get NIFTY 50 current data"""
        try:
            # Simulate NIFTY data - replace with actual Dhan API call
            return {
                'price': 19650,
                'change': 125,
                'change_pct': 0.64
            }
        except:
            return {'price': 19500, 'change': 0, 'change_pct': 0.0}
    
    def get_latest_signals(self):
        """Get latest AI trading signals"""
        # Simulate signals - replace with actual AI predictions
        return [
            {
                'symbol': 'RELIANCE',
                'type': 'BUY',
                'price': 2750.50,
                'target': 2900.00,
                'stop_loss': 2650.00,
                'confidence': 85.5,
                'entry_price': 2750.50,
                'current_price': 2755.30,
                'company_name': 'Reliance Industries Ltd',
                'technical_reason': 'Golden cross formation with RSI showing bullish divergence. Strong support at 2700.',
                'ai_reason': 'ML models indicate 85% probability of upward movement based on historical patterns and current market conditions.',
                'risk_reward': 2.8,
                'generated_time': '2 hours ago'
            },
            {
                'symbol': 'TCS',
                'type': 'HOLD',
                'price': 3890.20,
                'target': 4050.00,
                'stop_loss': 3750.00,
                'confidence': 72.3,
                'entry_price': 3850.00,
                'current_price': 3890.20,
                'company_name': 'Tata Consultancy Services',
                'technical_reason': 'Consolidating near resistance. Volume declining, waiting for breakout confirmation.',
                'ai_reason': 'Medium confidence signal. Market conditions suggest sideways movement in short term.',
                'risk_reward': 1.8,
                'generated_time': '1 hour ago'
            }
        ]
    
    def get_latest_news(self):
        """Get latest market news"""
        return [
            {
                'title': 'RBI maintains repo rate at 6.5%, focuses on growth',
                'summary': 'Reserve Bank of India keeps key interest rates unchanged as expected by market participants.',
                'time': '2 hours ago',
                'impact': 'High'
            },
            {
                'title': 'IT sector shows strong Q3 results',
                'summary': 'Major IT companies report better-than-expected earnings with strong guidance.',
                'time': '4 hours ago',
                'impact': 'Medium'
            },
            {
                'title': 'FII inflows continue for third consecutive day',
                'summary': 'Foreign institutional investors pump in ₹2,500 crores in Indian equities.',
                'time': '6 hours ago',
                'impact': 'Medium'
            }
        ]
    
    def get_confidence_color(self, confidence):
        """Get color based on confidence level"""
        if confidence >= 80:
            return "#10B981"  # Green
        elif confidence >= 65:
            return "#F59E0B"  # Amber
        else:
            return "#EF4444"  # Red
    
    def get_impact_color(self, impact):
        """Get color based on news impact"""
        if impact == "High":
            return "#EF4444"
        elif impact == "Medium":
            return "#F59E0B"
        else:
            return "#10B981"
    
    def get_filtered_signals(self, signal_filter, timeframe, confidence_filter):
        """Get filtered signals based on criteria"""
        all_signals = self.get_latest_signals()
        
        # Apply filters
        filtered = all_signals
        if signal_filter != "All":
            filtered = [s for s in filtered if s['type'] == signal_filter]
        
        filtered = [s for s in filtered if s['confidence'] >= confidence_filter]
        
        return filtered
    
    def get_portfolio_holdings(self):
        """Get current portfolio holdings"""
        return [
            {
                'symbol': 'RELIANCE',
                'quantity': 100,
                'avg_price': 2650.00,
                'current_price': 2750.50,
                'current_value': 275050,
                'pnl': 10050,
                'pnl_pct': 3.79
            },
            {
                'symbol': 'TCS',
                'quantity': 50,
                'avg_price': 3800.00,
                'current_price': 3890.20,
                'current_value': 194510,
                'pnl': 4510,
                'pnl_pct': 2.37
            },
            {
                'symbol': 'INFY',
                'quantity': 75,
                'avg_price': 1650.00,
                'current_price': 1678.90,
                'current_value': 125917,
                'pnl': 2167,
                'pnl_pct': 1.75
            }
        ]
    
    def get_stock_data(self, symbol, timeframe):
        """Get stock price data"""
        # Simulate stock data - replace with actual Dhan API call
        dates = pd.date_range(start='2024-01-01', end='2024-06-18', freq='D')
        np.random.seed(42)
        
        price = 2750 + np.cumsum(np.random.randn(len(dates)) * 10)
        volume = np.random.randint(1000000, 5000000, len(dates))
        
        return pd.DataFrame({
            'Date': dates,
            'Close': price,
            'Volume': volume,
            'High': price + np.random.rand(len(dates)) * 20,
            'Low': price - np.random.rand(len(dates)) * 20,
            'Open': price + np.random.randn(len(dates)) * 5
        })
    
    def create_technical_chart(self, data, symbol):
        """Create technical analysis chart"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Price Chart', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=symbol
            ),
            row=1, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=data['Date'],
                y=data['Volume'],
                name='Volume',
                marker_color='rgba(59, 130, 246, 0.6)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            showlegend=False,
            font=dict(family="Inter", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def get_technical_indicators(self, data):
        """Calculate technical indicators"""
        return {
            'rsi': 65.5,
            'macd': 12.5,
            'sma_20': data['Close'].iloc[-20:].mean(),
            'volume': data['Volume'].iloc[-1]
        }
    
    def render_market_chart(self):
        """Render market overview chart"""
        # Create sample market data
        dates = pd.date_range(start='2024-06-01', end='2024-06-18', freq='D')
        nifty_data = 19500 + np.cumsum(np.random.randn(len(dates)) * 50)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=nifty_data,
            mode='lines',
            name='NIFTY 50',
            line=dict(color='#1E3A8A', width=3)
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
            font=dict(family="Inter", size=10),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#F3F4F6'),
            yaxis=dict(showgrid=True, gridcolor='#F3F4F6')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def refresh_signals(self):
        """Refresh trading signals"""
        st.session_state.last_update = datetime.now()
        st.success("🔄 Signals refreshed!")
    
    def save_settings(self, settings):
        """Save user settings"""
        # In production, save to database
        pass
    
    def run(self):
        """Main application runner"""
        # Apply CSS
        st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)
        
        # Render header
        self.render_header()
        
        # Render navigation
        self.render_navigation()
        
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Render current page
        if st.session_state.current_page == 'Dashboard':
            self.render_dashboard()
        elif st.session_state.current_page == 'Signals':
            self.render_signals()
        elif st.session_state.current_page == 'Portfolio':
            self.render_portfolio()
        elif st.session_state.current_page == 'Analysis':
            self.render_analysis()
        elif st.session_state.current_page == 'Settings':
            self.render_settings()
        
        # Add bottom padding for mobile
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

# Initialize and run the app
if __name__ == "__main__":
    app = DhanAITradingApp()
    app.run()
