import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import time

# Page configuration for mobile-first design
st.set_page_config(
    page_title="🚀 Dhan AI Trading",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Mobile-First CSS
st.markdown("""
<style>
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #F8FAFC;
}

.main-header {
    background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
    padding: 1.5rem 1rem;
    border-radius: 0 0 20px 20px;
    color: white;
    text-align: center;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(30, 58, 138, 0.3);
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-left: 4px solid #1E3A8A;
    margin-bottom: 1rem;
}

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
</style>
""", unsafe_allow_html=True)

class DhanAITradingApp:
    def __init__(self):
        self.portfolio_value = 1050000
        self.daily_pnl = 25000
        
    def run(self):
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🚀 DHAN AI TRADING</h1>
            <p>Professional AI-Powered Trading Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("📊 Dashboard", use_container_width=True):
                st.session_state.current_page = 'Dashboard'
        with col2:
            if st.button("🎯 Signals", use_container_width=True):
                st.session_state.current_page = 'Signals'
        with col3:
            if st.button("💼 Portfolio", use_container_width=True):
                st.session_state.current_page = 'Portfolio'
        with col4:
            if st.button("📈 Analysis", use_container_width=True):
                st.session_state.current_page = 'Analysis'
        with col5:
            if st.button("⚙️ Settings", use_container_width=True):
                st.session_state.current_page = 'Settings'
        
        # Initialize current page
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Dashboard'
        
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

    def render_dashboard(self):
        st.markdown("## 📊 Dashboard")
        
        # Key metrics
        col1, col2 = st.columns(2)
        
        with col1:
            pnl_pct = (self.daily_pnl / self.portfolio_value) * 100
            pnl_color = "positive" if self.daily_pnl > 0 else "negative"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; color: #374151; font-size: 0.9rem;">Portfolio Value</h3>
                <div style="font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0; color: #111827;">
                    ₹{self.portfolio_value:,.0f}
                </div>
                <div style="font-size: 0.85rem; font-weight: 600; color: {'#10B981' if self.daily_pnl > 0 else '#EF4444'};">
                    Today: ₹{self.daily_pnl:+,.0f} ({pnl_pct:+.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Market overview (NIFTY)
            nifty_price = 19650
            nifty_change = 125
            nifty_change_pct = 0.64
            nifty_color = "positive" if nifty_change > 0 else "negative"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; color: #374151; font-size: 0.9rem;">NIFTY 50</h3>
                <div style="font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0; color: #111827;">
                    {nifty_price:,.0f}
                </div>
                <div style="font-size: 0.85rem; font-weight: 600; color: {'#10B981' if nifty_change > 0 else '#EF4444'};">
                    {nifty_change:+.0f} ({nifty_change_pct:+.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # AI Signals Section
        st.markdown("### 🎯 AI Signals")
        self.render_sample_signals()
        
        # Market Chart
        st.markdown("### 📈 Market Overview")
        self.render_market_chart()
        
        # News Section
        st.markdown("### 📰 Latest Market News")
        self.render_sample_news()
    
    def render_signals(self):
        st.markdown("## 🎯 AI Trading Signals")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        signal_filter = col1.selectbox("Filter", ["All", "BUY", "SELL", "OPTIONS"])
        timeframe = col2.selectbox("Timeframe", ["1D", "1W", "1M"])
        confidence_filter = col3.slider("Min Confidence", 0, 100, 70)
        
        # Sample signals with detailed cards
        signals = self.get_sample_signals()
        
        for signal in signals:
            if signal['confidence'] >= confidence_filter:
                self.render_detailed_signal_card(signal)
    
    def render_portfolio(self):
        st.markdown("## 💼 Portfolio Analysis")
        
        # Portfolio summary
        col1, col2, col3 = st.columns(3)
        
        total_invested = 950000
        returns = self.portfolio_value - total_invested
        returns_pct = (returns / total_invested) * 100
        
        col1.metric("Total Value", f"₹{self.portfolio_value:,.0f}", 
                   delta=f"₹{self.daily_pnl:+,.0f}")
        col2.metric("Total Returns", f"₹{returns:+,.0f}", delta=f"{returns_pct:+.1f}%")
        col3.metric("Active Positions", "12", delta="2")
        
        # Holdings
        st.markdown("### 📈 Current Holdings")
        holdings = self.get_sample_holdings()
        
        for holding in holdings:
            self.render_portfolio_card(holding)
    
    def render_analysis(self):
        st.markdown("## 📈 Technical Analysis")
        
        # Stock selector
        col1, col2 = st.columns([2, 1])
        symbol = col1.selectbox("Select Stock", ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"])
        timeframe = col2.selectbox("Timeframe", ["1D", "1W", "1M"])
        
        # Chart
        st.markdown(f"### {symbol} Price Chart")
        self.render_stock_chart(symbol)
        
        # Technical indicators
        st.markdown("### 📊 Technical Indicators")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RSI", "65.5", "Neutral")
        col2.metric("MACD", "12.5", "Bullish")
        col3.metric("SMA 20", "₹2,735", "Support")
        col4.metric("Volume", "2.5M", "+15%")
    
    def render_settings(self):
        st.markdown("## ⚙️ Settings")
        
        # API Status
        st.markdown("### 🔌 API Connections")
        
        # Check for Dhan API secrets
        try:
            dhan_config = st.secrets.get("dhan", {})
            if dhan_config.get("access_token"):
                st.success("✅ Dhan API: Connected")
                st.info(f"Client ID: {dhan_config.get('client_id', 'Not found')}")
            else:
                st.warning("⚠️ Dhan API: Not configured")
        except Exception as e:
            st.error("❌ Dhan API: Configuration error")
        
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
            st.success("✅ Settings saved successfully!")
    
    def render_sample_signals(self):
        signals = self.get_sample_signals()[:3]  # Top 3 signals
        
        for signal in signals:
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
    
    def render_detailed_signal_card(self, signal):
        signal_color = "#10B981" if signal['type'] == 'BUY' else "#EF4444"
        
        st.markdown(f"""
        <div class="signal-card" style="border-left: 4px solid {signal_color};">
            <h3>{signal['symbol']} - {signal['type']}</h3>
            <p><strong>Entry:</strong> ₹{signal['price']:.2f}</p>
            <p><strong>Target:</strong> ₹{signal['target']:.2f}</p>
            <p><strong>Stop Loss:</strong> ₹{signal['stop_loss']:.2f}</p>
            <p><strong>Confidence:</strong> {signal['confidence']:.1f}%</p>
            <p><strong>Risk-Reward:</strong> 1:{signal['risk_reward']:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_portfolio_card(self, holding):
        pnl_color = "#10B981" if holding['pnl'] > 0 else "#EF4444"
        
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid {pnl_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: #111827;">{holding['symbol']}</h4>
                    <p style="margin: 0.25rem 0; color: #6B7280; font-size: 0.85rem;">
                        {holding['quantity']} shares @ ₹{holding['avg_price']:.2f}
                    </p>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 0; font-size: 1.1rem; font-weight: 600;">₹{holding['current_value']:,.0f}</p>
                    <p style="margin: 0.25rem 0; color: {pnl_color}; font-weight: 600;">
                        ₹{holding['pnl']:+,.0f} ({holding['pnl_pct']:+.1f}%)
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sample_news(self):
        news_items = [
            {
                'title': 'RBI maintains repo rate at 6.5%, focuses on growth',
                'summary': 'Reserve Bank maintains key rates as expected by market participants.',
                'time': '2 hours ago',
                'impact': 'High'
            },
            {
                'title': 'IT sector shows strong Q3 results',
                'summary': 'Major IT companies report better-than-expected earnings.',
                'time': '4 hours ago',
                'impact': 'Medium'
            }
        ]
        
        for news in news_items:
            impact_color = "#EF4444" if news['impact'] == 'High' else "#F59E0B"
            
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid {impact_color};">
                <h5 style="margin: 0; color: #111827;">{news['title']}</h5>
                <p style="margin: 0.5rem 0; color: #6B7280; font-size: 0.85rem;">{news['summary']}</p>
                <div style="font-size: 0.75rem; color: #9CA3AF;">
                    {news['time']} | Impact: {news['impact']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_market_chart(self):
        # Create sample market data
        dates = pd.date_range(start='2024-06-01', end='2024-06-18', freq='D')
        prices = 19500 + np.cumsum(np.random.randn(len(dates)) * 50)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='NIFTY 50',
            line=dict(color='#1E3A8A', width=3)
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_stock_chart(self, symbol):
        # Generate sample stock data
        dates = pd.date_range(start='2024-05-01', end='2024-06-18', freq='D')
        base_price = 2750 if symbol == 'RELIANCE' else 3890
        prices = base_price + np.cumsum(np.random.randn(len(dates)) * 20)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name=symbol,
            line=dict(color='#1E3A8A', width=3)
        ))
        
        fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def get_sample_signals(self):
        return [
            {
                'symbol': 'RELIANCE',
                'type': 'BUY',
                'price': 2750.50,
                'target': 2900.00,
                'stop_loss': 2650.00,
                'confidence': 85.5,
                'risk_reward': 2.8
            },
            {
                'symbol': 'TCS',
                'type': 'HOLD',
                'price': 3890.20,
                'target': 4050.00,
                'stop_loss': 3750.00,
                'confidence': 72.3,
                'risk_reward': 1.8
            },
            {
                'symbol': 'INFY',
                'type': 'BUY',
                'price': 1678.90,
                'target': 1750.00,
                'stop_loss': 1620.00,
                'confidence': 78.9,
                'risk_reward': 2.1
            }
        ]
    
    def get_sample_holdings(self):
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
    
    def get_confidence_color(self, confidence):
        if confidence >= 80:
            return "#10B981"
        elif confidence >= 65:
            return "#F59E0B"
        else:
            return "#EF4444"

# Initialize and run the app
if __name__ == "__main__":
    app = DhanAITradingApp()
    app.run()
