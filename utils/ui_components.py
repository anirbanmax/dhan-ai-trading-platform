"""
Professional UI Components Module
Reusable UI components for mobile-first trading interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class UIComponents:
    """Professional UI Components for Trading Platform"""
    
    def __init__(self):
        """Initialize UI components"""
        self.colors = {
            'primary': '#1E3A8A',
            'secondary': '#3B82F6', 
            'success': '#10B981',
            'danger': '#EF4444',
            'warning': '#F59E0B',
            'info': '#06B6D4',
            'light': '#F8FAFC',
            'dark': '#1F2937',
            'muted': '#6B7280'
        }
        
        self.chart_config = {
            'height': 400,
            'margin': dict(l=10, r=10, t=30, b=10),
            'font': dict(family="Inter", size=12),
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white'
        }
    
    def metric_card(self, title: str, value: str, change: str = "", 
                   change_color: str = "neutral", icon: str = "") -> None:
        """Professional metric card component"""
        try:
            color_map = {
                'positive': '#10B981',
                'negative': '#EF4444', 
                'neutral': '#6B7280',
                'warning': '#F59E0B',
                'info': '#06B6D4'
            }
            
            border_color = color_map.get(change_color, '#6B7280')
            
            st.markdown(f"""
            <div style="
                background: white;
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.08);
                border-left: 4px solid {border_color};
                margin-bottom: 1rem;
                transition: transform 0.2s ease;
            ">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex: 1;">
                        <h3 style="margin: 0 0 0.5rem 0; color: #374151; font-size: 0.9rem; font-weight: 500;">
                            {icon} {title}
                        </h3>
                        <p style="margin: 0; font-size: 1.8rem; font-weight: 700; color: #111827; line-height: 1.2;">
                            {value}
                        </p>
                        {f'<p style="margin: 0.5rem 0 0 0; color: {border_color}; font-size: 0.85rem; font-weight: 600;">{change}</p>' if change else ''}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error rendering metric card: {str(e)}")
            st.error("Error rendering metric card")
    
    def signal_card(self, signal: Dict[str, Any], detailed: bool = False) -> None:
        """Professional signal card component"""
        try:
            signal_type = signal.get('type', 'HOLD').upper()
            confidence = signal.get('confidence', 50)
            
            # Signal type styling
            if signal_type == 'BUY':
                signal_color = '#10B981'
                signal_bg = '#ECFDF5'
                signal_icon = '📈'
            elif signal_type == 'SELL':
                signal_color = '#EF4444'
                signal_bg = '#FEF2F2'
                signal_icon = '📉'
            else:
                signal_color = '#6B7280'
                signal_bg = '#F9FAFB'
                signal_icon = '➡️'
            
            # Confidence color
            if confidence >= 80:
                conf_color = '#10B981'
            elif confidence >= 65:
                conf_color = '#F59E0B'
            else:
                conf_color = '#EF4444'
            
            # Basic signal card
            card_html = f"""
            <div style="
                background: white;
                border-radius: 16px;
                padding: 1.25rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 16px rgba(0,0,0,0.1);
                border: 1px solid #E5E7EB;
                position: relative;
                overflow: hidden;
            ">
                <div style="
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 4px;
                    background: {signal_color};
                "></div>
                
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                    <div>
                        <h3 style="margin: 0; color: #111827; font-size: 1.2rem; font-weight: 600;">
                            {signal_icon} {signal.get('symbol', 'N/A')}
                        </h3>
                        <p style="margin: 0.25rem 0; color: #6B7280; font-size: 0.9rem;">
                            {signal.get('company_name', '')}
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <span style="
                            background: {signal_color};
                            color: white;
                            padding: 6px 12px;
                            border-radius: 8px;
                            font-weight: 600;
                            font-size: 0.9rem;
                        ">
                            {signal_type}
                        </span>
                        <div style="margin-top: 0.5rem;">
                            <span style="
                                background: {conf_color};
                                color: white;
                                padding: 4px 8px;
                                border-radius: 6px;
                                font-size: 0.8rem;
                                font-weight: 500;
                            ">
                                {confidence:.0f}% Confidence
                            </span>
                        </div>
                    </div>
                </div>
            """
            
            # Add price information
            if 'price' in signal or 'entry_price' in signal:
                price = signal.get('entry_price', signal.get('price', 0))
                target = signal.get('target', signal.get('target_price', 0))
                stop_loss = signal.get('stop_loss', signal.get('stop_loss_price', 0))
                
                card_html += f"""
                <div style="
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
                    gap: 1rem;
                    margin-bottom: 1rem;
                    padding: 1rem;
                    background: {signal_bg};
                    border-radius: 8px;
                ">
                    <div style="text-align: center;">
                        <h5 style="margin: 0; color: #6B7280; font-size: 0.75rem; text-transform: uppercase;">Entry</h5>
                        <p style="margin: 0.25rem 0; font-size: 1rem; font-weight: 600; color: #111827;">₹{price:.2f}</p>
                    </div>
                    <div style="text-align: center;">
                        <h5 style="margin: 0; color: #10B981; font-size: 0.75rem; text-transform: uppercase;">Target</h5>
                        <p style="margin: 0.25rem 0; font-size: 1rem; font-weight: 600; color: #10B981;">₹{target:.2f}</p>
                    </div>
                    <div style="text-align: center;">
                        <h5 style="margin: 0; color: #EF4444; font-size: 0.75rem; text-transform: uppercase;">Stop Loss</h5>
                        <p style="margin: 0.25rem 0; font-size: 1rem; font-weight: 600; color: #EF4444;">₹{stop_loss:.2f}</p>
                    </div>
                </div>
                """
            
            # Add detailed information if requested
            if detailed:
                # Technical analysis reasoning
                if 'technical_reason' in signal:
                    card_html += f"""
                    <div style="
                        background: #F8FAFC;
                        padding: 1rem;
                        border-radius: 8px;
                        margin-bottom: 1rem;
                    ">
                        <h5 style="margin: 0 0 0.5rem 0; color: #374151; font-size: 0.85rem; font-weight: 600;">
                            📊 Technical Analysis
                        </h5>
                        <p style="margin: 0; color: #6B7280; font-size: 0.85rem; line-height: 1.4;">
                            {signal['technical_reason']}
                        </p>
                    </div>
                    """
                
                # AI analysis reasoning
                if 'ai_reason' in signal:
                    card_html += f"""
                    <div style="
                        background: #F0F9FF;
                        padding: 1rem;
                        border-radius: 8px;
                        margin-bottom: 1rem;
                    ">
                        <h5 style="margin: 0 0 0.5rem 0; color: #374151; font-size: 0.85rem; font-weight: 600;">
                            🧠 AI Analysis
                        </h5>
                        <p style="margin: 0; color: #6B7280; font-size: 0.85rem; line-height: 1.4;">
                            {signal['ai_reason']}
                        </p>
                    </div>
                    """
                
                # Risk metrics
                if 'risk_reward' in signal:
                    risk_reward = signal.get('risk_reward', 0)
                    generated_time = signal.get('generated_time', 'Unknown')
                    
                    card_html += f"""
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding-top: 1rem;
                        border-top: 1px solid #E5E7EB;
                        font-size: 0.8rem;
                        color: #6B7280;
                    ">
                        <span>Risk-Reward: 1:{risk_reward:.1f}</span>
                        <span>{generated_time}</span>
                    </div>
                    """
            
            # Action buttons
            card_html += f"""
                <div style="
                    display: flex;
                    gap: 0.5rem;
                    margin-top: 1rem;
                ">
                    <button style="
                        background: {signal_color};
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 6px;
                        font-size: 0.85rem;
                        font-weight: 500;
                        flex: 1;
                        cursor: pointer;
                    ">
                        📱 Set Alert
                    </button>
                    <button style="
                        background: transparent;
                        color: {signal_color};
                        border: 1px solid {signal_color};
                        padding: 8px 16px;
                        border-radius: 6px;
                        font-size: 0.85rem;
                        font-weight: 500;
                        flex: 1;
                        cursor: pointer;
                    ">
                        📊 Analyze
                    </button>
                </div>
            </div>
            """
            
            st.markdown(card_html, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error rendering signal card: {str(e)}")
            st.error("Error rendering signal card")
    
    def news_card(self, news_item: Dict[str, Any]) -> None:
        """Professional news card component"""
        try:
            impact = news_item.get('impact', 'Medium')
            sentiment = news_item.get('sentiment', 0)
            
            # Impact color coding
            impact_colors = {
                'High': '#EF4444',
                'Medium': '#F59E0B',
                'Low': '#10B981'
            }
            
            impact_color = impact_colors.get(impact, '#6B7280')
            
            # Sentiment indicator
            if sentiment > 0.2:
                sentiment_icon = '📈'
                sentiment_color = '#10B981'
                sentiment_text = 'Positive'
            elif sentiment < -0.2:
                sentiment_icon = '📉'
                sentiment_color = '#EF4444'
                sentiment_text = 'Negative'
            else:
                sentiment_icon = '➡️'
                sentiment_color = '#6B7280'
                sentiment_text = 'Neutral'
            
            st.markdown(f"""
            <div style="
                background: white;
                padding: 1rem;
                border-radius: 12px;
                margin-bottom: 0.75rem;
                border-left: 4px solid {impact_color};
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            ">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.75rem;">
                    <h5 style="margin: 0; color: #111827; font-size: 1rem; font-weight: 600; line-height: 1.3; flex: 1;">
                        {news_item.get('title', 'No Title')}
                    </h5>
                    <div style="margin-left: 1rem; text-align: right;">
                        <span style="
                            background: {impact_color};
                            color: white;
                            padding: 2px 6px;
                            border-radius: 4px;
                            font-size: 0.7rem;
                            font-weight: 500;
                        ">
                            {impact}
                        </span>
                    </div>
                </div>
                
                <p style="margin: 0 0 0.75rem 0; color: #6B7280; font-size: 0.85rem; line-height: 1.4;">
                    {news_item.get('summary', 'No summary available')}
                </p>
                
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: 0.75rem;
                    color: #9CA3AF;
                ">
                    <span>{news_item.get('time', 'Unknown time')} | {news_item.get('source', 'Unknown source')}</span>
                    <span style="color: {sentiment_color}; font-weight: 500;">
                        {sentiment_icon} {sentiment_text}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error rendering news card: {str(e)}")
            st.error("Error rendering news card")
    
    def portfolio_position_card(self, position: Dict[str, Any]) -> None:
        """Professional portfolio position card"""
        try:
            symbol = position.get('symbol', 'N/A')
            quantity = position.get('quantity', 0)
            avg_price = position.get('avg_price', 0)
            current_price = position.get('current_price', 0)
            pnl = position.get('pnl', 0)
            pnl_pct = position.get('pnl_pct', 0)
            current_value = position.get('current_value', 0)
            
            # PnL styling
            if pnl > 0:
                pnl_color = '#10B981'
                pnl_bg = '#ECFDF5'
                pnl_icon = '📈'
            elif pnl < 0:
                pnl_color = '#EF4444'
                pnl_bg = '#FEF2F2'
                pnl_icon = '📉'
            else:
                pnl_color = '#6B7280'
                pnl_bg = '#F9FAFB'
                pnl_icon = '➡️'
            
            st.markdown(f"""
            <div style="
                background: white;
                padding: 1.25rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                border-left: 4px solid {pnl_color};
                box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            ">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                    <div>
                        <h3 style="margin: 0; color: #111827; font-size: 1.1rem; font-weight: 600;">
                            {symbol}
                        </h3>
                        <p style="margin: 0.25rem 0; color: #6B7280; font-size: 0.85rem;">
                            {quantity} shares @ ₹{avg_price:.2f}
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <p style="margin: 0; font-size: 1.2rem; font-weight: 600; color: #111827;">
                            ₹{current_value:,.0f}
                        </p>
                        <p style="margin: 0.25rem 0; color: {pnl_color}; font-weight: 600; font-size: 0.9rem;">
                            {pnl_icon} ₹{pnl:+,.0f} ({pnl_pct:+.1f}%)
                        </p>
                    </div>
                </div>
                
                <div style="
                    background: {pnl_bg};
                    padding: 0.75rem;
                    border-radius: 8px;
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 1rem;
                ">
                    <div>
                        <h5 style="margin: 0; color: #6B7280; font-size: 0.75rem;">CURRENT PRICE</h5>
                        <p style="margin: 0.25rem 0; font-weight: 600; color: #111827;">₹{current_price:.2f}</p>
                    </div>
                    <div>
                        <h5 style="margin: 0; color: #6B7280; font-size: 0.75rem;">TOTAL VALUE</h5>
                        <p style="margin: 0.25rem 0; font-weight: 600; color: #111827;">₹{current_value:,.0f}</p>
                    </div>
                </div>
                
                <div style="
                    display: flex;
                    gap: 0.5rem;
                    margin-top: 1rem;
                ">
                    <button style="
                        background: #1E3A8A;
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 6px;
                        font-size: 0.8rem;
                        flex: 1;
                        cursor: pointer;
                    ">
                        📊 Analyze
                    </button>
                    <button style="
                        background: transparent;
                        color: #EF4444;
                        border: 1px solid #EF4444;
                        padding: 8px 16px;
                        border-radius: 6px;
                        font-size: 0.8rem;
                        flex: 1;
                        cursor: pointer;
                    ">
                        💰 Trade
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error rendering position card: {str(e)}")
            st.error("Error rendering position card")
    
    def create_candlestick_chart(self, data: pd.DataFrame, title: str = "Price Chart", 
                                indicators: Dict[str, pd.Series] = None, height: int = 400) -> go.Figure:
        """Create professional candlestick chart with indicators"""
        try:
            # Create subplots
            rows = 2 if indicators else 1
            fig = make_subplots(
                rows=rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(title, 'Volume') if indicators else (title,),
                row_width=[0.7, 0.3] if indicators else [1.0]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color=self.colors['success'],
                    decreasing_line_color=self.colors['danger']
                ),
                row=1, col=1
            )
            
            # Add technical indicators
            if indicators:
                for name, series in indicators.items():
                    if 'sma' in name.lower() or 'ema' in name.lower():
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=series,
                                mode='lines',
                                name=name.upper(),
                                line=dict(width=2)
                            ),
                            row=1, col=1
                        )
                
                # Volume chart
                if 'Volume' in data.columns:
                    colors = ['green' if close >= open else 'red' 
                             for close, open in zip(data['Close'], data['Open'])]
                    
                    fig.add_trace(
                        go.Bar(
                            x=data.index,
                            y=data['Volume'],
                            name='Volume',
                            marker_color=colors,
                            opacity=0.7
                        ),
                        row=2, col=1
                    )
            
            # Update layout
            fig.update_layout(
                height=height,
                xaxis_rangeslider_visible=False,
                showlegend=True,
                font=dict(family="Inter", size=11),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=10, r=10, t=50, b=10)
            )
            
            # Update axes
            fig.update_xaxes(showgrid=True, gridcolor='#F3F4F6')
            fig.update_yaxes(showgrid=True, gridcolor='#F3F4F6')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating candlestick chart: {str(e)}")
            return go.Figure()
    
    def create_options_chain_chart(self, options_data: pd.DataFrame, spot_price: float) -> go.Figure:
        """Create options chain visualization"""
        try:
            # Separate calls and puts
            calls = options_data[options_data['option_type'] == 'CALL']
            puts = options_data[options_data['option_type'] == 'PUT']
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Open Interest', 'Volume')
            )
            
            # Open Interest chart
            fig.add_trace(
                go.Bar(
                    x=calls['strike'],
                    y=calls['oi'],
                    name='Call OI',
                    marker_color=self.colors['success'],
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=puts['strike'],
                    y=-puts['oi'],  # Negative for puts
                    name='Put OI',
                    marker_color=self.colors['danger'],
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Volume chart
            fig.add_trace(
                go.Bar(
                    x=calls['strike'],
                    y=calls['volume'],
                    name='Call Volume',
                    marker_color=self.colors['success'],
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=puts['strike'],
                    y=-puts['volume'],  # Negative for puts
                    name='Put Volume',
                    marker_color=self.colors['danger'],
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Add current price line
            fig.add_vline(
                x=spot_price,
                line_dash="dash",
                line_color=self.colors['primary'],
                annotation_text=f"Spot: {spot_price}",
                annotation_position="top"
            )
            
            # Update layout
            fig.update_layout(
                height=500,
                showlegend=True,
                font=dict(family="Inter", size=11),
                plot_bgcolor='white',
                paper_bgcolor='white',
                title="Options Chain Analysis"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating options chart: {str(e)}")
            return go.Figure()
    
    def create_performance_chart(self, performance_data: pd.DataFrame, 
                                benchmark_data: pd.DataFrame = None) -> go.Figure:
        """Create portfolio performance chart"""
        try:
            fig = go.Figure()
            
            # Portfolio performance
            fig.add_trace(
                go.Scatter(
                    x=performance_data.index,
                    y=performance_data['portfolio_value'],
                    mode='lines',
                    name='Portfolio',
                    line=dict(color=self.colors['primary'], width=3)
                )
            )
            
            # Benchmark comparison
            if benchmark_data is not None:
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_data.index,
                        y=benchmark_data['value'],
                        mode='lines',
                        name='NIFTY 50',
                        line=dict(color=self.colors['muted'], width=2, dash='dash')
                    )
                )
            
            # Update layout
            fig.update_layout(
                title="Portfolio Performance",
                height=350,
                showlegend=True,
                font=dict(family="Inter", size=11),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=10, r=10, t=50, b=10)
            )
            
            # Update axes
            fig.update_xaxes(title="Date", showgrid=True, gridcolor='#F3F4F6')
            fig.update_yaxes(title="Value (₹)", showgrid=True, gridcolor='#F3F4F6')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating performance chart: {str(e)}")
            return go.Figure()
    
    def create_risk_gauge(self, risk_score: float, title: str = "Risk Score") -> go.Figure:
        """Create risk gauge chart"""
        try:
            # Define risk levels
            if risk_score <= 30:
                color = self.colors['success']
                risk_level = "Low Risk"
            elif risk_score <= 60:
                color = self.colors['warning']
                risk_level = "Medium Risk"
            else:
                color = self.colors['danger']
                risk_level = "High Risk"
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"{title}<br><span style='font-size:0.8em;color:gray'>{risk_level}</span>"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 30], 'color': "#ECFDF5"},
                        {'range': [30, 60], 'color': "#FEF3C7"},
                        {'range': [60, 100], 'color': "#FEE2E2"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                font={'color': "darkblue", 'family': "Inter"},
                paper_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating risk gauge: {str(e)}")
            return go.Figure()
    
    def status_badge(self, text: str, status: str = "neutral") -> str:
        """Create status badge HTML"""
        color_map = {
            'success': '#10B981',
            'danger': '#EF4444',
            'warning': '#F59E0B',
            'info': '#06B6D4',
            'neutral': '#6B7280'
        }
        
        bg_color = color_map.get(status, '#6B7280')
        
        return f"""
        <span style="
            background: {bg_color};
            color: white;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 500;
            display: inline-block;
        ">
            {text}
        </span>
        """
    
    def progress_bar(self, value: float, max_value: float = 100, 
                    color: str = "primary", height: int = 8) -> None:
        """Create progress bar"""
        try:
            percentage = min((value / max_value) * 100, 100)
            bar_color = self.colors.get(color, self.colors['primary'])
            
            st.markdown(f"""
            <div style="
                width: 100%;
                background-color: #E5E7EB;
                border-radius: {height//2}px;
                height: {height}px;
                margin: 0.5rem 0;
            ">
                <div style="
                    width: {percentage}%;
                    background-color: {bar_color};
                    height: 100%;
                    border-radius: {height//2}px;
                    transition: width 0.3s ease;
                "></div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error creating progress bar: {str(e)}")
    
    def loading_spinner(self, message: str = "Loading...") -> None:
        """Create loading spinner"""
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <div style="
                border: 3px solid #F3F4F6;
                border-top: 3px solid {self.colors['primary']};
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 1rem auto;
            "></div>
            <p style="color: {self.colors['muted']}; font-size: 0.9rem;">{message}</p>
            <style>
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
            </style>
        </div>
        """, unsafe_allow_html=True)
    
    def alert_box(self, message: str, alert_type: str = "info", 
                 dismissible: bool = False) -> None:
        """Create alert box"""
        try:
            color_map = {
                'success': {'bg': '#ECFDF5', 'border': '#10B981', 'text': '#065F46', 'icon': '✅'},
                'danger': {'bg': '#FEF2F2', 'border': '#EF4444', 'text': '#991B1B', 'icon': '❌'},
                'warning': {'bg': '#FEF3C7', 'border': '#F59E0B', 'text': '#92400E', 'icon': '⚠️'},
                'info': {'bg': '#EFF6FF', 'border': '#3B82F6', 'text': '#1E40AF', 'icon': 'ℹ️'}
            }
            
            style = color_map.get(alert_type, color_map['info'])
            
            st.markdown(f"""
            <div style="
                background: {style['bg']};
                border: 1px solid {style['border']};
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
                color: {style['text']};
                display: flex;
                align-items: center;
                gap: 0.5rem;
            ">
                <span style="font-size: 1.2rem;">{style['icon']}</span>
                <span style="flex: 1;">{message}</span>
                {f'<button style="background: none; border: none; font-size: 1.2rem; cursor: pointer;">✕</button>' if dismissible else ''}
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error creating alert box: {str(e)}")
    
    def data_table(self, data: pd.DataFrame, title: str = "", 
                  max_height: int = 400) -> None:
        """Create professional data table"""
        try:
            if title:
                st.markdown(f"### {title}")
            
            # Custom CSS for table
            st.markdown(f"""
            <style>
                .dataframe {{
                    border-radius: 8px;
                    overflow: hidden;
                    border: 1px solid #E5E7EB;
                    max-height: {max_height}px;
                    overflow-y: auto;
                }}
                .dataframe th {{
                    background: #F8FAFC;
                    padding: 12px;
                    font-weight: 600;
                    color: #374151;
                    border-bottom: 2px solid #E5E7EB;
                    position: sticky;
                    top: 0;
                    z-index: 1;
                }}
                .dataframe td {{
                    padding: 12px;
                    border-bottom: 1px solid #F3F4F6;
                }}
                .dataframe tr:hover {{
                    background: #F9FAFB;
                }}
            </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(data, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error creating data table: {str(e)}")
            st.error("Error displaying data table")

# Export main class
__all__ = ['UIComponents']
