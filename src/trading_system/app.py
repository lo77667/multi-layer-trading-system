"""Streamlit web dashboard for the trading system."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import asyncio
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="🤖 Multi-Layer Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success { color: #09ab56; }
    .danger { color: #ff2b2b; }
    .warning { color: #ffa421; }
</style>
""", unsafe_allow_html=True)


def load_config():
    """Load system configuration."""
    try:
        from trading_system.core.config import SystemConfig
        return SystemConfig.from_env()
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        return None


def main():
    """Main dashboard function."""
    st.title("🤖 Multi-Layer AI Trading System")
    st.markdown("---")
    
    # Initialize session state
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'trades_history' not in st.session_state:
        st.session_state.trades_history = []
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ System Controls")
        st.markdown("---")
        
        # Trading mode display
        if st.session_state.config:
            mode = st.session_state.config.trading_mode.value
            mode_color = "🟢" if mode == "paper" else "🔴"
            st.metric("Trading Mode", f"{mode_color} {mode.upper()}")
        
        # Symbol selection
        symbol = st.text_input(
            "Trading Symbol",
            value="EUR/USD",
            help="Enter the symbol to trade (e.g., EUR/USD, AAPL)"
        )
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "▶️ Start Trading",
                key="start_btn",
                help="Start automated trading"
            ):
                st.session_state.is_running = True
                st.success("Trading started!")
        
        with col2:
            if st.button(
                "⏸️ Stop Trading",
                key="stop_btn",
                help="Stop automated trading"
            ):
                st.session_state.is_running = False
                st.warning("Trading stopped!")
        
        # Kill switch
        if st.session_state.config:
            st.markdown("---")
            st.subheader("🚨 Risk Controls")
            kill_switch_status = st.checkbox(
                "Kill Switch (Emergency Stop)",
                value=st.session_state.config.risk.kill_switch_enabled,
                help="Enable emergency stop mechanism"
            )
            
            daily_loss_limit = st.slider(
                "Daily Loss Limit (%)",
                min_value=0.5,
                max_value=5.0,
                value=st.session_state.config.risk.daily_loss_limit * 100,
                step=0.1,
                help="Maximum daily loss before trading stops"
            )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "📈 Performance",
        "🤖 Agent Signals",
        "💾 Trade History",
        "⚙️ Settings",
    ])
    
    # Tab 1: Dashboard
    with tab1:
        st.subheader("Real-time Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Portfolio Value",
                "$100,000",
                "+2.5%",
            )
        
        with col2:
            st.metric(
                "Daily P&L",
                "$2,500",
                "+2.5%",
            )
        
        with col3:
            st.metric(
                "Open Positions",
                "3",
                help="Number of active trades"
            )
        
        with col4:
            st.metric(
                "Win Rate",
                "65%",
                "+5%",
            )
        
        st.markdown("---")
        
        # Current symbol analysis
        st.subheader(f"Analysis for {symbol}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Agent Signals**")
            signals_data = {
                'Agent': ['Fundamentals', 'Sentiment', 'Technical'],
                'Signal': ['BUY', 'BUY', 'HOLD'],
                'Confidence': [0.85, 0.72, 0.65],
            }
            signals_df = pd.DataFrame(signals_data)
            st.dataframe(signals_df, use_container_width=True)
        
        with col2:
            st.write("**Market Data**")
            market_data = {
                'Metric': ['Current Price', 'RSI(14)', 'MACD', 'ATR(14)', 'Volume Ratio'],
                'Value': ['1.0854', '65.3', '0.0045', '0.0032', '1.25'],
            }
            market_df = pd.DataFrame(market_data)
            st.dataframe(market_df, use_container_width=True)
        
        # Price chart with indicators
        st.subheader("Price Chart")
        fig = go.Figure()
        
        # Dummy data for demo
        dates = pd.date_range(start='2024-01-01', periods=100)
        prices = [1.08 + (i * 0.0001) for i in range(100)]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4')
        ))
        
        fig.update_layout(
            title=f"{symbol} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode='x unified',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Performance
    with tab2:
        st.subheader("Performance Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Return", "15.3%", "+2.1%")
            st.metric("Sharpe Ratio", "1.85")
        
        with col2:
            st.metric("Max Drawdown", "-3.2%")
            st.metric("Win Rate", "62%")
        
        with col3:
            st.metric("Profit Factor", "2.45")
            st.metric("Recovery Factor", "4.78")
        
        st.markdown("---")
        
        # Equity curve
        st.write("**Equity Curve**")
        equity_dates = pd.date_range(start='2024-01-01', periods=100)
        equity_curve = [100000 + (i * 150) for i in range(100)]
        
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=equity_dates,
            y=equity_curve,
            fill='tozeroy',
            name='Portfolio Value',
            line=dict(color='#2ca02c')
        ))
        fig_equity.update_layout(
            title="Portfolio Equity Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            hovermode='x unified',
            height=400,
        )
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Monthly returns
        st.write("**Monthly Returns**")
        monthly_data = {
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'Return': [2.5, 1.8, 3.2, -1.5, 2.1, 3.5],
        }
        monthly_df = pd.DataFrame(monthly_data)
        fig_monthly = go.Figure(data=[
            go.Bar(x=monthly_df['Month'], y=monthly_df['Return'])
        ])
        fig_monthly.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Tab 3: Agent Signals
    with tab3:
        st.subheader("Agent Analysis Details")
        
        agent_col1, agent_col2, agent_col3 = st.columns(3)
        
        with agent_col1:
            st.subheader("📊 Fundamentals Analyst")
            with st.expander("View Analysis"):
                st.write("""
                **P/E Ratio**: 18.5 (below market average)
                **Earnings Growth**: +12% YoY
                **Debt/Equity**: 0.45 (healthy)
                **Recommendation**: BUY (Confidence: 85%)
                """)
        
        with agent_col2:
            st.subheader("😊 Sentiment Analyst")
            with st.expander("View Analysis"):
                st.write("""
                **Positive Headlines**: 8
                **Negative Headlines**: 3
                **Neutral Headlines**: 5
                **Overall Sentiment**: Bullish (Confidence: 72%)
                **Trend**: 📈 Improving
                """)
        
        with agent_col3:
            st.subheader("📈 Technical Analyst")
            with st.expander("View Analysis"):
                st.write("""
                **Trend**: Uptrend
                **Support**: 1.0820
                **Resistance**: 1.0900
                **Momentum**: Strong
                **Recommendation**: HOLD (Confidence: 65%)
                """)
        
        st.markdown("---")
        
        # Debate summary
        st.subheader("🎯 Debate Engine Summary")
        with st.expander("View Structured Debate"):
            st.write("""
            **Round 1 - Bullish Argument:**
            The fundamentals are solid with strong earnings growth, and technical analysis shows
            an uptrend with healthy momentum indicators.
            
            **Round 1 - Bearish Counter:**
            While fundamentals look good, sentiment data shows mixed signals with
            recent negative headlines that could trigger selling pressure.
            
            **Final Consensus:**
            Combined analysis suggests a cautious BUY with stop-loss at 1.0820
            for risk management.
            """)
    
    # Tab 4: Trade History
    with tab4:
        st.subheader("Trade History")
        
        # Sample trade data
        trades = {
            'Trade ID': ['TRD001', 'TRD002', 'TRD003', 'TRD004'],
            'Symbol': ['EUR/USD', 'GBP/JPY', 'EUR/USD', 'AAPL'],
            'Entry Time': ['2024-02-01 10:30', '2024-02-02 14:15', '2024-02-03 09:45', '2024-02-03 16:20'],
            'Direction': ['BUY', 'SELL', 'BUY', 'BUY'],
            'Entry Price': [1.0820, 145.32, 1.0850, 185.50],
            'Exit Price': [1.0890, 145.10, 1.0875, 187.25],
            'P&L': ['+$700', '+$200', '+$250', '+$175'],
            'Status': ['CLOSED', 'CLOSED', 'CLOSED', 'OPEN'],
        }
        trades_df = pd.DataFrame(trades)
        st.dataframe(trades_df, use_container_width=True)
        
        # Export button
        csv = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade History (CSV)",
            data=csv,
            file_name="trades_history.csv",
            mime="text/csv"
        )
    
    # Tab 5: Settings
    with tab5:
        st.subheader("System Configuration")
        
        if st.session_state.config:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**LLM Configuration**")
                st.text_input(
                    "Deep Think Model",
                    value=st.session_state.config.deep_think_llm.model_name,
                    disabled=True
                )
                st.text_input(
                    "Quick Think Model",
                    value=st.session_state.config.quick_think_llm.model_name,
                    disabled=True
                )
            
            with col2:
                st.write("**Risk Management**")
                st.metric("Daily Loss Limit", f"{st.session_state.config.risk.daily_loss_limit*100}%")
                st.metric("Per-Trade Risk", f"{st.session_state.config.risk.per_trade_risk*100}%")
                st.metric("Min R:R Ratio", f"1:{st.session_state.config.risk.min_reward_risk_ratio}")
            
            st.markdown("---")
            st.write("**Agent Settings**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.checkbox(
                    "Fundamentals Agent",
                    value=st.session_state.config.agents.fundamentals_enabled,
                    disabled=True
                )
            with col2:
                st.checkbox(
                    "Sentiment Agent",
                    value=st.session_state.config.agents.sentiment_enabled,
                    disabled=True
                )
            with col3:
                st.checkbox(
                    "Technical Agent",
                    value=st.session_state.config.agents.technical_enabled,
                    disabled=True
                )
            
            st.markdown("---")
            st.write("**Data & API Settings**")
            st.text_input(
                "Primary Data Source",
                value=st.session_state.config.data_source.primary,
                disabled=True
            )
            st.text_input(
                "Backup Data Source",
                value=st.session_state.config.data_source.backup,
                disabled=True
            )
        
        # Save settings button
        if st.button("💾 Save Configuration"):
            st.success("Configuration saved!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
    🤖 Multi-Layer AI Trading System | Powered by LLM Agents | Paper Trading Mode
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
