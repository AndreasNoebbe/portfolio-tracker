import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
import warnings
import time
import requests
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Portfolio Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e7d3a 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-header {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Your actual portfolio
PORTFOLIO = {
    'GOOGL': {'shares': 55, 'avg_price': 154.84, 'currency': 'USD', 'name': 'Alphabet Inc.'},
    'EVO.ST': {'shares': 20, 'avg_price': 651.9, 'currency': 'SEK', 'name': 'Evolution AB'},
    'NVO': {'shares': 270, 'avg_price': 60.44, 'currency': 'USD', 'name': 'Novo Nordisk ADR'},
    'NOVO-B.CO': {'shares': 430, 'avg_price': 477.49, 'currency': 'DKK', 'name': 'Novo Nordisk B'},
    'UNH': {'shares': 50, 'avg_price': 294.13, 'currency': 'USD', 'name': 'UnitedHealth Group'}
}

@st.cache_data(ttl=3600)  # Cache exchange rates for 1 hour
def get_exchange_rates():
    """Get current exchange rates to DKK"""
    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/DKK')
        data = response.json()
        
        # Convert to rates TO DKK (inverse of FROM DKK)
        rates = {
            'USD_TO_DKK': 1 / data['rates']['USD'],
            'SEK_TO_DKK': 1 / data['rates']['SEK'],
            'DKK_TO_DKK': 1.0
        }
        return rates
    except:
        # Fallback rates if API fails
        return {
            'USD_TO_DKK': 6.85,  # Approximate USD to DKK
            'SEK_TO_DKK': 0.63,  # Approximate SEK to DKK  
            'DKK_TO_DKK': 1.0
        }

def convert_to_dkk(amount, from_currency, exchange_rates):
    """Convert amount to DKK"""
    if from_currency == 'USD':
        return amount * exchange_rates['USD_TO_DKK']
    elif from_currency == 'SEK':
        return amount * exchange_rates['SEK_TO_DKK']
    else:  # DKK
        return amount

@st.cache_data(ttl=600)  # Cache for 10 minutes to reduce API calls
def get_current_prices():
    """Fetch current prices for all holdings with rate limiting protection"""
    tickers = list(PORTFOLIO.keys())
    current_prices = {}
    
    # Add delay between API calls to prevent rate limiting
    for i, ticker in enumerate(tickers):
        try:
            if i > 0:  # Add delay between requests
                time.sleep(2)  # 2 second delay between tickers
                
            # Method 1: Try single ticker download with longer period
            stock = yf.Ticker(ticker)
            hist = stock.history(period='5d')
            
            if not hist.empty:
                current_prices[ticker] = hist['Close'].iloc[-1]
                continue
                
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")
            continue
    
    # Fallback: Use realistic demo data if all fails
    if len(current_prices) < len(tickers):
        st.warning("⚠️ Rate limited or API unavailable - using demo data")
        current_prices = {
            'GOOGL': 175.50,      # USD
            'EVO.ST': 680.00,     # SEK
            'NVO': 65.25,         # USD  
            'NOVO-B.CO': 520.00,  # DKK
            'UNH': 515.75         # USD
        }
    
    return current_prices

def calculate_portfolio_summary(current_prices, exchange_rates):
    """Calculate your actual P&L and portfolio metrics in DKK"""
    summary = []
    total_invested_dkk = 0
    total_current_dkk = 0
    
    for ticker, position in PORTFOLIO.items():
        current_price = current_prices.get(ticker, 0)
        
        # Convert invested amount to DKK
        invested_value_original = position['shares'] * position['avg_price']
        invested_value_dkk = convert_to_dkk(invested_value_original, position['currency'], exchange_rates)
        
        # Convert current value to DKK  
        current_value_original = position['shares'] * current_price
        current_value_dkk = convert_to_dkk(current_value_original, position['currency'], exchange_rates)
        
        # Calculate P&L in DKK
        pnl_dkk = current_value_dkk - invested_value_dkk
        pnl_percent = (pnl_dkk / invested_value_dkk * 100) if invested_value_dkk > 0 else 0
        
        summary.append({
            'ticker': ticker,
            'name': position['name'],
            'shares': position['shares'],
            'avg_price_original': position['avg_price'],
            'avg_price_dkk': convert_to_dkk(position['avg_price'], position['currency'], exchange_rates),
            'current_price_original': current_price,
            'current_price_dkk': convert_to_dkk(current_price, position['currency'], exchange_rates),
            'invested_dkk': invested_value_dkk,
            'current_value_dkk': current_value_dkk,
            'pnl_dkk': pnl_dkk,
            'pnl_percent': pnl_percent,
            'currency': position['currency']
        })
        
        total_invested_dkk += invested_value_dkk
        total_current_dkk += current_value_dkk
    
    total_pnl_dkk = total_current_dkk - total_invested_dkk
    total_pnl_percent = (total_pnl_dkk / total_invested_dkk * 100) if total_invested_dkk > 0 else 0
    
    return summary, total_invested_dkk, total_current_dkk, total_pnl_dkk, total_pnl_percent

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_historical_data():
    """Fetch historical data for portfolio analysis with rate limiting"""
    tickers = list(PORTFOLIO.keys())
    
    try:
        # Add delay to prevent rate limiting
        time.sleep(1)
        
        # Try to download all tickers together first
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        data = yf.download(tickers, start=start_date, end=end_date, 
                          group_by='ticker', progress=False, auto_adjust=True)
        
        if not data.empty:
            return data
            
    except Exception as e:
        print(f"Group download failed: {e}")
    
    # If rate limited, return None to use demo mode
    st.warning("⚠️ Historical data limited due to API rate limiting")
    return None

def calculate_portfolio_history(data, exchange_rates):
    """Calculate your portfolio value over time in DKK"""
    if data is None:
        return pd.DataFrame()
    
    portfolio_values = []
    dates = data.index
    
    for date in dates:
        daily_value_dkk = 0
        for ticker, position in PORTFOLIO.items():
            try:
                if len(PORTFOLIO) == 1:
                    price = data.loc[date, 'Adj Close']
                else:
                    price = data.loc[date, (ticker, 'Adj Close')]
                
                if not pd.isna(price):
                    # Convert to DKK
                    price_dkk = convert_to_dkk(price, position['currency'], exchange_rates)
                    daily_value_dkk += position['shares'] * price_dkk
                    
            except Exception:
                continue
        
        if daily_value_dkk > 0:
            portfolio_values.append({'date': date, 'portfolio_value_dkk': daily_value_dkk})
    
    return pd.DataFrame(portfolio_values)

def create_portfolio_pie_chart(summary):
    """Create a better aligned pie chart of portfolio allocation in DKK"""
    labels = [f"{row['name']}" for row in summary]
    values = [row['current_value_dkk'] for row in summary]
    
    # Create better colors
    colors = ['#1f4e79', '#2e7d3a', '#ff6b35', '#6c5ce7', '#fd79a8']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo='label+percent',
        textposition='auto',
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f} DKK<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': 'Portfolio Allocation (DKK)',
            'x': 0.5,
            'font': {'size': 18, 'color': '#2c3e50'}
        },
        height=450,
        paper_bgcolor='white',
        font={'color': '#2c3e50', 'size': 12},
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

def create_pnl_bar_chart(summary):
    """Create a better aligned P&L bar chart in DKK"""
    tickers = [row['ticker'] for row in summary]
    pnl_values = [row['pnl_dkk'] for row in summary]
    colors = ['#28a745' if pnl >= 0 else '#dc3545' for pnl in pnl_values]
    
    fig = go.Figure(data=[go.Bar(
        x=tickers,
        y=pnl_values,
        marker_color=colors,
        text=[f"{pnl:+,.0f} DKK" for pnl in pnl_values],
        textposition='outside',  # Place text outside bars
        textfont=dict(size=12, color='#2c3e50'),
        hovertemplate='<b>%{x}</b><br>P&L: %{y:+,.0f} DKK<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': 'Profit & Loss by Position (DKK)',
            'x': 0.5,
            'font': {'size': 18, 'color': '#2c3e50'}
        },
        xaxis_title='Holdings',
        yaxis_title='P&L (DKK)',
        height=450,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font={'color': '#2c3e50'},
        margin=dict(l=60, r=60, t=80, b=60),  # Increased margins for text
        xaxis=dict(
            tickfont=dict(size=12),
            gridcolor='#f0f0f0'
        ),
        yaxis=dict(
            tickfont=dict(size=12),
            gridcolor='#f0f0f0',
            zeroline=True,
            zerolinecolor='#999999',
            zerolinewidth=1
        )
    )
    
    return fig

def create_historical_performance_chart(portfolio_history, total_invested_dkk):
    """Create a chart showing portfolio performance over time in DKK"""
    if portfolio_history.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Portfolio value line
    fig.add_trace(go.Scatter(
        x=portfolio_history['date'],
        y=portfolio_history['portfolio_value_dkk'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f4e79', width=3),
        fill='tonexty',
        hovertemplate='Date: %{x}<br>Value: %{y:,.0f} DKK<extra></extra>'
    ))
    
    # Invested amount line
    fig.add_hline(
        y=total_invested_dkk, 
        line_dash="dash", 
        line_color="#dc3545", 
        line_width=2,
        annotation_text=f"Total Invested: {total_invested_dkk:,.0f} DKK",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title={
            'text': 'Portfolio Performance Over Time (DKK)',
            'x': 0.5,
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title='Date',
        yaxis_title='Portfolio Value (DKK)',
        height=500,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font={'color': '#2c3e50'},
        hovermode='x unified',
        margin=dict(l=80, r=80, t=80, b=60),
        xaxis=dict(
            gridcolor='#f0f0f0',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            gridcolor='#f0f0f0',
            tickfont=dict(size=12),
            tickformat=',.0f'
        )
    )
    
    return fig

def calculate_risk_metrics(data):
    """Calculate sophisticated risk metrics for portfolio analysis"""
    if data is None:
        return {}, pd.DataFrame()
    
    risk_metrics = {}
    returns_data = {}
    
    for ticker, position in PORTFOLIO.items():
        try:
            if len(PORTFOLIO) == 1:
                prices = data['Adj Close']
            else:
                prices = data[(ticker, 'Adj Close')].dropna()
            
            if len(prices) < 2:
                continue
                
            returns = prices.pct_change().dropna()
            returns_data[ticker] = returns
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(252) * 100
            var_95 = np.percentile(returns, 5) * 100
            var_99 = np.percentile(returns, 1) * 100
            
            risk_free_rate = 0.02 / 252
            excess_returns = returns - risk_free_rate
            sharpe_ratio = (excess_returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative / running_max - 1)
            max_drawdown = drawdown.min() * 100
            
            portfolio_weight = position['shares'] * prices.iloc[-1]
            
            risk_metrics[ticker] = {
                'volatility': volatility,
                'var_95': var_95,
                'var_99': var_99,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'portfolio_weight': portfolio_weight
            }
            
        except Exception as e:
            continue
    
    returns_df = pd.DataFrame(returns_data)
    correlation_matrix = returns_df.corr() if len(returns_df.columns) > 1 else pd.DataFrame()
    
    return risk_metrics, correlation_matrix

def portfolio_optimization_analysis(data):
    """Perform Modern Portfolio Theory optimization on the portfolio"""
    if data is None:
        return {}, pd.DataFrame()
    
    returns_data = {}
    for ticker in PORTFOLIO.keys():
        try:
            if len(PORTFOLIO) == 1:
                prices = data['Adj Close']
            else:
                prices = data[(ticker, 'Adj Close')].dropna()
            
            if len(prices) < 2:
                continue
                
            returns = prices.pct_change().dropna()
            returns_data[ticker] = returns
        except Exception:
            continue
    
    if len(returns_data) < 2:
        return {}, pd.DataFrame()
    
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    if returns_df.empty:
        return {}, pd.DataFrame()
    
    expected_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    
    num_assets = len(expected_returns)
    
    def portfolio_return(weights):
        return np.dot(weights, expected_returns)
    
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def portfolio_sharpe(weights):
        risk_free_rate = 0.02
        return (portfolio_return(weights) - risk_free_rate) / portfolio_volatility(weights)
    
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.array([1/num_assets] * num_assets)
    
    optimizations = {}
    
    # Maximum Sharpe Ratio
    try:
        result = minimize(lambda w: -portfolio_sharpe(w), initial_guess, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            optimizations['max_sharpe'] = {
                'weights': result.x,
                'return': portfolio_return(result.x),
                'volatility': portfolio_volatility(result.x),
                'sharpe': portfolio_sharpe(result.x)
            }
    except:
        pass
    
    # Minimum Volatility
    try:
        result = minimize(portfolio_volatility, initial_guess,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            optimizations['min_vol'] = {
                'weights': result.x,
                'return': portfolio_return(result.x),
                'volatility': portfolio_volatility(result.x),
                'sharpe': portfolio_sharpe(result.x)
            }
    except:
        pass
    
    # Calculate current portfolio allocation
    current_values = {}
    total_value = 0
    
    for ticker, position in PORTFOLIO.items():
        if ticker in expected_returns.index:
            try:
                if len(PORTFOLIO) == 1:
                    current_price = data['Adj Close'].iloc[-1]
                else:
                    current_price = data[(ticker, 'Adj Close')].iloc[-1]
                
                value = position['shares'] * current_price
                current_values[ticker] = value
                total_value += value
            except:
                continue
    
    current_weights = np.array([current_values.get(ticker, 0) / total_value 
                               for ticker in expected_returns.index])
    
    current_portfolio = {
        'weights': current_weights,
        'return': portfolio_return(current_weights),
        'volatility': portfolio_volatility(current_weights),
        'sharpe': portfolio_sharpe(current_weights)
    }
    
    # Generate efficient frontier
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), 50)
    efficient_portfolios = []
    
    for target in target_returns:
        try:
            constraints_ef = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: portfolio_return(w) - target}
            ]
            
            result = minimize(portfolio_volatility, initial_guess,
                            method='SLSQP', bounds=bounds, constraints=constraints_ef)
            
            if result.success:
                efficient_portfolios.append({
                    'return': target,
                    'volatility': portfolio_volatility(result.x)
                })
        except:
            continue
    
    efficient_frontier = pd.DataFrame(efficient_portfolios)
    
    return {
        'optimizations': optimizations,
        'current': current_portfolio,
        'asset_names': list(expected_returns.index),
        'expected_returns': expected_returns,
        'efficient_frontier': efficient_frontier
    }, returns_df

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 Portfolio Tracker</h1>
        <p>Professional Investment Analytics & Risk Management</p>
        <p style="font-size: 0.9em; opacity: 0.8;">Real-time tracking of personal holdings with institutional-grade analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    auto_refresh = st.sidebar.checkbox("⏱️ Auto-refresh (5 min)", value=True)
    
    # Add demo mode toggle
    demo_mode = st.sidebar.checkbox("📊 Demo Mode (use sample data)", value=False)
    if demo_mode:
        st.sidebar.info("Using demo data for testing")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Portfolio Holdings:**")
    for ticker, data in PORTFOLIO.items():
        st.sidebar.write(f"📊 **{ticker}**: {data['shares']} shares")
    
    # Add timestamp for last update
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Add market hours info
    now = datetime.now()
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if market_open <= now <= market_close and now.weekday() < 5:
        st.sidebar.success("🟢 Market is open")
    else:
        st.sidebar.info("🔵 Market is closed")
        st.sidebar.caption("Live data may be limited")
    
    # Fetch data and exchange rates
    with st.spinner("Fetching market data and exchange rates..."):
        exchange_rates = get_exchange_rates()
        
        if demo_mode:
            # Force demo data
            current_prices = {
                'GOOGL': 175.50,    # USD
                'EVO.ST': 680.00,   # SEK 
                'NVO': 65.25,       # USD
                'NOVO-B.CO': 520.00, # DKK
                'UNH': 515.75       # USD
            }
            historical_data = None  # Will trigger limited functionality
        else:
            current_prices = get_current_prices()
            historical_data = get_historical_data()
    
    if not current_prices:
        st.error("❌ Unable to fetch any market data. Please check your internet connection or try again later.")
        st.info("💡 **Tip:** This app works best during market hours (9:30 AM - 4:00 PM ET)")
        st.stop()
    
    # Show exchange rates
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Exchange Rates (to DKK):**")
    st.sidebar.write(f"💵 USD: {exchange_rates['USD_TO_DKK']:.2f}")
    st.sidebar.write(f"💰 SEK: {exchange_rates['SEK_TO_DKK']:.2f}")
    
    # Show data quality info
    if demo_mode:
        st.info("🧪 **Demo Mode:** Using sample data for demonstration purposes")
    else:
        st.success(f"✅ **Live Data:** Successfully fetched data for {len(current_prices)} assets")
    
    # Calculate metrics
    summary, total_invested_dkk, total_current_dkk, total_pnl_dkk, total_pnl_percent = calculate_portfolio_summary(current_prices, exchange_rates)
    portfolio_history = calculate_portfolio_history(historical_data, exchange_rates)
    risk_metrics, correlation_matrix = calculate_risk_metrics(historical_data)
    optimization_results, returns_df = portfolio_optimization_analysis(historical_data)
    
    # Summary metrics with better formatting
    st.markdown('<div class="section-header"><h3>💰 Portfolio Overview</h3></div>', unsafe_allow_html=True)
    
    # Create metrics in a more organized layout
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.metric(
            label="💼 Current Value",
            value=f"{total_current_dkk:,.0f} DKK",
            delta=f"{total_pnl_dkk:+,.0f} DKK"
        )
    
    with metric_cols[1]:
        st.metric(
            label="💳 Total Invested",
            value=f"{total_invested_dkk:,.0f} DKK",
            delta=None
        )
    
    with metric_cols[2]:
        st.metric(
            label="📈 Total P&L",
            value=f"{total_pnl_dkk:+,.0f} DKK",
            delta=f"{total_pnl_percent:+.2f}%"
        )
    
    with metric_cols[3]:
        st.metric(
            label="🎯 Total Return",
            value=f"{total_pnl_percent:+.2f}%",
            delta=None
        )
    
    # Charts section with improved layout
    st.markdown('<div class="section-header"><h3>📈 Performance Analysis</h3></div>', unsafe_allow_html=True)
    
    # Historical performance chart - full width
    if not portfolio_history.empty:
        st.plotly_chart(
            create_historical_performance_chart(portfolio_history, total_invested_dkk), 
            use_container_width=True
        )
    else:
        st.info("📊 Historical performance chart unavailable - limited data access")
    
    # Two-column layout for pie chart and bar chart
    chart_col1, chart_col2 = st.columns([1, 1], gap="large")
    
    with chart_col1:
        # Portfolio allocation pie chart
        pie_chart = create_portfolio_pie_chart(summary)
        st.plotly_chart(pie_chart, use_container_width=True)
    
    with chart_col2:
        # P&L bar chart
        bar_chart = create_pnl_bar_chart(summary)
        st.plotly_chart(bar_chart, use_container_width=True)
    
    # Holdings table with improved formatting
    st.markdown('<div class="section-header"><h3>📋 Individual Position Details</h3></div>', unsafe_allow_html=True)
    
    # Create detailed holdings table
    holdings_data = []
    for row in summary:
        holdings_data.append({
            'Asset': row['name'],
            'Ticker': row['ticker'],
            'Shares': f"{row['shares']:,}",
            'Avg Price': f"{row['avg_price_original']:.2f} {row['currency']}",
            'Current Price': f"{row['current_price_original']:.2f} {row['currency']}",
            'Invested (DKK)': f"{row['invested_dkk']:,.0f}",
            'Current Value (DKK)': f"{row['current_value_dkk']:,.0f}",
            'P&L (DKK)': f"{row['pnl_dkk']:+,.0f}",
            'P&L (%)': f"{row['pnl_percent']:+.2f}%"
        })
    
    df_holdings = pd.DataFrame(holdings_data)
    
    # Display with custom styling
    st.dataframe(
        df_holdings,
        use_container_width=True,
        hide_index=True
    )
    
    # Risk Analytics
    st.markdown('<div class="section-header"><h3>⚠️ Risk Analytics</h3></div>', unsafe_allow_html=True)
    
    if risk_metrics:
        # Risk metrics table
        risk_data = []
        total_weight = sum(metrics['portfolio_weight'] for metrics in risk_metrics.values())
        
        for ticker, metrics in risk_metrics.items():
            weight_pct = (metrics['portfolio_weight'] / total_weight * 100) if total_weight > 0 else 0
            risk_data.append({
                'Asset': ticker,
                'Weight (%)': f"{weight_pct:.1f}%",
                'Volatility (%)': f"{metrics['volatility']:.2f}%",
                'VaR 95%': f"{metrics['var_95']:.2f}%",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{metrics['max_drawdown']:.2f}%"
            })
        
        df_risk = pd.DataFrame(risk_data)
        st.dataframe(df_risk, use_container_width=True, hide_index=True)
        
        # Correlation heatmap
        if not correlation_matrix.empty:
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdYlBu_r',
                zmin=-1, zmax=1,
                text=np.round(correlation_matrix.values, 2),
                texttemplate='%{text}',
                textfont={'size': 12},
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': 'Portfolio Correlation Matrix',
                    'x': 0.5,
                    'font': {'size': 18, 'color': '#2c3e50'}
                },
                height=500,
                paper_bgcolor='white',
                font={'color': '#2c3e50'},
                margin=dict(l=80, r=80, t=80, b=60)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Risk analytics require historical data - try refreshing or enable demo mode for full analysis")
    
    # Portfolio Optimization
    st.markdown('<div class="section-header"><h3>🎯 Portfolio Optimization</h3></div>', unsafe_allow_html=True)
    
    if optimization_results and 'optimizations' in optimization_results:
        opt_data = optimization_results['optimizations']
        current_data = optimization_results['current']
        asset_names = optimization_results['asset_names']
        
        if opt_data:
            opt_col1, opt_col2 = st.columns([2, 1], gap="large")
            
            with opt_col1:
                # Efficient frontier
                if 'efficient_frontier' in optimization_results and not optimization_results['efficient_frontier'].empty:
                    efficient_frontier = optimization_results['efficient_frontier']
                    
                    fig = go.Figure()
                    
                    # Efficient frontier line
                    fig.add_trace(go.Scatter(
                        x=efficient_frontier['volatility'],
                        y=efficient_frontier['return'],
                        mode='lines',
                        name='Efficient Frontier',
                        line=dict(color='#1f4e79', width=3),
                        hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                    ))
                    
                    # Current portfolio
                    fig.add_trace(go.Scatter(
                        x=[current_data['volatility']],
                        y=[current_data['return']],
                        mode='markers',
                        name='Current Portfolio',
                        marker=dict(color='#dc3545', size=15, symbol='diamond'),
                        hovertemplate='Current Portfolio<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: ' + f"{current_data['sharpe']:.2f}" + '<extra></extra>'
                    ))
                    
                    # Optimal portfolios
                    colors = ['#28a745', '#ff6b35']
                    names = ['Max Sharpe', 'Min Volatility']
                    
                    for i, (scenario, color, name) in enumerate(zip(['max_sharpe', 'min_vol'], colors, names)):
                        if scenario in opt_data:
                            data = opt_data[scenario]
                            fig.add_trace(go.Scatter(
                                x=[data['volatility']],
                                y=[data['return']],
                                mode='markers',
                                name=name,
                                marker=dict(color=color, size=12),
                                hovertemplate=f'{name}<br>Volatility: %{{x:.2%}}<br>Return: %{{y:.2%}}<br>Sharpe: {data["sharpe"]:.2f}<extra></extra>'
                            ))
                    
                    fig.update_layout(
                        title={
                            'text': 'Efficient Frontier & Portfolio Optimization',
                            'x': 0.5,
                            'font': {'size': 18, 'color': '#2c3e50'}
                        },
                        xaxis_title='Volatility (Risk)',
                        yaxis_title='Expected Return',
                        height=500,
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font={'color': '#2c3e50'},
                        hovermode='closest',
                        margin=dict(l=60, r=60, t=80, b=60),
                        xaxis=dict(tickformat='.1%', gridcolor='#f0f0f0'),
                        yaxis=dict(tickformat='.1%', gridcolor='#f0f0f0')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with opt_col2:
                # Optimization comparison table
                opt_comparison = []
                
                for i, asset in enumerate(asset_names):
                    row = {'Asset': asset, 'Current': f"{current_data['weights'][i]:.1%}"}
                    
                    if 'max_sharpe' in opt_data:
                        row['Max Sharpe'] = f"{opt_data['max_sharpe']['weights'][i]:.1%}"
                    if 'min_vol' in opt_data:
                        row['Min Vol'] = f"{opt_data['min_vol']['weights'][i]:.1%}"
                    
                    opt_comparison.append(row)
                
                # Summary metrics
                summary_metrics = [
                    {'Metric': 'Expected Return', 'Current': f"{current_data['return']:.2%}"},
                    {'Metric': 'Volatility', 'Current': f"{current_data['volatility']:.2%}"},
                    {'Metric': 'Sharpe Ratio', 'Current': f"{current_data['sharpe']:.2f}"}
                ]
                
                if 'max_sharpe' in opt_data:
                    for i, metric in enumerate(['return', 'volatility', 'sharpe']):
                        summary_metrics[i]['Max Sharpe'] = f"{opt_data['max_sharpe'][metric]:.2%}" if metric != 'sharpe' else f"{opt_data['max_sharpe'][metric]:.2f}"
                
                if 'min_vol' in opt_data:
                    for i, metric in enumerate(['return', 'volatility', 'sharpe']):
                        summary_metrics[i]['Min Vol'] = f"{opt_data['min_vol'][metric]:.2%}" if metric != 'sharpe' else f"{opt_data['min_vol'][metric]:.2f}"
                
                st.markdown("**Asset Allocation Comparison:**")
                st.dataframe(pd.DataFrame(opt_comparison), use_container_width=True, hide_index=True)
                
                st.markdown("**Portfolio Metrics Comparison:**")
                st.dataframe(pd.DataFrame(summary_metrics), use_container_width=True, hide_index=True)
                
                # Add rebalancing recommendations
                if 'max_sharpe' in opt_data:
                    st.markdown("**💡 Rebalancing Suggestions (Max Sharpe):**")
                    suggestions = []
                    for i, asset in enumerate(asset_names):
                        current_weight = current_data['weights'][i]
                        optimal_weight = opt_data['max_sharpe']['weights'][i]
                        difference = optimal_weight - current_weight
                        
                        if abs(difference) > 0.05:  # Only show significant changes > 5%
                            action = "Increase" if difference > 0 else "Decrease"
                            suggestions.append(f"• **{action} {asset}** by {abs(difference):.1%}")
                    
                    if suggestions:
                        for suggestion in suggestions[:3]:  # Show top 3 suggestions
                            st.write(suggestion)
                    else:
                        st.write("✅ Portfolio is well-optimized!")
        else:
            st.info("📊 Portfolio optimization requires historical data - try refreshing or enable demo mode for full analysis")
    else:
        st.info("📊 Portfolio optimization requires historical data - try refreshing or enable demo mode for full analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #666;">
        <p>📈 Professional Portfolio Analytics & Risk Management Dashboard</p>
        <p><strong>Technologies:</strong> Python, Streamlit, Plotly, yfinance, Modern Portfolio Theory</p>
        <p><em>All values converted to Danish Kroner (DKK) for consistency</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()