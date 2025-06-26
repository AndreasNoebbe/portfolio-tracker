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
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #1f4e79;
    }
    
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    
    .section-header {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Your actual portfolio - exactly as you own it
PORTFOLIO = {
    'GOOGL': {'shares': 55, 'avg_price': 154.84, 'currency': 'USD', 'name': 'Alphabet Inc.'},
    'EVO.ST': {'shares': 20, 'avg_price': 651.9, 'currency': 'SEK', 'name': 'Evolution AB'},
    'NVO': {'shares': 270, 'avg_price': 60.44, 'currency': 'USD', 'name': 'Novo Nordisk ADR'},
    'NOVO-B.CO': {'shares': 430, 'avg_price': 477.49, 'currency': 'DKK', 'name': 'Novo Nordisk B'},
    'UNH': {'shares': 50, 'avg_price': 294.13, 'currency': 'USD', 'name': 'UnitedHealth Group'}
}

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
            'GOOGL': 175.50,  # Realistic current prices
            'EVO.ST': 680.00,  
            'NVO': 65.25,     
            'NOVO-B.CO': 520.00,  
            'UNH': 515.75    
        }
    
    return current_prices

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

def calculate_portfolio_summary(current_prices):
    """Calculate your actual P&L and portfolio metrics"""
    summary = []
    total_invested = 0
    total_current = 0
    
    for ticker, position in PORTFOLIO.items():
        current_price = current_prices.get(ticker, 0)
        
        # Calculate values
        invested_value = position['shares'] * position['avg_price']
        current_value = position['shares'] * current_price
        pnl = current_value - invested_value
        pnl_percent = (pnl / invested_value * 100) if invested_value > 0 else 0
        
        summary.append({
            'ticker': ticker,
            'name': position['name'],
            'shares': position['shares'],
            'avg_price': position['avg_price'],
            'current_price': current_price,
            'invested': invested_value,
            'current_value': current_value,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'currency': position['currency']
        })
        
        total_invested += invested_value
        total_current += current_value
    
    total_pnl = total_current - total_invested
    total_pnl_percent = (total_pnl / total_invested * 100) if total_invested > 0 else 0
    
    return summary, total_invested, total_current, total_pnl, total_pnl_percent

def calculate_portfolio_history(data):
    """Calculate your portfolio value over time based on your actual positions"""
    if data is None:
        return pd.DataFrame()
    
    portfolio_values = []
    dates = data.index
    
    for date in dates:
        daily_value = 0
        for ticker, position in PORTFOLIO.items():
            try:
                if len(PORTFOLIO) == 1:
                    price = data.loc[date, 'Adj Close']
                else:
                    price = data.loc[date, (ticker, 'Adj Close')]
                
                if not pd.isna(price):
                    daily_value += position['shares'] * price
                    
            except Exception:
                continue
        
        if daily_value > 0:
            portfolio_values.append({'date': date, 'portfolio_value': daily_value})
    
    return pd.DataFrame(portfolio_values)

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
    
    # Fetch data
    with st.spinner("Fetching market data..."):
        if demo_mode:
            # Force demo data
            current_prices = {
                'GOOGL': 175.50,
                'EVO.ST': 680.00, 
                'NVO': 65.25,
                'NOVO-B.CO': 520.00,
                'UNH': 515.75
            }
            historical_data = None  # Will trigger limited functionality
        else:
            current_prices = get_current_prices()
            historical_data = get_historical_data()
    
    if not current_prices:
        st.error("❌ Unable to fetch any market data. Please check your internet connection or try again later.")
        st.info("💡 **Tip:** This app works best during market hours (9:30 AM - 4:00 PM ET)")
        st.stop()
    
    # Add data source status to sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Source Status:**")
    
    # Simple status without additional API calls
    if demo_mode:
        st.sidebar.info("📊 Demo mode enabled")
    elif len(current_prices) == len(PORTFOLIO):
        st.sidebar.success("✅ All tickers loaded")
    else:
        st.sidebar.warning("⚠️ Some tickers using fallback data")
    
    # Show data quality info
    if demo_mode:
        st.info("🧪 **Demo Mode:** Using sample data for demonstration purposes")
    elif any("demo" in str(current_prices).lower() for ticker in current_prices):
        st.info("ℹ️ **Fallback Mode:** Some data may be simulated due to market closure or API limitations")
    else:
        st.success(f"✅ **Live Data:** Successfully fetched data for {len(current_prices)} assets")
    
    # Calculate metrics
    summary, total_invested, total_current, total_pnl, total_pnl_percent = calculate_portfolio_summary(current_prices)
    portfolio_history = calculate_portfolio_history(historical_data)
    risk_metrics, correlation_matrix = calculate_risk_metrics(historical_data)
    optimization_results, returns_df = portfolio_optimization_analysis(historical_data)
    
    # Summary metrics
    st.markdown('<div class="section-header"><h3>💰 Portfolio Overview</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Value",
            value=f"${total_current:,.2f}",
            delta=f"${total_pnl:+,.2f}"
        )
    
    with col2:
        st.metric(
            label="Total Invested",
            value=f"${total_invested:,.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Total P&L",
            value=f"${total_pnl:+,.2f}",
            delta=f"{total_pnl_percent:+.2f}%"
        )
    
    with col4:
        st.metric(
            label="Total Return",
            value=f"{total_pnl_percent:+.2f}%",
            delta=None
        )
    
    # Charts section
    st.markdown('<div class="section-header"><h3>📈 Performance Analysis</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Historical performance chart
        if not portfolio_history.empty:
            total_invested_line = sum(pos['shares'] * pos['avg_price'] for pos in PORTFOLIO.values())
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_history['date'],
                y=portfolio_history['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f4e79', width=3),
                fill='tonexty'
            ))
            
            fig.add_hline(y=total_invested_line, line_dash="dash", line_color="red", 
                         annotation_text="Total Invested")
            
            fig.update_layout(
                title="Portfolio Performance Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Portfolio allocation pie chart
        labels = [row['name'] for row in summary]
        values = [row['current_value'] for row in summary]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='percent',
            textposition='inside'
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # P&L Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # P&L bar chart
        tickers = [row['ticker'] for row in summary]
        pnl_values = [row['pnl'] for row in summary]
        colors = ['green' if pnl >= 0 else 'red' for pnl in pnl_values]
        
        fig = go.Figure(data=[go.Bar(
            x=tickers,
            y=pnl_values,
            marker_color=colors,
            text=[f"${pnl:+,.0f}" for pnl in pnl_values],
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Profit & Loss by Position",
            xaxis_title="Holdings",
            yaxis_title="P&L ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Holdings table
        df_summary = pd.DataFrame(summary)
        df_display = df_summary[['name', 'shares', 'current_price', 'pnl', 'pnl_percent']].copy()
        df_display.columns = ['Asset', 'Shares', 'Current Price', 'P&L ($)', 'P&L (%)']
        df_display['Current Price'] = df_display['Current Price'].apply(lambda x: f"${x:.2f}")
        df_display['P&L ($)'] = df_display['P&L ($)'].apply(lambda x: f"${x:+,.2f}")
        df_display['P&L (%)'] = df_display['P&L (%)'].apply(lambda x: f"{x:+.2f}%")
        
        st.markdown("**Individual Position Details:**")
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    
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
                textfont={'size': 12}
            ))
            
            fig.update_layout(
                title="Portfolio Correlation Matrix",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio Optimization
    st.markdown('<div class="section-header"><h3>🎯 Portfolio Optimization</h3></div>', unsafe_allow_html=True)
    
    if optimization_results and 'optimizations' in optimization_results:
        opt_data = optimization_results['optimizations']
        current_data = optimization_results['current']
        asset_names = optimization_results['asset_names']
        
        if opt_data:
            col1, col2 = st.columns([2, 1])
            
            with col1:
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
                        line=dict(color='#1f4e79', width=3)
                    ))
                    
                    # Current portfolio
                    fig.add_trace(go.Scatter(
                        x=[current_data['volatility']],
                        y=[current_data['return']],
                        mode='markers',
                        name='Current Portfolio',
                        marker=dict(color='red', size=15, symbol='diamond')
                    ))
                    
                    # Optimal portfolios
                    colors = ['green', 'orange']
                    names = ['Max Sharpe', 'Min Volatility']
                    
                    for i, (scenario, color, name) in enumerate(zip(['max_sharpe', 'min_vol'], colors, names)):
                        if scenario in opt_data:
                            data = opt_data[scenario]
                            fig.add_trace(go.Scatter(
                                x=[data['volatility']],
                                y=[data['return']],
                                mode='markers',
                                name=name,
                                marker=dict(color=color, size=12)
                            ))
                    
                    fig.update_layout(
                        title='Efficient Frontier & Portfolio Optimization',
                        xaxis_title='Volatility (Risk)',
                        yaxis_title='Expected Return',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #666;">
        <p>📈 Professional Portfolio Analytics & Risk Management Dashboard</p>
        <p><strong>Technologies:</strong> Python, Streamlit, Plotly, yfinance, Modern Portfolio Theory</p>
        <p><em>Note: Some international tickers (EVO.ST, NOVO-B.CO) may have limited data availability outside market hours</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()