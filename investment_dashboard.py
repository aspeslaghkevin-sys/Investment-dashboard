import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- Sidebar Inputs ---
st.sidebar.header("📊 Portfolio Configuration")
tickers_input = st.sidebar.text_input("Enter stock tickers (comma-separated)", "AAPL,MSFT,GOOGL")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# --- Fetch Data Safely ---
@st.cache_data
def get_data(tickers, start, end):
    raw_data = yf.download(tickers, start=start, end=end, group_by='ticker')
    if raw_data.empty:
        return None
    try:
        if len(tickers) == 1:
            return raw_data["Adj Close"].to_frame(name=tickers[0])
        else:
            return raw_data["Adj Close"]
    except KeyError:
        return None

data = get_data(tickers, start_date, end_date)

# --- Main Dashboard ---
st.title("💼 Investment Dashboard")
st.markdown("Track your portfolio performance with real-time data and interactive charts.")

if not tickers:
    st.warning("Please enter at least one valid ticker symbol.")
elif data is None or data.empty:
    st.error("❌ No data found. Please check your ticker symbols or date range.")
else:
    # --- Real-Time Price Display ---
    st.subheader("📌 Latest Prices")
    latest_prices = {ticker: round(data[ticker].dropna().iloc[-1], 2) for ticker in data.columns}
    st.write(pd.DataFrame(latest_prices.items(), columns=["Ticker", "Latest Price ($)"]))

    # --- Line Chart ---
    st.subheader("📈 Price History")
    st.line_chart(data)

    # --- Performance Metrics ---
    st.subheader("📊 Cumulative Returns")
    returns = data.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod() - 1
    st.dataframe(cumulative_returns.tail())

    # --- Interactive Plot ---
    st.subheader("📉 Interactive Chart")
    fig = go.Figure()
    for ticker in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[ticker], mode='lines', name=ticker))
    fig.update_layout(title="Stock Prices Over Time", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

    # --- Download Option ---
    st.subheader("📥 Download Data")
    st.download_button("Download CSV", data.to_csv(), file_name="portfolio_data.csv")
