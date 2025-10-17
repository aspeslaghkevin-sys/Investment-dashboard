import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- Sidebar Inputs ---
st.sidebar.header("Portfolio Configuration")
tickers = st.sidebar.text_input("Enter stock tickers (comma-separated)", "AAPL,MSFT,GOOGL").upper().split(",")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# --- Fetch Data ---
@st.cache_data
def get_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    return data

data = get_data(tickers, start_date, end_date)

# --- Portfolio Overview ---
st.title("📈 Investment Dashboard")
st.subheader("Portfolio Performance Overview")

st.line_chart(data)

# --- Performance Metrics ---
st.subheader("Performance Metrics")
returns = data.pct_change().dropna()
cumulative_returns = (1 + returns).cumprod() - 1
st.dataframe(cumulative_returns.tail())

# --- Interactive Plot ---
st.subheader("Interactive Price Chart")
fig = go.Figure()
for ticker in tickers:
    fig.add_trace(go.Scatter(x=data.index, y=data[ticker], mode='lines', name=ticker))
fig.update_layout(title="Stock Prices Over Time", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig)

# --- Download Option ---
st.subheader("Download Data")
st.download_button("Download CSV", data.to_csv(), file_name="portfolio_data.csv")
