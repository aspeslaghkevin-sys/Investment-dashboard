# investment_dashboard_enhanced.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta

st.set_page_config(page_title="Enhanced Investment Dashboard", layout="wide")

# -----------------------
# Helpers and caching
# -----------------------
@st.cache_data(show_spinner=False)
def fetch_history(tickers, start, end, interval):
    if not tickers:
        return pd.DataFrame()
    try:
        # yf.download handles multiple tickers; use auto_adjust to get adjusted prices if desired
        df = yf.download(tickers, start=start, end=end, interval=interval, group_by='ticker', auto_adjust=True, threads=True)
        if df.empty:
            return pd.DataFrame()
        # Normalize dataframe to Close prices: produce DataFrame with columns = tickers, index = dates
        if isinstance(tickers, str) or len(tickers) == 1:
            # single ticker: df may be single-level
            col = "Close" if "Close" in df.columns else "Adj Close" if "Adj Close" in df.columns else None
            if col:
                return pd.DataFrame({tickers if isinstance(tickers, str) else tickers[0]: df[col]})
            else:
                return pd.DataFrame()
        # multiple tickers: df is multiindex columns like ('AAPL','Close') or ('Adj Close')
        close_frames = {}
        # support both styles: either top-level tickers or 'Adj Close' single level
        if isinstance(df.columns, pd.MultiIndex):
            # try first with 'Close' then 'Adj Close'
            for t in tickers:
                try:
                    if ('Close' in df[t].columns):
                        close_frames[t] = df[t]['Close']
                    elif ('Adj Close' in df[t].columns):
                        close_frames[t] = df[t]['Adj Close']
                    else:
                        # pick Close-like column if available
                        cols = list(df[t].columns)
                        if cols:
                            close_frames[t] = df[t][cols[0]]
                except Exception:
                    continue
        else:
            # single-level columns for multiple tickers: columns may be tickers already if user passed string "A B C"
            # attempt to convert directly
            for c in df.columns:
                close_frames[c] = df[c]
        if not close_frames:
            return pd.DataFrame()
        close_df = pd.concat(close_frames, axis=1)
        # ensure consistent datetime index
        close_df.index = pd.to_datetime(close_df.index)
        return close_df.sort_index()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_latest_price(ticker):
    try:
        t = yf.Ticker(ticker)
        # try fast_info if present then fallback
        info = getattr(t, "fast_info", None)
        if info and info.get("last_price") is not None:
            return float(info["last_price"])
        hist = t.history(period="5d", interval="1d", auto_adjust=True)
        if not hist.empty:
            return float(hist['Close'].dropna().iloc[-1])
        return None
    except Exception:
        return None

def safe_round(x):
    try:
        return round(float(x), 2)
    except Exception:
        return x

# -----------------------
# Session state init
# -----------------------
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "MSFT", "GOOGL"]
if 'holdings' not in st.session_state:
    # holdings structure: { 'AAPL': {'qty': 0.0, 'cost': 0.0} }
    st.session_state.holdings = {t: {"qty": 0.0, "cost": 0.0} for t in st.session_state.watchlist}
if 'cash' not in st.session_state:
    st.session_state.cash = 0.0

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Watchlist Controls")
    new_ticker = st.text_input("Add ticker (comma-separated to add many)", value="")
    if st.button("Add ticker(s)"):
        entries = [x.strip().upper() for x in new_ticker.replace(";", ",").split(",") if x.strip()]
        for e in entries:
            if e and e not in st.session_state.watchlist:
                st.session_state.watchlist.append(e)
                st.session_state.holdings.setdefault(e, {"qty": 0.0, "cost": 0.0})
    remove = st.multiselect("Remove tickers", st.session_state.watchlist)
    if st.button("Remove selected"):
        for r in remove:
            if r in st.session_state.watchlist:
                st.session_state.watchlist.remove(r)
                st.session_state.holdings.pop(r, None)

    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV (tickers, optional qty, optional cost)", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded, header=None)
            for i, row in df_up.iterrows():
                t = str(row[0]).strip().upper()
                if not t or t.lower() in ["nan", "none"]:
                    continue
                qty = float(row[1]) if len(row) > 1 and pd.notna(row[1]) else 0.0
                cost = float(row[2]) if len(row) > 2 and pd.notna(row[2]) else 0.0
                if t not in st.session_state.watchlist:
                    st.session_state.watchlist.append(t)
                st.session_state.holdings[t] = {"qty": qty, "cost": cost}
            st.success("CSV processed and added to watchlist.")
        except Exception:
            st.error("Failed to parse CSV. Expect rows like: TICKER,qty,cost")

    st.markdown("---")
    st.subheader("Time range and interval")
    today = date.today()
    end_default = today
    start_default = today - timedelta(days=365)
    start_date = st.date_input("Start date", start_default)
    end_date = st.date_input("End date", end_default)
    if start_date >= end_date:
        st.error("Start date must be before end date.")
    interval = st.selectbox("Interval", options=["1d", "1wk", "1mo"], index=0)
    st.markdown("---")
    st.subheader("Chart options")
    chart_mode = st.selectbox("Chart mode", ["Combined line", "Normalized comparison", "Single ticker (candlestick)"])
    show_sma = st.checkbox("Show SMA", value=False)
    sma_window = st.slider("SMA window", 5, 200, 20) if show_sma else None
    show_ema = st.checkbox("Show EMA", value=False)
    ema_window = st.slider("EMA window", 5, 200, 50) if show_ema else None
    show_volume = st.checkbox("Show volume on single ticker", value=True)
    st.markdown("---")
    st.subheader("Net worth")
    st.session_state.cash = st.number_input("Cash balance", value=float(st.session_state.cash), step=100.0, format="%.2f")
    st.markdown("Holdings (enter qty and cost basis below in the main area).")

# -----------------------
# Main layout
# -----------------------
st.title("Enhanced Interactive Investment Dashboard")
left, right = st.columns([3, 1])

with left:
    st.header("Watchlist and Holdings")
    # show editable holdings table
    if st.session_state.watchlist:
        holdings_df = pd.DataFrame([
            {"Ticker": t, "Qty": st.session_state.holdings.get(t, {}).get("qty", 0.0),
             "Cost": st.session_state.holdings.get(t, {}).get("cost", 0.0)}
            for t in st.session_state.watchlist
        ])
        edited = st.experimental_data_editor(holdings_df, num_rows="dynamic")
        # sync back to session_state holdings
        for idx, row in edited.iterrows():
            t = str(row["Ticker"]).strip().upper()
            try:
                qty = float(row["Qty"])
            except Exception:
                qty = 0.0
            try:
                cost = float(row["Cost"])
            except Exception:
                cost = 0.0
            if t not in st.session_state.watchlist:
                st.session_state.watchlist.append(t)
            st.session_state.holdings[t] = {"qty": qty, "cost": cost}
    else:
        st.info("Watchlist is empty. Add tickers in the sidebar.")

    st.markdown("---")
    # Fetch price history for all watchlist
    fetch_tickers = list(st.session_state.watchlist)
    data = fetch_history(fetch_tickers, start_date.isoformat(), end_date.isoformat(), interval)

    if data.empty:
        st.error("No price data available for the selected tickers/timeframe. Verify tickers and date range.")
    else:
        st.subheader("Charts")
        # chart selector for which tickers to plot
        to_plot = st.multiselect("Select tickers to plot (for combined/normalized)", options=list(data.columns), default=list(data.columns)[:3])
        if not to_plot:
            st.warning("Select at least one ticker to plot.")
        else:
            plot_df = data[to_plot].copy()
            if chart_mode == "Combined line":
                fig = go.Figure()
                for t in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[t], mode='lines', name=t))
                if show_sma:
                    for t in plot_df.columns:
                        sma = data[t].rolling(window=sma_window, min_periods=1).mean()
                        fig.add_trace(go.Scatter(x=sma.index, y=sma, mode='lines', name=f"{t} SMA{str(sma_window)}", line=dict(dash='dash')))
                if show_ema:
                    for t in plot_df.columns:
                        ema = data[t].ewm(span=ema_window, adjust=False).mean()
                        fig.add_trace(go.Scatter(x=ema.index, y=ema, mode='lines', name=f"{t} EMA{str(ema_window)}", line=dict(dash='dot')))
                fig.update_layout(title="Combined Price Chart", xaxis_title="Date", yaxis_title="Price", legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)

            elif chart_mode == "Normalized comparison":
                norm = plot_df.divide(plot_df.iloc[0]).fillna(method='ffill')
                fig = px.line(norm, x=norm.index, y=norm.columns, labels={"value":"Normalized", "index":"Date"})
                fig.update_layout(title="Normalized Comparison (base = 1 on first date)", legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)

            else:  # Single ticker candlestick/ohlc
                active = st.selectbox("Select ticker for OHLC/Candlestick", options=list(data.columns))
                tdf = None
                try:
                    # attempt to fetch raw OHLC for active ticker (full history)
                    raw = yf.download(active, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval, auto_adjust=True)
                    if raw.empty:
                        st.error(f"No OHLC data for {active}")
                    else:
                        raw.index = pd.to_datetime(raw.index)
                        tdf = raw
                except Exception:
                    tdf = None

                if tdf is not None and not tdf.empty:
                    type_choice = st.radio("Plot type", ["Candlestick", "OHLC"], horizontal=True)
                    fig2 = go.Figure()
                    if type_choice == "Candlestick":
                        fig2.add_trace(go.Candlestick(x=tdf.index, open=tdf['Open'], high=tdf['High'], low=tdf['Low'], close=tdf['Close'], name=active))
                    else:
                        fig2.add_trace(go.Ohlc(x=tdf.index, open=tdf['Open'], high=tdf['High'], low=tdf['Low'], close=tdf['Close'], name=active))
                    if show_sma:
                        sma = tdf['Close'].rolling(window=sma_window, min_periods=1).mean()
                        fig2.add_trace(go.Scatter(x=sma.index, y=sma, mode='lines', name=f"SMA{str(sma_window)}", line=dict(color='orange')))
                    if show_ema:
                        ema = tdf['Close'].ewm(span=ema_window, adjust=False).mean()
                        fig2.add_trace(go.Scatter(x=ema.index, y=ema, mode='lines', name=f"EMA{str(ema_window)}", line=dict(color='purple')))
                    if show_volume and 'Volume' in tdf.columns:
                        vol = go.Bar(x=tdf.index, y=tdf['Volume'], name='Volume', marker=dict(color='lightgray'), yaxis='y2')
                        fig2.add_trace(vol)
                        fig2.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Volume'))
                        fig2.update_layout(xaxis_rangeslider_visible=True)
                    fig2.update_layout(height=600)
                    st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("Raw Close Prices (sample)")
        st.dataframe(data.tail(10))

        # download combined CSV
        csv = data.to_csv().encode('utf-8')
        st.download_button("Download All Close Prices CSV", csv, "close_prices.csv", "text/csv")

with right:
    st.header("Net Worth & Snapshot")
    # compute latest prices for each ticker
    latest = {}
    for t in st.session_state.watchlist:
        try:
            # take last non-null from fetched data if available, else fetch individually
            if t in data.columns and not data[t].dropna().empty:
                latest_price = data[t].dropna().iloc[-1]
            else:
                latest_price = fetch_latest_price(t)
        except Exception:
            latest_price = fetch_latest_price(t)
        latest[t] = latest_price

    # compute holdings valuation
    rows = []
    total_holdings_value = 0.0
    unrealized_pl = 0.0
    for t in st.session_state.watchlist:
        qty = st.session_state.holdings.get(t, {}).get("qty", 0.0)
        cost = st.session_state.holdings.get(t, {}).get("cost", 0.0)
        price = latest.get(t)
        mv = None
        pl = None
        if price is not None:
            mv = qty * price
            total_holdings_value += mv
            if qty and cost:
                pl = mv - (qty * cost)
                unrealized_pl += pl
        rows.append({"Ticker": t, "Qty": qty, "Price": safe_round(price) if price is not None else "N/A", "Market Value": safe_round(mv) if mv is not None else "N/A", "Unrealized P/L": safe_round(pl) if pl is not None else "N/A"})

    holdings_table = pd.DataFrame(rows).set_index("Ticker")
    st.subheader("Holdings")
    st.dataframe(holdings_table)

    st.markdown("---")
    st.subheader("Net Worth")
    cash = float(st.session_state.cash)
    total_net_worth = total_holdings_value + cash
    st.metric("Holdings Value", f"${total_holdings_value:,.2f}")
    st.metric("Cash", f"${cash:,.2f}")
    st.metric("Total Net Worth", f"${total_net_worth:,.2f}")
    st.markdown(f"**Unrealized P/L:** ${unrealized_pl:,.2f}")

    st.markdown("---")
    # allocation pie chart
    alloc_df = pd.DataFrame([
        {"Ticker": t, "Value": (r["Market Value"] if isinstance(r["Market Value"], (int, float)) else 0.0)}
        for t, r in zip(st.session_state.watchlist, rows)
    ])
    if alloc_df["Value"].sum() > 0:
        fig_alloc = px.pie(alloc_df, names='Ticker', values='Value', title='Holdings Allocation', hole=0.35)
        st.plotly_chart(fig_alloc, use_container_width=True)
    else:
        st.info("No market value found to compute allocation.")

    st.markdown("---")
    st.subheader("Quick actions")
    if st.button("Refresh Prices"):
        # clear caches, then re-run to refresh (cache_data has no direct clear method in older streamlit)
        st.cache_data.clear()
        st.experimental_rerun()

    st.download_button("Download Holdings CSV", holdings_table.reset_index().to_csv(index=False).encode('utf-8'), "holdings.csv", "text/csv")

st.caption("Enhanced dashboard with adjustable timeframe, multi-ticker plotting, and net worth aggregation. Data via Yahoo Finance.")
