# investment_dashboard_full.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import io
import os

st.set_page_config(page_title="Investment Dashboard", layout="wide", initial_sidebar_state="expanded")

# -----------------------
# CSS: load styles.css if present, else inject default
# -----------------------
def inject_css():
    css_path = "styles.css"
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        css = """
        <style>
        :root { --bg:#0b1220; --panel:#071126; --accent:#06b6d4; --muted:#9fb7c9; }
        .stApp { background: linear-gradient(180deg,#071126 0%, #0b1b2b 100%); }
        [data-testid="stSidebar"] { background: #071126 !important; color: var(--muted) !important; }
        .stButton>button { background: var(--accent); color: #022; border-radius: 8px; padding: 6px 10px; }
        .css-1d391kg { box-shadow: 0 6px 24px rgba(2,6,23,0.6); border-radius: 10px; }
        .stDownloadButton>button { background: #0ea5a4; color: #022; }
        .streamlit-expanderHeader { color: #cfe7ff; }
        table { border-collapse: collapse; width: 100%; }
        table th, table td { padding: 6px 8px; border: 1px solid rgba(255,255,255,0.06); }
        footer { visibility: hidden; height: 0px; }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

inject_css()

# -----------------------
# Compatibility wrapper for data editor
# -----------------------
def data_editor_wrapper(df: pd.DataFrame, key: str = None, num_rows="fixed", height=None):
    """
    Use st.data_editor if available, else st.experimental_data_editor, else CSV textarea fallback.
    Returns edited DataFrame.
    """
    if hasattr(st, "data_editor"):
        return st.data_editor(df, num_rows=num_rows, key=key, height=height)
    if hasattr(st, "experimental_data_editor"):
        return st.experimental_data_editor(df, num_rows=num_rows, key=key, height=height)
    st.warning("Interactive data editor not available in this Streamlit version. Use the CSV editor below to edit holdings.")
    csv = df.to_csv(index=False)
    edited = st.text_area("Edit CSV (header included)", value=csv, height=220, key=(key or "csv") + "_fallback")
    try:
        new_df = pd.read_csv(io.StringIO(edited))
        return new_df
    except Exception:
        st.error("CSV parsing failed. No changes applied.")
        return df

# -----------------------
# Data fetchers and helpers (cached)
# -----------------------
@st.cache_data(show_spinner=False)
def fetch_history(tickers, start, end, interval):
    """
    Fetch historical OHLCV and return a DataFrame of Close prices with columns=tickers.
    tickers: list[str]
    """
    if not tickers:
        return pd.DataFrame()
    try:
        # Yahoo allows list of tickers
        df = yf.download(tickers, start=start, end=end, interval=interval, group_by='ticker', auto_adjust=True, threads=True)
        if df.empty:
            return pd.DataFrame()
        # Single ticker case: df may be single-level
        if isinstance(tickers, (list, tuple)) and len(tickers) == 1:
            t0 = tickers[0]
            col = "Close" if "Close" in df.columns else "Adj Close" if "Adj Close" in df.columns else None
            if col:
                return pd.DataFrame({t0: df[col]})
            return pd.DataFrame()
        if not isinstance(tickers, (list, tuple)) and isinstance(df.columns, pd.Index):
            # user passed a single string without list and df is simple
            col = "Close" if "Close" in df.columns else "Adj Close" if "Adj Close" in df.columns else None
            if col:
                return pd.DataFrame({tickers: df[col]})
            return pd.DataFrame()

        close_frames = {}
        if isinstance(df.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    # prefer 'Close' then 'Adj Close', then fallback to first numeric column
                    if 'Close' in df[t].columns:
                        close_frames[t] = df[t]['Close']
                    elif 'Adj Close' in df[t].columns:
                        close_frames[t] = df[t]['Adj Close']
                    else:
                        cols = [c for c in df[t].columns if c in ['Close','Adj Close','close','adjclose']]
                        if cols:
                            close_frames[t] = df[t][cols[0]]
                        else:
                            first_col = list(df[t].columns)[0]
                            close_frames[t] = df[t][first_col]
                except Exception:
                    continue
        else:
            # single-level columns containing multiple tickers (rare)
            for c in df.columns:
                close_frames[c] = df[c]

        if not close_frames:
            return pd.DataFrame()
        close_df = pd.concat(close_frames, axis=1)
        close_df.index = pd.to_datetime(close_df.index)
        return close_df.sort_index()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_latest_price(ticker):
    try:
        t = yf.Ticker(ticker)
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
    st.session_state.holdings = {t: {"qty": 0.0, "cost": 0.0} for t in st.session_state.watchlist}
if 'cash' not in st.session_state:
    st.session_state.cash = 0.0

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Add / Manage Tickers and Holdings")
    new_ticker = st.text_input("Ticker (single or comma-separated)", value="")
    col_a, col_b = st.columns([1,1])
    with col_a:
        new_qty = st.number_input("Quantity (single add)", value=0.0, step=1.0, format="%.4f")
    with col_b:
        new_cost = st.number_input("Cost basis per share", value=0.0, step=0.01, format="%.2f")
    if st.button("Add ticker(s) with qty"):
        entries = [x.strip().upper() for x in new_ticker.replace(";", ",").split(",") if x.strip()]
        for i, e in enumerate(entries):
            if e and e not in st.session_state.watchlist:
                st.session_state.watchlist.append(e)
                if len(entries) == 1:
                    st.session_state.holdings[e] = {"qty": float(new_qty), "cost": float(new_cost)}
                else:
                    st.session_state.holdings.setdefault(e, {"qty": 0.0, "cost": 0.0})
    st.markdown("---")
    remove = st.multiselect("Remove tickers", options=st.session_state.watchlist)
    if st.button("Remove selected"):
        for r in remove:
            if r in st.session_state.watchlist:
                st.session_state.watchlist.remove(r)
                st.session_state.holdings.pop(r, None)
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV (ticker,qty,cost) or (ticker)", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded, header=None)
            for _, row in df_up.iterrows():
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
    default_end = today
    default_start = today - timedelta(days=365)
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=default_end)
    if start_date >= end_date:
        st.error("Start date must be before end date.")
    interval = st.selectbox("Interval", options=["1d", "1wk", "1mo"], index=0)
    st.markdown("---")
    st.subheader("Chart Settings")
    chart_mode = st.selectbox("Chart mode", ["Combined line", "Normalized comparison", "Single ticker (candlestick)"])
    normalize = st.checkbox("Normalize prices (for comparison)", value=False)
    show_sma = st.checkbox("Show SMA", value=False)
    sma_window = st.slider("SMA window", 5, 200, 20) if show_sma else None
    show_ema = st.checkbox("Show EMA", value=False)
    ema_window = st.slider("EMA window", 5, 200, 50) if show_ema else None
    show_volume = st.checkbox("Show volume on single ticker", value=True)
    st.markdown("---")
    st.subheader("Net worth")
    st.session_state.cash = st.number_input("Cash balance", value=float(st.session_state.cash), step=100.0, format="%.2f")
    st.markdown("---")
    if st.button("Clear watchlist"):
        st.session_state.watchlist = []
        st.session_state.holdings = {}
        st.success("Watchlist cleared.")

# -----------------------
# Main layout
# -----------------------
st.title("Interactive Investment Dashboard — Full")
left, right = st.columns([3, 1])

with left:
    st.header("Holdings Editor")
    holdings_df = pd.DataFrame([
        {"Ticker": t,
         "Qty": st.session_state.holdings.get(t, {}).get("qty", 0.0),
         "Cost": st.session_state.holdings.get(t, {}).get("cost", 0.0)}
        for t in st.session_state.watchlist
    ])
    if holdings_df.empty:
        st.info("No holdings yet. Add tickers in the sidebar.")
    edited = data_editor_wrapper(holdings_df, key="holdings_editor", num_rows="dynamic", height=220)
    try:
        # Ensure 'Ticker' column exists and sync back to session_state
        if not edited.empty:
            for _, row in edited.iterrows():
                t = str(row.get("Ticker") or "").strip().upper()
                if not t:
                    continue
                try:
                    qty = float(row.get("Qty") or 0.0)
                except Exception:
                    qty = 0.0
                try:
                    cost = float(row.get("Cost") or 0.0)
                except Exception:
                    cost = 0.0
                if t not in st.session_state.watchlist:
                    st.session_state.watchlist.append(t)
                st.session_state.holdings[t] = {"qty": qty, "cost": cost}
    except Exception:
        st.error("Failed to process edited holdings. No changes applied.")

    st.markdown("---")
    fetch_tickers = list(st.session_state.watchlist)
    data = fetch_history(fetch_tickers, start_date.isoformat(), end_date.isoformat(), interval)

    if data.empty:
        st.error("No price data available for the selected tickers/timeframe. Verify tickers and date range.")
    else:
        st.subheader("Charts")
        to_plot = st.multiselect("Select tickers to plot", options=list(data.columns), default=list(data.columns)[:4])
        if not to_plot:
            st.warning("Select at least one ticker to plot.")
        else:
            plot_df = data[to_plot].copy()
            if normalize:
                # Base normalization to 1 at first valid date
                first_vals = plot_df.iloc[0]
                plot_df = plot_df.divide(first_vals).fillna(method='ffill')
            if chart_mode == "Combined line":
                fig = go.Figure()
                for t in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[t], mode='lines', name=t))
                if show_sma:
                    for t in plot_df.columns:
                        sma = data[t].rolling(window=sma_window, min_periods=1).mean()
                        fig.add_trace(go.Scatter(x=sma.index, y=sma, mode='lines', name=f"{t} SMA{sma_window}", line=dict(dash='dash')))
                if show_ema:
                    for t in plot_df.columns:
                        ema = data[t].ewm(span=ema_window, adjust=False).mean()
                        fig.add_trace(go.Scatter(x=ema.index, y=ema, mode='lines', name=f"{t} EMA{ema_window}", line=dict(dash='dot')))
                fig.update_layout(title="Combined Price Chart", xaxis_title="Date", yaxis_title="Price", legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)

            elif chart_mode == "Normalized comparison":
                norm = plot_df  # already normalized above if normalize True; else divide by first row for normalized mode
                if not normalize:
                    norm = plot_df.divide(plot_df.iloc[0]).fillna(method='ffill')
                fig = px.line(norm, x=norm.index, y=norm.columns, labels={"value": "Normalized", "index": "Date"})
                fig.update_layout(title="Normalized Comparison (base = 1 on first date)", legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)

            else:
                active = st.selectbox("Select ticker for OHLC/Candlestick", options=list(data.columns))
                raw = yf.download(active, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval, auto_adjust=True)
                if raw.empty:
                    st.error(f"No OHLC data for {active}")
                else:
                    raw.index = pd.to_datetime(raw.index)
                    type_choice = st.radio("Plot type", ["Candlestick", "OHLC"], horizontal=True)
                    fig2 = go.Figure()
                    if type_choice == "Candlestick":
                        fig2.add_trace(go.Candlestick(x=raw.index, open=raw['Open'], high=raw['High'], low=raw['Low'], close=raw['Close'], name=active))
                    else:
                        fig2.add_trace(go.Ohlc(x=raw.index, open=raw['Open'], high=raw['High'], low=raw['Low'], close=raw['Close'], name=active))
                    if show_sma:
                        sma = raw['Close'].rolling(window=sma_window, min_periods=1).mean()
                        fig2.add_trace(go.Scatter(x=sma.index, y=sma, mode='lines', name=f"SMA{sma_window}", line=dict(color='orange')))
                    if show_ema:
                        ema = raw['Close'].ewm(span=ema_window, adjust=False).mean()
                        fig2.add_trace(go.Scatter(x=ema.index, y=ema, mode='lines', name=f"EMA{ema_window}", line=dict(color='purple')))
                    if show_volume and 'Volume' in raw.columns:
                        vol = go.Bar(x=raw.index, y=raw['Volume'], name='Volume', marker=dict(color='lightgray'), yaxis='y2')
                        fig2.add_trace(vol)
                        fig2.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Volume'))
                        fig2.update_layout(xaxis_rangeslider_visible=True)
                    fig2.update_layout(height=600)
                    st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("Close prices (sample)")
        st.dataframe(data.tail(10))
        st.download_button("Download All Close Prices CSV", data.to_csv().encode('utf-8'), file_name="close_prices.csv", mime="text/csv")
        for t in data.columns:
            st.download_button(f"Download {t} CSV", data[[t]].to_csv().encode('utf-8'), file_name=f"{t}_close.csv", mime="text/csv")

with right:
    st.header("Net Worth & Snapshot")
    latest = {}
    total_holdings_value = 0.0
    unrealized_pl = 0.0
    rows = []
    for t in st.session_state.watchlist:
        price = None
        if t in data.columns and not data[t].dropna().empty:
            price = data[t].dropna().iloc[-1]
        else:
            price = fetch_latest_price(t)
        qty = float(st.session_state.holdings.get(t, {}).get("qty", 0.0))
        cost = float(st.session_state.holdings.get(t, {}).get("cost", 0.0))
        mv = qty * price if (price is not None) else None
        pl = (mv - (qty * cost)) if (mv is not None and qty and cost) else None
        if mv:
            total_holdings_value += mv
        if pl:
            unrealized_pl += pl
        rows.append({"Ticker": t, "Qty": qty, "Price": safe_round(price) if price is not None else "N/A", "Market Value": safe_round(mv) if mv is not None else "N/A", "Unrealized P/L": safe_round(pl) if pl is not None else "N/A"})

    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Ticker"))
    st.markdown("---")
    cash = float(st.session_state.cash)
    total_net_worth = total_holdings_value + cash
    st.metric("Holdings Value", f"${total_holdings_value:,.2f}")
    st.metric("Cash", f"${cash:,.2f}")
    st.metric("Total Net Worth", f"${total_net_worth:,.2f}")
    st.markdown(f"**Unrealized P/L:** ${unrealized_pl:,.2f}")
    st.markdown("---")
    alloc_df = pd.DataFrame([{"Ticker": r["Ticker"], "Value": (r["Market Value"] if isinstance(r["Market Value"], (int, float)) else 0.0)} for r in rows])
    if not alloc_df.empty and alloc_df["Value"].sum() > 0:
        fig_alloc = px.pie(alloc_df, names='Ticker', values='Value', title='Holdings Allocation', hole=0.35)
        st.plotly_chart(fig_alloc, use_container_width=True)
    else:
        st.info("No allocation data available.")
    st.markdown("---")
    if st.button("Refresh (clear caches)"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.experimental_rerun()
    st.download_button("Download Holdings CSV", pd.DataFrame(rows).set_index("Ticker").reset_index().to_csv(index=False).encode('utf-8'), file_name="holdings.csv", mime="text/csv")

st.caption("Interactive Investment Dashboard. Data from Yahoo Finance (yfinance).")
