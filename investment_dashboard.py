# investment_dashboard_compat.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import io

st.set_page_config(page_title="Investment Dashboard (Compat)", layout="wide")

# -----------------------
# Compatibility helpers
# -----------------------
def data_editor_wrapper(df: pd.DataFrame, key: str = None, num_rows="fixed"):
    """
    Try st.data_editor, then st.experimental_data_editor, else fallback to editable CSV text area.
    Returns the edited dataframe (or original if no edits parsed).
    """
    if hasattr(st, "data_editor"):
        return st.data_editor(df, num_rows=num_rows, key=key)
    if hasattr(st, "experimental_data_editor"):
        return st.experimental_data_editor(df, num_rows=num_rows, key=key)
    # Fallback: show CSV in text area for manual edits
    st.warning("Interactive data editor not available in this Streamlit version. Use the CSV editor below to edit holdings.")
    csv = df.to_csv(index=False)
    edited = st.text_area("Edit CSV (rows as CSV, header included)", value=csv, height=220, key=(key or "csv")+ "_fallback")
    try:
        new_df = pd.read_csv(io.StringIO(edited))
        return new_df
    except Exception:
        st.error("CSV parsing failed. No changes applied.")
        return df

# -----------------------
# Data fetchers (cached)
# -----------------------
@st.cache_data(show_spinner=False)
def fetch_history(tickers, start, end, interval):
    if not tickers:
        return pd.DataFrame()
    try:
        df = yf.download(tickers, start=start, end=end, interval=interval, group_by='ticker', auto_adjust=True, threads=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(tickers, (str,)) or (hasattr(tickers, "__len__") and len(tickers) == 1):
            # single ticker: try Close or Adj Close
            col = "Close" if "Close" in df.columns else "Adj Close" if "Adj Close" in df.columns else None
            if col:
                name = tickers if isinstance(tickers, str) else tickers[0]
                return pd.DataFrame({name: df[col]})
            return pd.DataFrame()
        # multiple tickers -> multiindex columns
        close_frames = {}
        if isinstance(df.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    if 'Close' in df[t].columns:
                        close_frames[t] = df[t]['Close']
                    elif 'Adj Close' in df[t].columns:
                        close_frames[t] = df[t]['Adj Close']
                    else:
                        cols = list(df[t].columns)
                        if cols:
                            close_frames[t] = df[t][cols[0]]
                except Exception:
                    continue
        else:
            # single-level columns (rare)
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
    st.header("Watchlist Controls")
    new_ticker = st.text_input("Add ticker (comma-separated)", value="")
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
    uploaded = st.file_uploader("Upload CSV (ticker,qty,cost) - optional header", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            # try to detect headers; accept both formats
            if {'ticker','qty','cost'}.issubset(set(map(str.lower, df_up.columns))):
                df_up.columns = [c.lower() for c in df_up.columns]
                for _, row in df_up.iterrows():
                    t = str(row['ticker']).strip().upper()
                    qty = float(row.get('qty', 0.0) or 0.0)
                    cost = float(row.get('cost', 0.0) or 0.0)
                    if t and t not in st.session_state.watchlist:
                        st.session_state.watchlist.append(t)
                    st.session_state.holdings[t] = {"qty": qty, "cost": cost}
            else:
                # fallback: assume columns order ticker, qty, cost
                for _, row in df_up.iterrows():
                    t = str(row.iloc[0]).strip().upper()
                    qty = float(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else 0.0
                    cost = float(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else 0.0
                    if t and t not in st.session_state.watchlist:
                        st.session_state.watchlist.append(t)
                    st.session_state.holdings[t] = {"qty": qty, "cost": cost}
            st.success("CSV processed and added.")
        except Exception:
            st.error("Failed to parse CSV. Expected columns: ticker, qty, cost or plain rows.")

    st.markdown("---")
    st.subheader("Date and interval")
    today = date.today()
    start_default = today - timedelta(days=365)
    start_date = st.date_input("Start date", start_default)
    end_date = st.date_input("End date", today)
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

# -----------------------
# Main layout
# -----------------------
st.title("Robust Interactive Investment Dashboard")
left, right = st.columns([3, 1])

with left:
    st.header("Holdings Editor")
    holdings_df = pd.DataFrame([
        {"Ticker": t, "Qty": st.session_state.holdings.get(t, {}).get("qty", 0.0),
         "Cost": st.session_state.holdings.get(t, {}).get("cost", 0.0)}
        for t in st.session_state.watchlist
    ])
    # Use compatibility wrapper instead of direct experimental_data_editor
    edited = data_editor_wrapper(holdings_df, key="holdings_editor", num_rows="dynamic")
    # Sync edited back into session_state
    try:
        for _, row in edited.iterrows():
            t = str(row["Ticker"]).strip().upper()
            qty = float(row.get("Qty") or 0.0)
            cost = float(row.get("Cost") or 0.0)
            if t not in st.session_state.watchlist:
                st.session_state.watchlist.append(t)
            st.session_state.holdings[t] = {"qty": qty, "cost": cost}
    except Exception:
        st.error("Failed to process edited holdings. No changes applied.")

    st.markdown("---")
    fetch_tickers = list(st.session_state.watchlist)
    data = fetch_history(fetch_tickers, start_date.isoformat(), end_date.isoformat(), interval)

    if data.empty:
        st.error("No price data for the selected tickers/timeframe.")
    else:
        st.subheader("Charts")
        to_plot = st.multiselect("Select tickers to plot", options=list(data.columns), default=list(data.columns)[:3])
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

            else:  # Single ticker mode
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
                        fig2.add_trace(go.Scatter(x=sma.index, y=sma, mode='lines', name=f"SMA{str(sma_window)}", line=dict(color='orange')))
                    if show_ema:
                        ema = raw['Close'].ewm(span=ema_window, adjust=False).mean()
                        fig2.add_trace(go.Scatter(x=ema.index, y=ema, mode='lines', name=f"EMA{str(ema_window)}", line=dict(color='purple')))
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
        st.download_button("Download Close CSV", data.to_csv().encode('utf-8'), "close_prices.csv", "text/csv")

with right:
    st.header("Net Worth Snapshot")
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
        qty = st.session_state.holdings.get(t, {}).get("qty", 0.0)
        cost = st.session_state.holdings.get(t, {}).get("cost", 0.0)
        mv = qty * price if price is not None else None
        pl = (mv - qty * cost) if (mv is not None and qty and cost) else None
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
        fig_alloc = px.pie(alloc_df, names='Ticker', values='Value', title='Allocation', hole=0.35)
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

st.caption("Compatibility version of dashboard. If you still see data-editor errors, upgrade Streamlit or use the CSV editor fallback.")
