# investment_dashboard_pro.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import io
import os

# Page config
st.set_page_config(page_title="Professional Investment Dashboard", layout="wide", initial_sidebar_state="expanded")

# ----------------------
# CSS and theming
# ----------------------
def load_css():
    css_path = "styles.css"
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # polished dark theme defaults
        base_css = """
        <style>
        :root{--bg:#0b1220;--panel:#071126;--accent:#06b6d4;--muted:#9fb7c9;--card:#071a2a}
        .stApp { background: linear-gradient(180deg, #061821 0%, #08131f 100%); color: #e6eef8; }
        [data-testid="stSidebar"] { background: var(--panel) !important; color: var(--muted) !important; }
        header { display:none; }
        .big-title { font-weight:700; font-size:22px; color: #e6eef8; }
        .subtle { color: #9fb7c9; font-size:12px }
        .kpi { background: linear-gradient(180deg, rgba(6,22,33,0.6), rgba(6,18,28,0.4)); padding:12px; border-radius:10px; }
        .stButton>button { background:var(--accent); color:#012226; border-radius:8px; padding:6px 10px; }
        .stDownloadButton>button { background:#0ea5a4; color:#012226; border-radius:8px; }
        footer { visibility:hidden; height:0px; }
        </style>
        """
        st.markdown(base_css, unsafe_allow_html=True)

load_css()

# ----------------------
# Compatibility data editor wrapper
# ----------------------
import io
def data_editor_wrapper(df: pd.DataFrame, key: str = None, num_rows="fixed", height=None):
    if hasattr(st, "data_editor"):
        return st.data_editor(df, num_rows=num_rows, key=key, height=height)
    if hasattr(st, "experimental_data_editor"):
        return st.experimental_data_editor(df, num_rows=num_rows, key=key, height=height)
    st.warning("Interactive table not available. Use CSV textarea below to edit holdings.")
    csv = df.to_csv(index=False)
    edited = st.text_area("Edit CSV (header included)", value=csv, height=220, key=(key or "csv") + "_fallback")
    try:
        return pd.read_csv(io.StringIO(edited))
    except Exception:
        st.error("CSV parsing failed. No changes applied.")
        return df

# ----------------------
# Data fetch with caching
# ----------------------
@st.cache_data(show_spinner=False)
def fetch_close_series(tickers, start, end, interval):
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(tickers, start=start, end=end, interval=interval, group_by='ticker', auto_adjust=True, threads=True)
        if raw.empty:
            return pd.DataFrame()
        # Normalize to a Close DataFrame with columns = tickers
        if isinstance(tickers, (list,tuple)) and len(tickers) == 1:
            t0 = tickers[0]
            col = "Close" if "Close" in raw.columns else ("Adj Close" if "Adj Close" in raw.columns else None)
            if col:
                return pd.DataFrame({t0: raw[col]})
            return pd.DataFrame()
        close_frames = {}
        if isinstance(raw.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    if 'Close' in raw[t].columns:
                        close_frames[t] = raw[t]['Close']
                    elif 'Adj Close' in raw[t].columns:
                        close_frames[t] = raw[t]['Adj Close']
                    else:
                        first_col = list(raw[t].columns)[0]
                        close_frames[t] = raw[t][first_col]
                except Exception:
                    continue
        else:
            # single-level columns containing tickers
            for c in raw.columns:
                close_frames[c] = raw[c]
        if not close_frames:
            return pd.DataFrame()
        close_df = pd.concat(close_frames, axis=1)
        close_df.index = pd.to_datetime(close_df.index)
        return close_df.sort_index()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_ohlcv(ticker, start, end, interval):
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        return df[['Open','High','Low','Close','Volume']]
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_latest(ticker):
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

# ----------------------
# Small analytics helpers
# ----------------------
def pct_format(x):
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "N/A"

def safe_round(x):
    try:
        return round(float(x), 2)
    except Exception:
        return "N/A"

def portfolio_metrics(close_df, holdings, cash):
    # Weighted returns and simple analytics
    metrics = {}
    if close_df.empty or not holdings:
        return metrics
    latest = close_df.iloc[-1]
    prev = close_df.iloc[-2] if len(close_df) > 1 else close_df.iloc[-1]
    # holdings: dict ticker-> {qty, cost}
    values = {}
    total = 0.0
    total_cost = 0.0
    weights = {}
    for t, h in holdings.items():
        qty = float(h.get("qty",0) or 0.0)
        price = latest.get(t, None) if t in latest.index or t in latest.index else (None if t not in close_df.columns else latest[t])
        if price is None or np.isnan(price):
            price = fetch_latest(t)
        mv = qty * price if price is not None else 0.0
        cost_val = qty * float(h.get("cost",0) or 0.0)
        values[t] = {"mv": mv, "price": price, "cost_val": cost_val}
        total += mv
        total_cost += cost_val
    total_net = total + float(cash or 0.0)
    # returns
    period_returns = {}
    for t in close_df.columns:
        try:
            r = (latest[t] / prev[t]) - 1 if prev[t] != 0 else 0
            period_returns[t] = r
        except Exception:
            period_returns[t] = 0.0
    # compute portfolio weighted return
    port_return = 0.0
    for t, v in values.items():
        if total and v["mv"]:
            w = v["mv"] / total
            port_return += w * period_returns.get(t, 0.0)
    metrics["values"] = values
    metrics["total_holdings"] = total
    metrics["total_net"] = total_net
    metrics["period_return"] = port_return
    metrics["total_unrealized"] = total - total_cost
    return metrics

# ----------------------
# Session state defaults
# ----------------------
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "MSFT", "GOOGL"]
if 'holdings' not in st.session_state:
    st.session_state.holdings = {t: {"qty": 0.0, "cost": 0.0} for t in st.session_state.watchlist}
if 'cash' not in st.session_state:
    st.session_state.cash = 0.0

# ----------------------
# Sidebar controls
# ----------------------
with st.sidebar:
    st.markdown("## Portfolio Controls")
    new_ticker = st.text_input("Add tickers (comma-separated)", value="")
    cols = st.columns([1,1,1])
    with cols[0]:
        add_qty = st.number_input("Quantity (single add)", value=0.0, format="%.4f", step=1.0)
    with cols[1]:
        add_cost = st.number_input("Cost basis per share", value=0.0, format="%.2f", step=0.01)
    with cols[2]:
        if st.button("Add"):
            entries = [x.strip().upper() for x in new_ticker.replace(";",",").split(",") if x.strip()]
            for e in entries:
                if e and e not in st.session_state.watchlist:
                    st.session_state.watchlist.append(e)
                    # apply qty/cost only if single entry provided
                    if len(entries) == 1:
                        st.session_state.holdings[e] = {"qty": float(add_qty), "cost": float(add_cost)}
                    else:
                        st.session_state.holdings.setdefault(e, {"qty": 0.0, "cost": 0.0})
            st.experimental_rerun()
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV (ticker,qty,cost)", type=["csv"])
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded, header=None)
            for _, r in df_up.iterrows():
                t = str(r[0]).strip().upper()
                if not t:
                    continue
                qty = float(r[1]) if len(r) > 1 and pd.notna(r[1]) else 0.0
                cost = float(r[2]) if len(r) > 2 and pd.notna(r[2]) else 0.0
                if t not in st.session_state.watchlist:
                    st.session_state.watchlist.append(t)
                st.session_state.holdings[t] = {"qty": qty, "cost": cost}
            st.success("CSV imported")
            st.experimental_rerun()
        except Exception:
            st.error("CSV parse failed: expected columns ticker, qty, cost")
    st.markdown("---")
    st.subheader("Date and interval")
    today = date.today()
    start_default = today - timedelta(days=365)
    start_date = st.date_input("Start date", value=start_default)
    end_date = st.date_input("End date", value=today)
    if start_date >= end_date:
        st.error("Start date must be before end date.")
    interval = st.selectbox("Interval", options=["1d","1wk","1mo"], index=0)
    st.markdown("---")
    st.subheader("Chart & Indicators")
    chart_mode = st.selectbox("Chart mode", ["Combined line","Normalized comparison","Single ticker candlestick"])
    normalize = st.checkbox("Normalize for comparison", value=False)
    sma = st.checkbox("SMA", value=False)
    sma_window = st.slider("SMA days", 5, 200, 20) if sma else None
    ema = st.checkbox("EMA", value=False)
    ema_window = st.slider("EMA days", 5, 200, 50) if ema else None
    show_volume = st.checkbox("Show volume", value=True)
    st.markdown("---")
    st.subheader("Cash & quick actions")
    st.session_state.cash = st.number_input("Cash balance", value=float(st.session_state.cash), format="%.2f", step=100.0)
    if st.button("Clear watchlist"):
        st.session_state.watchlist = []
        st.session_state.holdings = {}
        st.success("Cleared")
        st.experimental_rerun()

# ----------------------
# Main layout
# ----------------------
st.markdown("<div class='big-title'>Professional Investment Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtle'>Clean UI · Instant interactivity · Exportable insights</div>", unsafe_allow_html=True)
st.markdown("---")

# top KPIs
left_kpi, mid_kpi, right_kpi, ctrl_kpi = st.columns([2,2,2,2])
# fetch data
fetch_tickers = list(st.session_state.watchlist)
data = fetch_close_series(fetch_tickers, start_date.isoformat(), end_date.isoformat(), interval)

metrics = portfolio_metrics(data, st.session_state.holdings, st.session_state.cash)
total_net = metrics.get("total_net", 0.0)
period_return = metrics.get("period_return", 0.0)
total_holdings = metrics.get("total_holdings", 0.0)

with left_kpi:
    st.markdown("<div class='kpi'><strong>Total Net Worth</strong></div>", unsafe_allow_html=True)
    st.metric(label="", value=f"${total_net:,.2f}")

with mid_kpi:
    st.markdown("<div class='kpi'><strong>Holdings Value</strong></div>", unsafe_allow_html=True)
    st.metric(label="", value=f"${total_holdings:,.2f}")

with right_kpi:
    st.markdown("<div class='kpi'><strong>Period Return</strong></div>", unsafe_allow_html=True)
    st.metric(label="", value=pct_format(period_return))

with ctrl_kpi:
    st.markdown("<div class='kpi'><strong>Cash</strong></div>", unsafe_allow_html=True)
    st.metric(label="", value=f"${float(st.session_state.cash):,.2f}")

# two-column main
main_col, side_col = st.columns([3,1])

with main_col:
    st.subheader("Charts")
    if data.empty:
        st.warning("No price data for selected tickers or date range. Add tickers or expand date range.")
    else:
        to_plot = st.multiselect("Tickers to plot", options=list(data.columns), default=list(data.columns)[:4])
        if not to_plot:
            st.info("Select tickers to plot.")
        else:
            plot_df = data[to_plot].copy()
            if normalize:
                base = plot_df.iloc[0]
                plot_df = plot_df.divide(base).fillna(method='ffill')

            if chart_mode == "Combined line":
                fig = go.Figure()
                for t in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[t], name=t, mode='lines', hovertemplate='%{y:$,.2f}<extra>%{text}</extra>', text=[t]*len(plot_df)))
                if sma:
                    for t in plot_df.columns:
                        sma_series = data[t].rolling(window=sma_window, min_periods=1).mean()
                        fig.add_trace(go.Scatter(x=sma_series.index, y=sma_series, name=f"{t} SMA{str(sma_window)}", line=dict(dash='dash')))
                if ema:
                    for t in plot_df.columns:
                        ema_series = data[t].ewm(span=ema_window, adjust=False).mean()
                        fig.add_trace(go.Scatter(x=ema_series.index, y=ema_series, name=f"{t} EMA{str(ema_window)}", line=dict(dash='dot')))
                fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(t=40))
                fig.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig, use_container_width=True, theme="plotly_dark")

            elif chart_mode == "Normalized comparison":
                norm = plot_df.divide(plot_df.iloc[0]).fillna(method='ffill') if not normalize else plot_df
                fig = px.line(norm, x=norm.index, y=norm.columns, labels={"value":"Normalized", "index":"Date"})
                fig.update_layout(legend=dict(orientation="h"), margin=dict(t=40))
                st.plotly_chart(fig, use_container_width=True, theme="plotly_dark")

            else:  # single ticker candlestick
                active = st.selectbox("Select ticker for OHLC", options=list(data.columns))
                raw = fetch_ohlcv(active, start_date.isoformat(), end_date.isoformat(), interval)
                if raw.empty:
                    st.error(f"No OHLC data for {active}")
                else:
                    # candlestick with SMA/EMA and volume on secondary axis
                    fig2 = go.Figure()
                    fig2.add_trace(go.Candlestick(x=raw.index, open=raw['Open'], high=raw['High'], low=raw['Low'], close=raw['Close'], name=active))
                    if sma:
                        sma_s = raw['Close'].rolling(window=sma_window, min_periods=1).mean()
                        fig2.add_trace(go.Scatter(x=sma_s.index, y=sma_s, name=f"SMA{str(sma_window)}", line=dict(color='orange')))
                    if ema:
                        ema_s = raw['Close'].ewm(span=ema_window, adjust=False).mean()
                        fig2.add_trace(go.Scatter(x=ema_s.index, y=ema_s, name=f"EMA{str(ema_window)}", line=dict(color='purple')))
                    if show_volume and 'Volume' in raw.columns:
                        fig2.add_trace(go.Bar(x=raw.index, y=raw['Volume'], name='Volume', marker=dict(color='lightgray'), yaxis='y2'))
                        fig2.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Volume'))
                    fig2.update_layout(xaxis_rangeslider_visible=True, height=600, margin=dict(t=40))
                    st.plotly_chart(fig2, use_container_width=True, theme="plotly_dark")

    st.markdown("---")
    st.subheader("Holdings and Data Export")
    holdings_df = pd.DataFrame([{"Ticker": t, "Qty": st.session_state.holdings.get(t,{}).get("qty",0.0), "Cost": st.session_state.holdings.get(t,{}).get("cost",0.0)} for t in st.session_state.watchlist])
    if holdings_df.empty:
        st.info("No holdings yet")
    edited = data_editor_wrapper(holdings_df, key="holdings_editor", num_rows="dynamic", height=220)
    # sync back
    try:
        for _, r in edited.iterrows():
            t = str(r.get("Ticker") or "").strip().upper()
            if not t:
                continue
            qty = float(r.get("Qty") or 0.0)
            cost = float(r.get("Cost") or 0.0)
            if t not in st.session_state.watchlist:
                st.session_state.watchlist.append(t)
            st.session_state.holdings[t] = {"qty": qty, "cost": cost}
    except Exception:
        st.error("Failed to update holdings")

    # export
    if not data.empty:
        st.download_button("Download close prices CSV", data.to_csv().encode('utf-8'), "close_prices.csv", mime="text/csv")
    st.download_button("Download holdings CSV", pd.DataFrame(st.session_state.holdings).T.reset_index().rename(columns={"index":"Ticker"}).to_csv(index=False).encode('utf-8'), "holdings.csv", mime="text/csv")

with side_col:
    st.header("Net Worth Snapshot")
    rows = []
    total_val = 0.0
    total_cost = 0.0
    for t in st.session_state.watchlist:
        qty = float(st.session_state.holdings.get(t,{}).get("qty",0.0) or 0.0)
        cost = float(st.session_state.holdings.get(t,{}).get("cost",0.0) or 0.0)
        price = None
        if t in data.columns and not data[t].dropna().empty:
            price = data[t].dropna().iloc[-1]
        else:
            price = fetch_latest(t)
        mv = qty * price if price is not None else 0.0
        pl = mv - qty*cost if qty and cost else 0.0
        total_val += mv
        total_cost += qty*cost
        rows.append({"Ticker":t, "Qty": qty, "Price": safe_round(price) if price is not None else "N/A", "Market Value": safe_round(mv), "Unrealized P/L": safe_round(pl)})
    df_rows = pd.DataFrame(rows).set_index("Ticker") if rows else pd.DataFrame()
    if not df_rows.empty:
        st.table(df_rows)
    st.markdown("---")
    cash = float(st.session_state.cash)
    total_net = total_val + cash
    st.metric("Holdings", f"${total_val:,.2f}")
    st.metric("Cash", f"${cash:,.2f}")
    st.metric("Total Net Worth", f"${total_net:,.2f}")
    st.markdown(f"**Total Unrealized P/L:** ${total_val - total_cost:,.2f}")
    st.markdown("---")
    # allocation chart
    alloc = pd.DataFrame([{"Ticker": r["Ticker"], "Value": r["Market Value"]} for r in rows])
    if not alloc.empty and alloc["Value"].sum() > 0:
        fig_alloc = px.pie(alloc, names='Ticker', values='Value', title='Allocation', hole=0.35)
        st.plotly_chart(fig_alloc, use_container_width=True, theme="plotly_dark")
    st.markdown("---")
    if st.button("Refresh data"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.experimental_rerun()

st.caption("Design follows dashboard best practices for clarity, context, and interactivity.")
