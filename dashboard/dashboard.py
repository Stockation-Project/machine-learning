# =============================================================================
# IDX80 STOCK DASHBOARD — Production-Ready Single File
# Data Source : REST API (https://dashboard-psu155-production.up.railway.app)
# Pages       : Market Overview · Stock Exploration
# =============================================================================

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE CONFIG  (must be first st call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDX80 Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME / COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":           "#0A0E1A",
    "bg_card":      "#0F1629",
    "bg_chart":     "#0D1220",
    "border":       "#1E2D4A",
    "accent":       "#00D4FF",
    "accent2":      "#7B61FF",
    "accent3":      "#FF6B35",
    "green":        "#00E676",
    "red":          "#FF3D57",
    "yellow":       "#FFD600",
    "text_primary": "#E8EDF5",
    "text_muted":   "#6B7FA3",
    "text_dim":     "#3A4A6B",
    "gridline":     "#152035",
}

SECTOR_PALETTE = [
    "#00D4FF", "#7B61FF", "#FF6B35", "#00E676", "#FFD600",
    "#FF3D57", "#00BFA5", "#FF80AB", "#82B1FF", "#CCFF90",
    "#FFD180", "#EA80FC", "#A7FFEB", "#FF9E80", "#B9F6CA",
]

# ─── Plotly template ──────────────────────────────────────────────────────────
_tpl = go.layout.Template()
_tpl.layout = go.Layout(
    paper_bgcolor=C["bg_chart"],
    plot_bgcolor=C["bg_chart"],
    font=dict(family="JetBrains Mono, monospace", color=C["text_primary"], size=12),
    title=dict(
        font=dict(size=14, color=C["text_primary"], family="JetBrains Mono, monospace"),
        x=0.02, xanchor="left",
    ),
    xaxis=dict(
        gridcolor=C["gridline"], zerolinecolor=C["border"],
        tickcolor=C["text_muted"], linecolor=C["border"],
        tickfont=dict(size=10, color=C["text_muted"]),
    ),
    yaxis=dict(
        gridcolor=C["gridline"], zerolinecolor=C["border"],
        tickcolor=C["text_muted"], linecolor=C["border"],
        tickfont=dict(size=10, color=C["text_muted"]),
    ),
    legend=dict(
        bgcolor="rgba(15,22,41,0.85)", bordercolor=C["border"],
        borderwidth=1, font=dict(size=10, color=C["text_muted"]),
    ),
    margin=dict(l=50, r=20, t=44, b=40),
    hoverlabel=dict(
        bgcolor=C["bg_card"], bordercolor=C["accent"],
        font=dict(size=11, color=C["text_primary"], family="JetBrains Mono, monospace"),
    ),
    colorway=SECTOR_PALETTE,
)
pio.templates["idx80"] = _tpl
pio.templates.default = "idx80"


def _fig(height=380):
    """Shorthand: return kwargs for apply_default_layout."""
    return dict(height=height, margin=dict(l=50, r=20, t=44, b=40))


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Root reset ── */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {C["bg"]};
    color: {C["text_primary"]};
}}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{visibility: hidden;}}
.block-container {{padding: 0 2rem 2rem 2rem; max-width: 100%;}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background-color: {C["bg_card"]};
    border-right: 1px solid {C["border"]};
}}
section[data-testid="stSidebar"] * {{color: {C["text_primary"]} !important;}}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div {{
    background-color: {C["bg"]} !important;
    border-color: {C["border"]} !important;
}}
section[data-testid="stSidebar"] .stDateInput input {{
    background-color: {C["bg"]} !important;
    border-color: {C["border"]} !important;
    color: {C["text_primary"]} !important;
}}

/* ── Top header bar ── */
.top-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 0 10px 0;
    border-bottom: 1px solid {C["border"]};
    margin-bottom: 24px;
}}
.top-bar-left {{
    display: flex;
    align-items: center;
    gap: 14px;
}}
.top-bar-logo {{
    font-size: 28px;
    line-height: 1;
}}
.top-bar-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: {C["text_primary"]};
    letter-spacing: -0.5px;
}}
.top-bar-sub {{
    font-size: 12px;
    color: {C["text_muted"]};
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
}}
.last-updated {{
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: {C["text_muted"]};
    line-height: 1.6;
}}
.last-updated .lu-label {{
    color: {C["text_dim"]};
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 9px;
}}
.last-updated .lu-date {{
    color: {C["accent"]};
    font-weight: 600;
    font-size: 13px;
}}

/* ── KPI cards ── */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 20px;
}}
.kpi-card {{
    background: {C["bg_card"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}}
.kpi-card:hover {{ border-color: {C["accent"]}44; }}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {C["accent"]}, {C["accent2"]});
    opacity: 0.6;
}}
.kpi-icon {{
    font-size: 20px;
    margin-bottom: 6px;
    opacity: 0.8;
}}
.kpi-label {{
    font-size: 10px;
    color: {C["text_muted"]};
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 6px;
}}
.kpi-value {{
    font-size: 22px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: {C["text_primary"]};
    line-height: 1.1;
}}
.kpi-delta {{
    font-size: 11px;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}}
.kpi-pos {{ color: {C["green"]}; }}
.kpi-neg {{ color: {C["red"]}; }}

/* ── Section labels ── */
.section-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: {C["text_dim"]};
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 24px 0 10px 0;
    padding-left: 2px;
    border-left: 3px solid {C["accent"]};
    padding-left: 8px;
}}

/* ── Divider ── */
.divider {{
    border: none;
    border-top: 1px solid {C["border"]};
    margin: 20px 0;
}}

/* ── Stock info card ── */
.stock-info-card {{
    background: {C["bg_card"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
}}
.stock-info-ticker {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: {C["accent"]};
}}
.stock-info-name {{
    font-size: 15px;
    color: {C["text_muted"]};
    margin-top: 2px;
    margin-bottom: 12px;
}}
.stock-badge {{
    display: inline-block;
    background: {C["bg"]};
    border: 1px solid {C["border"]};
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 11px;
    color: {C["text_muted"]};
    margin-right: 6px;
    font-family: 'JetBrains Mono', monospace;
}}
.status-gain      {{ color: {C["green"]}; font-weight: 700; }}
.status-loss      {{ color: {C["red"]}; font-weight: 700; }}
.status-neutral   {{ color: {C["yellow"]}; font-weight: 700; }}

/* ── Streamlit metric override ── */
[data-testid="metric-container"] {{
    background: {C["bg_card"]};
    border: 1px solid {C["border"]};
    border-radius: 8px;
    padding: 14px 18px;
}}
[data-testid="metric-container"] label {{
    color: {C["text_muted"]} !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {C["text_primary"]} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 20px !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricDelta"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {C["bg_card"]};
    border-bottom: 1px solid {C["border"]};
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: {C["text_muted"]};
    background-color: transparent;
    border: none;
    padding: 8px 16px;
}}
.stTabs [aria-selected="true"] {{
    color: {C["accent"]} !important;
    border-bottom: 2px solid {C["accent"]} !important;
    background-color: transparent !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {C["border"]};
    border-radius: 8px;
    overflow: hidden;
}}

/* ── Spinner ── */
.stSpinner > div {{color: {C["accent"]} !important;}}

/* ── Sidebar nav ── */
.nav-item {{
    display: block;
    padding: 10px 14px;
    border-radius: 8px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    margin-bottom: 4px;
    transition: background 0.15s;
    text-decoration: none;
}}
.nav-active {{
    background: {C["accent"]}22;
    color: {C["accent"]} !important;
    border-left: 3px solid {C["accent"]};
}}
.nav-inactive {{
    color: {C["text_muted"]} !important;
    border-left: 3px solid transparent;
}}
.nav-inactive:hover {{
    background: {C["bg"]};
    color: {C["text_primary"]} !important;
}}

/* ── Selectbox & input ── */
div[data-baseweb="select"] > div {{
    background-color: {C["bg"]} !important;
    border-color: {C["border"]} !important;
    color: {C["text_primary"]} !important;
}}
div[data-baseweb="select"] span {{color: {C["text_primary"]} !important;}}
div[data-baseweb="popover"] {{background-color: {C["bg_card"]} !important;}}
li[role="option"] {{
    background-color: {C["bg_card"]} !important;
    color: {C["text_primary"]} !important;
}}
li[role="option"]:hover {{
    background-color: {C["bg"]} !important;
    color: {C["accent"]} !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached)
# ─────────────────────────────────────────────────────────────────────────────
BASE_URL = "https://dashboard-psu155-production.up.railway.app"

@st.cache_data(ttl=300, show_spinner=False)
def load_prices() -> pd.DataFrame:
    resp = requests.get(f"{BASE_URL}/stocks", timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_metadata() -> pd.DataFrame:
    resp = requests.get(f"{BASE_URL}/metadata", timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


@st.cache_data(ttl=300, show_spinner=False)
def load_features() -> pd.DataFrame:
    resp = requests.get(f"{BASE_URL}/features", timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(features: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Add rolling_volatility, cum_max, drawdown to features df."""
    df = features.copy()

    df["rolling_volatility"] = (
        df.groupby("ticker")["daily_return"]
        .transform(lambda x: x.rolling(30).std() * np.sqrt(252))
    )

    price_cum_max = (
        prices.sort_values(["ticker", "date"])
        .groupby("ticker")["close"]
        .cummax()
        .values
    )
    prices_sorted = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices_sorted["cum_max"] = price_cum_max
    prices_sorted["drawdown"] = (prices_sorted["close"] - prices_sorted["cum_max"]) / prices_sorted["cum_max"]

    df = df.merge(
        prices_sorted[["ticker", "date", "cum_max", "drawdown"]],
        on=["ticker", "date"],
        how="left",
    )
    return df


def update_market_cap(metadata: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Recalculate market cap using latest close price × shares outstanding."""
    latest_close = (
        prices.sort_values("date")
        .groupby("ticker")["close"]
        .last()
        .reset_index()
        .rename(columns={"close": "latest_close"})
    )
    latest_close.columns = ["ticker", "latest_close"]
    meta = metadata.merge(latest_close, on="ticker", how="left")
    meta["market_cap"] = meta["shares_outstanding"] * meta["latest_close"]
    return meta


def build_plot_df(metadata: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Merge metadata with aggregated avg_return + volatility per ticker."""
    avg_return = (
        features.groupby("ticker")["daily_return"]
        .mean()
        .reset_index()
    )
    avg_return.columns = ["ticker", "avg_return"]

    volatility = (
        features.groupby("ticker")["daily_return"]
        .std()
        .reset_index()
    )
    volatility.columns = ["ticker", "volatility"]

    return metadata.merge(avg_return, on="ticker").merge(volatility, on="ticker")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_large(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    v = float(v)
    if abs(v) >= 1e12:
        return f"{v/1e12:.2f} T"
    if abs(v) >= 1e9:
        return f"{v/1e9:.2f} B"
    if abs(v) >= 1e6:
        return f"{v/1e6:.2f} M"
    return f"{v:,.0f}"


def pct(v, d=3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{float(v)*100:.{d}f}%"


def status_label(r: float) -> tuple[str, str]:
    """Return (label, css_class)."""
    if r > 0.02:
        return "⬆ Strong Gain", "status-gain"
    elif r > 0:
        return "▲ Gain", "status-gain"
    elif r < -0.02:
        return "⬇ Strong Loss", "status-loss"
    elif r < 0:
        return "▼ Loss", "status-loss"
    return "● Neutral", "status-neutral"


def add_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cum_return, rolling_vol, drawdown for single stock."""
    df = df.sort_values("date").copy()
    df["cum_return"] = (1 + df["daily_return"].fillna(0)).cumprod()
    df["rolling_vol"] = df["daily_return"].rolling(20).std() * np.sqrt(252)
    df["cum_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_return"] / df["cum_max"] - 1
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS — shared layout
# ─────────────────────────────────────────────────────────────────────────────
CHART_CFG = {"displayModeBar": False, "scrollZoom": False}


def _lay(fig: go.Figure, h: int = 380, **kw) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=50, r=20, t=44, b=40), **kw)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MARKET OVERVIEW CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def chart_scatter(df: pd.DataFrame) -> go.Figure:
    plot = df.dropna(subset=["volatility", "avg_return", "market_cap"]).copy()
    plot["vol_pct"]  = plot["volatility"] * 100
    plot["ret_pct"]  = plot["avg_return"]  * 100
    plot["mcap_B"]   = plot["market_cap"]  / 1e9

    fig = px.scatter(
        plot,
        x="vol_pct",
        y="ret_pct",
        size="market_cap",
        color="sector",
        hover_name="ticker",
        title="Risk vs Return — Bubble = Market Cap",
        size_max=60,
        color_discrete_sequence=SECTOR_PALETTE,
        custom_data=["ticker", "company_name", "vol_pct", "ret_pct", "mcap_B", "sector"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
            "Sector: %{customdata[5]}<br>"
            "Volatility: %{customdata[2]:.3f}%<br>"
            "Avg Return: %{customdata[3]:.4f}%<br>"
            "Market Cap: %{customdata[4]:.2f}B IDR<extra></extra>"
        ),
        marker=dict(opacity=0.75, line=dict(width=0.5, color=C["bg"])),
    )
    fig.add_hline(y=0, line_dash="dot", line_color=C["text_dim"], line_width=1)
    fig.update_layout(
        xaxis_title="Volatility (Daily Std %)",
        yaxis_title="Avg Daily Return (%)",
        legend=dict(orientation="v", x=1.01, y=1, font=dict(size=9)),
    )
    return _lay(fig, 480)


def chart_sector_bar(df: pd.DataFrame) -> go.Figure:
    sec = (
        df.groupby("sector")
        .agg(avg_return=("avg_return", "mean"), market_cap=("market_cap", "sum"))
        .reset_index()
        .sort_values("avg_return", ascending=True)
    )
    colors = [C["green"] if v >= 0 else C["red"] for v in sec["avg_return"]]

    fig = go.Figure(go.Bar(
        x=sec["avg_return"] * 100,
        y=sec["sector"],
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v*100:.4f}%" for v in sec["avg_return"]],
        textposition="outside",
        textfont=dict(size=9, color=C["text_muted"]),
        hovertemplate="<b>%{y}</b><br>Avg Return: %{x:.4f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Sector Performance — Avg Daily Return",
        xaxis_title="Avg Daily Return (%)",
        yaxis_title=None,
        xaxis=dict(ticksuffix="%"),
    )
    return _lay(fig, 420)


def chart_treemap(df: pd.DataFrame) -> go.Figure:
    plot = df.dropna(subset=["market_cap", "sector"]).copy()
    plot["ret_pct"] = plot["avg_return"] * 100
    plot["mcap_B"]  = plot["market_cap"] / 1e9

    fig = px.treemap(
        plot,
        path=["sector", "ticker"],
        values="market_cap",
        color="ret_pct",
        hover_data={"ret_pct": ":.4f", "mcap_B": ":.2f"},
        color_continuous_scale=[[0, C["red"]], [0.5, "#152035"], [1, C["green"]]],
        color_continuous_midpoint=0,
        title="IDX80 Market Treemap — Size: Market Cap · Color: Avg Return",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Avg Return: %{customdata[0]:.4f}%<br>"
            "Market Cap: %{customdata[1]:.2f}B IDR<extra></extra>"
        ),
        marker_line_color=C["bg"],
        marker_line_width=1.5,
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Return %",
            tickfont=dict(size=9, color=C["text_muted"]),
            len=0.5,
        ),
    )
    return _lay(fig, 480)


def chart_return_histogram(features: pd.DataFrame) -> go.Figure:
    returns = features["daily_return"].dropna() * 100

    fig = go.Figure(go.Histogram(
        x=returns,
        nbinsx=100,
        marker_color=C["accent"],
        marker_line_width=0.3,
        marker_line_color=C["bg"],
        opacity=0.85,
        hovertemplate="Return: %{x:.3f}%<br>Count: %{y}<extra></extra>",
        name="Daily Return",
    ))
    mean_v = returns.mean()
    fig.add_vline(x=mean_v, line_dash="dash", line_color=C["yellow"], line_width=1.5,
                  annotation_text=f"μ={mean_v:.4f}%",
                  annotation_font_color=C["yellow"], annotation_font_size=10)
    fig.update_layout(
        title="Distribution of Daily Returns — All IDX80",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
    )
    return _lay(fig, 340)


def chart_market_vol(features: pd.DataFrame) -> go.Figure:
    mv = (
        features.groupby("date")["daily_return"]
        .std()
        .reset_index()
    )
    mv.columns = ["date", "market_volatility"]

    fig = go.Figure(go.Scatter(
        x=mv["date"],
        y=mv["market_volatility"] * 100,
        mode="lines",
        line=dict(color=C["accent2"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(123,97,255,0.10)",
        hovertemplate="%{x|%d %b %Y}<br>Market Vol: %{y:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Market Volatility Through Time",
        xaxis_title=None,
        yaxis_title="Cross-Sectional Std of Returns (%)",
        yaxis_ticksuffix="%",
    )
    return _lay(fig, 320)


def chart_top_volatility(df: pd.DataFrame) -> go.Figure:
    top = df.nlargest(10, "volatility").copy()
    top["vol_pct"] = top["volatility"] * 100

    fig = go.Figure(go.Bar(
        x=top["ticker"],
        y=top["vol_pct"],
        marker_color=C["accent3"],
        marker_line_width=0,
        text=[f"{v:.2f}%" for v in top["vol_pct"]],
        textposition="outside",
        textfont=dict(size=9, color=C["text_muted"]),
        hovertemplate="<b>%{x}</b><br>Volatility: %{y:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Top 10 Most Volatile Stocks",
        xaxis_title=None,
        yaxis_title="Volatility (Daily Std %)",
        yaxis_ticksuffix="%",
    )
    return _lay(fig, 320)


def chart_top_mcap(df: pd.DataFrame) -> go.Figure:
    top = df.nlargest(10, "market_cap").copy()
    top["mcap_T"] = top["market_cap"] / 1e12

    fig = go.Figure(go.Bar(
        x=top["ticker"],
        y=top["mcap_T"],
        marker_color=C["accent"],
        marker_line_width=0,
        text=[f"{v:.2f}T" for v in top["mcap_T"]],
        textposition="outside",
        textfont=dict(size=9, color=C["text_muted"]),
        hovertemplate="<b>%{x}</b><br>Market Cap: %{y:.3f}T IDR<extra></extra>",
    ))
    fig.update_layout(
        title="Top 10 Largest Market Cap Stocks",
        xaxis_title=None,
        yaxis_title="Market Cap (Trillion IDR)",
    )
    return _lay(fig, 320)


def chart_correlation_heatmap(features: pd.DataFrame, plot_df: pd.DataFrame) -> go.Figure:
    top_tickers = (
        plot_df.sort_values("market_cap", ascending=False)["ticker"]
        .head(15)
        .tolist()
    )
    pivot = features.pivot(index="date", columns="ticker", values="daily_return")
    available = [t for t in top_tickers if t in pivot.columns]
    corr = pivot[available].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        title="Return Correlation Heatmap — Top 15 by Market Cap",
        color_continuous_scale=[[0, C["red"]], [0.5, C["bg_chart"]], [1, C["green"]]],
        zmin=-1, zmax=1,
    )
    fig.update_traces(
        hovertemplate="%{x} × %{y}<br>ρ = %{z:.3f}<extra></extra>",
        textfont=dict(size=8),
    )
    fig.update_layout(
        xaxis=dict(tickfont=dict(size=9), tickangle=45),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
        margin=dict(l=80, r=20, t=50, b=90),
        coloraxis_colorbar=dict(
            title="ρ", len=0.8,
            tickfont=dict(size=9, color=C["text_muted"]),
        ),
    )
    return _lay(fig, 500)


# ─────────────────────────────────────────────────────────────────────────────
# STOCK EXPLORATION CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def chart_price(df_price: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df_price["date"],
        y=df_price["close"],
        mode="lines",
        name="Close",
        line=dict(color=C["accent"], width=1.8),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.07)",
        hovertemplate="%{x|%d %b %Y}<br>Close: Rp %{y:,.0f}<extra></extra>",
    ))
    ma20 = df_price["close"].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=df_price["date"], y=ma20, name="MA20",
        line=dict(color=C["yellow"], width=1, dash="dot"),
        hovertemplate="MA20: Rp %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(title=f"{ticker} — Price Chart", yaxis_title="Close Price (IDR)")
    return _lay(fig, 340)


def chart_daily_return(df_feat: pd.DataFrame, ticker: str) -> go.Figure:
    returns = df_feat["daily_return"] * 100
    colors = [C["green"] if v >= 0 else C["red"] for v in returns]
    fig = go.Figure(go.Bar(
        x=df_feat["date"], y=returns,
        marker_color=colors, marker_line_width=0,
        name="Daily Return",
        hovertemplate="%{x|%d %b %Y}<br>Return: %{y:.3f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=C["text_dim"], line_width=1)
    fig.update_layout(
        title=f"{ticker} — Daily Return",
        yaxis_title="Return (%)", yaxis_ticksuffix="%",
    )
    return _lay(fig, 300)


def chart_cumulative_return(df_feat: pd.DataFrame, ticker: str) -> go.Figure:
    cum = (df_feat["cum_return"] - 1) * 100
    last = cum.iloc[-1] if not cum.empty else 0
    clr = C["green"] if last >= 0 else C["red"]

    fig = go.Figure(go.Scatter(
        x=df_feat["date"], y=cum,
        mode="lines", name="Cum. Return",
        line=dict(color=clr, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.12)",
        hovertemplate="%{x|%d %b %Y}<br>Cum. Return: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=C["text_dim"], line_width=1)
    fig.update_layout(
        title=f"{ticker} — Cumulative Return",
        yaxis_title="Cumulative Return (%)", yaxis_ticksuffix="%",
    )
    return _lay(fig, 300)


def chart_rolling_vol(df_feat: pd.DataFrame, ticker: str) -> go.Figure:
    vol = df_feat["rolling_vol"] * 100
    fig = go.Figure(go.Scatter(
        x=df_feat["date"], y=vol,
        mode="lines", name="Rolling Vol (20d)",
        line=dict(color=C["accent2"], width=1.8),
        fill="tozeroy",
        fillcolor="rgba(123,97,255,0.10)",
        hovertemplate="%{x|%d %b %Y}<br>Vol (Ann.): %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=f"{ticker} — Rolling Volatility (20d, Annualised)",
        yaxis_title="Annualised Vol (%)", yaxis_ticksuffix="%",
    )
    return _lay(fig, 300)


def chart_drawdown(df_feat: pd.DataFrame, ticker: str) -> go.Figure:
    dd = df_feat["drawdown"] * 100
    fig = go.Figure(go.Scatter(
        x=df_feat["date"], y=dd,
        mode="lines", name="Drawdown",
        line=dict(color=C["red"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255,61,87,0.13)",
        hovertemplate="%{x|%d %b %Y}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=C["text_dim"], line_width=1)
    fig.update_layout(
        title=f"{ticker} — Drawdown",
        yaxis_title="Drawdown (%)", yaxis_ticksuffix="%",
    )
    return _lay(fig, 280)


def chart_stock_hist(df_feat: pd.DataFrame, ticker: str) -> go.Figure:
    returns = df_feat["daily_return"].dropna() * 100
    mean_v, std_v = returns.mean(), returns.std()

    fig = go.Figure(go.Histogram(
        x=returns, nbinsx=80,
        marker_color=C["accent"], marker_line_width=0.3,
        marker_line_color=C["bg"], opacity=0.85,
        hovertemplate="Return: %{x:.3f}%<br>Count: %{y}<extra></extra>",
    ))
    for mult, col, label in [
        (0,  C["accent"],  f"μ={mean_v:.4f}%"),
        (-1, C["yellow"],  "-1σ"),  (1, C["yellow"], "+1σ"),
        (-2, C["red"],     "-2σ"),  (2, C["red"],    "+2σ"),
    ]:
        fig.add_vline(x=mean_v + mult * std_v, line_dash="dot",
                      line_color=col, line_width=1,
                      annotation_text=label,
                      annotation_font_size=9, annotation_font_color=col)
    fig.update_layout(
        title=f"{ticker} — Return Distribution",
        xaxis_title="Daily Return (%)", yaxis_title="Frequency",
    )
    return _lay(fig, 300)


# ─────────────────────────────────────────────────────────────────────────────
# TOP-BAR RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def render_topbar(last_date):
    if last_date is not None:
        date_str = pd.Timestamp(last_date).strftime("%d %b %Y")
        time_str = datetime.now().strftime("%H:%M WIB")
    else:
        date_str = "—"
        time_str = "—"

    st.markdown(f"""
    <div class="top-bar">
        <div class="top-bar-left">
            <div class="top-bar-logo">📈</div>
            <div>
                <div class="top-bar-title">IDX80 Dashboard</div>
                <div class="top-bar-sub">Indonesia Stock Exchange · Blue Chip Index</div>
            </div>
        </div>
        <div class="last-updated">
            <div class="lu-label">Terakhir Diperbarui</div>
            <div class="lu-date">{date_str}</div>
            <div>Diakses {time_str}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px 0; font-family:'JetBrains Mono',monospace;
                font-size:11px; color:#3A4A6B; text-transform:uppercase;
                letter-spacing:2px;">Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        label="Page",
        options=["🏦  Market Overview", "🔍  Stock Exploration"],
        label_visibility="collapsed",
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ALL DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Memuat data IDX80..."):
    try:
        prices_raw   = load_prices()
        metadata_raw = load_metadata()
        features_raw = load_features()
        load_ok = True
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {e}")
        st.stop()

# Derive shared datasets
features_eng = engineer_features(features_raw, prices_raw)
metadata_upd  = update_market_cap(metadata_raw, prices_raw)
plot_df        = build_plot_df(metadata_upd, features_eng)

last_date = prices_raw["date"].max() if not prices_raw.empty else None


# ─────────────────────────────────────────────────────────────────────────────
# ██████████  PAGE 1: MARKET OVERVIEW  ██████████
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏦  Market Overview":

    render_topbar(last_date)

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Filter")
        sectors = ["Semua Sektor"] + sorted(plot_df["sector"].dropna().unique().tolist())
        sel_sector = st.selectbox("Sektor", sectors)

    filtered = plot_df.copy()
    if sel_sector != "Semua Sektor":
        filtered = filtered[filtered["sector"] == sel_sector]
    filtered_feat = features_eng.copy()
    if sel_sector != "Semua Sektor":
        tickers_in = filtered["ticker"].tolist()
        filtered_feat = filtered_feat[filtered_feat["ticker"].isin(tickers_in)]

    # ── KPI Metrics ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)

    total_stocks     = filtered["ticker"].nunique()
    avg_mkt_return   = filtered["avg_return"].mean()
    avg_volatility   = filtered["volatility"].mean()
    total_market_cap = filtered["market_cap"].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📌 Total Saham", str(total_stocks))
    with col2:
        st.metric(
            "📈 Avg Return",
            pct(avg_mkt_return, 4),
            delta=f"{avg_mkt_return*100:+.4f}%",
        )
    with col3:
        st.metric("〰 Avg Volatility", pct(avg_volatility, 4))
    with col4:
        st.metric("💰 Total Market Cap", f"IDR {fmt_large(total_market_cap)}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Treemap + Sector Bar ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Komposisi Pasar & Performa Sektor</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        st.plotly_chart(chart_treemap(filtered), use_container_width=True, config=CHART_CFG)
    with c2:
        st.plotly_chart(chart_sector_bar(filtered), use_container_width=True, config=CHART_CFG)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Risk vs Return ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Risk vs Return</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_scatter(filtered), use_container_width=True, config=CHART_CFG)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Top Gainers / Losers ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Top Movers</div>', unsafe_allow_html=True)
    tab_g, tab_l, tab_v, tab_mc = st.tabs(["🚀 Top Gainers", "📉 Top Losers", "🌪 Highest Volatility", "💰 Largest Market Cap"])

    def _format_tbl(df_sub: pd.DataFrame, cols: list) -> pd.DataFrame:
        out = df_sub[cols].copy()
        if "avg_return" in out.columns:
            out["avg_return"] = (out["avg_return"] * 100).map("{:+.4f}%".format)
        if "volatility" in out.columns:
            out["volatility"] = (out["volatility"] * 100).map("{:.4f}%".format)
        if "market_cap" in out.columns:
            out["market_cap"] = out["market_cap"].apply(lambda v: f"IDR {fmt_large(v)}")
        return out.rename(columns={
            "ticker": "Ticker", "company_name": "Perusahaan",
            "sector": "Sektor", "avg_return": "Avg Return",
            "volatility": "Volatility", "market_cap": "Market Cap",
        })

    with tab_g:
        top10g = filtered.nlargest(10, "avg_return")
        st.dataframe(_format_tbl(top10g, ["ticker","company_name","sector","avg_return","volatility"]),
                     use_container_width=True, hide_index=True)
    with tab_l:
        top10l = filtered.nsmallest(10, "avg_return")
        st.dataframe(_format_tbl(top10l, ["ticker","company_name","sector","avg_return","volatility"]),
                     use_container_width=True, hide_index=True)
    with tab_v:
        top10v = filtered.nlargest(10, "volatility")
        st.dataframe(_format_tbl(top10v, ["ticker","company_name","sector","avg_return","volatility"]),
                     use_container_width=True, hide_index=True)
    with tab_mc:
        top10mc = filtered.nlargest(10, "market_cap")
        st.dataframe(_format_tbl(top10mc, ["ticker","company_name","sector","market_cap","avg_return"]),
                     use_container_width=True, hide_index=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Volatility charts ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Analisis Volatilitas</div>', unsafe_allow_html=True)
    cv1, cv2 = st.columns(2, gap="medium")
    with cv1:
        st.plotly_chart(chart_market_vol(filtered_feat), use_container_width=True, config=CHART_CFG)
    with cv2:
        st.plotly_chart(chart_top_volatility(filtered), use_container_width=True, config=CHART_CFG)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Distribution + Correlation ────────────────────────────────────────────
    st.markdown('<div class="section-label">Distribusi Return & Korelasi</div>', unsafe_allow_html=True)
    cd1, cd2 = st.columns([2, 3], gap="medium")
    with cd1:
        st.plotly_chart(chart_return_histogram(filtered_feat), use_container_width=True, config=CHART_CFG)
        st.plotly_chart(chart_top_mcap(filtered), use_container_width=True, config=CHART_CFG)
    with cd2:
        st.plotly_chart(chart_correlation_heatmap(filtered_feat, filtered), use_container_width=True, config=CHART_CFG)


# ─────────────────────────────────────────────────────────────────────────────
# ██████████  PAGE 2: STOCK EXPLORATION  ██████████
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍  Stock Exploration":

    render_topbar(last_date)

    # ── Sidebar: stock selector + date range ──────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔍 Stock Selector")

        all_tickers = sorted(prices_raw["ticker"].unique().tolist())
        selected_ticker = st.selectbox("Pilih Saham", all_tickers, index=all_tickers.index("BBCA") if "BBCA" in all_tickers else 0)

        st.markdown("### 📅 Rentang Waktu")
        min_date = prices_raw["date"].min().date()
        max_date = prices_raw["date"].max().date()
        default_start = max(min_date, (pd.Timestamp(max_date) - pd.DateOffset(years=2)).date())

        start_date = st.date_input("Dari", value=default_start, min_value=min_date, max_value=max_date)
        end_date   = st.date_input("Hingga", value=max_date,   min_value=min_date, max_value=max_date)

        if start_date > end_date:
            st.error("Tanggal mulai harus sebelum tanggal akhir.")
            st.stop()

    # ── Filter data for selected ticker ──────────────────────────────────────
    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)

    df_price = prices_raw[
        (prices_raw["ticker"] == selected_ticker) &
        (prices_raw["date"] >= start_ts) &
        (prices_raw["date"] <= end_ts)
    ].copy().sort_values("date").reset_index(drop=True)

    df_feat = features_eng[
        (features_eng["ticker"] == selected_ticker) &
        (features_eng["date"] >= start_ts) &
        (features_eng["date"] <= end_ts)
    ].copy().sort_values("date").reset_index(drop=True)

    if df_price.empty or df_feat.empty:
        st.warning("⚠️ Tidak ada data untuk ticker dan rentang waktu yang dipilih.")
        st.stop()

    # Add derived features for stock-level charts
    df_feat = add_stock_features(df_feat)

    # ── Company info ──────────────────────────────────────────────────────────
    meta_row = metadata_upd[metadata_upd["ticker"] == selected_ticker]
    company_name = meta_row["company_name"].values[0] if not meta_row.empty else selected_ticker
    sector       = meta_row["sector"].values[0]       if not meta_row.empty else "—"
    industry     = meta_row["industry"].values[0]     if not meta_row.empty else "—"
    market_cap   = meta_row["market_cap"].values[0]   if not meta_row.empty else None
    beta         = meta_row["beta"].values[0]         if not meta_row.empty else None
    shares_out   = meta_row["shares_outstanding"].values[0] if not meta_row.empty else None

    latest_price  = df_price.iloc[-1]["close"]
    latest_ret    = df_feat.iloc[-1]["daily_return"]
    status_txt, status_cls = status_label(latest_ret)
    max_dd = df_feat["drawdown"].min() * 100

    # ── Stock Info Card ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="stock-info-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
            <div>
                <div class="stock-info-ticker">{selected_ticker}</div>
                <div class="stock-info-name">{company_name}</div>
                <div>
                    <span class="stock-badge">🏭 {sector}</span>
                    <span class="stock-badge">🔧 {industry}</span>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:28px;
                            font-weight:700; color:{C["text_primary"]};">
                    Rp {latest_price:,.0f}
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:14px;
                            margin-top:4px;">
                    <span class="{status_cls}">{latest_ret:+.2%} &nbsp; {status_txt}</span>
                </div>
                <div style="font-size:11px; color:{C["text_muted"]}; margin-top:4px;">
                    Return terbaru · {df_feat.iloc[-1]["date"].strftime("%d %b %Y")}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Secondary metrics row ─────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("💰 Market Cap", f"IDR {fmt_large(market_cap)}")
    with m2:
        st.metric("β Beta", f"{float(beta):.3f}" if beta is not None and str(beta) not in ["None","nan"] else "N/A")
    with m3:
        st.metric("📊 Shares Outstanding", fmt_large(shares_out))
    with m4:
        st.metric("📉 Max Drawdown", f"{max_dd:.2f}%")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Price & Daily Return ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Pergerakan Harga</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_price(df_price, selected_ticker), use_container_width=True, config=CHART_CFG)
    st.plotly_chart(chart_daily_return(df_feat, selected_ticker), use_container_width=True, config=CHART_CFG)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Cumulative Return + Rolling Vol ───────────────────────────────────────
    st.markdown('<div class="section-label">Return & Volatilitas</div>', unsafe_allow_html=True)
    cr1, cr2 = st.columns(2, gap="medium")
    with cr1:
        st.plotly_chart(chart_cumulative_return(df_feat, selected_ticker), use_container_width=True, config=CHART_CFG)
    with cr2:
        st.plotly_chart(chart_rolling_vol(df_feat, selected_ticker), use_container_width=True, config=CHART_CFG)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Drawdown + Histogram ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Risiko & Distribusi</div>', unsafe_allow_html=True)
    rd1, rd2 = st.columns(2, gap="medium")
    with rd1:
        st.plotly_chart(chart_drawdown(df_feat, selected_ticker), use_container_width=True, config=CHART_CFG)
    with rd2:
        st.plotly_chart(chart_stock_hist(df_feat, selected_ticker), use_container_width=True, config=CHART_CFG)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Correlation with other stocks ─────────────────────────────────────────
    st.markdown('<div class="section-label">Korelasi dengan Saham Lain</div>', unsafe_allow_html=True)

    pivot = features_eng.pivot(index="date", columns="ticker", values="daily_return")
    if selected_ticker in pivot.columns:
        corr_series = (
            pivot.corr()[selected_ticker]
            .drop(selected_ticker)
            .dropna()
            .sort_values(ascending=True)
        )
        colors = [C["green"] if v >= 0 else C["red"] for v in corr_series.values]

        fig_corr = go.Figure(go.Bar(
            x=corr_series.values,
            y=corr_series.index.tolist(),
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            hovertemplate="%{y}<br>ρ = %{x:.3f}<extra></extra>",
        ))
        fig_corr.update_layout(
            title=f"Korelasi Return {selected_ticker} dengan Saham IDX80 Lainnya",
            xaxis_title="Pearson Correlation (ρ)",
            xaxis=dict(range=[-1.1, 1.3]),
            yaxis_title=None,
            height=max(380, len(corr_series) * 18),
            margin=dict(l=70, r=20, t=44, b=40),
        )
        st.plotly_chart(fig_corr, use_container_width=True, config=CHART_CFG)
    else:
        st.info("Data korelasi tidak tersedia untuk ticker ini.")