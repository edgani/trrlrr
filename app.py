"""
TRR/LRR Macro Scanner — God Tier
===================================
Port of MQAwam-V3 Pine Script TRR/LRR engine + MacroRegime Pro v7 macro filter.
Only shows macro-aligned long / short / watch signals.

Run: streamlit run scanner_trrlrr.py
"""
from __future__ import annotations

import concurrent.futures
import datetime
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    st.error("Install yfinance: pip install yfinance")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TRR/LRR Scanner",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
html,[class*="css"]{font-family:'DM Sans',sans-serif}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:.5rem;padding-bottom:2rem}
.topbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:8px 12px;
  border-radius:8px;background:rgba(255,255,255,0.03);
  border:1px solid rgba(255,255,255,0.07);margin-bottom:10px;font-size:11px}
.qb{display:inline-block;padding:3px 12px;border-radius:20px;
  font-family:'DM Mono',monospace;font-weight:500;font-size:11px}
.q1{background:#d4edda;color:#155724}.q2{background:#fff3cd;color:#856404}
.q3{background:#ffeeba;color:#7d4e00}.q4{background:#f8d7da;color:#721c24}
.qunk{background:#e2e3e5;color:#495057}
.mc{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
  border-radius:8px;padding:10px 12px;margin-bottom:4px}
.mc .lb{font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  opacity:.4;margin-bottom:2px}
.mc .vl{font-family:'Syne',sans-serif;font-size:16px;font-weight:700}
.mc .sb{font-size:10px;opacity:.5}
.sh{font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  opacity:.35;padding:6px 0 3px;border-bottom:1px solid rgba(255,255,255,0.06);
  margin-bottom:6px}
.good{color:#3dbb6c}.bad{color:#e05252}.warn{color:#e5a020}.neu{color:#888}
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ENGINE PARAMETERS (matches MQAwam-V3 Pine defaults)
# ══════════════════════════════════════════════════════════════════════════════
TRADE_LEN   = 15
TREND_LEN   = 63
TAIL_LEN    = 252   # reduced from 756; keeps conceptual role; adj tail_atr_mult
NORM_LEN    = 63
RV_LEN      = 20
VOL_ROC_LEN = 5
ATR_LEN     = 14
DMI_LEN     = 14
ADX_SMOOTH  = 14
ER_LEN      = 20
R2_LEN      = 30

TRADE_ATR_MULT     = 1.35
TREND_ATR_MULT     = 2.05
TAIL_ATR_MULT      = 2.80   # scaled down from 4.20 to compensate shorter TAIL_LEN
CENTER_BIAS        = 0.18
SHOCK_BOOST        = 0.22
TREND_SHOCK_BOOST  = 0.15
TAIL_SHOCK_BOOST   = 0.12

TRADE_THRESH   = 0.20;  TRADE_NEUTRAL  = 0.06
TREND_THRESH   = 0.14;  TREND_NEUTRAL  = 0.05
TAIL_THRESH    = 0.10;  TAIL_NEUTRAL   = 0.03

ASYM_BOOST       = 0.22
TRANS_MIN        = 61.0
ACTIVITY_FLOOR   = 42.0
COMPRESSION_WARN = 62.0
CONFLUENCE_MIN   = 58.0
LAUNCH_MIN       = 62.0
WATCH_BUFFER     = 8.0
CHASE_ATR_LIMIT  = 0.60

TRANS_BREAK_W   = 0.30
TRANS_FOLLOW_W  = 0.22
TRANS_RECLAIM_W = 0.16
TRANS_ALIGN_W   = 0.20
TRANS_QUAL_W    = 0.12

QUALITY_STRONG = 55.0
QUALITY_WEAK   = 40.0

# Volume mult by asset class (Pine: Forex=0.35, Crypto=0.80, else 1.0)
VOL_MULTS: Dict[str, float] = {
    "us_equity": 1.00,
    "ihsg":      1.00,
    "forex":     0.35,
    "commodities": 1.00,
    "crypto":    0.80,
}

MIN_BARS = 130          # default — need NORM_LEN + headroom
MIN_BARS_AC: Dict[str, int] = {   # per-asset-class overrides
    "us_equity":  130,
    "ihsg":        80,   # some JK stocks have shorter history
    "forex":       80,   # some exotic pairs thinner
    "commodities": 80,   # grains futures can roll-gap
    "crypto":      80,
}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — UNIVERSE  &  MACRO QUAD POLICY
# ══════════════════════════════════════════════════════════════════════════════

# (asset_class, display_style)
UNIVERSE: Dict[str, Tuple[str, str]] = {
    # ── US Equities ──────────────────────────────────────────────────────────
    "SPY":    ("us_equity", "Broad equities"),
    "QQQ":    ("us_equity", "Growth/Tech"),
    "IWM":    ("us_equity", "Small caps"),
    "RSP":    ("us_equity", "Broad equities"),
    "XLE":    ("us_equity", "Energy/Value"),
    "XLF":    ("us_equity", "Financials"),
    "XLI":    ("us_equity", "Industrials"),
    "XLB":    ("us_equity", "Materials"),
    "XLK":    ("us_equity", "Growth/Tech"),
    "XLV":    ("us_equity", "Defensives"),
    "XLY":    ("us_equity", "Consumer disc"),
    "XLP":    ("us_equity", "Defensives"),
    "XLU":    ("us_equity", "Defensives"),
    "XLRE":   ("us_equity", "Real estate"),
    "XLC":    ("us_equity", "Growth/Tech"),
    "GLD":    ("us_equity", "Gold"),
    "TLT":    ("us_equity", "Long bonds"),
    "IEF":    ("us_equity", "Bonds"),
    "SHY":    ("us_equity", "Short bonds"),
    "HYG":    ("us_equity", "HY Credit"),
    "LQD":    ("us_equity", "IG Credit"),
    "UUP":    ("us_equity", "USD"),
    "EEM":    ("us_equity", "EM Equities"),
    "AAPL":   ("us_equity", "Growth/Tech"),
    "MSFT":   ("us_equity", "Growth/Tech"),
    "NVDA":   ("us_equity", "Semis/AI"),
    "AMZN":   ("us_equity", "Growth/Tech"),
    "META":   ("us_equity", "Growth/Tech"),
    "GOOGL":  ("us_equity", "Growth/Tech"),
    "TSLA":   ("us_equity", "Semis/AI"),
    "AVGO":   ("us_equity", "Semis/AI"),
    "AMD":    ("us_equity", "Semis/AI"),
    "NFLX":   ("us_equity", "Growth/Tech"),
    "JPM":    ("us_equity", "Financials"),
    "BAC":    ("us_equity", "Financials"),
    "GS":     ("us_equity", "Financials"),
    "XOM":    ("us_equity", "Energy/Value"),
    "CVX":    ("us_equity", "Energy/Value"),
    # ── IHSG ────────────────────────────────────────────────────────────────
    "^JKSE":   ("ihsg", "Index"),
    "BBCA.JK": ("ihsg", "Bank"),
    "BBRI.JK": ("ihsg", "Bank"),
    "BMRI.JK": ("ihsg", "Bank"),
    "BBNI.JK": ("ihsg", "Bank"),
    "BRIS.JK": ("ihsg", "Bank"),
    "ADRO.JK": ("ihsg", "Batu Bara/Energi"),
    "PTBA.JK": ("ihsg", "Batu Bara/Energi"),
    "ITMG.JK": ("ihsg", "Batu Bara/Energi"),
    "HRUM.JK": ("ihsg", "Batu Bara/Energi"),
    "ANTM.JK": ("ihsg", "Logam"),
    "INCO.JK": ("ihsg", "Logam"),
    "MDKA.JK": ("ihsg", "Logam"),
    "TLKM.JK": ("ihsg", "Telco/Infra"),
    "EXCL.JK": ("ihsg", "Telco/Infra"),
    "ASII.JK": ("ihsg", "Consumer"),
    "ICBP.JK": ("ihsg", "Consumer"),
    "INDF.JK": ("ihsg", "Consumer"),
    "KLBF.JK": ("ihsg", "Consumer"),
    "AMRT.JK": ("ihsg", "Consumer"),
    "CTRA.JK": ("ihsg", "Properti/Health"),
    "BSDE.JK": ("ihsg", "Properti/Health"),
    "HEAL.JK": ("ihsg", "Properti/Health"),
    # ── Forex ────────────────────────────────────────────────────────────────
    "EURUSD=X": ("forex", "EUR/USD"),
    "GBPUSD=X": ("forex", "GBP/USD"),
    "AUDUSD=X": ("forex", "AUD/USD"),
    "NZDUSD=X": ("forex", "NZD/USD"),
    "JPY=X":    ("forex", "USD/JPY"),
    "CHF=X":    ("forex", "USD/CHF"),
    "CAD=X":    ("forex", "USD/CAD"),
    "IDR=X":    ("forex", "USD/IDR"),
    "CNH=X":    ("forex", "USD/CNH"),
    "SGD=X":    ("forex", "USD/SGD"),
    # ── Commodities ──────────────────────────────────────────────────────────
    "GC=F": ("commodities", "Gold"),
    "SI=F": ("commodities", "Silver"),
    "CL=F": ("commodities", "WTI Oil"),
    "BZ=F": ("commodities", "Brent Oil"),
    "NG=F": ("commodities", "Natural Gas"),
    "HG=F": ("commodities", "Copper"),
    "ZC=F": ("commodities", "Corn"),
    "ZW=F": ("commodities", "Wheat"),
    "ZS=F": ("commodities", "Soybeans"),
    # ── Crypto ───────────────────────────────────────────────────────────────
    "BTC-USD":  ("crypto", "BTC"),
    "ETH-USD":  ("crypto", "ETH"),
    "SOL-USD":  ("crypto", "SOL"),
    "BNB-USD":  ("crypto", "BNB"),
    "XRP-USD":  ("crypto", "XRP"),
    "ADA-USD":  ("crypto", "ADA"),
    "AVAX-USD": ("crypto", "AVAX"),
    "LINK-USD": ("crypto", "LINK"),
}

DISPLAY_NAME: Dict[str, str] = {
    "SPY": "SPY (S&P 500)", "QQQ": "QQQ (Nasdaq)", "IWM": "IWM (Small Cap)",
    "RSP": "RSP (Equal Wt)", "XLE": "XLE (Energy)", "XLF": "XLF (Financials)",
    "XLI": "XLI (Industrials)", "XLB": "XLB (Materials)", "XLK": "XLK (Tech)",
    "XLV": "XLV (Health)", "XLY": "XLY (Cons. Disc)", "XLP": "XLP (Cons. Staples)",
    "XLU": "XLU (Utilities)", "XLRE": "XLRE (REIT)", "XLC": "XLC (Comm.)",
    "GLD": "GLD (Gold ETF)", "TLT": "TLT (20Y Bond)", "IEF": "IEF (7-10Y Bond)",
    "SHY": "SHY (1-3Y Bond)", "HYG": "HYG (HY Credit)", "LQD": "LQD (IG Credit)",
    "UUP": "UUP (USD Index)", "EEM": "EEM (EM Equity)",
    "AAPL": "AAPL", "MSFT": "MSFT", "NVDA": "NVDA", "AMZN": "AMZN",
    "META": "META", "GOOGL": "GOOGL", "TSLA": "TSLA", "AVBO": "AVBO",
    "AMD": "AMD", "NFLX": "NFLX", "JPM": "JPM", "BAC": "BAC",
    "GS": "GS", "XOM": "XOM", "CVX": "CVX",
    "^JKSE": "IHSG", "BBCA.JK": "BBCA", "BBRI.JK": "BBRI", "BMRI.JK": "BMRI",
    "BBNI.JK": "BBNI", "BRIS.JK": "BRIS", "ADRO.JK": "ADRO", "PTBA.JK": "PTBA",
    "ITMG.JK": "ITMG", "HRUM.JK": "HRUM", "ANTM.JK": "ANTM", "INCO.JK": "INCO",
    "MDKA.JK": "MDKA", "TLKM.JK": "TLKM", "EXCL.JK": "EXCL", "ASII.JK": "ASII",
    "ICBP.JK": "ICBP", "INDF.JK": "INDF", "KLBF.JK": "KLBF", "AMRT.JK": "AMRT",
    "CTRA.JK": "CTRA", "BSDE.JK": "BSDE", "HEAL.JK": "HEAL",
    "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "AUDUSD=X": "AUD/USD",
    "NZDUSD=X": "NZD/USD", "JPY=X": "USD/JPY", "CHF=X": "USD/CHF",
    "CAD=X": "USD/CAD", "IDR=X": "USD/IDR", "CNH=X": "USD/CNH", "SGD=X": "USD/SGD",
    "GC=F": "Gold (XAU)", "SI=F": "Silver (XAG)", "CL=F": "WTI Oil",
    "BZ=F": "Brent Oil", "NG=F": "Natural Gas", "HG=F": "Copper",
    "ZC=F": "Corn", "ZW=F": "Wheat", "ZS=F": "Soybeans",
    "BTC-USD": "BTC/USD", "ETH-USD": "ETH/USD", "SOL-USD": "SOL/USD",
    "BNB-USD": "BNB/USD", "XRP-USD": "XRP/USD", "ADA-USD": "ADA/USD",
    "AVAX-USD": "AVAX/USD", "LINK-USD": "LINK/USD",
}

# ── Macro quad policy: per quad → per asset class → {long, short, avoid} ────
QUAD_POLICY: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "Q1": {
        "us_equity":   {
            "long":  ["Growth/Tech", "Semis/AI", "Gold", "EM Equities", "IG Credit", "Short bonds", "Bonds"],
            "short": ["Energy/Value"],
            "avoid": ["HY Credit"],
        },
        "ihsg":        {
            "long":  ["Bank", "Consumer", "Telco/Infra", "Index"],
            "short": ["Batu Bara/Energi"],
            "avoid": ["Logam", "Properti/Health"],
        },
        "forex":       {
            "long":  ["EUR/USD", "GBP/USD", "AUD/USD", "NZD/USD"],
            "short": ["USD/JPY", "USD/CHF"],
            "avoid": ["USD/IDR", "USD/CNH"],
        },
        "commodities": {
            "long":  ["Gold", "Silver"],
            "short": ["WTI Oil", "Brent Oil", "Copper"],
            "avoid": ["Natural Gas"],
        },
        "crypto":      {
            "long":  ["BTC", "ETH", "SOL", "LINK"],
            "short": [],
            "avoid": ["ADA", "AVAX"],
        },
    },
    "Q2": {
        "us_equity":   {
            "long":  ["Growth/Tech", "Semis/AI", "Energy/Value", "Financials",
                      "Industrials", "Materials", "EM Equities", "Broad equities"],
            "short": ["Defensives", "Long bonds", "Bonds"],
            "avoid": ["Short bonds", "IG Credit"],
        },
        "ihsg":        {
            "long":  ["Bank", "Batu Bara/Energi", "Logam", "Consumer", "Index"],
            "short": [],
            "avoid": ["Properti/Health"],
        },
        "forex":       {
            "long":  ["AUD/USD", "NZD/USD", "USD/CAD"],
            "short": ["USD/JPY", "USD/CHF"],
            "avoid": ["USD/IDR"],
        },
        "commodities": {
            "long":  ["WTI Oil", "Brent Oil", "Copper", "Corn", "Wheat", "Soybeans", "Natural Gas"],
            "short": ["Gold", "Silver"],
            "avoid": [],
        },
        "crypto":      {
            "long":  ["BTC", "ETH", "SOL", "LINK", "ADA", "AVAX", "BNB", "XRP"],
            "short": [],
            "avoid": [],
        },
    },
    "Q3": {
        "us_equity":   {
            "long":  ["Defensives", "Energy/Value", "Gold", "USD", "Short bonds"],
            "short": ["Consumer disc", "Small caps", "Growth/Tech", "Semis/AI",
                      "Real estate", "HY Credit"],
            "avoid": ["EM Equities", "Long bonds", "Broad equities"],
        },
        "ihsg":        {
            "long":  ["Batu Bara/Energi", "Logam", "Telco/Infra"],
            "short": ["Consumer", "Properti/Health"],
            "avoid": ["Bank", "Index"],
        },
        "forex":       {
            "long":  ["USD/JPY", "USD/CHF", "USD/IDR", "USD/CNH"],
            "short": ["AUD/USD", "NZD/USD", "EUR/USD", "GBP/USD"],
            "avoid": ["USD/SGD"],
        },
        "commodities": {
            "long":  ["Gold", "Silver", "WTI Oil", "Brent Oil"],
            "short": ["Copper", "Corn", "Wheat", "Soybeans"],
            "avoid": ["Natural Gas"],
        },
        "crypto":      {
            "long":  ["BTC"],
            "short": ["ETH", "SOL", "LINK", "ADA", "AVAX"],
            "avoid": ["BNB", "XRP"],
        },
    },
    "Q4": {
        "us_equity":   {
            "long":  ["Defensives", "Long bonds", "Bonds", "Gold", "USD", "Short bonds"],
            "short": ["Consumer disc", "Small caps", "Energy/Value", "Materials", "HY Credit"],
            "avoid": ["Financials", "EM Equities", "Industrials", "Broad equities"],
        },
        "ihsg":        {
            "long":  ["Telco/Infra"],
            "short": ["Consumer", "Logam", "Batu Bara/Energi"],
            "avoid": ["Bank", "Properti/Health", "Index"],
        },
        "forex":       {
            "long":  ["USD/JPY", "USD/CHF", "USD/IDR"],
            "short": ["AUD/USD", "NZD/USD", "EUR/USD", "GBP/USD"],
            "avoid": ["USD/CNH", "USD/SGD"],
        },
        "commodities": {
            "long":  ["Gold"],
            "short": ["WTI Oil", "Brent Oil", "Copper", "Natural Gas"],
            "avoid": ["Silver", "Corn", "Wheat", "Soybeans"],
        },
        "crypto":      {
            "long":  ["BTC"],
            "short": ["ETH", "SOL", "ADA", "AVAX", "LINK"],
            "avoid": ["BNB", "XRP"],
        },
    },
}

QUAD_LABEL = {
    "Q1": "Risk-On Goldilocks ↑G ↓I",
    "Q2": "Reflation / Boom ↑G ↑I",
    "Q3": "Stagflation ↓G ↑I",
    "Q4": "Deflation / Recession ↓G ↓I",
}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MATH HELPERS (exact Pine port)
# ══════════════════════════════════════════════════════════════════════════════

def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def _squash(x: float) -> float:
    """Pine: f_squash = x / (1 + |x|)"""
    return x / (1.0 + abs(x))

def _z(s: pd.Series, n: int) -> pd.Series:
    """Rolling z-score. Pine ta.stdev = sample std (ddof=1)."""
    ma = s.rolling(n, min_periods=max(n // 3, 10)).mean()
    sd = s.rolling(n, min_periods=max(n // 3, 10)).std()
    return ((s - ma) / sd.replace(0.0, np.nan)).fillna(0.0)

def _er(s: pd.Series, n: int) -> pd.Series:
    """Efficiency ratio: net move / total path over n bars."""
    direction = (s - s.shift(n)).abs()
    path = s.diff().abs().rolling(n, min_periods=max(n // 3, 5)).sum()
    return (direction / path.replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)

def _rv(close: pd.Series, n: int = 20) -> pd.Series:
    """Realized volatility annualized. Pine: ta.stdev(log(c/c[1]),n)*sqrt(252)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(n, min_periods=max(n // 3, 5)).std() * math.sqrt(252.0)

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's ATR (EWM alpha=1/n, adjust=False — matches Pine ta.atr)."""
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

def _adx(
    high: pd.Series, low: pd.Series, close: pd.Series,
    dmi_n: int = 14, smooth_n: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder ADX, +DI, -DI."""
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    up   = high.diff()
    down = -low.diff()
    plus_dm  = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=high.index
    )
    atr_s    = tr.ewm(alpha=1.0 / dmi_n, adjust=False, min_periods=dmi_n).mean()
    plus_di  = (
        100.0 * plus_dm.ewm(alpha=1.0 / dmi_n, adjust=False, min_periods=dmi_n).mean()
        / atr_s.replace(0.0, np.nan)
    ).fillna(0.0)
    minus_di = (
        100.0 * minus_dm.ewm(alpha=1.0 / dmi_n, adjust=False, min_periods=dmi_n).mean()
        / atr_s.replace(0.0, np.nan)
    ).fillna(0.0)
    dx  = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1.0 / smooth_n, adjust=False, min_periods=smooth_n).mean().fillna(0.0)
    return adx, plus_di, minus_di

def _r2(close: pd.Series, n: int = 30) -> pd.Series:
    """Rolling R² of close vs bar index (Pine: ta.correlation(close, bar_index, n)²)."""
    idx  = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    corr = close.rolling(n, min_periods=n // 2).corr(idx)
    return (corr ** 2).fillna(0.0)

def _score_blend(
    pc: pd.Series, bc: pd.Series, vc: pd.Series, rvc: pd.Series,
    sc: pd.Series, prc: pd.Series,
    wp: float, wb: float, wvb: float, wrv: float, ws: float, wpr: float,
    vol_mult: float = 1.0,
) -> pd.Series:
    """
    Exact Pine f_score_blend:
      wVolume = wVolumeBase * volMult
      result  = (wP*price + wB*basis + wV*vol + wRV*(-rv) + wS*slope + wPR*persist)
                / (wP + wB + wV + wRV + wS + wPR)
    Note: wRV is ADDED to denominator but term is SUBTRACTED in numerator.
    """
    wv    = wvb * vol_mult
    denom = max(wp + wb + wv + wrv + ws + wpr, 1e-9)
    return (wp * pc + wb * bc + wv * vc - wrv * rvc + ws * sc + wpr * prc) / denom

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SCORE ENGINES (Trade / Trend / Tail)
# ══════════════════════════════════════════════════════════════════════════════

def _trade_series(h, l, c, v, vm=1.0) -> pd.Series:
    basis = c.ewm(span=TRADE_LEN, adjust=False).mean()
    rv_s  = _rv(c, RV_LEN)
    pc  = _z(c.pct_change(TRADE_LEN) * 100.0, NORM_LEN)
    bc  = _z((c / basis - 1.0) * 100.0, NORM_LEN)
    vc  = _z(
        v.pct_change(VOL_ROC_LEN).ewm(span=2, adjust=False).mean() * 100.0,
        NORM_LEN,
    )
    rvc = _z(rv_s.pct_change(1) * 100.0, NORM_LEN)
    sc  = _z(basis.pct_change(3) * 100.0, NORM_LEN)
    ern = max(5, round(TRADE_LEN * 0.7))
    prc = _z(_er(c, ern), NORM_LEN)
    return _score_blend(pc, bc, vc, rvc, sc, prc, 0.28, 0.22, 0.16, 0.12, 0.12, 0.10, vm)

def _trend_series(h, l, c, v, vm=1.0) -> pd.Series:
    basis = c.ewm(span=TREND_LEN, adjust=False).mean()
    rv_s  = _rv(c, RV_LEN)
    pc  = _z(c.pct_change(TREND_LEN) * 100.0, NORM_LEN)
    bc  = _z((c / basis - 1.0) * 100.0, NORM_LEN)
    vc  = _z(
        v.pct_change(VOL_ROC_LEN).ewm(span=5, adjust=False).mean() * 100.0,
        NORM_LEN,
    )
    rvc = _z(rv_s.pct_change(5) * 100.0, NORM_LEN)
    sc  = _z(basis.pct_change(10) * 100.0, NORM_LEN)
    prc = _z(_er(c, TREND_LEN), NORM_LEN)
    return _score_blend(pc, bc, vc, rvc, sc, prc, 0.18, 0.30, 0.10, 0.08, 0.20, 0.14, vm)

def _tail_series(h, l, c, v, vm=1.0) -> pd.Series:
    basis_raw = c.ewm(span=TAIL_LEN, adjust=False).mean()
    basis_252 = c.ewm(span=252, adjust=False).mean()
    basis = basis_raw.where(basis_raw.notna(), basis_252)
    rv_s  = _rv(c, RV_LEN)
    pc  = _z(c.pct_change(min(252, len(c) - 1)) * 100.0, NORM_LEN)
    bc  = _z((c / basis - 1.0) * 100.0, NORM_LEN)
    vc  = _z(
        v.pct_change(VOL_ROC_LEN).ewm(span=10, adjust=False).mean() * 100.0,
        NORM_LEN,
    )
    rvc = _z(rv_s.pct_change(10) * 100.0, NORM_LEN)
    sc  = _z(basis.pct_change(21) * 100.0, NORM_LEN)
    prc = _z(_er(c, min(63, len(c) - 1)), NORM_LEN)
    return _score_blend(pc, bc, vc, rvc, sc, prc, 0.10, 0.38, 0.06, 0.06, 0.24, 0.16, vm)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — QUALITY & ACTIVITY
# ══════════════════════════════════════════════════════════════════════════════

def _quality(h, l, c) -> pd.Series:
    adx, _, _ = _adx(h, l, c, DMI_LEN, ADX_SMOOTH)
    adx_n = ((adx - 12.0) / 28.0).clip(0.0, 1.0)
    er_s  = _er(c, ER_LEN).clip(0.0, 1.0)
    r2_s  = _r2(c, R2_LEN).clip(0.0, 1.0)
    return 100.0 * (0.45 * adx_n + 0.35 * er_s + 0.20 * r2_s)

def _activity_compression(h, l, c, v, vm=1.0) -> Tuple[pd.Series, pd.Series]:
    atr_s    = _atr(h, l, c, ATR_LEN)
    atr_pct  = (atr_s / c.replace(0.0, np.nan) * 100.0).fillna(0.0)
    rv_s     = _rv(c, RV_LEN).fillna(0.0)
    atr_base = atr_pct.rolling(50, min_periods=20).mean().replace(0.0, np.nan)
    rv_base  = rv_s.rolling(50, min_periods=20).mean().replace(0.0, np.nan)
    vol_base = v.rolling(20, min_periods=10).mean().replace(0.0, np.nan)
    act_atr  = (atr_pct / (atr_base * 1.25)).clip(0.0, 1.0).fillna(0.5)
    act_rv   = (rv_s / (rv_base * 1.20)).clip(0.0, 1.0).fillna(0.5)
    act_vraw = (v / (vol_base * 1.35)).clip(0.0, 1.0).fillna(0.5)
    act_v    = 0.35 + 0.65 * act_vraw * vm
    act  = 100.0 * (0.42 * act_atr + 0.43 * act_rv + 0.15 * act_v).clip(0.0, 1.0)
    comp = 100.0 * (1.0 - (0.50 * act_atr + 0.38 * act_rv + 0.12 * act_v)).clip(0.0, 1.0)
    return act, comp

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — STATE HYSTERESIS
# ══════════════════════════════════════════════════════════════════════════════

def _state_hysteresis(score: float, thresh: float, neutral: float, prev: int = 0) -> int:
    if score > thresh:             return 1
    if score < -thresh:            return -1
    if abs(score) <= neutral:      return 0
    return prev

def _state_series_last(score_series: pd.Series, thresh: float, neutral: float) -> int:
    """Simulate state carry-over over last 40 bars (avoids full-series loop)."""
    prev = 0
    tail = score_series.dropna().values[-40:]
    for s in tail:
        prev = _state_hysteresis(float(s), thresh, neutral, prev)
    return prev

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FULL TRR/LRR CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def compute_trrlrr(df: pd.DataFrame, vol_mult: float = 1.0) -> Optional[Dict]:
    """
    Complete TRR/LRR signal computation for one ticker.
    Input: daily OHLCV DataFrame (columns: Open, High, Low, Close, Volume).
    Returns: result dict for the LAST bar, or None if insufficient data.
    """
    if df is None or len(df) < 63:  # absolute floor = TREND_LEN
        return None

    h = df["High"].ffill()
    l = df["Low"].ffill()
    c = df["Close"].ffill()
    v = df["Volume"].fillna(0.0)

    # Volume fallback for zero-volume assets (some FX tickers)
    if v.sum() == 0.0:
        v = c.diff().abs().fillna(0.0) + 1.0

    # ── Scores ───────────────────────────────────────────────────────────────
    trd_s = _trade_series(h, l, c, v, vol_mult)
    trn_s = _trend_series(h, l, c, v, vol_mult)
    tal_s = _tail_series(h, l, c, v, vol_mult)

    # ── ATR & bases ──────────────────────────────────────────────────────────
    atr_s  = _atr(h, l, c, ATR_LEN)
    tb_s   = c.ewm(span=TRADE_LEN, adjust=False).mean()
    trb_s  = c.ewm(span=TREND_LEN, adjust=False).mean()
    tab_s  = c.ewm(span=TAIL_LEN,  adjust=False).mean()

    # ── Shock (realized vol z-score) ─────────────────────────────────────────
    rv_s   = _rv(c, RV_LEN)
    shock_s = _z(rv_s.pct_change(1) * 100.0, NORM_LEN)

    # ── Quality & Activity ───────────────────────────────────────────────────
    qual_s      = _quality(h, l, c)
    act_s, comp_s = _activity_compression(h, l, c, v, vol_mult)

    # ── Helper: extract last finite float ───────────────────────────────────
    def last(s: pd.Series) -> float:
        vals = s.dropna()
        return float(vals.iloc[-1]) if not vals.empty else 0.0

    ts_v   = last(trd_s)
    trn_v  = last(trn_s)
    tal_v  = last(tal_s)
    atr_v  = last(atr_s)
    shock_v = max(last(shock_s), 0.0)
    tb_v   = last(tb_s)
    trb_v  = last(trb_s)
    tab_v  = last(tab_s)
    q_v    = last(qual_s)
    act_v  = last(act_s)
    comp_v = last(comp_s)
    cl_v   = float(c.iloc[-1])
    op_v   = float(df["Open"].ffill().iloc[-1])
    hi_v   = float(h.iloc[-1])
    lo_v   = float(l.iloc[-1])
    pc_v   = float(c.iloc[-2]) if len(c) >= 2 else cl_v  # prev close

    if atr_v <= 0.0 or cl_v <= 0.0:
        return None

    # ── States ───────────────────────────────────────────────────────────────
    trade_state = _state_series_last(trd_s, TRADE_THRESH, TRADE_NEUTRAL)
    trend_state = _state_series_last(trn_s, TREND_THRESH, TREND_NEUTRAL)
    tail_state  = _state_series_last(tal_s, TAIL_THRESH,  TAIL_NEUTRAL)

    # ── Quality components ───────────────────────────────────────────────────
    qual_bull = _clip((q_v - QUALITY_WEAK) / max(QUALITY_STRONG - QUALITY_WEAK, 1.0))
    qual_bear = _clip((QUALITY_STRONG - q_v) / max(QUALITY_STRONG - QUALITY_WEAK, 1.0))

    # ── Pressure vectors ─────────────────────────────────────────────────────
    trd_up = _clip(0.60 * max(ts_v, 0) + 0.25 * max(trn_v, 0) + 0.15 * qual_bull, 0, 1.5)
    trd_dn = _clip(0.60 * max(-ts_v, 0) + 0.25 * max(-trn_v, 0) + 0.15 * qual_bear, 0, 1.5)
    trn_up = _clip(0.55 * max(trn_v, 0) + 0.25 * max(tal_v, 0) + 0.20 * qual_bull, 0, 1.5)
    trn_dn = _clip(0.55 * max(-trn_v, 0) + 0.25 * max(-tal_v, 0) + 0.20 * qual_bear, 0, 1.5)
    tal_up = _clip(0.60 * max(tal_v, 0) + 0.20 * max(trn_v, 0) + 0.20 * qual_bull, 0, 1.5)
    tal_dn = _clip(0.60 * max(-tal_v, 0) + 0.20 * max(-trn_v, 0) + 0.20 * qual_bear, 0, 1.5)

    # ── TRR / LRR ────────────────────────────────────────────────────────────
    compress = max(-shock_v, 0.0)  # shock compression term

    # TRADE
    trade_anchor = 0.65 * tb_v + 0.35 * pc_v
    trade_center = trade_anchor + CENTER_BIAS * atr_v * _squash(ts_v)
    tw_up = atr_v * TRADE_ATR_MULT * max(
        0.55,
        1.0 + 0.18 * max(ts_v, 0) + SHOCK_BOOST * shock_v
        + ASYM_BOOST * trd_up - 0.08 * trd_dn - 0.08 * compress,
    )
    tw_dn = atr_v * TRADE_ATR_MULT * max(
        0.55,
        1.0 + 0.18 * max(-ts_v, 0) + SHOCK_BOOST * shock_v
        + ASYM_BOOST * trd_dn - 0.08 * trd_up - 0.08 * compress,
    )
    trade_trr = trade_center + tw_up
    trade_lrr = trade_center - tw_dn

    # TREND
    rn_up = atr_v * TREND_ATR_MULT * max(
        0.70,
        1.0 + 0.14 * max(trn_v, 0) + 0.08 * max(tal_v, 0)
        + TREND_SHOCK_BOOST * shock_v + ASYM_BOOST * trn_up - 0.06 * trn_dn,
    )
    rn_dn = atr_v * TREND_ATR_MULT * max(
        0.70,
        1.0 + 0.14 * max(-trn_v, 0) + 0.08 * max(-tal_v, 0)
        + TREND_SHOCK_BOOST * shock_v + ASYM_BOOST * trn_dn - 0.06 * trn_up,
    )
    trend_trr = trb_v + rn_up
    trend_lrr = trb_v - rn_dn

    # TAIL
    ta_up = atr_v * TAIL_ATR_MULT * max(
        0.80,
        1.0 + 0.10 * max(tal_v, 0) + 0.06 * max(trn_v, 0)
        + TAIL_SHOCK_BOOST * shock_v + ASYM_BOOST * tal_up - 0.05 * tal_dn,
    )
    ta_dn = atr_v * TAIL_ATR_MULT * max(
        0.80,
        1.0 + 0.10 * max(-tal_v, 0) + 0.06 * max(-trn_v, 0)
        + TAIL_SHOCK_BOOST * shock_v + ASYM_BOOST * tal_dn - 0.05 * tal_up,
    )
    tail_trr = tab_v + ta_up
    tail_lrr = tab_v - ta_dn

    # ── Core bias ─────────────────────────────────────────────────────────────
    trd_conf   = _clip(ts_v  / max(TRADE_THRESH, 1e-9), -2.0, 2.0)
    trn_conf   = _clip(trn_v / max(TREND_THRESH, 1e-9), -2.0, 2.0)
    tal_conf   = _clip(tal_v / max(TAIL_THRESH,  1e-9), -2.0, 2.0)
    bias_score = 0.50 * _squash(trd_conf) + 0.30 * _squash(trn_conf) + 0.20 * _squash(tal_conf)
    core_bias  = 1 if bias_score > 0.15 else (-1 if bias_score < -0.15 else 0)

    # ── Candle characteristics (last bar) ─────────────────────────────────────
    candle_rng    = max(hi_v - lo_v, atr_v * 0.01)
    body_strength = _clip(abs(cl_v - op_v) / candle_rng)
    # Range expansion vs 20-bar average
    last20_rng = (h.iloc[-22:] - l.iloc[-22:]).mean() if len(h) >= 22 else atr_v
    range_exp  = _clip((candle_rng / (max(last20_rng, atr_v * 0.01) * 1.15)) / 2.0)
    strong_bull = cl_v > op_v and body_strength >= 0.45
    strong_bear = cl_v < op_v and body_strength >= 0.45

    # ── Transition score components ───────────────────────────────────────────
    c3 = c.iloc[-3:]
    bull_break = float(
        (c3 > trade_trr).mean() * 0.40 + (c3 > trend_trr).mean() * 0.20
        + range_exp * 0.20 + body_strength * 0.20
    )
    bear_break = float(
        (c3 < trade_lrr).mean() * 0.40 + (c3 < trend_lrr).mean() * 0.20
        + range_exp * 0.20 + body_strength * 0.20
    )
    bull_follow = (
        _clip(0.40 + 0.30 * (1 if cl_v > op_v else 0) + 0.15 * body_strength + 0.15 * range_exp)
        if cl_v > trade_trr else 0.0
    )
    bear_follow = (
        _clip(0.40 + 0.30 * (1 if cl_v < op_v else 0) + 0.15 * body_strength + 0.15 * range_exp)
        if cl_v < trade_lrr else 0.0
    )
    bull_reclaim = (
        _clip(0.40 + 0.30 * (1 if strong_bull else 0) + 0.30 * body_strength)
        if lo_v <= trade_lrr and cl_v > trade_lrr and strong_bull else 0.0
    )
    bear_reject = (
        _clip(0.40 + 0.30 * (1 if strong_bear else 0) + 0.30 * body_strength)
        if hi_v >= trade_trr and cl_v < trade_trr and strong_bear else 0.0
    )

    bull_align = _clip(
        (max(trade_state, 0) + max(trend_state, 0) + max(tail_state, 0)) / 3.0
        + 0.20 * qual_bull + 0.15 * range_exp + 0.05 * body_strength
    )
    bear_align = _clip(
        (max(-trade_state, 0) + max(-trend_state, 0) + max(-tail_state, 0)) / 3.0
        + 0.20 * qual_bear + 0.15 * range_exp + 0.05 * body_strength
    )
    qual_norm = _clip(q_v / 100.0)

    def trans(brk, flw, rcl, aln, qn):
        return 100.0 * _clip(
            TRANS_BREAK_W * brk + TRANS_FOLLOW_W * flw
            + TRANS_RECLAIM_W * rcl + TRANS_ALIGN_W * aln + TRANS_QUAL_W * qn
        )

    act_mult    = _clip(act_v / max(ACTIVITY_FLOOR, 1.0), 0.20, 1.0)
    comp_pen    = _clip(comp_v / 100.0, 0.0, 0.85)

    bull_trans_raw = trans(bull_break, bull_follow, bull_reclaim, bull_align, qual_norm)
    bear_trans_raw = trans(bear_break, bear_follow, bear_reject, bear_align, qual_norm)
    bull_trans = bull_trans_raw * act_mult * (1.0 - 0.55 * comp_pen)
    bear_trans = bear_trans_raw * act_mult * (1.0 - 0.55 * comp_pen)

    # ── Confluence (6 conditions; VWAP excluded — not available on daily scanner) ──
    bull_conf = 100.0 * sum([
        core_bias >= 0, trade_state >= 0, trend_state >= 0, tail_state >= 0,
        cl_v >= tb_v, cl_v >= trb_v,
    ]) / 6.0
    bear_conf = 100.0 * sum([
        core_bias <= 0, trade_state <= 0, trend_state <= 0, tail_state <= 0,
        cl_v <= tb_v, cl_v <= trb_v,
    ]) / 6.0

    # ── Slope agree (last 2 bars of each basis) ───────────────────────────────
    def _prev(s): return float(s.dropna().iloc[-2]) if len(s.dropna()) >= 2 else float(s.dropna().iloc[-1])
    bull_slope = tb_v >= _prev(tb_s) and trb_v >= _prev(trb_s) and tab_v >= _prev(tab_s)
    bear_slope = tb_v <= _prev(tb_s) and trb_v <= _prev(trb_s) and tab_v <= _prev(tab_s)

    # ── Chase check ───────────────────────────────────────────────────────────
    bull_over = max(cl_v - trade_trr, 0.0) / atr_v
    bear_over = max(trade_lrr - cl_v, 0.0) / atr_v
    bull_chase = bull_over > CHASE_ATR_LIMIT
    bear_chase = bear_over > CHASE_ATR_LIMIT

    # Fresh = always 100 in scanner (no freeze cadence)
    fresh = 100.0

    # ── Launch scores ─────────────────────────────────────────────────────────
    bull_launch = (
        0.28 * bull_trans + 0.22 * bull_conf + 0.14 * act_v + 0.14 * q_v
        + 0.10 * fresh + 0.06 * (100.0 if bull_slope else 0.0)
        + 0.06 * (0.0 if bull_chase else 100.0)
    )
    bear_launch = (
        0.28 * bear_trans + 0.22 * bear_conf + 0.14 * act_v + 0.14 * q_v
        + 0.10 * fresh + 0.06 * (100.0 if bear_slope else 0.0)
        + 0.06 * (0.0 if bear_chase else 100.0)
    )

    # ── Hard wait ─────────────────────────────────────────────────────────────
    hard_wait = (
        q_v <= QUALITY_WEAK or act_v < ACTIVITY_FLOOR or comp_v >= COMPRESSION_WARN
    )

    # ── Structural alignment (no counter-trend in default) ────────────────────
    bull_struct = trend_state >= 0 and tail_state >= 0 and bull_slope
    bear_struct = trend_state <= 0 and tail_state <= 0 and bear_slope

    eff_conf_min  = CONFLUENCE_MIN
    eff_launch    = LAUNCH_MIN
    eff_trans     = TRANS_MIN
    eff_watch_min = LAUNCH_MIN - WATCH_BUFFER

    bull_trigger = (
        bull_trans >= eff_trans and bull_launch >= eff_launch
        and bull_trans > bear_trans and bull_conf >= eff_conf_min
        and not hard_wait and bull_struct and not bull_chase
    )
    bear_trigger = (
        bear_trans >= eff_trans and bear_launch >= eff_launch
        and bear_trans > bull_trans and bear_conf >= eff_conf_min
        and not hard_wait and bear_struct and not bear_chase
    )
    bull_watch = (
        bull_launch >= eff_watch_min and bull_trans > bear_trans
        and bull_conf >= max(35.0, eff_conf_min - 8.0)
        and not hard_wait and bull_struct and not bull_chase
        and not bull_trigger
    )
    bear_watch = (
        bear_launch >= eff_watch_min and bear_trans > bull_trans
        and bear_conf >= max(35.0, eff_conf_min - 8.0)
        and not hard_wait and bear_struct and not bear_chase
        and not bear_trigger
    )

    signal: Optional[str]
    if   bull_trigger: signal = "LONG"
    elif bear_trigger: signal = "SHORT"
    elif bull_watch:   signal = "WATCH LONG"
    elif bear_watch:   signal = "WATCH SHORT"
    else:              signal = None

    # ── Returns ───────────────────────────────────────────────────────────────
    ret_1m = float((c.iloc[-1] / c.iloc[-22]  - 1) * 100) if len(c) >= 22  else float("nan")
    ret_3m = float((c.iloc[-1] / c.iloc[-63]  - 1) * 100) if len(c) >= 63  else float("nan")
    ret_1w = float((c.iloc[-1] / c.iloc[-6]   - 1) * 100) if len(c) >= 6   else float("nan")

    return {
        "signal":       signal,
        "price":        cl_v,
        "atr":          atr_v,
        # Levels
        "trade_trr":    trade_trr,
        "trade_lrr":    trade_lrr,
        "trend_trr":    trend_trr,
        "trend_lrr":    trend_lrr,
        "tail_trr":     tail_trr,
        "tail_lrr":     tail_lrr,
        # States
        "trade_state":  trade_state,
        "trend_state":  trend_state,
        "tail_state":   tail_state,
        # Scores
        "trade_score":  ts_v,
        "trend_score":  trn_v,
        "tail_score":   tal_v,
        "quality":      q_v,
        "activity":     act_v,
        "compression":  comp_v,
        # Transition
        "bull_trans":   bull_trans,
        "bear_trans":   bear_trans,
        "bull_conf":    bull_conf,
        "bear_conf":    bear_conf,
        "bull_launch":  bull_launch,
        "bear_launch":  bear_launch,
        # Flags
        "core_bias":    core_bias,
        "hard_wait":    hard_wait,
        "bull_chase":   bull_chase,
        "bear_chase":   bear_chase,
        "bull_slope":   bull_slope,
        "bear_slope":   bear_slope,
        "bull_struct":  bull_struct,
        "bear_struct":  bear_struct,
        # Returns
        "ret_1w":       ret_1w,
        "ret_1m":       ret_1m,
        "ret_3m":       ret_3m,
        # Distances (in ATR)
        "dist_trr_atr": (trade_trr - cl_v) / atr_v if atr_v > 0 else float("nan"),
        "dist_lrr_atr": (cl_v - trade_lrr) / atr_v if atr_v > 0 else float("nan"),
    }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

BENCHMARK_TICKERS = ["SPY", "IWM", "GLD", "GC=F", "CL=F", "TLT", "^VIX", "HYG", "LQD"]

def _extract_ticker_df(raw: pd.DataFrame, tk: str) -> Optional[pd.DataFrame]:
    """
    Extract single-ticker DataFrame from yfinance batch download result.

    yfinance <= 0.1.x : MultiIndex (Ticker, Field)  → raw[tk]
    yfinance >= 0.2.x : MultiIndex (Field, Ticker)  → raw.xs(tk, axis=1, level=1)
    yfinance single   : flat columns                → raw directly
    """
    if not isinstance(raw.columns, pd.MultiIndex):
        # Single-ticker download or flat columns
        return raw.copy()

    lvl0 = raw.columns.get_level_values(0).unique().tolist()
    lvl1 = raw.columns.get_level_values(1).unique().tolist()

    # yfinance 1.x: level-0 = field names (Close, Open …), level-1 = tickers
    if tk in lvl1:
        try:
            df = raw.xs(tk, axis=1, level=1)
            # Rename "Adj Close" → "Close" if needed
            if "Adj Close" in df.columns and "Close" not in df.columns:
                df = df.rename(columns={"Adj Close": "Close"})
            return df.copy()
        except Exception:
            pass

    # yfinance old style: level-0 = tickers, level-1 = field names
    if tk in lvl0:
        try:
            return raw[tk].copy()
        except Exception:
            pass

    return None


def _clean_df(df: pd.DataFrame, min_bars: int) -> Optional[pd.DataFrame]:
    """Standardise OHLCV, forward-fill, drop NaN, enforce min bar count."""
    if df is None or df.empty:
        return None
    # Accept "Adj Close" as "Close" fallback
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if "Close" not in needed:
        return None
    df = df[needed].copy()
    # Ensure Volume column exists (forex has no volume)
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df.ffill().dropna(subset=["Close"])
    if len(df) < min_bars:
        return None
    return df


def _fetch_single(tk: str, period: str) -> Optional[pd.DataFrame]:
    """Individual ticker download — used as fallback."""
    try:
        raw = yf.download(tk, period=period, auto_adjust=True,
                          progress=False, threads=False)
        ac, _ = UNIVERSE.get(tk, ("us_equity", ""))
        return _clean_df(raw, MIN_BARS_AC.get(ac, MIN_BARS))
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all(period: str = "5y") -> Dict[str, pd.DataFrame]:
    """
    Download daily OHLCV for all universe + benchmark tickers.

    Strategy:
      1. Batch download all tickers at once (fast path).
      2. For any ticker that fails or has <min_bars, retry individually (slow path).
    Compatible with yfinance 0.1.x, 0.2.x, and 1.x MultiIndex layouts.
    """
    all_tickers = list(set(list(UNIVERSE.keys()) + BENCHMARK_TICKERS))
    result: Dict[str, pd.DataFrame] = {}
    need_fallback: list = []

    # ── Pass 1: batch download ────────────────────────────────────────────────
    try:
        raw = yf.download(
            all_tickers, period=period, auto_adjust=True,
            progress=False, threads=True,
        )
        for tk in all_tickers:
            ac, _ = UNIVERSE.get(tk, ("us_equity", ""))
            mb = MIN_BARS_AC.get(ac, MIN_BARS)
            try:
                df_raw = _extract_ticker_df(raw, tk)
                df = _clean_df(df_raw, mb)
                if df is not None:
                    result[tk] = df
                else:
                    need_fallback.append(tk)
            except Exception:
                need_fallback.append(tk)
    except Exception as e:
        # Batch failed entirely — fall back all
        need_fallback = all_tickers

    # ── Pass 2: individual fallback for missing tickers ───────────────────────
    if need_fallback:
        def _fetch(tk: str) -> Tuple[str, Optional[pd.DataFrame]]:
            return tk, _fetch_single(tk, period)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(_fetch, tk): tk for tk in need_fallback}
            for fut in concurrent.futures.as_completed(futs):
                tk, df = fut.result()
                if df is not None:
                    result[tk] = df

    return result

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MACRO QUAD ENGINE (price-based; standalone)
# ══════════════════════════════════════════════════════════════════════════════

def determine_quad(data: Dict[str, pd.DataFrame]) -> Tuple[str, float, Dict]:
    """
    Infer macro quad from price signals.
    Growth proxy : SPY + IWM trend vs TLT (bonds up = growth fear).
    Inflation proxy: GLD + CL=F 3-month return.
    Returns (quad, confidence 0-1, detail_dict).
    """
    def _ret(tk, n):
        df = data.get(tk)
        if df is None or len(df) < n + 1:
            return float("nan")
        c = df["Close"].ffill()
        return float(c.iloc[-1] / c.iloc[-(n + 1)] - 1)

    def _trend_score(tk, n=200):
        df = data.get(tk)
        if df is None or len(df) < n:
            return 0.5
        c = df["Close"].ffill()
        ma = c.rolling(n).mean().iloc[-1]
        return 1.0 if float(c.iloc[-1]) > ma else 0.0

    spy_3m = _ret("SPY", 63);  spy_1m = _ret("SPY", 21)
    iwm_3m = _ret("IWM", 63);  iwm_1m = _ret("IWM", 21)
    tlt_3m = _ret("TLT", 63)
    gld_3m = _ret("GLD", 63)  if not math.isnan(_ret("GLD", 63)) else _ret("GC=F", 63)
    oil_3m = _ret("CL=F", 63)
    hyg_3m = _ret("HYG", 63)

    vix_df = data.get("^VIX")
    vix_now = float(vix_df["Close"].iloc[-1]) if vix_df is not None and not vix_df.empty else 20.0

    # Growth signal: + = accelerating
    g_parts = []
    if math.isfinite(spy_3m): g_parts.append((spy_3m, 0.40))
    if math.isfinite(iwm_3m): g_parts.append((iwm_3m, 0.30))
    if math.isfinite(tlt_3m): g_parts.append((-tlt_3m, 0.15))
    if math.isfinite(hyg_3m): g_parts.append((hyg_3m, 0.15))
    g_score  = sum(v * w for v, w in g_parts) / max(sum(w for _, w in g_parts), 0.01)
    g_acc    = g_score > 0

    # Inflation signal: + = rising
    i_parts = []
    if math.isfinite(gld_3m): i_parts.append((gld_3m, 0.55))
    if math.isfinite(oil_3m): i_parts.append((oil_3m, 0.45))
    i_score  = sum(v * w for v, w in i_parts) / max(sum(w for _, w in i_parts), 0.01)
    i_acc    = i_score > 0

    if   g_acc and not i_acc: quad = "Q1"
    elif g_acc and i_acc:     quad = "Q2"
    elif not g_acc and i_acc: quad = "Q3"
    else:                     quad = "Q4"

    g_conf = min(abs(g_score) / 0.04, 1.0)
    i_conf = min(abs(i_score) / 0.03, 1.0)
    confidence = (g_conf + i_conf) / 2.0

    detail = {
        "growth_score": round(g_score * 100, 1),
        "infl_score":   round(i_score * 100, 1),
        "spy_3m":       f"{spy_3m*100:.1f}%" if math.isfinite(spy_3m) else "N/A",
        "gld_3m":       f"{gld_3m*100:.1f}%" if math.isfinite(gld_3m) else "N/A",
        "oil_3m":       f"{oil_3m*100:.1f}%" if math.isfinite(oil_3m) else "N/A",
        "tlt_3m":       f"{tlt_3m*100:.1f}%" if math.isfinite(tlt_3m) else "N/A",
        "vix":          round(vix_now, 1),
        "growth_acc":   g_acc,
        "infl_acc":     i_acc,
    }
    return quad, _clip(confidence), detail

def vix_exec_state(data: Dict[str, pd.DataFrame]) -> Tuple[str, float]:
    """Simple exec mode from VIX."""
    vix_df = data.get("^VIX")
    if vix_df is None or vix_df.empty:
        return "Mixed", 0.55
    vix = float(vix_df["Close"].iloc[-1])
    if vix < 19:  return "🟢 Investable", 0.80
    if vix < 29:  return "🟡 Chop Zone",  0.55
    return "🔴 Defensive", 0.25

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — MACRO ALIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def get_macro_alignment(asset_class: str, style: str, quad: str, signal: str) -> str:
    """Returns 'aligned', 'neutral', 'against', or 'avoid'."""
    policy = QUAD_POLICY.get(quad, {}).get(asset_class, {})
    long_styles  = policy.get("long",  [])
    short_styles = policy.get("short", [])
    avoid_styles = policy.get("avoid", [])

    if style in avoid_styles:
        return "avoid"
    is_long  = "LONG"  in signal
    is_short = "SHORT" in signal
    if is_long:
        if style in long_styles:  return "aligned"
        if style in short_styles: return "against"
        return "neutral"
    if is_short:
        if style in short_styles: return "aligned"
        if style in long_styles:  return "against"
        return "neutral"
    return "neutral"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — SCANNER ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def scan_one(ticker: str, data: Dict[str, pd.DataFrame], quad: str) -> Optional[Dict]:
    """Compute full signal for one ticker. Returns row dict or None."""
    ac, style = UNIVERSE.get(ticker, (None, None))
    if ac is None:
        return None
    df = data.get(ticker)
    if df is None:
        return None
    vm   = VOL_MULTS.get(ac, 1.0)
    res  = compute_trrlrr(df, vm)
    if res is None or res["signal"] is None:
        return None

    sig    = res["signal"]
    align  = get_macro_alignment(ac, style, quad, sig)

    # Pull leading score and conf by direction
    if "LONG" in sig:
        launch = res["bull_launch"]
        trans  = res["bull_trans"]
        conf   = res["bull_conf"]
    else:
        launch = res["bear_launch"]
        trans  = res["bear_trans"]
        conf   = res["bear_conf"]

    def fmt_state(s): return "↑" if s > 0 else ("↓" if s < 0 else "→")
    def pct(v): return f"{v:+.1f}%" if math.isfinite(v) else "N/A"
    def lvl(v): return f"{v:.4g}" if math.isfinite(v) else "N/A"

    return {
        "Ticker":       ticker,
        "Name":         DISPLAY_NAME.get(ticker, ticker),
        "Class":        ac,
        "Style":        style,
        "Signal":       sig,
        "Macro":        align,
        "Launch":       round(launch, 1),
        "Trans":        round(trans, 1),
        "Conf%":        round(conf, 1),
        "Quality":      round(res["quality"], 1),
        "Activity":     round(res["activity"], 1),
        "T/Tr/Ta":      f"{fmt_state(res['trade_state'])}/{fmt_state(res['trend_state'])}/{fmt_state(res['tail_state'])}",
        "TrScore":      round(res["trade_score"], 3),
        "TnScore":      round(res["trend_score"], 3),
        "TaScore":      round(res["tail_score"], 3),
        "Price":        round(res["price"], 4 if res["price"] < 10 else 2),
        "Trade TRR":    lvl(res["trade_trr"]),
        "Trade LRR":    lvl(res["trade_lrr"]),
        "Trend TRR":    lvl(res["trend_trr"]),
        "Trend LRR":    lvl(res["trend_lrr"]),
        "ATR":          round(res["atr"], 4 if res["atr"] < 1 else 2),
        "ΔtoTRR(ATR)":  round(res["dist_trr_atr"], 2) if math.isfinite(res["dist_trr_atr"]) else 999.0,
        "ΔtoLRR(ATR)":  round(res["dist_lrr_atr"], 2) if math.isfinite(res["dist_lrr_atr"]) else 999.0,
        "1W%":          pct(res["ret_1w"]),
        "1M%":          pct(res["ret_1m"]),
        "3M%":          pct(res["ret_3m"]),
        # raw for sorting / coloring
        "_launch":      launch,
        "_trans":       trans,
        "_align":       align,
        "_signal":      sig,
        "_class":       ac,
        "_hard_wait":   res["hard_wait"],
    }

@st.cache_data(ttl=3600, show_spinner=False)
def run_scan(period: str = "5y") -> Tuple[pd.DataFrame, str, float, Dict, str, float, Dict]:
    """
    Full scan: fetch → quad → compute all tickers → return results df.
    Returns: (df, quad, conf, detail, exec_state, exec_score, data)
    """
    data = fetch_all(period)
    quad, conf, detail = determine_quad(data)
    exec_state, exec_score = vix_exec_state(data)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futs = {
            pool.submit(scan_one, tk, data, quad): tk
            for tk in UNIVERSE
        }
        for fut in concurrent.futures.as_completed(futs):
            try:
                row = fut.result()
                if row is not None:
                    results.append(row)
            except Exception:
                pass

    if not results:
        return pd.DataFrame(), quad, conf, detail, exec_state, exec_score, data

    df = pd.DataFrame(results).sort_values("_launch", ascending=False)
    return df, quad, conf, detail, exec_state, exec_score, data

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

SIGNAL_EMOJI = {
    "LONG":        "🟢 LONG",
    "SHORT":       "🔴 SHORT",
    "WATCH LONG":  "🟡 WATCH ▲",
    "WATCH SHORT": "🟡 WATCH ▼",
}
ALIGN_EMOJI = {
    "aligned": "✅ Aligned",
    "neutral": "⚠️ Neutral",
    "against": "❌ Against",
    "avoid":   "🚫 Avoid",
}
CLASS_LABEL = {
    "us_equity":   "🇺🇸 US Equity",
    "ihsg":        "🇮🇩 IHSG",
    "forex":       "💱 Forex",
    "commodities": "🛢️ Commodity",
    "crypto":      "₿ Crypto",
}
QUAD_EMOJI = {"Q1": "🟩", "Q2": "🟨", "Q3": "🟧", "Q4": "🟥"}

def signal_color(sig: str) -> str:
    if sig == "LONG":        return "#3dbb6c"
    if sig == "SHORT":       return "#e05252"
    if sig == "WATCH LONG":  return "#e5a020"
    if sig == "WATCH SHORT": return "#e5a020"
    return "#888"

def align_color(a: str) -> str:
    return {"aligned": "#3dbb6c", "neutral": "#e5a020", "against": "#e05252", "avoid": "#888"}.get(a, "#888")

def color_df(df: pd.DataFrame) -> pd.DataFrame.style:
    """Apply row color based on signal."""
    def row_bg(row):
        sig = row.get("_signal", "")
        if sig == "LONG":        return ["background-color: rgba(61,187,108,0.08)"] * len(row)
        if sig == "SHORT":       return ["background-color: rgba(224,82,82,0.08)"]  * len(row)
        if "WATCH" in sig:       return ["background-color: rgba(229,160,32,0.06)"] * len(row)
        return [""] * len(row)
    return df.style.apply(row_bg, axis=1)

def mc(label, value, sub="", cls=""):
    st.markdown(
        f'<div class="mc"><div class="lb">{label}</div>'
        f'<div class="vl {cls}">{value}</div>'
        f'{"<div class=sb>" + sub + "</div>" if sub else ""}</div>',
        unsafe_allow_html=True,
    )

def sh(t):
    st.markdown(f'<div class="sh">{t}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        '<span style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;'
        'letter-spacing:-.03em">🔬 TRR/LRR Macro Scanner</span>'
        '<span style="font-size:10px;opacity:.3;margin-left:8px;font-family:DM Mono,monospace">'
        "v1.0 · MQAwam-V3 Engine · MacroRegime Pro Filter</span>",
        unsafe_allow_html=True,
    )

    # ── Sidebar controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Scanner Controls")

        period = st.selectbox("Data period", ["3y", "5y", "2y"], index=1)

        if st.button("🔄 Refresh Scan", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### 🔭 Filters")

        show_signal = st.multiselect(
            "Signal type",
            ["LONG", "SHORT", "WATCH LONG", "WATCH SHORT"],
            default=["LONG", "SHORT", "WATCH LONG", "WATCH SHORT"],
        )
        show_align = st.multiselect(
            "Macro alignment",
            ["aligned", "neutral", "against", "avoid"],
            default=["aligned", "neutral"],
            help="'aligned' = quad policy confirms direction.\n'neutral' = no explicit policy.\n'against' = quad says opposite.",
        )
        show_class = st.multiselect(
            "Asset class",
            ["us_equity", "ihsg", "forex", "commodities", "crypto"],
            default=["us_equity", "ihsg", "forex", "commodities", "crypto"],
            format_func=lambda x: CLASS_LABEL.get(x, x),
        )

        min_launch = st.slider("Min launch score", 0, 100, 54, 1)
        min_trans  = st.slider("Min transition score", 0, 100, 45, 1)
        min_quality= st.slider("Min quality score", 0, 100, 35, 1)

        allow_counter_trend = st.checkbox(
            "Allow counter-trend", value=False,
            help="If ON: show signals where trend/tail state disagrees with signal direction."
        )

        override_quad = st.selectbox(
            "Override macro quad (optional)",
            ["Auto-detect", "Q1", "Q2", "Q3", "Q4"],
        )

        st.markdown("---")
        st.caption(
            "TRR = Top of Risk Range (resistance / short bias above)\n\n"
            "LRR = Low of Risk Range (support / long bias below)\n\n"
            "Launch score = composite signal quality (0–100)\n\n"
            "Trans = transition score — momentum of breakout/reclaim\n\n"
            "T/Tr/Ta = Trade / Trend / Tail state (↑↓→)"
        )

    # ── Run scan ─────────────────────────────────────────────────────────────
    with st.spinner("Running TRR/LRR scan across all universes…"):
        df_all, quad_auto, quad_conf, quad_detail, exec_state, exec_score, data = run_scan(period)

    quad = override_quad if override_quad != "Auto-detect" else quad_auto

    # ── Top bar ───────────────────────────────────────────────────────────────
    vix_val = quad_detail.get("vix", "—")
    q_badge = f'<span class="qb {quad.lower()}">{quad}</span>'
    st.markdown(
        f'<div class="topbar">'
        f'{q_badge} <strong>{QUAD_LABEL.get(quad, quad)}</strong>'
        f'<span style="opacity:.2">|</span>'
        f'Conf: <strong>{quad_conf:.0%}</strong>'
        f'<span style="opacity:.2">|</span>'
        f'Growth: <strong style="color:{"#3dbb6c" if quad_detail.get("growth_acc") else "#e05252"}">'
        f'{"▲" if quad_detail.get("growth_acc") else "▼"} {quad_detail.get("growth_score","?")}%</strong>'
        f'<span style="opacity:.2">|</span>'
        f'Inflation: <strong style="color:{"#e05252" if quad_detail.get("infl_acc") else "#3dbb6c"}">'
        f'{"▲" if quad_detail.get("infl_acc") else "▼"} {quad_detail.get("infl_score","?")}%</strong>'
        f'<span style="opacity:.2">|</span>'
        f'VIX: <strong>{vix_val}</strong>'
        f'<span style="opacity:.2">|</span>'
        f'Exec: <strong>{exec_state}</strong>'
        f'<span style="opacity:.25;margin-left:auto">{datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if df_all.empty:
        st.warning("No signals found or data fetch failed. Try refreshing.")
        return

    # ── Apply filters ─────────────────────────────────────────────────────────
    df = df_all.copy()
    df = df[df["_signal"].isin(show_signal)]
    df = df[df["_align"].isin(show_align)]
    df = df[df["_class"].isin(show_class)]
    df = df[df["_launch"] >= min_launch]
    df = df[df["Trans"] >= min_trans]
    df = df[df["Quality"] >= min_quality]

    if not allow_counter_trend:
        # Remove rows where signal direction contradicts trend/tail state
        def _struct_ok(row):
            sig = row["_signal"]
            ts  = row.get("T/Tr/Ta", "→/→/→").split("/")
            if len(ts) < 3: return True
            trd, trn, tal = ts
            if "LONG"  in sig: return trn in ("↑", "→") and tal in ("↑", "→")
            if "SHORT" in sig: return trn in ("↓", "→") and tal in ("↓", "→")
            return True
        df = df[df.apply(_struct_ok, axis=1)]

    # ── Summary metrics ───────────────────────────────────────────────────────
    n_long  = len(df[df["_signal"] == "LONG"])
    n_short = len(df[df["_signal"] == "SHORT"])
    n_wl    = len(df[df["_signal"] == "WATCH LONG"])
    n_ws    = len(df[df["_signal"] == "WATCH SHORT"])

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    loaded = sum(1 for tk in UNIVERSE if tk in data)
    with c1: mc("Data Coverage", f"{loaded}/{len(UNIVERSE)}", f"tickers loaded", "neu" if loaded == len(UNIVERSE) else "warn")
    with c2: mc("Scan Universe", f"{len(UNIVERSE)} tickers", f"{len(df_all)} had signals")
    with c3: mc("🟢 LONG",     str(n_long),  f"Triggered signals",       "good")
    with c4: mc("🔴 SHORT",    str(n_short), f"Triggered signals",        "bad")
    with c5: mc("🟡 WATCH ▲▼", f"{n_wl}▲ / {n_ws}▼", "Near-trigger",    "warn")
    with c6: mc("Showing",     str(len(df)), f"after filters",            "neu")

    st.markdown("---")

    if df.empty:
        st.info("No results match current filters. Try relaxing the filter thresholds.")
        return

    # ── Tabs: Summary | Table | Detail ───────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📋 Signal Cards", "📊 Full Table", "🔍 Detail View"])

    # ── TAB 1: Signal Cards ───────────────────────────────────────────────────
    with tab1:
        # Split by signal type
        for sig_type, emoji in [
            ("LONG",        "🟢 LONG TRIGGERS"),
            ("SHORT",       "🔴 SHORT TRIGGERS"),
            ("WATCH LONG",  "🟡 WATCH — LONG"),
            ("WATCH SHORT", "🟡 WATCH — SHORT"),
        ]:
            sub = df[df["_signal"] == sig_type]
            if sub.empty:
                continue
            sh(emoji)
            cols = st.columns(min(4, len(sub)))
            for i, (_, row) in enumerate(sub.iterrows()):
                col = cols[i % len(cols)]
                align_badge = ALIGN_EMOJI.get(row["_align"], row["_align"])
                sc = signal_color(sig_type)
                with col:
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);'
                        f'border-left:3px solid {sc};border-radius:8px;padding:10px 12px;margin-bottom:8px">'
                        f'<div style="font-family:DM Mono,monospace;font-size:13px;font-weight:700;color:{sc}">'
                        f'{SIGNAL_EMOJI.get(sig_type,"")}</div>'
                        f'<div style="font-size:15px;font-weight:700;margin:2px 0">{row["Name"]}</div>'
                        f'<div style="font-size:10px;opacity:.5;margin-bottom:6px">'
                        f'{CLASS_LABEL.get(row["Class"],row["Class"])} · {row["Style"]}</div>'
                        f'<div style="font-size:11px;color:{align_color(row["_align"])};margin-bottom:4px">'
                        f'{align_badge}</div>'
                        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:11px">'
                        f'<span style="opacity:.5">Launch</span><span style="font-family:DM Mono,monospace;font-weight:600">{row["Launch"]}</span>'
                        f'<span style="opacity:.5">Trans</span><span style="font-family:DM Mono,monospace">{row["Trans"]}</span>'
                        f'<span style="opacity:.5">Quality</span><span style="font-family:DM Mono,monospace">{row["Quality"]}</span>'
                        f'<span style="opacity:.5">T/Tr/Ta</span><span style="font-family:DM Mono,monospace">{row["T/Tr/Ta"]}</span>'
                        f'<span style="opacity:.5">Price</span><span style="font-family:DM Mono,monospace">{row["Price"]}</span>'
                        f'<span style="opacity:.5">ATR</span><span style="font-family:DM Mono,monospace">{row["ATR"]}</span>'
                        f'<span style="opacity:.5">Trade TRR</span><span style="font-family:DM Mono,monospace;color:#e5a020">{row["Trade TRR"]}</span>'
                        f'<span style="opacity:.5">Trade LRR</span><span style="font-family:DM Mono,monospace;color:#6496ff">{row["Trade LRR"]}</span>'
                        f'<span style="opacity:.5">1W</span><span style="font-family:DM Mono,monospace">{row["1W%"]}</span>'
                        f'<span style="opacity:.5">1M</span><span style="font-family:DM Mono,monospace">{row["1M%"]}</span>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

    # ── TAB 2: Full Table ─────────────────────────────────────────────────────
    with tab2:
        display_cols = [
            "Name", "Class", "Style", "Signal", "Macro",
            "Launch", "Trans", "Conf%", "Quality", "Activity",
            "T/Tr/Ta", "Price", "Trade TRR", "Trade LRR",
            "Trend TRR", "Trend LRR", "ATR",
            "ΔtoTRR(ATR)", "ΔtoLRR(ATR)", "1W%", "1M%", "3M%",
        ]
        view = df[display_cols].copy()
        # Rename for display
        view["Signal"] = view["Signal"].map(lambda s: SIGNAL_EMOJI.get(s, s))
        view["Macro"]  = view["Macro"].map(lambda a: ALIGN_EMOJI.get(a, a))
        view["Class"]  = view["Class"].map(lambda c: CLASS_LABEL.get(c, c))

        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Launch":     st.column_config.ProgressColumn("Launch", min_value=0, max_value=100, format="%.0f"),
                "Trans":      st.column_config.ProgressColumn("Trans",  min_value=0, max_value=100, format="%.0f"),
                "Quality":    st.column_config.ProgressColumn("Quality",min_value=0, max_value=100, format="%.0f"),
                "Activity":   st.column_config.ProgressColumn("Activity",min_value=0, max_value=100, format="%.0f"),
                "Conf%":      st.column_config.NumberColumn("Conf%", format="%.0f%%"),
                "ΔtoTRR(ATR)": st.column_config.NumberColumn("→TRR (ATR)", format="%.2f"),
                "ΔtoLRR(ATR)": st.column_config.NumberColumn("→LRR (ATR)", format="%.2f"),
            },
        )

        st.download_button(
            "⬇️ Export CSV",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name=f"trrlrr_scan_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

    # ── TAB 3: Detail View ────────────────────────────────────────────────────
    with tab3:
        ticker_list = df["Ticker"].tolist()
        if not ticker_list:
            st.info("No results to inspect.")
        else:
            sel = st.selectbox("Select ticker for detail", ticker_list)
            row = df[df["Ticker"] == sel].iloc[0]
            sig  = row["_signal"]
            sc   = signal_color(sig)

            st.markdown(
                f'<div style="font-size:20px;font-weight:700;color:{sc};margin-bottom:4px">'
                f'{SIGNAL_EMOJI.get(sig,"")} {row["Name"]} · {row["Style"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'Macro: **{ALIGN_EMOJI.get(row["_align"],row["_align"])}** · '
                f'Quad: **{quad}** — {QUAD_LABEL.get(quad,"")}',
                unsafe_allow_html=True,
            )
            st.divider()

            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                sh("📐 TRR / LRR LEVELS")
                mc("Trade TRR", row["Trade TRR"], "Short-term resistance")
                mc("Trade LRR", row["Trade LRR"], "Short-term support")
                mc("Trend TRR", row["Trend TRR"], "Medium-term resistance")
                mc("Trend LRR", row["Trend LRR"], "Medium-term support")
                mc("Price",     str(row["Price"]), f'ATR: {row["ATR"]}')
                mc("→ TRR (ATR)", str(row["ΔtoTRR(ATR)"]), "Distance in ATR units")
                mc("→ LRR (ATR)", str(row["ΔtoLRR(ATR)"]), "Distance in ATR units")
            with dc2:
                sh("📊 SIGNAL SCORES")
                mc("Launch Score",     str(row["Launch"]),    "0–100; needs ≥62 for trigger")
                mc("Transition Score", str(row["Trans"]),     "0–100; needs ≥61 for trigger")
                mc("Confluence %",     str(row["Conf%"]) + "%", "0–100; needs ≥58")
                mc("Quality",          str(row["Quality"]),   "ADX+ER+R² composite")
                mc("Activity",         str(row["Activity"]),  "ATR+RV+Volume composite")
                mc("T / Tr / Ta",      row["T/Tr/Ta"],        "Trade/Trend/Tail state")
            with dc3:
                sh("📈 MOMENTUM SCORES")
                mc("Trade Score",  f'{row["TrScore"]:+.3f}', f"Threshold ±{TRADE_THRESH}")
                mc("Trend Score",  f'{row["TnScore"]:+.3f}', f"Threshold ±{TREND_THRESH}")
                mc("Tail Score",   f'{row["TaScore"]:+.3f}', f"Threshold ±{TAIL_THRESH}")
                mc("1W Return",    row["1W%"])
                mc("1M Return",    row["1M%"])
                mc("3M Return",    row["3M%"])

            sh("📚 INTERPRETATION")
            is_long = "LONG" in sig
            bias_word = "bullish" if is_long else "bearish"
            level_ref = row["Trade LRR"] if is_long else row["Trade TRR"]
            level_label = "Trade LRR" if is_long else "Trade TRR"
            alert_word = "above" if is_long else "below"
            inv_word = "breaks below" if is_long else "breaks above"
            inv_level = row["Trade LRR"] if is_long else row["Trade TRR"]

            st.markdown(
                f"**Signal rationale:** {row['Name']} shows a **{sig}** signal with launch score "
                f"**{row['Launch']}** and transition score **{row['Trans']}**. "
                f"The macro quad (**{quad}**) alignment is **{row['_align']}** for this asset class and style. "
                f"All three horizons (Trade/Trend/Tail) are {row['T/Tr/Ta']}, indicating a "
                f"{'multi-horizon ' + bias_word + ' stack' if row['T/Tr/Ta'].count('↑' if is_long else '↓') >= 2 else 'partial ' + bias_word + ' structure'}.\n\n"
                f"**Key level to hold:** Price must stay {alert_word} **{level_ref}** ({level_label}). "
                f"Signal is invalidated if price **{inv_word} {inv_level}** on a closing basis."
            )

    # ── Quad context footer ───────────────────────────────────────────────────
    with st.expander(f"📋 Quad {quad} — What to prefer per asset class"):
        policy = QUAD_POLICY.get(quad, {})
        for ac_key, ac_label in CLASS_LABEL.items():
            p = policy.get(ac_key, {})
            if not p:
                continue
            st.markdown(f"**{ac_label}**")
            cols = st.columns(3)
            with cols[0]:
                st.markdown("🟢 **Long-biased styles**")
                for s in p.get("long", []):
                    st.markdown(f"- {s}")
            with cols[1]:
                st.markdown("🔴 **Short-biased styles**")
                for s in p.get("short", []):
                    st.markdown(f"- {s}")
            with cols[2]:
                st.markdown("🚫 **Avoid**")
                for s in p.get("avoid", []):
                    st.markdown(f"- {s}")
            st.divider()

if __name__ == "__main__":
    main()
