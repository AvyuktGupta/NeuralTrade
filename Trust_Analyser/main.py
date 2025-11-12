"""
Company Trustworthiness Analyzer (Fully automated)
- Fetches annual financial statements (income, balance sheet, cashflow) via yfinance
- Computes core metrics (EPS growth, Revenue CAGR, D/E, ROE, margins, FCF trend, liquidity)
- Applies heuristic checks to detect suspicious signs (sudden spikes, inconsistent growth, rising leverage)
- Produces a Trust Score 0-100 and a short verdict + reasons

Usage:
    pip install yfinance pandas numpy requests
    python fundamental_trust_analyzer.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import math
import requests
from datetime import datetime

# -------------------- USER CONFIG --------------------
TICKERS = ["TVSMOTOR.NS"]  # list of tickers to analyze
YEARS_TO_CHECK = 3         # number of past annual periods to analyze (best-effort)
TELEGRAM_BOT_TOKEN = "7914157083:AAGCCCHsdF0RQJGF2ezNnxN0mYszRQXfNOg"  # set "1234:ABC" or None to disable telegram alerts
TELEGRAM_CHAT_ID = "5678086807"
MIN_DATA_YEARS = 2         # minimum years of data required to compute core metrics
# -----------------------------------------------------

# Default weights for scoring (sum should ideally be 100, but normalized anyway)
WEIGHTS = {
    "eps_growth": 15,
    "revenue_growth": 15,
    "roe": 12,
    "de_ratio": 12,
    "profit_margin": 10,
    "fcf_trend": 12,
    "current_ratio": 8,
    "marketcap_stability": 6,
    "insider_flow": 10  # placeholder — likely unavailable in yfinance; will be neutral if missing
}

# -------------------- HELPERS --------------------
def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("(Telegram disabled) Message would be:", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    r = requests.post(url, data=payload)
    print("Telegram status:", r.status_code, r.text)


def safe_get_first_matching_index(df: pd.DataFrame, keywords):
    """
    Given a DataFrame whose index contains field names (like 'Total Revenue', 'Net Income'),
    try to find the best matching row using keywords (list of keywords).
    Returns the row name (index) if found, else None.
    """
    if df is None or df.empty:
        return None
    idx_lower = [str(i).lower() for i in df.index]
    for kw in keywords:
        for i_name, i_lower in zip(df.index, idx_lower):
            if kw.lower() in i_lower:
                return i_name
    return None


def cagr(start, end, periods):
    # handle zero or negative start carefully:
    try:
        start = float(start)
        end = float(end)
        if periods <= 0:
            return None
        if start <= 0:
            # use simple annualized growth if start <= 0: fallback to (end/start) not meaningful
            # approximate using arithmetic average growth
            return ((end - start) / max(abs(start), 1e-6)) / periods * 100
        return ( (end / start) ** (1.0 / periods) - 1.0 ) * 100.0
    except Exception:
        return None


def percent_change_series(series):
    """Return percent change from first to last as %"""
    try:
        s = pd.Series(series).dropna().astype(float)
        if len(s) < 2:
            return None
        return (s.iloc[-1] / s.iloc[0] - 1.0) * 100.0
    except Exception:
        return None


def normalize_score(value, good_min=None, good_max=None, invert=False):
    """
    Convert value to 0-100.
    If good_min/good_max specify a target range, values inside get higher score.
    If invert=True, smaller values are better (e.g. Debt/Equity)
    """
    if value is None or (isinstance(value, (float, int)) and (math.isnan(value) or math.isinf(value))):
        return 50  # neutral when unknown
    try:
        v = float(value)
    except Exception:
        return 50
    # If target range provided:
    if good_min is not None and good_max is not None:
        if good_min == good_max:
            return 100 if v == good_min else 0
        # Score drops linearly outside the range; inside range -> 100
        if good_min <= v <= good_max:
            return 100
        # if invert is True then lower is better: swap logic
        if invert:
            # for invert, lower values closer to good_min are better
            # map [good_max*3 (bad)] -> 0 and [good_min] -> 100
            bad = good_max * 3 if good_max != 0 else good_max + 1
            if v <= good_min:
                return 100
            elif v >= bad:
                return 0
            else:
                return max(0, min(100, int(100 * (1 - (v - good_min) / (bad - good_min)))))
        else:
            # normal: higher is better beyond good_min up to some high bound
            low = good_min / 3 if good_min != 0 else good_min - 1
            if v <= low:
                return 0
            elif v >= good_min:
                return 100
            else:
                return max(0, min(100, int(100 * (v - low) / (good_min - low))))
    else:
        # no target range: map using a heuristic. Use tanh scaling around 0
        return int(max(0, min(100, 50 + (v / (abs(v) + 10)) * 50)))  # crude fallback


# -------------------- CORE ANALYSIS --------------------
def fetch_fundamentals(ticker):
    """Return dict with financial DataFrames and basic info"""
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    # annual tables (yfinance returns columns as years; transpose for easier handling)
    fin = tk.financials  # income statement (year columns)
    bal = tk.balance_sheet
    cash = tk.cashflow
    # yfinance often returns quarterly tables too; attempt to also get 'quarterly' if needed
    return {"ticker": ticker, "info": info, "financials": fin, "balance_sheet": bal, "cashflow": cash}


def extract_time_series(df):
    """
    Given yfinance DataFrame with columns as periods (e.g., '2023-03-31', ...),
    return a cleaned DataFrame with columns sorted oldest->newest and numeric values.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    # yfinance columns are timestamps; convert to str and sort ascending:
    try:
        cols = list(df.columns)
        # ensure column order oldest -> newest
        cols_sorted = sorted(cols, key=lambda x: pd.to_datetime(str(x)))
        df2 = df[cols_sorted].astype(float)
        return df2
    except Exception:
        return df.astype(float)


def build_metrics_bundle(fund_data, years_to_check=YEARS_TO_CHECK):
    info = fund_data["info"]
    fin_raw = extract_time_series(fund_data["financials"])
    bal_raw = extract_time_series(fund_data["balance_sheet"])
    cash_raw = extract_time_series(fund_data["cashflow"])

    # convert columns to year ints or labels
    cols = list(fin_raw.columns) if not fin_raw.empty else []
    n_periods = len(cols)

    # If no annual financials present, try using 'quarterly' and aggregate yearly -> fallback (not implemented here)
    # We'll proceed with what's available.
    # Find keys in income statement
    revenue_key = safe_get_first_matching_index(fin_raw, ["total revenue", "revenue", "net sales", "sales"])
    net_income_key = safe_get_first_matching_index(fin_raw, ["net income", "netIncome", "net earnings", "profit attributable", "net loss"])
    eps_key = safe_get_first_matching_index(fin_raw, ["basic eps", "earnings per share", "eps", "diluted eps"])
    # Balance
    equity_key = safe_get_first_matching_index(bal_raw, ["total shareholders' equity", "total stockholders' equity", "total equity", "stockholders' equity", "total equity"])
    total_debt_key = safe_get_first_matching_index(bal_raw, ["total debt", "long term debt", "total liabilities", "debt"])
    current_assets_key = safe_get_first_matching_index(bal_raw, ["total current assets", "current assets"])
    current_liabilities_key = safe_get_first_matching_index(bal_raw, ["total current liabilities", "current liabilities"])
    # Cashflow
    fcf_key = safe_get_first_matching_index(cash_raw, ["free cash flow", "free cash", "fcf", "cash from operating activities"])

    # Build time series arrays (oldest -> newest)
    def series_values(df, key):
        if df is None or df.empty or key is None:
            return []
        s = df.loc[key].dropna().astype(float)
        # reverse to oldest -> newest if current ordering is newest -> oldest
        # extract columns and ensure order oldest->newest
        return list(s.values[::-1]) if len(s.index) > 0 and pd.to_datetime(df.columns[0]) > pd.to_datetime(df.columns[-1]) else list(s.values)

    revenue_ts = series_values(fin_raw, revenue_key)
    netincome_ts = series_values(fin_raw, net_income_key)
    eps_ts = series_values(fin_raw, eps_key)
    equity_ts = series_values(bal_raw, equity_key)
    debt_ts = series_values(bal_raw, total_debt_key)
    current_assets_ts = series_values(bal_raw, current_assets_key)
    current_liab_ts = series_values(bal_raw, current_liabilities_key)
    fcf_ts = series_values(cash_raw, fcf_key)

    # Trim to last YEARS_TO_CHECK + maybe one extra for growth calc
    def trim_to_years(ts, years):
        if not ts:
            return []
        return ts[-(years+1):] if len(ts) >= (years+1) else ts

    revenue_ts = trim_to_years(revenue_ts, years_to_check)
    netincome_ts = trim_to_years(netincome_ts, years_to_check)
    eps_ts = trim_to_years(eps_ts, years_to_check)
    equity_ts = trim_to_years(equity_ts, years_to_check)
    debt_ts = trim_to_years(debt_ts, years_to_check)
    current_assets_ts = trim_to_years(current_assets_ts, years_to_check)
    current_liab_ts = trim_to_years(current_liab_ts, years_to_check)
    fcf_ts = trim_to_years(fcf_ts, years_to_check)

    # Basic info from 'info' dict
    market_cap = info.get("marketCap", None)
    trailing_pe = info.get("trailingPE", None)
    forward_pe = info.get("forwardPE", None)
    # Try to get D/E directly from info if possible
    info_de = info.get("debtToEquity", None)

    bundle = {
        "revenue_ts": revenue_ts,
        "netincome_ts": netincome_ts,
        "eps_ts": eps_ts,
        "equity_ts": equity_ts,
        "debt_ts": debt_ts,
        "current_assets_ts": current_assets_ts,
        "current_liab_ts": current_liab_ts,
        "fcf_ts": fcf_ts,
        "market_cap": market_cap,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "info_de": info_de,
        "available_periods": n_periods
    }
    return bundle


def compute_ratios_and_scores(bundle):
    reasons = []
    # compute growths and ratios
    revenue = bundle["revenue_ts"]
    netincome = bundle["netincome_ts"]
    eps = bundle["eps_ts"]
    equity = bundle["equity_ts"]
    debt = bundle["debt_ts"]
    fcf = bundle["fcf_ts"]
    curr_assets = bundle["current_assets_ts"]
    curr_liab = bundle["current_liab_ts"]

    periods = max(len(revenue), len(netincome), len(eps), len(equity), len(debt), len(fcf))
    periods = max(periods, 0)
    # For growths we need >=2 points
    eps_growth = None
    revenue_cagr = None
    netincome_cagr = None

    if len(eps) >= 2:
        eps_growth = cagr(eps[0], eps[-1], len(eps)-1)
    if len(revenue) >= 2:
        revenue_cagr = cagr(revenue[0], revenue[-1], len(revenue)-1)
    if len(netincome) >= 2:
        netincome_cagr = cagr(netincome[0], netincome[-1], len(netincome)-1)

    # Debt to equity: prefer info_de if available
    de_ratio = None
    if bundle["info_de"] is not None:
        try:
            de_ratio = float(bundle["info_de"])
        except Exception:
            de_ratio = None

    # Fallback compute D/E from last known debt and equity
    if de_ratio is None and debt and equity and equity[-1] != 0:
        try:
            de_ratio = (debt[-1] if len(debt) > 0 else 0) / (equity[-1] if len(equity) > 0 else 1)
        except Exception:
            de_ratio = None

    # ROE: NetIncome / Equity (take last year if possible)
    roe = None
    if netincome and equity and equity[-1] not in (None, 0):
        roe = (netincome[-1] / equity[-1]) * 100.0

    # Profit Margin = NetIncome / Revenue
    profit_margin = None
    if netincome and revenue and revenue[-1] not in (None, 0):
        profit_margin = (netincome[-1] / revenue[-1]) * 100.0

    # Free cash flow trend: compute percent change from oldest to newest (if available)
    fcf_trend = None
    if len(fcf) >= 2:
        fcf_trend = percent_change_series(fcf)

    # Current ratio
    current_ratio = None
    if curr_assets and curr_liab and curr_liab[-1] not in (None, 0):
        current_ratio = curr_assets[-1] / curr_liab[-1]

    # Market cap stability: we can approximate by using info/trailing marketCap presence. If unavailable, neutral score.
    marketcap = bundle.get("market_cap", None)

    # Now convert to normalized scores (0-100)
    scores = {}
    scores["eps_growth"] = normalize_score(eps_growth, good_min=10, good_max=100)    # >10% is good
    scores["revenue_growth"] = normalize_score(revenue_cagr, good_min=8, good_max=100)
    scores["roe"] = normalize_score(roe, good_min=15, good_max=100)
    # For D/E lower is better -> invert True, target <1
    scores["de_ratio"] = normalize_score(de_ratio, good_min=0, good_max=1, invert=True)
    scores["profit_margin"] = normalize_score(profit_margin, good_min=10, good_max=100)
    # FCF trend: positive rising FCF is good (map >0 -> good)
    scores["fcf_trend"] = normalize_score(fcf_trend, good_min=5, good_max=100)
    scores["current_ratio"] = normalize_score(current_ratio, good_min=1.2, good_max=5)
    scores["marketcap_stability"] = 50 if marketcap is None else (100 if marketcap > 1e10 else 70 if marketcap > 1e9 else 50)
    # insider flow not available -> neutral
    scores["insider_flow"] = 50

    # Build reasons / flags
    # 1. Sudden revenue or profit spikes
    if revenue and len(revenue) >= 3:
        # compute year-over-year percent changes
        yoy = []
        for i in range(1, len(revenue)):
            prev = revenue[i-1] if revenue[i-1] != 0 else 1e-6
            yoy.append((revenue[i] / prev - 1) * 100.0)
        if any(abs(x) > 150 for x in yoy):  # extremely large jump
            reasons.append("⚠️ Very large revenue spike (>150%) detected — investigate one-off items or mergers.")
    if netincome and len(netincome) >= 3:
        yoy_n = []
        for i in range(1, len(netincome)):
            prev = netincome[i-1] if netincome[i-1] != 0 else 1e-6
            yoy_n.append((netincome[i] / prev - 1) * 100.0)
        if any(abs(x) > 300 for x in yoy_n):
            reasons.append("⚠️ Very large net income spike (>300%) detected — could indicate extraordinary items or accounting adjustments.")

    # 2. EPS vs Revenue mismatch (EPS falling while revenue rising)
    if revenue_cagr is not None and eps_growth is not None:
        if revenue_cagr > 10 and (eps_growth is not None and eps_growth < 0):
            reasons.append("⚠️ Revenue growing while EPS is declining — possible margin compression or accounting issues.")

    # 3. Rising leverage trend
    if debt and len(debt) >= 2 and equity and len(equity) >= 2:
        # compute D/E trend last vs prev
        try:
            last_de = (debt[-1] / equity[-1]) if equity[-1] != 0 else None
            prev_de = (debt[-2] / equity[-2]) if equity[-2] != 0 else None
            if last_de is not None and prev_de is not None and last_de > prev_de * 1.5:
                reasons.append("⚠️ Rapid increase in debt-to-equity ratio detected.")
        except Exception:
            pass

    # 4. Negative FCF and declining
    if fcf_trend is not None and fcf_trend < -20:
        reasons.append("⚠️ Free cash flow dropped a lot in recent years — this could be risky.")

    # 5. Low liquidity
    if current_ratio is not None and current_ratio < 0.8:
        reasons.append("⚠️ Current ratio < 0.8 — potential short-term liquidity issues.")

    # 6. Low ROE
    if roe is not None and roe < 5:
        reasons.append("⚠️ ROE is very low (<5%) — poor capital efficiency.")

    # Compose weighted aggregate score
    total_weight = sum(WEIGHTS.values())
    weighted_sum = 0.0
    for k, w in WEIGHTS.items():
        weighted_sum += scores.get(k, 50) * w
    trust_score = weighted_sum / total_weight
    trust_score = max(0, min(100, trust_score))

    # Translate to verdict
    if trust_score >= 75:
        verdict = "Legit / Financially Strong"
    elif trust_score >= 50:
        verdict = "Caution / Mixed Signals"
    else:
        verdict = "Suspicious / Risky — investigate further"

    # Summary details for report:
    details = {
        "eps_growth": eps_growth,
        "revenue_cagr": revenue_cagr,
        "netincome_cagr": netincome_cagr,
        "de_ratio": de_ratio,
        "roe": roe,
        "profit_margin": profit_margin,
        "fcf_trend_pct": fcf_trend,
        "current_ratio": current_ratio,
        "market_cap": bundle.get("market_cap", None) if (bundle := bundle) else None,
        "scores": scores,
        "trust_score": trust_score,
        "verdict": verdict,
        "reasons": reasons
    }

    return details


def analyze_ticker(ticker):
    print(f"\n==== ANALYZING {ticker} ====")
    fund_data = fetch_fundamentals(ticker)
    bundle = build_metrics_bundle(fund_data)
    # quick check for sufficient data
    available = bundle.get("available_periods", 0)
    if available < MIN_DATA_YEARS:
        print(f"⚠️ Warning: Only {available} annual periods found in yfinance for {ticker}. Results will be best-effort.")
    results = compute_ratios_and_scores(bundle)

    # Build printable summary
    msg_lines = []
    msg_lines.append(f"*{ticker}* — Trust Score: *{results['trust_score']:.1f}/100*")
    msg_lines.append(f"Verdict: *{results['verdict']}*")
    msg_lines.append("")
    # Key metrics
    def fmt(v, suffix=""):
        return "N/A" if v is None else (f"{v:.2f}{suffix}" if isinstance(v, (int, float)) else str(v))
    msg_lines.append(f"- EPS growth (approx): {fmt(results['eps_growth'], '%')}")
    msg_lines.append(f"- Revenue CAGR (approx): {fmt(results['revenue_cagr'], '%')}")
    msg_lines.append(f"- Net Income CAGR (approx): {fmt(results['netincome_cagr'], '%')}")
    msg_lines.append(f"- Debt/Equity (last): {fmt(results['de_ratio'])}")
    msg_lines.append(f"- ROE (last): {fmt(results['roe'], '%')}")
    msg_lines.append(f"- Profit margin (last): {fmt(results['profit_margin'], '%')}")
    msg_lines.append(f"- Free cash flow trend: {fmt(results['fcf_trend_pct'], '%')}")
    msg_lines.append(f"- Current ratio (last): {fmt(results['current_ratio'])}")
    msg_lines.append("")

    if results['reasons']:
        msg_lines.append("Flags / Warnings:")
        for r in results['reasons']:
            msg_lines.append(f"  • {r}")
    else:
        msg_lines.append("No major red flags detected in the heuristics.")

    summary_text = "\n".join(msg_lines)
    # Print once
    print(summary_text)

    # Optionally send to Telegram
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        send_telegram(summary_text)
    else:
        print("(Telegram disabled)")

    return {"ticker": ticker, "results": results, "summary": summary_text}


# -------------------- MAIN --------------------
if __name__ == "__main__":
    final_reports = []
    for t in TICKERS:
        try:
            r = analyze_ticker(t)
            final_reports.append(r)
        except Exception as e:
            print("Error analyzing", t, e)

    # Optionally: combine with technical bot outputs or store results
    # Example: print all trust scores quickly
    print("\n--- Summary ---")
    for rep in final_reports:
        t = rep["ticker"]
        sc = rep["results"]["trust_score"]
        print(f"{t}: {sc:.1f}/100 — {rep['results']['verdict']}")
