# fundamental_module.py
"""
Company Fundamental Analyzer — part of M3A Fusion System
Evaluates company trustworthiness & financial stability.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import math

# ---------------- CONFIG ----------------
YEARS_TO_CHECK = 3
MIN_DATA_YEARS = 2

WEIGHTS = {
    "eps_growth": 15,
    "revenue_growth": 15,
    "roe": 12,
    "de_ratio": 12,
    "profit_margin": 10,
    "fcf_trend": 12,
    "current_ratio": 8,
    "marketcap_stability": 6,
    "insider_flow": 10
}

# -------------- HELPERS --------------
def cagr(start, end, years):
    try:
        if start <= 0 or years <= 0:
            return None
        return (end / start) ** (1 / years) - 1
    except:
        return None

def normalize_score(value, target, upper_is_good=True):
    if value is None: return 50
    if upper_is_good:
        if value < target: return int(50 * (value / target))
        if value >= target: return min(100, int(70 + (value - target) * 30 / target))
    else:
        if value > target: return int(100 - min(100, (value - target) * 50))
        if value <= target: return 100
    return 50

# -------------- MAIN ANALYZER --------------
def analyze_fundamentals(ticker):
    print(f"🧠 Analyzing fundamentals for {ticker}...")

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        fin = tk.financials
        bal = tk.balance_sheet
        cash = tk.cashflow

        # extract values safely
        def get_val(df, row):
            if df is None or df.empty: return []
            matches = [i for i in df.index if row.lower() in i.lower()]
            if not matches: return []
            s = df.loc[matches[0]].dropna().astype(float)
            return list(s.values[::-1])

        rev = get_val(fin, "total revenue")
        ni = get_val(fin, "net income")
        eps = get_val(fin, "eps") or get_val(fin, "earnings per share")
        debt = get_val(bal, "total debt") or get_val(bal, "long term debt")
        eq = get_val(bal, "total stockholders' equity")
        ca = get_val(bal, "total current assets")
        cl = get_val(bal, "total current liabilities")
        fcf = get_val(cash, "free cash flow") or get_val(cash, "cash from operating activities")

        # derive ratios
        eps_growth = cagr(eps[0], eps[-1], len(eps)-1) * 100 if len(eps) > 1 else None
        rev_growth = cagr(rev[0], rev[-1], len(rev)-1) * 100 if len(rev) > 1 else None
        roe = (ni[-1]/eq[-1])*100 if eq and eq[-1]!=0 else None
        de = (debt[-1]/eq[-1]) if eq and eq[-1]!=0 else None
        margin = (ni[-1]/rev[-1])*100 if rev and rev[-1]!=0 else None
        fcf_trend = ((fcf[-1]/fcf[0])-1)*100 if len(fcf)>1 and fcf[0]!=0 else None
        curr_ratio = (ca[-1]/cl[-1]) if cl and cl[-1]!=0 else None

        # scoring
        scores = {
            "eps_growth": normalize_score(eps_growth, 10),
            "revenue_growth": normalize_score(rev_growth, 8),
            "roe": normalize_score(roe, 15),
            "de_ratio": normalize_score(de, 1, upper_is_good=False),
            "profit_margin": normalize_score(margin, 10),
            "fcf_trend": normalize_score(fcf_trend, 5),
            "current_ratio": normalize_score(curr_ratio, 1.2),
            "marketcap_stability": 70 if info.get("marketCap",0)>1e9 else 50,
            "insider_flow": 50
        }

        trust_score = sum(scores[k]*w for k,w in WEIGHTS.items()) / sum(WEIGHTS.values())
        if trust_score>=75: verdict="Legit / Strong"
        elif trust_score>=55: verdict="Caution / Mixed"
        else: verdict="Risky / Weak"

        result = {
            "module": "Fundamental Analyzer",
            "ticker": ticker,
            "trust_score": round(trust_score,1),
            "verdict": verdict,
            "metrics": {
                "EPS Growth %": eps_growth,
                "Revenue Growth %": rev_growth,
                "ROE %": roe,
                "Debt/Equity": de,
                "Margin %": margin,
                "FCF Trend %": fcf_trend,
                "Current Ratio": curr_ratio
            }
        }
        return result
    except Exception as e:
        print("❌ Fundamental analysis failed:", e)
        return None
