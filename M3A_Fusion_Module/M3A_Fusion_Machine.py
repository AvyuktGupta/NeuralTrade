# M3A_Fusion_Machine.py
"""
M³A Fusion Machine (v1.0)
Author: Avyukt Gupta

Combines multiple stock analysis AIs (technical + fundamental + more)
into a unified market intelligence system.
"""

import traceback
from fundamental_module import analyze_fundamentals
from technical_module import analyze_technical
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------
# 🔧 CONFIG
FUSION_WEIGHTS = {
    "technical": 0.35,
    "fundamental": 0.35,
    "sentiment": 0.15,
    "risk": 0.15
}

STOCKS = ["TVSMOTOR.NS", "TCS.NS", "RELIANCE.NS"]

# -----------------------------------------------------
# 🧠 Fusion Logic
def fuse_models(ticker):
    fundamental = analyze_fundamentals(ticker)
    technical = analyze_technical(ticker)

    # Placeholder modules
    sentiment = {"verdict": "Neutral", "score": 50}
    risk = {"verdict": "Moderate", "score": 60}

    # Extract scores
    f_score = fundamental["trust_score"] if fundamental else 50
    t_conf = technical["confidence"] if technical else 50
    s_score = sentiment["score"]
    r_score = risk["score"]

    # Weighted average
    fusion_score = (
        FUSION_WEIGHTS["fundamental"] * f_score
        + FUSION_WEIGHTS["technical"] * t_conf
        + FUSION_WEIGHTS["sentiment"] * s_score
        + FUSION_WEIGHTS["risk"] * r_score
    )

    # Final verdict
    if fusion_score >= 75:
        verdict = "BUY ✅"
    elif fusion_score >= 55:
        verdict = "HOLD ⚖️"
    else:
        verdict = "SELL 🔻"

    # Determine short sentiment labels
    fund_label = "Strong" if f_score >= 75 else "Mixed" if f_score >= 55 else "Weak"
    tech_label = technical["signal"] if technical else "N/A"

    # Clean single-line summary
    print(f"{ticker:<12} — {verdict:<6} ({fusion_score:.1f}/100) | "
          f"Fundamental: {fund_label:<6} | Technical: {tech_label:<4} | "
          f"Conf: {t_conf:.0f}%")
    
    # Prepare detailed report for Telegram/logging
    detailed_report = (
    f"📊 {ticker}\n"
    f"Fundamentals: {fundamental['verdict']} ({f_score:.1f})\n"
    f"Technicals: {technical['signal']} ({t_conf:.1f})\n"
    f"Sentiment: {sentiment['verdict']} ({s_score})\n"
    f"Risk: {risk['verdict']} ({r_score})\n"
    f"Final: {verdict} — Score: {fusion_score:.1f}/100"
    )
# (You can send this string to Telegram or save to a log file)


    # Return structured data for summary aggregation
    return {"ticker": ticker, "fusion_score": fusion_score, "verdict": verdict}


# -----------------------------------------------------
# 🚀 MAIN EXECUTION
if __name__ == "__main__":
    print("🚀 M³A Fusion Machine Started\n")
    all_reports = []

    for stock in STOCKS:
        report = fuse_models(stock)
        if report:
            all_reports.append(report)

    # Final summary
    buy_count = sum(1 for r in all_reports if "BUY" in r["verdict"])
    hold_count = sum(1 for r in all_reports if "HOLD" in r["verdict"])
    sell_count = sum(1 for r in all_reports if "SELL" in r["verdict"])

    print(f"\n📊 Summary: {buy_count} BUY, {hold_count} HOLD, {sell_count} SELL")
