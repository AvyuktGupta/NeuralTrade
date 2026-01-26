"""
Indicator Module - Technical Analysis
Provides technical analysis using multi-timeframe indicators.
"""
import yfinance as yf
import pandas as pd
import ta
from APIs.indicator_calculators.indicators import (
    compute_rsi, compute_stochrsi, compute_macd, compute_bollinger_bands
)
from ta.trend import ADXIndicator, CCIIndicator
from ta.momentum import ROCIndicator, WilliamsRIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import KeltnerChannel

# Weights for different indicators
WEIGHTS = {
    "RSI": 10,
    "StochRSI": 7,
    "MACD": 15,
    "MA Crossover": 15,
    "ADX": 10,
    "Bollinger": 10,
    "Keltner": 8,
    "OBV": 8,
    "CCI": 7,
    "ROC": 5
}


def analyze_single_timeframe(ticker, interval):
    """Analyze technical indicators for one timeframe"""
    try:
        df = yf.download(ticker, period="60d", interval=interval, progress=False)
        if df.empty:
            return None

        # Flatten to 1D Series (fix for pandas/ta error)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.squeeze()

        # Compute indicators
        rsi = ta.momentum.RSIIndicator(close).rsi()
        macd = ta.trend.MACD(close)
        stoch = ta.momentum.StochRSIIndicator(close)
        bb = ta.volatility.BollingerBands(close)

        buy, sell = 0, 0

        # RSI
        if rsi.iloc[-1] < 30:
            buy += 1
        elif rsi.iloc[-1] > 70:
            sell += 1

        # MACD
        if macd.macd_diff().iloc[-1] > 0:
            buy += 1
        else:
            sell += 1

        # StochRSI
        if stoch.stochrsi_k().iloc[-1] < 0.2:
            buy += 1
        elif stoch.stochrsi_k().iloc[-1] > 0.8:
            sell += 1

        # Bollinger Bands
        if close.iloc[-1] < bb.bollinger_lband().iloc[-1]:
            buy += 1
        elif close.iloc[-1] > bb.bollinger_hband().iloc[-1]:
            sell += 1

        confidence = round((abs(buy - sell) / 4) * 100, 1)
        signal = "BUY" if buy > sell else "SELL" if sell > buy else "HOLD"

        return {
            "timeframe": interval,
            "signal": signal,
            "confidence": confidence
        }

    except Exception as e:
        print(f"⚠️ Error in {ticker} {interval}: {e}")
        return None


def analyze_technical(ticker):
    """Analyze across all timeframes and fuse results"""
    TIMEFRAMES = ["5m", "15m", "1h", "1d"]
    TIMEFRAME_WEIGHTS = {
        "5m": 0.15,
        "15m": 0.25,
        "1h": 0.3,
        "1d": 0.3
    }

    results = []
    total_weight = 0
    weighted_sum = 0

    # analyze each timeframe
    for tf in TIMEFRAMES:
        res = analyze_single_timeframe(ticker, tf)
        if not res:
            continue
        results.append(res)

        # convert signal to numeric
        val = 0
        if res["signal"] == "BUY":
            val = 1
        elif res["signal"] == "SELL":
            val = -1
        else:
            val = 0

        # apply weight
        w = TIMEFRAME_WEIGHTS.get(tf, 0.25)
        weighted_sum += val * w * res["confidence"]
        total_weight += w * 100

    if total_weight == 0:
        return {
            "module": "Technical Analyzer",
            "ticker": ticker,
            "signal": "NO DATA",
            "confidence": 0,
            "details": []
        }

    final_score = weighted_sum / total_weight * 100  # normalized -100 to 100 range
    final_conf = abs(final_score)
    signal = "BUY" if final_score > 15 else "SELL" if final_score < -15 else "HOLD"

    # summary string
    summary_lines = [
        f"{r['timeframe']:>4}: {r['signal']:<4} ({r['confidence']}%)" for r in results
    ]
    summary = " | ".join(summary_lines)

    return {
        "module": "Technical Analyzer",
        "ticker": ticker,
        "signal": signal,
        "confidence": round(final_conf, 1),
        "details": results
    }


def run_indicator_module(input_data: dict) -> dict:
    """
    Main entry point for indicator module.
    
    Args:
        input_data: Dictionary containing:
            - ticker: Stock ticker symbol (e.g., "TVSMOTOR.NS")
            - interval: Optional, default "15m"
            - period: Optional, default "60d"
    
    Returns:
        dict: Technical analysis results with signal, confidence, etc.
    """
    ticker = input_data.get("ticker")
    if not ticker:
        raise ValueError("ticker is required in input_data")
    
    # Use the multi-timeframe analysis
    result = analyze_technical(ticker)
    
    return result
