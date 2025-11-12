import yfinance as yf
import pandas as pd
import ta
import requests
import time
from indicators import compute_rsi, compute_stochrsi, compute_macd, compute_bollinger_bands

from ta.trend import ADXIndicator, CCIIndicator
from ta.momentum import ROCIndicator, WilliamsRIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import KeltnerChannel


# Telegram bot
bot_token = "7914157083:AAGCCCHsdF0RQJGF2ezNnxN0mYszRQXfNOg"
chat_id = "5678086807"
text = "Test message from StockBot!"

url = f"https://api.telegram.org/bot7322733470:AAHJ8dFnAVjaqXkwjEweIEjSXnP7GCOO9Cc/sendMessage"
payload = {"chat_id": chat_id, "text": text}

r = requests.post(url, data=payload)
print(r.json())

# Stock symbols
symbols = ["TVSMOTOR.NS"]
interval = "15m"
period = "60d"

# Weights 
weights = {
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
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("✅ Message sent to Telegram!")
    else:
        print("❌ Failed to send message:", response.text)

def run_stock_analysis():
    all_messages = []

    for symbol in symbols:
        print(f"\n🔍 Analyzing: {symbol}")
        df = yf.download(symbol, interval=interval, period=period)

        if df.empty:
            print(f"⚠️ No data for {symbol}")
            continue

        df = df.dropna()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()


        # Compute indicators
        rsi = compute_rsi(close)
        stoch_k, stoch_d = compute_stochrsi(close)
        ma50 = close.rolling(window=50).mean()
        ma200 = close.rolling(window=200).mean()
        macd_line, macd_signal, _ = compute_macd(close)
        bb_upper, _, bb_lower = compute_bollinger_bands(close)
        adx = ADXIndicator(high, low, close).adx()
        cci = CCIIndicator(high, low, close).cci()
        roc = ROCIndicator(close).roc()
        williams = WilliamsRIndicator(high, low, close).williams_r()
        obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        keltner = KeltnerChannel(high, low, close)

        buy_signals = []
        sell_signals = []
        confidence = 0

        # --- Signal ---
        if stoch_k.iloc[-1] < 20 and stoch_k.iloc[-1] > stoch_d.iloc[-1]:
            buy_signals.append("StochRSI")
            confidence += weights["StochRSI"]
        elif stoch_k.iloc[-1] > 80 and stoch_k.iloc[-1] < stoch_d.iloc[-1]:
            sell_signals.append("StochRSI")
            confidence += weights["StochRSI"]

        if rsi.iloc[-1] < 30:
            buy_signals.append("RSI")
            confidence += weights["RSI"]
        elif rsi.iloc[-1] > 70:
            sell_signals.append("RSI")
            confidence += weights["RSI"]

        if ma50.iloc[-1] > ma200.iloc[-1]:
            buy_signals.append("MA Crossover")
            confidence += weights["MA Crossover"]
        elif ma50.iloc[-1] < ma200.iloc[-1]:
            sell_signals.append("MA Crossover")
            confidence += weights["MA Crossover"]

        if macd_line.iloc[-1] > macd_signal.iloc[-1]:
            buy_signals.append("MACD")
            confidence += weights["MACD"]
        elif macd_line.iloc[-1] < macd_signal.iloc[-1]:
            sell_signals.append("MACD")
            confidence += weights["MACD"]

        if close.iloc[-1] < bb_lower.iloc[-1]:
            buy_signals.append("Bollinger")
            confidence += weights["Bollinger"]
        elif close.iloc[-1] > bb_upper.iloc[-1]:
            sell_signals.append("Bollinger")
            confidence += weights["Bollinger"]

        if adx.iloc[-1] > 25:
            buy_signals.append("ADX")
            confidence += weights["ADX"]

        if cci.iloc[-1] > 100:
            buy_signals.append("CCI")
            confidence += weights["CCI"]
        elif cci.iloc[-1] < -100:
            sell_signals.append("CCI")
            confidence += weights["CCI"]

        if roc.iloc[-1] > 0:
            buy_signals.append("ROC")
            confidence += weights["ROC"]
        elif roc.iloc[-1] < 0:
            sell_signals.append("ROC")
            confidence += weights["ROC"]

        if williams.iloc[-1] < -80:
            buy_signals.append("Williams %R")
            confidence += weights["Williams %R"]
        elif williams.iloc[-1] > -20:
            sell_signals.append("Williams %R")
            confidence += weights["Williams %R"]

        if obv.iloc[-1] > obv.iloc[-2]:
            buy_signals.append("OBV")
            confidence += weights["OBV"]
        elif obv.iloc[-1] < obv.iloc[-2]:
            sell_signals.append("OBV")
            confidence += weights["OBV"]

        kc_upper = keltner.keltner_channel_hband().iloc[-1]
        kc_lower = keltner.keltner_channel_lband().iloc[-1]
        if close.iloc[-1] < kc_lower:
            buy_signals.append("Keltner")
            confidence += weights["Keltner"]
        elif close.iloc[-1] > kc_upper:
            sell_signals.append("Keltner")
            confidence += weights["Keltner"]

        # --- Final Output ---
        if confidence >= 60 and len(buy_signals) > len(sell_signals):
            decision = f"📈 *{symbol}* — BUY ({', '.join(buy_signals)})\nConfidence: {confidence}% 🚀"
        elif confidence >= 60 and len(sell_signals) > len(buy_signals):
            decision = f"🔻 *{symbol}* — SELL ({', '.join(sell_signals)})\nConfidence: {confidence}% ⚠️"
        else:
            decision = f"⏳ *{symbol}* — Hold / No Clear Signal ({', '.join(buy_signals + sell_signals)})\nConfidence: {confidence}%"

        print(decision)
        all_messages.append(decision)

    #  Send Telegram Msg
    if all_messages:
        final_msg = "📊 *Stock Summary Update*\n\n" + "\n\n".join(all_messages)
        send_telegram_alert(final_msg)

#  Auto run every 1 minute
print("📡 Stock bot started... polling every 15 minutes.")
while True:
    run_stock_analysis()
    time.sleep(60)  # 1 minute
