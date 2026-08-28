from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Tracked Assets grouped by Category
SYMBOLS = {
    # INDICES
    "NIFTY 50": {"ticker": "^NSEI", "category": "Indices"},
    "NIFTY BANK": {"ticker": "^NSEBANK", "category": "Indices"},
    "SENSEX": {"ticker": "^BSESN", "category": "Indices"},
    "NIFTY MIDCAP": {"ticker": "^NSEMDCP50", "category": "Indices"},
    "INDIA VIX": {"ticker": "^INDIAVIX", "category": "Indices"},

    # COMMODITIES
    "GOLD (MCX)": {"ticker": "GOLDBEES.NS", "category": "Commodities"},
    "SILVER (MCX)": {"ticker": "SILVERBEES.NS", "category": "Commodities"},
    "CRUDE OIL": {"ticker": "CL=F", "category": "Commodities"},
    "NATURAL GAS": {"ticker": "NG=F", "category": "Commodities"},
    "COPPER": {"ticker": "HG=F", "category": "Commodities"},

    # INTRADAY STOCKS
    "RELIANCE": {"ticker": "RELIANCE.NS", "category": "Equities"},
    "TCS": {"ticker": "TCS.NS", "category": "Equities"},
    "HDFC BANK": {"ticker": "HDFCBANK.NS", "category": "Equities"},
    "INFOSYS": {"ticker": "INFY.NS", "category": "Equities"},
    "ICICI BANK": {"ticker": "ICICIBANK.NS", "category": "Equities"},
    "SBIN": {"ticker": "SBIN.NS", "category": "Equities"},
    "BHARTI AIRTEL": {"ticker": "BHARTIARTL.NS", "category": "Equities"},
    "TATA STEEL": {"ticker": "TATASTEEL.NS", "category": "Equities"}
}

def calculate_indicators(df):
    # EMA 9 and EMA 21
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()

    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # VWAP
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

    # MACD (12, 26, 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df

@app.route('/api/live-data', methods=['GET'])
def get_live_data():
    results = {}

    for name, item in SYMBOLS.items():
        ticker = item['ticker']
        category = item['category']

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1d", interval="5m")

            if not df.empty and len(df) >= 26:
                df = calculate_indicators(df)
                latest = df.iloc[-1]
                
                ltp = round(float(latest['Close']), 2)
                ema21 = round(float(latest['EMA21']), 2)
                ema9 = round(float(latest['EMA9']), 2)
                rsi = round(float(latest['RSI']), 2)
                vwap = round(float(latest['VWAP']), 2) if not np.isnan(latest['VWAP']) else ema21
                macd = round(float(latest['MACD']), 2)
                macd_signal = round(float(latest['MACD_Signal']), 2)
                
                ema_bullish = ema9 > ema21
                vwap_above = ltp > vwap
                macd_bullish = macd > macd_signal
                
                # Intraday Signal Rules
                signal = "NEUTRAL"
                entry, sl, target = 0.0, 0.0, 0.0

                if ema_bullish and vwap_above and rsi > 52 and macd_bullish:
                    signal = "BUY"
                    entry = ltp
                    sl = ema21
                    risk = entry - sl
                    target = round(entry + (risk * 2), 2) if risk > 0 else round(entry * 1.01, 2)
                elif not ema_bullish and not vwap_above and rsi < 48 and not macd_bullish:
                    signal = "SELL"
                    entry = ltp
                    sl = ema21
                    risk = sl - entry
                    target = round(entry - (risk * 2), 2) if risk > 0 else round(entry * 0.99, 2)

                results[name] = {
                    "category": category,
                    "ltp": ltp,
                    "ema21": ema21,
                    "rsi": rsi,
                    "macdBullish": macd_bullish,
                    "vwapAbove": vwap_above,
                    "emaBullish": ema_bullish,
                    "signal": signal,
                    "entry": entry,
                    "sl": sl,
                    "target": target
                }
        except Exception as e:
            print(f"Error fetching {name}: {e}")

    return jsonify(results)

if __name__ == '__main__':
    print("Starting Multi-Asset Intraday Trading Bridge on http://localhost:5000")
    app.run(port=5000, debug=True)