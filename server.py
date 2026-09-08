import logging
import sys
import threading
import time

import numpy as np
import yfinance as yf
from curl_cffi import requests as curl_requests
from flask import Flask, jsonify
from flask_cors import CORS

# ----------------------------------------------------------------------------
# Logging setup — this is the #1 fix. The old code used bare print() inside a
# try/except, but empty dataframes from yfinance don't raise exceptions, so
# nothing was ever printed. We now log at every stage so failures are visible
# in Render's Logs tab instead of silently vanishing into an empty {}.
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("trade-scanner")

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
    "TATA STEEL": {"ticker": "TATASTEEL.NS", "category": "Equities"},
}

# ----------------------------------------------------------------------------
# yfinance on cloud hosts (Render, Heroku, AWS, etc.) frequently gets blocked
# or silently returns empty dataframes because Yahoo Finance fingerprints and
# throttles datacenter IPs. curl_cffi lets us impersonate a real Chrome TLS/
# HTTP fingerprint, which is yfinance's own documented workaround for this.
# We create ONE shared session and reuse it for every ticker.
# ----------------------------------------------------------------------------
def make_session():
    return curl_requests.Session(impersonate="chrome")


SESSION = make_session()

# ----------------------------------------------------------------------------
# Server-side cache. Instead of re-fetching 18 tickers from Yahoo on every
# single browser poll (previously every 10s from the frontend -> 18 requests
# each time, which looks like scraping and invites rate-limiting), a single
# background thread refreshes the cache on its own schedule. All incoming
# /api/live-data requests just read the cache instantly, no matter how often
# clients poll.
# ----------------------------------------------------------------------------
CACHE_LOCK = threading.Lock()
CACHE = {
    "data": {},
    "last_updated": None,
    "last_error": None,
}

REFRESH_INTERVAL_SECONDS = 45   # how often we hit Yahoo for fresh data
PER_SYMBOL_DELAY_SECONDS = 0.4  # small stagger so 18 calls don't fire as a burst


def calculate_indicators(df):
    # EMA 9 and EMA 21
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()

    # RSI (14)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # VWAP
    df["VWAP"] = (
        df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3
    ).cumsum() / df["Volume"].cumsum()

    # MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


def fetch_one(name, ticker, category, session):
    """Fetch + compute indicators for a single symbol. Returns a result dict
    or None. Logs the reason on every failure path so nothing fails silently."""
    try:
        stock = yf.Ticker(ticker, session=session)
        df = stock.history(period="1d", interval="5m")

        if df.empty:
            log.warning("%s (%s): yfinance returned an EMPTY dataframe", name, ticker)
            return None

        if len(df) < 26:
            log.warning(
                "%s (%s): only %d bars available, need >= 26 for MACD/indicators",
                name, ticker, len(df),
            )
            return None

        df = calculate_indicators(df)
        latest = df.iloc[-1]

        ltp = round(float(latest["Close"]), 2)
        ema21 = round(float(latest["EMA21"]), 2)
        ema9 = round(float(latest["EMA9"]), 2)
        rsi = round(float(latest["RSI"]), 2)
        vwap = round(float(latest["VWAP"]), 2) if not np.isnan(latest["VWAP"]) else ema21
        macd = round(float(latest["MACD"]), 2)
        macd_signal = round(float(latest["MACD_Signal"]), 2)

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

        log.info("%s (%s): OK ltp=%s signal=%s", name, ticker, ltp, signal)

        return {
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
            "target": target,
        }

    except Exception as e:
        log.error("%s (%s): EXCEPTION during fetch: %s", name, ticker, e, exc_info=True)
        return None


def refresh_cache():
    """Runs in a background thread forever, refreshing CACHE every
    REFRESH_INTERVAL_SECONDS. This decouples Yahoo request volume from
    however often browsers poll the API."""
    global SESSION
    consecutive_all_empty = 0

    while True:
        start = time.time()
        results = {}
        empty_count = 0

        for name, item in SYMBOLS.items():
            result = fetch_one(name, item["ticker"], item["category"], SESSION)
            if result is not None:
                results[name] = result
            else:
                empty_count += 1
            time.sleep(PER_SYMBOL_DELAY_SECONDS)

        with CACHE_LOCK:
            if results:
                CACHE["data"] = results
                CACHE["last_error"] = None
            # If literally everything failed, keep serving the last good
            # cache instead of overwriting it with {} — better a slightly
            # stale table than a blank one.
            if empty_count == len(SYMBOLS):
                CACHE["last_error"] = "All symbols failed this cycle — likely Yahoo Finance blocking/rate-limiting this server's IP."
                log.error(CACHE["last_error"])
            CACHE["last_updated"] = time.time()

        if empty_count == len(SYMBOLS):
            consecutive_all_empty += 1
            log.error(
                "ALL %d symbols failed (%d consecutive full-failure cycles). "
                "This strongly suggests Yahoo Finance is blocking this server's IP, "
                "not a code bug. Consider a different data provider if this persists.",
                len(SYMBOLS), consecutive_all_empty,
            )
            # Recreate the session in case it's the specific TLS session that
            # got flagged.
            if consecutive_all_empty % 3 == 0:
                log.info("Recreating yfinance session after repeated full failures")
                SESSION = make_session()
        else:
            consecutive_all_empty = 0
            log.info(
                "Cycle complete: %d/%d symbols OK", len(results), len(SYMBOLS)
            )

        elapsed = time.time() - start
        sleep_for = max(REFRESH_INTERVAL_SECONDS - elapsed, 5)
        time.sleep(sleep_for)


@app.route("/api/live-data", methods=["GET"])
def get_live_data():
    with CACHE_LOCK:
        return jsonify(CACHE["data"])


@app.route("/api/status", methods=["GET"])
def get_status():
    """Diagnostic endpoint: hit this in your browser to see cache health
    without guessing from the raw data endpoint."""
    with CACHE_LOCK:
        return jsonify({
            "last_updated": CACHE["last_updated"],
            "symbols_cached": len(CACHE["data"]),
            "symbols_expected": len(SYMBOLS),
            "last_error": CACHE["last_error"],
        })


# Start the background refresh thread once, at import time, so it runs under
# both `python server.py` and a production WSGI server like gunicorn.
_refresh_thread = threading.Thread(target=refresh_cache, daemon=True)
_refresh_thread.start()

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting Multi-Asset Intraday Trading Bridge on port %d", port)
    app.run(host="0.0.0.0", port=port)
