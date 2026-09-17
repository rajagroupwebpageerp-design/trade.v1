import json
import logging
import os
import sys
import tempfile
import threading
import time

import numpy as np
import yfinance as yf
from curl_cffi import requests as curl_requests
from flask import Flask, jsonify
from flask_cors import CORS

from smc import compute_smc_all_timeframes

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
#
# IMPORTANT: gunicorn can run multiple worker PROCESSES, each with its own
# separate Python memory. An in-memory-only cache means each worker has its
# own independent copy — one worker's background thread can be happily
# fetching and logging success, while a request that happens to land on a
# *different* worker sees a permanently empty cache. That was exactly the
# bug: logs showed successful cycles, but /api/live-data kept returning {}.
#
# Fix: persist the cache to a JSON file on disk (shared filesystem within
# the same container/instance). Every worker's background thread writes to
# this file, and every request reads straight from the file — so it doesn't
# matter which worker handles which request, they all see the same data.
# Writes are atomic (write to a temp file, then os.replace) so a reader
# never sees a half-written file.
# ----------------------------------------------------------------------------
CACHE_FILE = os.path.join(tempfile.gettempdir(), "trade_scanner_cache.json")
CACHE_WRITE_LOCK = threading.Lock()  # only guards this process's own writes

REFRESH_INTERVAL_SECONDS = 45   # how often we hit Yahoo for fresh data
PER_SYMBOL_DELAY_SECONDS = 0.4  # small stagger so 18 calls don't fire as a burst

# 5-minute bars are the base timeframe for everything, including SMC: higher
# timeframes (15m/1h/4h/1d) are built by resampling this same dataframe, so
# we only ever make ONE Yahoo request per symbol per cycle regardless of how
# many timeframes we analyze. 60d is Yahoo's maximum lookback for 5m bars,
# and gives enough daily candles for the 1D SMC view to be meaningful.
HISTORY_PERIOD = "60d"


def read_cache():
    """Read the shared cache file. Safe to call from any worker process."""
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"data": {}, "last_updated": None, "last_error": None}


def write_cache(data=None, last_error=None):
    """Atomically write the shared cache file so readers never see a
    half-written/corrupt file mid-write."""
    with CACHE_WRITE_LOCK:
        current = read_cache()
        if data is not None:
            current["data"] = data
        if last_error is not None or data is not None:
            current["last_error"] = last_error
        current["last_updated"] = time.time()

        fd, tmp_path = tempfile.mkstemp(dir=tempfile.gettempdir())
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(current, f)
            os.replace(tmp_path, CACHE_FILE)  # atomic on POSIX
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


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

    # VWAP must reset every trading day — it's meaningless accumulated across
    # multiple days. Now that fetch_one() pulls several days of history (for
    # EMA/RSI/MACD warm-up), we group by calendar date so each day gets its
    # own independent cumulative VWAP, exactly like the backtest does.
    df["_date"] = df.index.date

    def _day_vwap(g):
        typical = (g["High"] + g["Low"] + g["Close"]) / 3
        return (g["Volume"] * typical).cumsum() / g["Volume"].cumsum()

    df["VWAP"] = df.groupby("_date", group_keys=False).apply(_day_vwap)
    df.drop(columns=["_date"], inplace=True)

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
        # Pull several weeks of 5-minute bars, not just "today". EMA/RSI/MACD
        # need ~26 bars of warm-up; at 5-minute bars that's ~2+ hours, so a
        # period="1d" fetch has NO valid signal for the first couple of
        # hours after market open every single day. Pulling prior days'
        # bars lets the indicators warm up using yesterday's data, so
        # today's very first bars already have valid EMA/RSI/MACD values.
        # It also gives the SMC (OB/FVG/QML) detection enough bars to build
        # meaningful 15m/1h/4h/1d structure via resampling, without any
        # extra Yahoo requests. VWAP is unaffected — calculate_indicators()
        # resets it per calendar day regardless of how many days we pass in.
        df = stock.history(period=HISTORY_PERIOD, interval="5m")

        if df.empty:
            log.warning("%s (%s): yfinance returned an EMPTY dataframe (hard failure)", name, ticker)
            return None, "hard_failure"

        if len(df) < 26:
            # With a 60-day window this should be rare (e.g. a symbol with a
            # very short trading history, or a market holiday gap). This is
            # NOT the same as Yahoo blocking us, so it's tracked separately.
            log.warning(
                "%s (%s): only %d bars available even with %s window, need >= 26",
                name, ticker, len(df), HISTORY_PERIOD,
            )
            return None, "insufficient_data"

        # Run SMC detection on the raw, unmodified OHLCV bars BEFORE
        # calculate_indicators() adds its own columns — SMC only needs
        # Open/High/Low/Close/Volume, and computing it first keeps the two
        # concerns cleanly separated.
        try:
            smc_data = compute_smc_all_timeframes(df, symbol_name=name)
        except Exception as e:
            log.warning("%s (%s): SMC computation failed entirely: %s", name, ticker, e)
            smc_data = {}

        df = calculate_indicators(df)
        latest = df.iloc[-1]

        ltp = round(float(latest["Close"]), 2)
        ema21 = round(float(latest["EMA21"]), 2)
        ema9 = round(float(latest["EMA9"]), 2)
        rsi = round(float(latest["RSI"]), 2)
        vwap = round(float(latest["VWAP"]), 2) if not np.isnan(latest["VWAP"]) else ema21
        macd = round(float(latest["MACD"]), 2)
        macd_signal = round(float(latest["MACD_Signal"]), 2)

        if any(np.isnan(x) for x in [ema21, ema9, rsi, macd, macd_signal]):
            log.warning("%s (%s): latest bar still has NaN indicator(s), skipping this cycle", name, ticker)
            return None, "insufficient_data"

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
            "smc": smc_data,
        }, None

    except Exception as e:
        log.error("%s (%s): EXCEPTION during fetch: %s", name, ticker, e, exc_info=True)
        return None, "hard_failure"


def refresh_cache():
    """Runs in a background thread forever, refreshing the shared cache file
    every REFRESH_INTERVAL_SECONDS. This decouples Yahoo request volume from
    however often browsers poll the API."""
    global SESSION
    consecutive_all_hard_failed = 0
    pid = os.getpid()
    log.info("Background refresh thread started in worker pid=%d", pid)

    while True:
        start = time.time()
        results = {}
        hard_failures = 0
        insufficient_data = 0

        for name, item in SYMBOLS.items():
            result, fail_type = fetch_one(name, item["ticker"], item["category"], SESSION)
            if result is not None:
                results[name] = result
            elif fail_type == "hard_failure":
                hard_failures += 1
            elif fail_type == "insufficient_data":
                insufficient_data += 1
            time.sleep(PER_SYMBOL_DELAY_SECONDS)

        total = len(SYMBOLS)

        if results:
            # Keep serving the last good cache if this cycle produced
            # nothing new — better a slightly stale table than a blank one.
            write_cache(data=results, last_error=None)

        if hard_failures == total:
            # Every symbol had a genuine fetch failure (empty df / exception)
            # — THIS is the real signature of Yahoo blocking/rate-limiting,
            # not "not enough bars yet" which is expected early in the day.
            err = "All symbols had genuine fetch failures this cycle — likely Yahoo Finance blocking/rate-limiting this server's IP."
            write_cache(data=None, last_error=err)
            log.error("[pid=%d] %s", pid, err)
            consecutive_all_hard_failed += 1
            if consecutive_all_hard_failed % 3 == 0:
                log.info("[pid=%d] Recreating yfinance session after repeated hard failures", pid)
                SESSION = make_session()
        elif insufficient_data == total:
            # Expected in the first ~2 hours after market open before EMA/
            # RSI/MACD have enough bars to warm up. Not an error condition.
            consecutive_all_hard_failed = 0
            log.info(
                "[pid=%d] All %d symbols still warming up (insufficient bars) — "
                "normal in the first couple hours after market open.",
                pid, total,
            )
        else:
            consecutive_all_hard_failed = 0
            log.info(
                "[pid=%d] Cycle complete: %d/%d OK, %d insufficient data, %d hard failures",
                pid, len(results), total, insufficient_data, hard_failures,
            )

        try:
            os.utime(LOCK_FILE, None)  # prove this worker is still alive
        except FileNotFoundError:
            pass

        elapsed = time.time() - start
        sleep_for = max(REFRESH_INTERVAL_SECONDS - elapsed, 5)
        time.sleep(sleep_for)


@app.route("/api/live-data", methods=["GET"])
def get_live_data():
    cache = read_cache()
    return jsonify(cache["data"])


@app.route("/api/status", methods=["GET"])
def get_status():
    """Diagnostic endpoint: hit this in your browser to see cache health
    without guessing from the raw data endpoint."""
    cache = read_cache()
    return jsonify({
        "last_updated": cache["last_updated"],
        "symbols_cached": len(cache["data"]),
        "symbols_expected": len(SYMBOLS),
        "last_error": cache["last_error"],
        "served_by_pid": os.getpid(),
    })


# ----------------------------------------------------------------------------
# Only ONE worker process should run the background fetch loop — otherwise
# every worker hits Yahoo independently, multiplying request volume and
# raising the odds of getting rate-limited again. We use a simple lock FILE
# (not the cache file) as a cross-process mutex: whichever worker process
# creates it first "wins" and runs the loop; the rest skip starting their
# own thread and just serve reads from the shared cache file.
# ----------------------------------------------------------------------------
LOCK_FILE = os.path.join(tempfile.gettempdir(), "trade_scanner_refresh.lock")
LOCK_STALE_SECONDS = REFRESH_INTERVAL_SECONDS * 4  # if not refreshed in this long, assume the owner died


def try_become_refresher():
    # If a lock file exists but hasn't been touched in a while, its owner
    # likely crashed/restarted without cleaning up — reclaim it so the cache
    # doesn't stay frozen forever.
    try:
        age = time.time() - os.path.getmtime(LOCK_FILE)
        if age > LOCK_STALE_SECONDS:
            log.warning("Refresh lock is stale (%.0fs old) — reclaiming it", age)
            os.remove(LOCK_FILE)
    except FileNotFoundError:
        pass

    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return True
    except FileExistsError:
        return False


def refresh_loop_with_lock():
    """Wraps refresh_cache so it periodically re-touches the lock file
    (proves this worker is still alive) and releases the lock if it ever
    exits, so another worker can take over."""
    try:
        refresh_cache()
    finally:
        try:
            os.remove(LOCK_FILE)
        except FileNotFoundError:
            pass


if try_become_refresher():
    _refresh_thread = threading.Thread(target=refresh_loop_with_lock, daemon=True)
    _refresh_thread.start()
else:
    log.info("Another worker already owns the refresh loop; pid=%d will only serve reads", os.getpid())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting Multi-Asset Intraday Trading Bridge on port %d", port)
    app.run(host="0.0.0.0", port=port)
