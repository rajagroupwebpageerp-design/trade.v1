"""
paper_trader.py — a fully simulated (no real orders placed anywhere) paper
trading bot.

Rule: on the 5-minute base indicators —
  - ALL of EMA(9>21), RSI, MACD, VWAP green  -> open a virtual LONG
  - ALL of them red                          -> open a virtual SHORT
  - while in a position, the INSTANT any single one of those four flips
    against it (not necessarily all four reversing) -> close the position

One open position per symbol at a time. State (open positions + a capped
closed-trade log) persists to a shared JSON file the same way the market
data cache does in server.py, so every gunicorn worker process sees the
same state regardless of which one handled a given request.
"""

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone

log = logging.getLogger("trade-scanner")

PAPER_FILE = os.path.join(tempfile.gettempdir(), "trade_scanner_paper.json")
PAPER_LOCK = threading.Lock()

MAX_CLOSED_TRADES = 200  # bounded log; oldest trades drop off past this
MAX_DAILY_SUMMARIES = 90  # keep roughly a trading quarter's worth of days

# Flip to False to go long-only (SHORT entries simply never trigger).
ALLOW_SHORTS = True

# Fixed UTC+5:30 offset — India doesn't observe DST, so this is correct
# year-round without needing the zoneinfo package. Used to group closed
# trades into calendar trading days for the daily P&L summary.
IST = timezone(timedelta(hours=5, minutes=30))


def _ist_date_str(unix_ts):
    return datetime.fromtimestamp(unix_ts, tz=IST).strftime("%Y-%m-%d")


def read_state():
    """Read the shared paper-trading state file. Safe from any worker."""
    try:
        with open(PAPER_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"positions": {}, "closedTrades": []}


def write_state(state):
    """Atomically write the shared state file (write temp, then replace)."""
    with PAPER_LOCK:
        fd, tmp_path = tempfile.mkstemp(dir=tempfile.gettempdir())
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f)
            os.replace(tmp_path, PAPER_FILE)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


def _all_green(item):
    return item["emaBullish"] and item["vwapAbove"] and item["rsi"] > 52 and item["macdBullish"]


def _all_red(item):
    return (not item["emaBullish"]) and (not item["vwapAbove"]) and item["rsi"] < 48 and (not item["macdBullish"])


def _record_daily_stats(state, trade):
    """Roll a just-closed trade into its exit day's running summary. Stored
    separately from the (capped) closed-trade log so day-level P&L survives
    even after hundreds of trades push individual trades out of that log."""
    daily = state.setdefault("dailyStats", {})
    day_key = _ist_date_str(trade["exitTime"])
    day = daily.setdefault(day_key, {"totalTrades": 0, "wins": 0, "losses": 0, "totalPnlPct": 0.0})
    day["totalTrades"] += 1
    if trade["pnlPct"] > 0:
        day["wins"] += 1
    else:
        day["losses"] += 1
    day["totalPnlPct"] = round(day["totalPnlPct"] + trade["pnlPct"], 3)

    # Trim to the most recent MAX_DAILY_SUMMARIES calendar days so this
    # dict can't grow forever over months/years of uptime.
    if len(daily) > MAX_DAILY_SUMMARIES:
        for old_key in sorted(daily.keys())[: len(daily) - MAX_DAILY_SUMMARIES]:
            del daily[old_key]


def get_daily_summaries(state=None, limit=30):
    """Most recent `limit` days' P&L summaries, newest first, each with
    win rate and avg PnL/trade derived on read."""
    if state is None:
        state = read_state()
    daily = state.get("dailyStats", {})
    out = []
    for day_key in sorted(daily.keys(), reverse=True)[:limit]:
        d = daily[day_key]
        total = d["totalTrades"]
        win_rate = round(d["wins"] / total * 100, 1) if total else 0.0
        avg_pnl = round(d["totalPnlPct"] / total, 3) if total else 0.0
        out.append({
            "date": day_key,
            "totalTrades": total,
            "wins": d["wins"],
            "losses": d["losses"],
            "winRatePct": win_rate,
            "totalPnlPct": d["totalPnlPct"],
            "avgPnlPct": avg_pnl,
        })
    return out


def process_cycle(results):
    """Call once per refresh cycle with the {symbol: item} dict just
    fetched. Mutates each item in place to add a 'paperPosition' field
    (None when flat), updates/persists open positions and the closed-trade
    log, and returns the state that was written."""
    state = read_state()
    positions = state.get("positions", {})
    closed = state.get("closedTrades", [])
    now = time.time()

    for symbol, item in results.items():
        pos = positions.get(symbol)
        ltp = item["ltp"]

        if pos is None:
            if _all_green(item):
                pos = {"side": "LONG", "entryPrice": ltp, "entryTime": now}
                positions[symbol] = pos
                log.info("PAPER BOT: opened LONG %s @ %s (all indicators green)", symbol, ltp)
            elif ALLOW_SHORTS and _all_red(item):
                pos = {"side": "SHORT", "entryPrice": ltp, "entryTime": now}
                positions[symbol] = pos
                log.info("PAPER BOT: opened SHORT %s @ %s (all indicators red)", symbol, ltp)
        else:
            exit_reason = None
            if pos["side"] == "LONG" and not _all_green(item):
                exit_reason = "an indicator turned red"
            elif pos["side"] == "SHORT" and not _all_red(item):
                exit_reason = "an indicator turned green"

            if exit_reason:
                entry_price = pos["entryPrice"]
                if pos["side"] == "LONG":
                    pnl_pct = round((ltp - entry_price) / entry_price * 100, 3)
                else:
                    pnl_pct = round((entry_price - ltp) / entry_price * 100, 3)

                closed.append({
                    "symbol": symbol,
                    "side": pos["side"],
                    "entryPrice": entry_price,
                    "exitPrice": ltp,
                    "entryTime": pos["entryTime"],
                    "exitTime": now,
                    "pnlPct": pnl_pct,
                    "reason": exit_reason,
                })
                _record_daily_stats(state, closed[-1])
                log.info(
                    "PAPER BOT: closed %s %s @ %s (entry %s) pnl=%.3f%% (%s)",
                    pos["side"], symbol, ltp, entry_price, pnl_pct, exit_reason,
                )
                del positions[symbol]
                pos = None

        if pos:
            entry_price = pos["entryPrice"]
            if pos["side"] == "LONG":
                unrealized = round((ltp - entry_price) / entry_price * 100, 3)
            else:
                unrealized = round((entry_price - ltp) / entry_price * 100, 3)
            item["paperPosition"] = {
                "side": pos["side"],
                "entryPrice": entry_price,
                "unrealizedPnlPct": unrealized,
            }
        else:
            item["paperPosition"] = None

    if len(closed) > MAX_CLOSED_TRADES:
        closed = closed[-MAX_CLOSED_TRADES:]

    state = {
        "positions": positions,
        "closedTrades": closed,
        "dailyStats": state.get("dailyStats", {}),
    }
    write_state(state)
    return state


def get_stats(state=None):
    """Summary stats over the closed-trade log."""
    if state is None:
        state = read_state()
    closed = state.get("closedTrades", [])
    total = len(closed)
    wins = sum(1 for t in closed if t["pnlPct"] > 0)
    losses = total - wins
    total_pnl = round(sum(t["pnlPct"] for t in closed), 3)
    avg_pnl = round(total_pnl / total, 3) if total else 0.0
    win_rate = round(wins / total * 100, 1) if total else 0.0
    return {
        "totalTrades": total,
        "wins": wins,
        "losses": losses,
        "winRatePct": win_rate,
        "totalPnlPct": total_pnl,
        "avgPnlPct": avg_pnl,
    }
