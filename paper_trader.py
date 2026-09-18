"""
paper_trader.py — simulated (no real orders) paper trading bot running ONE
strategy: "precision" — weighted multi-timeframe scoring across 7
conditions (EMA, RSI, MACD, VWAP, Order Block, FVG, QML), each counted
independently per timeframe:

  - 1H  (macro bias):           >= 5 of 7 conditions green
  - 15M (intermediate momentum): >= 6 of 7 conditions green
  - 5M  (execution trigger):     exactly 7 of 7 conditions green

All three timeframe thresholds must be true AT THE SAME TIME to open a
position (LONG if the bullish side qualifies, SHORT if the bearish side
does). Every entry is logged as a real-time alert line in the server logs.

Exit — whichever happens first:
  (a) price hits the fixed 5/10 point stop-loss/target (1:2 reward:risk), or
  (b) the SAME 3-timeframe confluence that justified entry breaks: checked
      every cycle, exits the instant ANY ONE of 5M/15M/1H drops below its
      required threshold on the position's side — doesn't wait for all
      three to fail, and doesn't wait for price to move either.

State (open positions, closed-trade log, daily IST P&L summary) persists
to a shared JSON file (same atomic-write pattern as the market data
cache), so every gunicorn worker process sees the same state.
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

MAX_CLOSED_TRADES = 200
MAX_DAILY_SUMMARIES = 90

# Set to False to go long-only (SHORT entries simply never trigger).
ALLOW_SHORTS = True

# Fixed price-POINTS (not %, not zone-based) stop-loss/target — 1:2
# reward:risk as specified. NOTE: a flat point value behaves very
# differently across assets at very different price scales (e.g. 5 points
# is enormous for Natural Gas at ~₹2.91, tiny for an index in the tens of
# thousands) — revisit if that turns out to be a problem in practice.
FIXED_SL_POINTS = 5
FIXED_TARGET_POINTS = 10

# Per-timeframe minimum green-condition counts (out of 7).
PRECISION_MIN_1H = 5    # macro bias: at most 2 of 7 may fail
PRECISION_MIN_15M = 6   # intermediate momentum: at most 1 of 7 may fail
PRECISION_MIN_5M = 7    # execution trigger: exactly all 7 must be green

STRATEGIES = ["precision"]

# Fixed UTC+5:30 — India doesn't observe DST, so this is correct year-round
# without needing the zoneinfo package. Used to group closed trades into
# calendar trading days for the daily P&L summary.
IST = timezone(timedelta(hours=5, minutes=30))


def _ist_date_str(unix_ts):
    return datetime.fromtimestamp(unix_ts, tz=IST).strftime("%Y-%m-%d")


def _empty_strategy_state():
    return {"positions": {}, "closedTrades": [], "dailyStats": {}}


def read_state():
    """Read the shared paper-trading state file. Safe from any worker."""
    try:
        with open(PAPER_FILE, "r") as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    strategies = state.get("strategies", {})
    for key in STRATEGIES:
        strategies.setdefault(key, _empty_strategy_state())
    state["strategies"] = strategies
    return state


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


def _record_daily_stats(strat_state, trade):
    daily = strat_state.setdefault("dailyStats", {})
    day_key = _ist_date_str(trade["exitTime"])
    day = daily.setdefault(day_key, {"totalTrades": 0, "wins": 0, "losses": 0, "totalPnlPct": 0.0})
    day["totalTrades"] += 1
    if trade["pnlPct"] > 0:
        day["wins"] += 1
    else:
        day["losses"] += 1
    day["totalPnlPct"] = round(day["totalPnlPct"] + trade["pnlPct"], 3)

    if len(daily) > MAX_DAILY_SUMMARIES:
        for old_key in sorted(daily.keys())[: len(daily) - MAX_DAILY_SUMMARIES]:
            del daily[old_key]


def _close_trade(strat_state, symbol, pos, exit_price, reason, now):
    entry_price = pos["entryPrice"]
    if pos["side"] == "LONG":
        pnl_pct = round((exit_price - entry_price) / entry_price * 100, 3)
    else:
        pnl_pct = round((entry_price - exit_price) / entry_price * 100, 3)

    trade = {
        "symbol": symbol,
        "side": pos["side"],
        "entryPrice": entry_price,
        "exitPrice": exit_price,
        "entryTime": pos["entryTime"],
        "exitTime": now,
        "pnlPct": pnl_pct,
        "reason": reason,
    }
    strat_state["closedTrades"].append(trade)
    _record_daily_stats(strat_state, trade)
    log.info(
        "PAPER BOT [precision]: closed %s %s @ %s (entry %s) pnl=%.3f%% (%s)",
        pos["side"], symbol, exit_price, entry_price, pnl_pct, reason,
    )
    return trade


def _position_snapshot(pos, ltp):
    if not pos:
        return None
    entry_price = pos["entryPrice"]
    if pos["side"] == "LONG":
        unrealized = round((ltp - entry_price) / entry_price * 100, 3)
    else:
        unrealized = round((entry_price - ltp) / entry_price * 100, 3)
    return {"side": pos["side"], "entryPrice": entry_price, "unrealizedPnlPct": unrealized}


def _has_unmitigated(zones, zone_type):
    return any(z["type"] == zone_type and not z["mitigated"] for z in (zones or []))


def _seven_flags_for_tf(item, tf, side):
    """The 7 conditions (EMA, RSI, MACD, VWAP, OB, FVG, QML) as booleans
    for ONE specific timeframe ('5m', '15m', or '1h'), matching `side`
    ('LONG' expects every condition bullish, 'SHORT' expects every
    condition bearish). Any indicator that hasn't warmed up yet on that
    timeframe (None) counts as a miss rather than raising — an incomplete
    timeframe just fails to qualify, it never crashes the strategy."""
    want_bullish = side == "LONG"

    ind = ((item.get("indicatorFlags") or {}).get(tf)) or {}
    ema_bullish = ind.get("emaBullish")
    macd_bullish = ind.get("macdBullish")
    vwap_above = ind.get("vwapAbove")
    rsi_val = ind.get("rsi")

    ema_ok = bool(ema_bullish) if want_bullish else (ema_bullish is False)
    macd_ok = bool(macd_bullish) if want_bullish else (macd_bullish is False)
    vwap_ok = bool(vwap_above) if want_bullish else (vwap_above is False)
    rsi_ok = (rsi_val is not None) and ((rsi_val > 52) if want_bullish else (rsi_val < 48))

    smc_tf = ((item.get("smc") or {}).get(tf)) or {}
    obs = smc_tf.get("orderBlocks", [])
    fvgs = smc_tf.get("fvg", [])
    qml_b = smc_tf.get("qmlBullish")
    qml_s = smc_tf.get("qmlBearish")
    zone_type = "bullish" if want_bullish else "bearish"

    ob_ok = _has_unmitigated(obs, zone_type)
    fvg_ok = _has_unmitigated(fvgs, zone_type)
    qml_ok = bool(qml_b and qml_b.get("status") == "confirmed") if want_bullish else bool(qml_s and qml_s.get("status") == "confirmed")

    return [ema_ok, rsi_ok, macd_ok, vwap_ok, ob_ok, fvg_ok, qml_ok]


def _run_precision_strategy(strat_state, symbol, item, now):
    """Entry: 1H >= 5/7, 15M >= 6/7, 5M == 7/7 (same side) all at once.
    Exit: whichever comes first —
      (a) price hits the fixed 5/10 point stop-loss/target, or
      (b) the SAME 3-timeframe confluence that justified entry breaks:
          re-checked every cycle, exits the instant ANY ONE of 5M, 15M,
          or 1H drops below its required threshold on the position's side."""
    positions = strat_state["positions"]
    pos = positions.get(symbol)
    ltp = item["ltp"]

    if pos is None:
        sides = ["LONG", "SHORT"] if ALLOW_SHORTS else ["LONG"]
        for side in sides:
            count_1h = sum(_seven_flags_for_tf(item, "1h", side))
            count_15m = sum(_seven_flags_for_tf(item, "15m", side))
            count_5m = sum(_seven_flags_for_tf(item, "5m", side))

            if count_1h >= PRECISION_MIN_1H and count_15m >= PRECISION_MIN_15M and count_5m >= PRECISION_MIN_5M:
                if side == "LONG":
                    positions[symbol] = {
                        "side": "LONG", "entryPrice": ltp, "entryTime": now,
                        "stopLoss": round(ltp - FIXED_SL_POINTS, 2),
                        "target": round(ltp + FIXED_TARGET_POINTS, 2),
                    }
                else:
                    positions[symbol] = {
                        "side": "SHORT", "entryPrice": ltp, "entryTime": now,
                        "stopLoss": round(ltp + FIXED_SL_POINTS, 2),
                        "target": round(ltp - FIXED_TARGET_POINTS, 2),
                    }
                log.info(
                    "PAPER BOT [precision]: ENTRY ALERT %s %s @ %s (1H=%d/7 15M=%d/7 5M=%d/7)",
                    side, symbol, ltp, count_1h, count_15m, count_5m,
                )
                break  # only one side can open per cycle
    else:
        exit_reason = None
        if pos["side"] == "LONG":
            if ltp <= pos["stopLoss"]:
                exit_reason = "hit stop loss"
            elif ltp >= pos["target"]:
                exit_reason = "hit target"
        else:
            if ltp >= pos["stopLoss"]:
                exit_reason = "hit stop loss"
            elif ltp <= pos["target"]:
                exit_reason = "hit target"

        if exit_reason is None:
            # Re-check the SAME 3-timeframe confluence that justified
            # entry. Exit the instant ANY ONE of the three falls below its
            # required threshold on this position's side — checked every
            # cycle, independent of whether price has moved yet.
            count_1h = sum(_seven_flags_for_tf(item, "1h", pos["side"]))
            count_15m = sum(_seven_flags_for_tf(item, "15m", pos["side"]))
            count_5m = sum(_seven_flags_for_tf(item, "5m", pos["side"]))

            if count_1h < PRECISION_MIN_1H:
                exit_reason = f"1H confluence dropped ({count_1h}/7, needs >={PRECISION_MIN_1H})"
            elif count_15m < PRECISION_MIN_15M:
                exit_reason = f"15M confluence dropped ({count_15m}/7, needs >={PRECISION_MIN_15M})"
            elif count_5m < PRECISION_MIN_5M:
                exit_reason = f"5M confluence dropped ({count_5m}/7, needs {PRECISION_MIN_5M})"

        if exit_reason:
            _close_trade(strat_state, symbol, pos, ltp, exit_reason, now)
            del positions[symbol]

    return _position_snapshot(positions.get(symbol), ltp)


STRATEGY_RUNNERS = {
    "precision": _run_precision_strategy,
}


def _log_diagnostics(results):
    """One line per cycle showing how many symbols currently satisfy each
    INDIVIDUAL 5M condition — lets you see in the Render logs whether the
    bottleneck is the base indicators, the SMC zones, or just needing time
    for all three timeframes to line up at once."""
    counts = {"all7_bull_5m": 0, "all7_bear_5m": 0}
    for item in results.values():
        if sum(_seven_flags_for_tf(item, "5m", "LONG")) == 7:
            counts["all7_bull_5m"] += 1
        if sum(_seven_flags_for_tf(item, "5m", "SHORT")) == 7:
            counts["all7_bear_5m"] += 1

    log.info(
        "PAPER BOT DIAGNOSTICS (of %d symbols): 5M-all-7-bullish=%d 5M-all-7-bearish=%d "
        "(these still need 15M>=6/7 and 1H>=5/7 on the same side to actually enter)",
        len(results), counts["all7_bull_5m"], counts["all7_bear_5m"],
    )


def process_cycle(results):
    """Run the precision strategy over this cycle's {symbol: item} results.
    Mutates each item to add item['paperPositions'] = {'precision': ...}
    (None when flat). Persists state in one write. A failure on one symbol
    is logged and just leaves that symbol's position untouched this cycle."""
    state = read_state()
    now = time.time()

    try:
        _log_diagnostics(results)
    except Exception as e:
        log.warning("Paper bot diagnostics failed: %s", e)

    for symbol, item in results.items():
        snapshots = {}
        for key, runner in STRATEGY_RUNNERS.items():
            try:
                snapshots[key] = runner(state["strategies"][key], symbol, item, now)
            except Exception as e:
                log.warning("Paper bot [%s] failed on %s: %s", key, symbol, e)
                snapshots[key] = None
        item["paperPositions"] = snapshots

    for key in STRATEGIES:
        closed = state["strategies"][key]["closedTrades"]
        if len(closed) > MAX_CLOSED_TRADES:
            state["strategies"][key]["closedTrades"] = closed[-MAX_CLOSED_TRADES:]

    write_state(state)
    return state


def get_stats(strat_state):
    """Summary stats over the closed-trade log."""
    closed = strat_state.get("closedTrades", [])
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


def get_daily_summaries(strat_state, limit=30):
    """Most recent `limit` days' P&L, newest first."""
    daily = strat_state.get("dailyStats", {})
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


def get_full_report(state=None):
    """{"precision": {openPositions, closedTrades, stats, dailySummaries}}"""
    if state is None:
        state = read_state()
    report = {}
    for key in STRATEGIES:
        strat_state = state["strategies"][key]
        report[key] = {
            "openPositions": strat_state.get("positions", {}),
            "closedTrades": strat_state.get("closedTrades", [])[-50:],
            "stats": get_stats(strat_state),
            "dailySummaries": get_daily_summaries(strat_state, limit=30),
        }
    return report
