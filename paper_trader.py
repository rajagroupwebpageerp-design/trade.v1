"""
paper_trader.py — simulated (no real orders) paper trading bot running ONE
consolidated strategy: "master" — combines the strongest, most justified
piece of every strategy explored earlier into a single rule, instead of
tracking several separate ones.

ENTRY — ALL THREE of these must be true at the same time:
  1. MTF agrees:        the 5M+15M+1H base-indicator signal (EMA/RSI/MACD/
                         VWAP) all point the same direction.
  2. SMC setup exists:  the nearest unmitigated OB/FVG zone (or a
                         confirmed QML) on 5M is in that same direction.
  3. Weighted 7-condition scoring, checked independently per timeframe
     across EMA, RSI, MACD, VWAP, Order Block, FVG, and QML (QML counts as
     green if "confirmed" OR "forming"):
       - 1H  (macro bias):            >= 5 of 7 green
       - 15M (intermediate momentum): >= 6 of 7 green
       - 5M  (execution trigger):     >= 6 of 7 green

EXIT — whichever happens first:
  (a) price hits the ATR(14)-based stop-loss/target (1.5x ATR risk, 3x ATR
      reward — a 1:2 ratio that scales to each symbol's own volatility,
      instead of a flat point value that doesn't fit every asset equally), or
  (b) the SAME 3-timeframe confluence that justified entry breaks: exits
      the instant any ONE of 1H/15M/5M drops below its required threshold
      on the position's side, checked every cycle regardless of price.

After a loss, that symbol is skipped for 15 minutes before a new entry is
allowed, so the bot can't immediately re-enter the same failing setup.

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

# Per-timeframe minimum green-condition counts (out of 7).
MIN_1H = 5
MIN_15M = 6
MIN_5M = 6

# ATR multiples for stop-loss/target — keeps a 1:2 reward:risk ratio while
# scaling to each symbol's own recent volatility. Widened from 1.5/3.0 to
# 2.0/4.0 (same 1:2 ratio, same 33% breakeven win rate) specifically to
# give trades more room before ordinary 5-minute noise stops them out —
# this is the one lever that can genuinely raise win rate without secretly
# making the risk:reward math worse.
ATR_SL_MULT = 2.0
ATR_TARGET_MULT = 4.0

# After a losing trade, that symbol is skipped for this many seconds
# before a new entry is allowed.
COOLDOWN_SECONDS = 900  # 15 minutes

STRATEGIES = ["master"]

# Fixed UTC+5:30 — India doesn't observe DST, so this is correct year-round
# without needing the zoneinfo package.
IST = timezone(timedelta(hours=5, minutes=30))


def _ist_date_str(unix_ts):
    return datetime.fromtimestamp(unix_ts, tz=IST).strftime("%Y-%m-%d")


def _empty_strategy_state():
    return {"positions": {}, "closedTrades": [], "dailyStats": {}, "cooldowns": {}}


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
        strategies[key].setdefault("cooldowns", {})
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
        "PAPER BOT [master]: closed %s %s @ %s (entry %s) pnl=%.3f%% (%s)",
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


def _all_green(item):
    return item["emaBullish"] and item["vwapAbove"] and item["rsi"] > 52 and item["macdBullish"]


def _all_red(item):
    return (not item["emaBullish"]) and (not item["vwapAbove"]) and item["rsi"] < 48 and (not item["macdBullish"])


def _has_unmitigated(zones, zone_type):
    return any(z["type"] == zone_type and not z["mitigated"] for z in (zones or []))


def _seven_flags_for_tf(item, tf, side):
    """7 booleans (EMA, RSI, MACD, VWAP, OB, FVG, QML) for ONE specific
    timeframe ('5m'/'15m'/'1h'), matching `side`. QML counts as green if
    'confirmed' OR 'forming'. Missing/not-yet-available data counts as a
    miss, never raises — an incomplete timeframe just fails to qualify."""
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
    qml_ok = bool(qml_b and qml_b.get("status") in ("confirmed", "forming")) if want_bullish else bool(qml_s and qml_s.get("status") in ("confirmed", "forming"))

    return [ema_ok, rsi_ok, macd_ok, vwap_ok, ob_ok, fvg_ok, qml_ok]


def _run_master_strategy(strat_state, symbol, item, now):
    positions = strat_state["positions"]
    cooldowns = strat_state.setdefault("cooldowns", {})
    pos = positions.get(symbol)
    ltp = item["ltp"]

    if pos is None:
        if now < cooldowns.get(symbol, 0):
            return _position_snapshot(None, ltp)  # still cooling down after a recent loss

        atr = item.get("atr14")
        if not atr or atr <= 0:
            return _position_snapshot(None, ltp)  # ATR not warmed up yet

        mtf_signal = item.get("mtfSignal", "NEUTRAL")
        setup = ((item.get("smc") or {}).get("5m") or {}).get("tradeSetup")

        sides = ["LONG", "SHORT"] if ALLOW_SHORTS else ["LONG"]
        for side in sides:
            mtf_ok = (mtf_signal == "BUY") if side == "LONG" else (mtf_signal == "SELL")
            setup_ok = bool(setup and setup.get("type") == ("bullish" if side == "LONG" else "bearish"))
            if not (mtf_ok and setup_ok):
                continue

            count_1h = sum(_seven_flags_for_tf(item, "1h", side))
            count_15m = sum(_seven_flags_for_tf(item, "15m", side))
            count_5m = sum(_seven_flags_for_tf(item, "5m", side))

            if count_1h >= MIN_1H and count_15m >= MIN_15M and count_5m >= MIN_5M:
                if side == "LONG":
                    positions[symbol] = {
                        "side": "LONG", "entryPrice": ltp, "entryTime": now,
                        "stopLoss": round(ltp - ATR_SL_MULT * atr, 2),
                        "target": round(ltp + ATR_TARGET_MULT * atr, 2),
                    }
                else:
                    positions[symbol] = {
                        "side": "SHORT", "entryPrice": ltp, "entryTime": now,
                        "stopLoss": round(ltp + ATR_SL_MULT * atr, 2),
                        "target": round(ltp - ATR_TARGET_MULT * atr, 2),
                    }
                log.info(
                    "PAPER BOT [master]: ENTRY ALERT %s %s @ %s (MTF=%s SMC=%s 1H=%d/7 15M=%d/7 5M=%d/7)",
                    side, symbol, ltp, mtf_signal, setup["source"], count_1h, count_15m, count_5m,
                )
                break
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
            count_1h = sum(_seven_flags_for_tf(item, "1h", pos["side"]))
            count_15m = sum(_seven_flags_for_tf(item, "15m", pos["side"]))
            count_5m = sum(_seven_flags_for_tf(item, "5m", pos["side"]))

            if count_1h < MIN_1H:
                exit_reason = f"1H confluence dropped ({count_1h}/7, needs >={MIN_1H})"
            elif count_15m < MIN_15M:
                exit_reason = f"15M confluence dropped ({count_15m}/7, needs >={MIN_15M})"
            elif count_5m < MIN_5M:
                exit_reason = f"5M confluence dropped ({count_5m}/7, needs >={MIN_5M})"

        if exit_reason:
            trade = _close_trade(strat_state, symbol, pos, ltp, exit_reason, now)
            del positions[symbol]
            if trade["pnlPct"] <= 0:
                cooldowns[symbol] = now + COOLDOWN_SECONDS

    return _position_snapshot(positions.get(symbol), ltp)


STRATEGY_RUNNERS = {
    "master": _run_master_strategy,
}


def _log_diagnostics(results):
    """One line per cycle: how many symbols currently have MTF agreement,
    an SMC setup, and 5M>=6/7 individually — shows which piece is the
    bottleneck for the combined entry, rather than a black box."""
    mtf_buy = mtf_sell = 0
    setup_bull = setup_bear = 0
    five_m_bull = five_m_bear = 0

    for item in results.values():
        if item.get("mtfSignal") == "BUY":
            mtf_buy += 1
        elif item.get("mtfSignal") == "SELL":
            mtf_sell += 1

        setup = ((item.get("smc") or {}).get("5m") or {}).get("tradeSetup")
        if setup and setup.get("type") == "bullish":
            setup_bull += 1
        elif setup and setup.get("type") == "bearish":
            setup_bear += 1

        if sum(_seven_flags_for_tf(item, "5m", "LONG")) >= MIN_5M:
            five_m_bull += 1
        if sum(_seven_flags_for_tf(item, "5m", "SHORT")) >= MIN_5M:
            five_m_bear += 1

    log.info(
        "PAPER BOT DIAGNOSTICS (of %d symbols): MTF-BUY=%d MTF-SELL=%d | "
        "smc-setup-bull=%d smc-setup-bear=%d | 5m>=%d-bull=%d 5m>=%d-bear=%d "
        "(all three must overlap on the SAME symbol to enter)",
        len(results), mtf_buy, mtf_sell, setup_bull, setup_bear,
        MIN_5M, five_m_bull, MIN_5M, five_m_bear,
    )


def process_cycle(results):
    """Run the master strategy over this cycle's {symbol: item} results.
    Mutates each item to add item['paperPositions'] = {'master': ...}
    (None when flat). Persists state in one write."""
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
    closed = strat_state.get("closedTrades", [])
    total = len(closed)
    wins = sum(1 for t in closed if t["pnlPct"] > 0)
    losses = total - wins
    total_pnl = round(sum(t["pnlPct"] for t in closed), 3)
    avg_pnl = round(total_pnl / total, 3) if total else 0.0
    win_rate = round(wins / total * 100, 1) if total else 0.0
    return {
        "totalTrades": total, "wins": wins, "losses": losses,
        "winRatePct": win_rate, "totalPnlPct": total_pnl, "avgPnlPct": avg_pnl,
    }


def get_daily_summaries(strat_state, limit=30):
    daily = strat_state.get("dailyStats", {})
    out = []
    for day_key in sorted(daily.keys(), reverse=True)[:limit]:
        d = daily[day_key]
        total = d["totalTrades"]
        win_rate = round(d["wins"] / total * 100, 1) if total else 0.0
        avg_pnl = round(d["totalPnlPct"] / total, 3) if total else 0.0
        out.append({
            "date": day_key, "totalTrades": total, "wins": d["wins"], "losses": d["losses"],
            "winRatePct": win_rate, "totalPnlPct": d["totalPnlPct"], "avgPnlPct": avg_pnl,
        })
    return out


def get_full_report(state=None):
    """{"master": {openPositions, closedTrades, stats, dailySummaries}}"""
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
