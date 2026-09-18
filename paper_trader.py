"""
paper_trader.py — simulated (no real orders) paper trading, run as FIVE
independent strategies so their P&L can be tracked and compared separately:

  "base"       — the original rule: ALL of EMA(9>21)/RSI/MACD/VWAP (5M)
                 green opens a LONG, all red opens a SHORT; exits the
                 instant any ONE of the four flips against the position.
  "smc"        — opens off the 5M SMC trade setup (the nearest unmitigated
                 OB/FVG zone, or a confirmed QML) at the live price, using
                 that setup's own stop-loss/target; exits when price hits
                 either.
  "mtf"        — opens when the 5M+15M+1H multi-timeframe signal agrees
                 (BUY or SELL); exits the instant that agreement breaks.
  "confluence" — opens ONLY when the base indicators are all green/red AND
                 an unmitigated OB AND an unmitigated FVG of the matching
                 direction are all present together on 5M. Uses a fixed
                 points-based stop-loss/target (not percentage, not zone-
                 derived) — see FIXED_SL_POINTS / FIXED_TARGET_POINTS.
  "strict"     — opens ONLY when ALL 7 conditions align: EMA, RSI, MACD,
                 VWAP, Order Block, FVG, and QML (confirmed). Same fixed
                 points SL/target as "confluence", but a smarter exit:
                 holds through minor noise and only exits early — before
                 price reaches SL/target — if MORE THAN 2 of the 7
                 conditions flip against the position; otherwise it just
                 waits for price to hit the stop-loss or target.

Each strategy keeps its own open positions, closed-trade log, and daily
(IST) P&L summary — all five persisted together in one shared JSON state
file (same atomic-write pattern as the market data cache), so every
gunicorn worker process sees the same state.
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

MAX_CLOSED_TRADES = 200      # per-strategy bounded log
MAX_DAILY_SUMMARIES = 90     # per-strategy, roughly a trading quarter

# Flip to False to go long-only across all strategies.
ALLOW_SHORTS = True

# Fixed price-POINTS (not %, not zone-based) for the "confluence" and
# "strict" strategies' stop-loss/target — 1:2 reward:risk as requested.
# NOTE: a flat point value behaves very differently across assets at very
# different price scales (e.g. 5 points is enormous for Natural Gas at
# ~₹2.91, tiny for an index in the tens of thousands) — revisit if that
# turns out to be a problem.
FIXED_SL_POINTS = 5
FIXED_TARGET_POINTS = 10

# "strict" exits early (before price reaches SL/target) once MORE than this
# many of its 7 conditions flip against the open position.
STRICT_MAX_FLIPPED = 2

STRATEGIES = ["base", "smc", "mtf", "confluence", "strict"]

# Fixed UTC+5:30 — India doesn't observe DST, so this is correct year-round
# without needing the zoneinfo package. Used to group closed trades into
# calendar trading days for the daily P&L summary.
IST = timezone(timedelta(hours=5, minutes=30))


def _ist_date_str(unix_ts):
    return datetime.fromtimestamp(unix_ts, tz=IST).strftime("%Y-%m-%d")


def _empty_strategy_state():
    return {"positions": {}, "closedTrades": [], "dailyStats": {}}


def read_state():
    """Read the shared paper-trading state file. Safe from any worker.
    Always returns a dict with all three strategy keys present, even on a
    fresh file, so callers never need to guard for missing keys."""
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


def _all_green(item):
    return item["emaBullish"] and item["vwapAbove"] and item["rsi"] > 52 and item["macdBullish"]


def _all_red(item):
    return (not item["emaBullish"]) and (not item["vwapAbove"]) and item["rsi"] < 48 and (not item["macdBullish"])


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


def _close_trade(strategy_key, strat_state, symbol, pos, exit_price, reason, now):
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
        "PAPER BOT [%s]: closed %s %s @ %s (entry %s) pnl=%.3f%% (%s)",
        strategy_key, pos["side"], symbol, exit_price, entry_price, pnl_pct, reason,
    )
    return trade


def _position_snapshot(pos, ltp, extra=None):
    if not pos:
        return None
    entry_price = pos["entryPrice"]
    if pos["side"] == "LONG":
        unrealized = round((ltp - entry_price) / entry_price * 100, 3)
    else:
        unrealized = round((entry_price - ltp) / entry_price * 100, 3)
    snap = {"side": pos["side"], "entryPrice": entry_price, "unrealizedPnlPct": unrealized}
    if extra:
        snap.update(extra)
    return snap


def _run_base_strategy(strat_state, symbol, item, now):
    """All 4 base indicators green -> LONG, all red -> SHORT. Exit the
    instant any ONE of the four flips against the open position."""
    positions = strat_state["positions"]
    pos = positions.get(symbol)
    ltp = item["ltp"]

    if pos is None:
        if _all_green(item):
            positions[symbol] = {"side": "LONG", "entryPrice": ltp, "entryTime": now}
        elif ALLOW_SHORTS and _all_red(item):
            positions[symbol] = {"side": "SHORT", "entryPrice": ltp, "entryTime": now}
    else:
        exit_reason = None
        if pos["side"] == "LONG" and not _all_green(item):
            exit_reason = "an indicator turned red"
        elif pos["side"] == "SHORT" and not _all_red(item):
            exit_reason = "an indicator turned green"
        if exit_reason:
            _close_trade("base", strat_state, symbol, pos, ltp, exit_reason, now)
            del positions[symbol]

    return _position_snapshot(positions.get(symbol), ltp)


def _run_smc_strategy(strat_state, symbol, item, now):
    """Opens off the 5M SMC trade setup (nearest unmitigated OB/FVG zone,
    or a confirmed QML) at the live price, carrying over that setup's own
    stop-loss/target. Exits when price actually hits either — a real
    trade-management exit, unlike the flip-based exits of the other two
    strategies."""
    positions = strat_state["positions"]
    pos = positions.get(symbol)
    ltp = item["ltp"]
    setup = ((item.get("smc") or {}).get("5m") or {}).get("tradeSetup")

    if pos is None:
        if setup:
            side = "LONG" if setup["type"] == "bullish" else "SHORT"
            positions[symbol] = {
                "side": side,
                "entryPrice": ltp,
                "entryTime": now,
                "stopLoss": setup["stopLoss"],
                "target": setup["target"],
                "source": setup["source"],
            }
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
        if exit_reason:
            _close_trade("smc", strat_state, symbol, pos, ltp, exit_reason, now)
            del positions[symbol]

    pos = positions.get(symbol)
    extra = {"source": pos["source"]} if pos else None
    return _position_snapshot(pos, ltp, extra=extra)


def _run_mtf_strategy(strat_state, symbol, item, now):
    """Opens when the 5M+15M+1H multi-timeframe signal agrees (BUY/SELL).
    Exits the instant that agreement breaks (signal no longer matches the
    side that was entered)."""
    positions = strat_state["positions"]
    pos = positions.get(symbol)
    ltp = item["ltp"]
    mtf_signal = item.get("mtfSignal", "NEUTRAL")

    if pos is None:
        if mtf_signal == "BUY":
            positions[symbol] = {"side": "LONG", "entryPrice": ltp, "entryTime": now}
        elif ALLOW_SHORTS and mtf_signal == "SELL":
            positions[symbol] = {"side": "SHORT", "entryPrice": ltp, "entryTime": now}
    else:
        exit_reason = None
        if pos["side"] == "LONG" and mtf_signal != "BUY":
            exit_reason = "MTF signal no longer BUY"
        elif pos["side"] == "SHORT" and mtf_signal != "SELL":
            exit_reason = "MTF signal no longer SELL"
        if exit_reason:
            _close_trade("mtf", strat_state, symbol, pos, ltp, exit_reason, now)
            del positions[symbol]

    return _position_snapshot(positions.get(symbol), ltp)


def _has_unmitigated(zones, zone_type):
    return any(z["type"] == zone_type and not z["mitigated"] for z in (zones or []))


def _run_confluence_strategy(strat_state, symbol, item, now):
    """Only enters when ALL base indicators are green/red AND an
    unmitigated OB AND an unmitigated FVG of the matching direction are all
    present together on 5M — a stricter combined confirmation than either
    'base' or 'smc' alone. Fixed points-based SL/target (1:2)."""
    positions = strat_state["positions"]
    pos = positions.get(symbol)
    ltp = item["ltp"]
    smc_5m = ((item.get("smc") or {}).get("5m")) or {}
    obs = smc_5m.get("orderBlocks", [])
    fvgs = smc_5m.get("fvg", [])

    if pos is None:
        bullish_confluence = _all_green(item) and _has_unmitigated(obs, "bullish") and _has_unmitigated(fvgs, "bullish")
        bearish_confluence = ALLOW_SHORTS and _all_red(item) and _has_unmitigated(obs, "bearish") and _has_unmitigated(fvgs, "bearish")

        if bullish_confluence:
            positions[symbol] = {
                "side": "LONG", "entryPrice": ltp, "entryTime": now,
                "stopLoss": round(ltp - FIXED_SL_POINTS, 2),
                "target": round(ltp + FIXED_TARGET_POINTS, 2),
            }
        elif bearish_confluence:
            positions[symbol] = {
                "side": "SHORT", "entryPrice": ltp, "entryTime": now,
                "stopLoss": round(ltp + FIXED_SL_POINTS, 2),
                "target": round(ltp - FIXED_TARGET_POINTS, 2),
            }
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
        if exit_reason:
            _close_trade("confluence", strat_state, symbol, pos, ltp, exit_reason, now)
            del positions[symbol]

    return _position_snapshot(positions.get(symbol), ltp)


def _condition_flags(item, side):
    """The 7 conditions (EMA, RSI, MACD, VWAP, OB, FVG, QML) as booleans:
    True if that condition currently matches `side` ('LONG' expects every
    condition bullish, 'SHORT' expects every condition bearish)."""
    smc_5m = ((item.get("smc") or {}).get("5m")) or {}
    obs = smc_5m.get("orderBlocks", [])
    fvgs = smc_5m.get("fvg", [])
    qml_b = smc_5m.get("qmlBullish")
    qml_s = smc_5m.get("qmlBearish")

    want_bullish = side == "LONG"
    zone_type = "bullish" if want_bullish else "bearish"

    return [
        item["emaBullish"] if want_bullish else not item["emaBullish"],
        (item["rsi"] > 52) if want_bullish else (item["rsi"] < 48),
        item["macdBullish"] if want_bullish else not item["macdBullish"],
        item["vwapAbove"] if want_bullish else not item["vwapAbove"],
        _has_unmitigated(obs, zone_type),
        _has_unmitigated(fvgs, zone_type),
        bool(qml_b and qml_b["status"] == "confirmed") if want_bullish else bool(qml_s and qml_s["status"] == "confirmed"),
    ]


def _run_strict_strategy(strat_state, symbol, item, now):
    """Enters only when ALL 7 conditions align. Same fixed points SL/target
    as 'confluence', but holds through minor noise: exits early (before
    price reaches SL/target) only once MORE than STRICT_MAX_FLIPPED of the
    7 conditions flip against the position — otherwise just waits for
    price to hit the stop-loss or target."""
    positions = strat_state["positions"]
    pos = positions.get(symbol)
    ltp = item["ltp"]

    if pos is None:
        if all(_condition_flags(item, "LONG")):
            positions[symbol] = {
                "side": "LONG", "entryPrice": ltp, "entryTime": now,
                "stopLoss": round(ltp - FIXED_SL_POINTS, 2),
                "target": round(ltp + FIXED_TARGET_POINTS, 2),
            }
        elif ALLOW_SHORTS and all(_condition_flags(item, "SHORT")):
            positions[symbol] = {
                "side": "SHORT", "entryPrice": ltp, "entryTime": now,
                "stopLoss": round(ltp + FIXED_SL_POINTS, 2),
                "target": round(ltp - FIXED_TARGET_POINTS, 2),
            }
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
            false_count = _condition_flags(item, pos["side"]).count(False)
            if false_count > STRICT_MAX_FLIPPED:
                exit_reason = f"{false_count}/7 conditions flipped against position"

        if exit_reason:
            _close_trade("strict", strat_state, symbol, pos, ltp, exit_reason, now)
            del positions[symbol]

    return _position_snapshot(positions.get(symbol), ltp)


STRATEGY_RUNNERS = {
    "base": _run_base_strategy,
    "smc": _run_smc_strategy,
    "mtf": _run_mtf_strategy,
    "confluence": _run_confluence_strategy,
    "strict": _run_strict_strategy,
}


def process_cycle(results):
    """Run all three strategies over this cycle's {symbol: item} results.
    Mutates each item to add item['paperPositions'] = {'base':.., 'smc':..,
    'mtf':..} (each None when flat on that strategy). Persists all state in
    one write. A failure in one strategy on one symbol is logged and just
    leaves that symbol's position untouched this cycle — it can't corrupt
    the other strategies or symbols."""
    state = read_state()
    now = time.time()

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
    """Summary stats over one strategy's closed-trade log."""
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
    """Most recent `limit` days' P&L for one strategy, newest first."""
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
    """Everything the frontend needs for all three strategies in one call:
    {"base": {...}, "smc": {...}, "mtf": {...}}, each with openPositions,
    the most recent closed trades, overall stats, and daily summaries."""
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
