"""
smc.py — Smart Money Concepts detection: Order Blocks (OB), Fair Value Gaps
(FVG), and the QML / Quasimodo reversal pattern.

Design notes (read this before changing thresholds):

- Everything here operates on a plain OHLCV pandas DataFrame with columns
  Open/High/Low/Close/Volume and a datetime index. It does not care what
  timeframe the candles are — the caller resamples first.

- These are simplified, rules-based versions of concepts that don't have a
  single universally-agreed definition (different SMC/ICT educators define
  OB/FVG/QML slightly differently). The versions here are the common,
  widely-used definitions:
    * Swing point  -> a fractal: a high/low that is the most extreme point
                       within `left` bars before and `right` bars after it.
    * Order Block  -> the last opposite-colored candle before a candle that
                       closes beyond ("breaks the structure of") the most
                       recent unbroken swing high/low.
    * FVG          -> a 3-candle imbalance: candle[i-1].high < candle[i+1].low
                       (bullish) or candle[i-1].low > candle[i+1].high
                       (bearish) — the gap between the two outer candles.
    * QML          -> a 4-point liquidity-sweep reversal: swing (P1) ->
                       opposite swing (P2, the "neckline") -> a swing beyond
                       P1 that sweeps liquidity (P3, the "head") -> price
                       closing back beyond the neckline (P2) confirms it.

- All loops are bounded (capped bar counts, capped backward searches) so
  this stays fast even though it's plain Python, not vectorized.
"""

import logging

log = logging.getLogger("trade-scanner")

# Only look at the most recent N candles per timeframe for SMC. Older
# structure is rarely actionable for intraday/swing entries, and this keeps
# every loop below bounded and fast regardless of how much history we feed in.
MAX_BARS_FOR_SMC = 300

# How many bars on each side must be more extreme for a bar to count as a
# swing high/low (fractal length). 2 is a common, fairly responsive default.
SWING_LEFT = 2
SWING_RIGHT = 2

# How far back an Order Block search will look for the last opposite candle
# before a break-of-structure candle.
OB_LOOKBACK = 15

# Max zones returned per type (bullish/bearish) for OB and FVG, so the API
# payload stays small — unmitigated (still "live") zones are preferred.
MAX_ZONES = 3

# Timeframes to compute, and the pandas resample rule for each. "5m" (None)
# means "use the base dataframe as-is" — no resampling needed.
TF_RULES = {
    "5m": None,
    "15m": "15min",
    "1h": "60min",
    "4h": "240min",
    "1d": "1D",
}


def resample_ohlc(df, rule):
    """Resample a 5-minute OHLCV dataframe up to a higher timeframe."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df.resample(rule).agg(agg)
    out.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    return out


def detect_swings(df, left=SWING_LEFT, right=SWING_RIGHT):
    """Mark fractal swing highs/lows. Adds boolean 'SwingHigh'/'SwingLow'
    columns and returns the same df (operates on a copy the caller passed)."""
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    swing_high = [False] * n
    swing_low = [False] * n

    for i in range(left, n - right):
        window_high = highs[i - left:i + right + 1]
        if highs[i] == window_high.max() and (window_high == highs[i]).sum() == 1:
            swing_high[i] = True
        window_low = lows[i - left:i + right + 1]
        if lows[i] == window_low.min() and (window_low == lows[i]).sum() == 1:
            swing_low[i] = True

    df["SwingHigh"] = swing_high
    df["SwingLow"] = swing_low
    return df


def _swings_list(df):
    """Chronological list of (index_position, 'high'|'low', price)."""
    swings = []
    for i in range(len(df)):
        if df["SwingHigh"].iat[i]:
            swings.append((i, "high", float(df["High"].iat[i])))
        elif df["SwingLow"].iat[i]:
            swings.append((i, "low", float(df["Low"].iat[i])))
    return swings


def detect_order_blocks(df, swings, max_zones=MAX_ZONES):
    """Bullish OB: last bearish candle before a close breaks above the most
    recent unbroken swing high. Bearish OB: mirror, on swing lows."""
    n = len(df)
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    swing_highs = [(i, v) for (i, t, v) in swings if t == "high"]
    swing_lows = [(i, v) for (i, t, v) in swings if t == "low"]

    bullish_raw, bearish_raw = [], []
    last_broken_high_idx = -1
    last_broken_low_idx = -1

    for i in range(n):
        prior_highs = [v for (si, v) in swing_highs if last_broken_high_idx < si < i]
        if prior_highs and closes[i] > max(prior_highs):
            for j in range(i - 1, max(i - OB_LOOKBACK, -1), -1):
                if closes[j] < opens[j]:
                    bullish_raw.append({"index": j, "top": round(float(highs[j]), 4), "bottom": round(float(lows[j]), 4)})
                    break
            last_broken_high_idx = i

        prior_lows = [v for (si, v) in swing_lows if last_broken_low_idx < si < i]
        if prior_lows and closes[i] < min(prior_lows):
            for j in range(i - 1, max(i - OB_LOOKBACK, -1), -1):
                if closes[j] > opens[j]:
                    bearish_raw.append({"index": j, "top": round(float(highs[j]), 4), "bottom": round(float(lows[j]), 4)})
                    break
            last_broken_low_idx = i

    def finalize(raw, is_bullish):
        out = []
        for ob in raw[-max_zones * 3:]:
            top, bottom, idx = ob["top"], ob["bottom"], ob["index"]
            mitigated = False
            for k in range(idx + 1, n):
                if lows[k] <= top and highs[k] >= bottom:
                    mitigated = True
                    break
            out.append({"type": "bullish" if is_bullish else "bearish", "top": top, "bottom": bottom, "mitigated": mitigated})
        out.sort(key=lambda z: z["mitigated"])  # unmitigated (False) first
        return out[:max_zones]

    return finalize(bullish_raw, True) + finalize(bearish_raw, False)


def detect_fvg(df, max_zones=MAX_ZONES):
    """3-candle imbalance gaps."""
    n = len(df)
    highs = df["High"].values
    lows = df["Low"].values

    bullish_raw, bearish_raw = [], []
    for i in range(1, n - 1):
        if lows[i + 1] > highs[i - 1]:
            bullish_raw.append({"index": i, "top": round(float(lows[i + 1]), 4), "bottom": round(float(highs[i - 1]), 4)})
        elif highs[i + 1] < lows[i - 1]:
            bearish_raw.append({"index": i, "top": round(float(lows[i - 1]), 4), "bottom": round(float(highs[i + 1]), 4)})

    def finalize(raw, is_bullish):
        out = []
        for fvg in raw[-max_zones * 3:]:
            top, bottom, idx = fvg["top"], fvg["bottom"], fvg["index"]
            mitigated = False
            for k in range(idx + 2, n):
                if lows[k] <= top and highs[k] >= bottom:
                    mitigated = True
                    break
            out.append({"type": "bullish" if is_bullish else "bearish", "top": top, "bottom": bottom, "mitigated": mitigated})
        out.sort(key=lambda z: z["mitigated"])
        return out[:max_zones]

    return finalize(bullish_raw, True) + finalize(bearish_raw, False)


def _find_qml(df, swings, bullish=True, max_p1_checked=100):
    """Most recent QML setup (confirmed takes priority, else 'forming').
    Bullish: P1 low -> P2 high (neckline) -> P3 low below P1 (sweep/head) ->
    close back above P2 confirms. Bearish is the mirror on highs/lows."""
    closes = df["Close"].values
    n = len(df)

    p1_pool = [s for s in swings if s[1] == ("low" if bullish else "high")][-max_p1_checked:]
    p2_pool = [s for s in swings if s[1] == ("high" if bullish else "low")]
    p3_pool = p1_pool_full = [s for s in swings if s[1] == ("low" if bullish else "high")]

    for i1, _, v1 in reversed(p1_pool):
        p2 = next(((i, v) for (i, _t, v) in p2_pool if i > i1), None)
        if not p2:
            continue
        i2, v2 = p2

        p3 = next(
            ((i, v) for (i, _t, v) in p3_pool if i > i2 and (v < v1 if bullish else v > v1)),
            None,
        )
        if not p3:
            continue
        i3, v3 = p3

        confirm_idx = None
        for j in range(i3 + 1, n):
            if (bullish and closes[j] > v2) or (not bullish and closes[j] < v2):
                confirm_idx = j
                break

        zone = sorted([round(float(v1), 4), round(float(v3), 4)])
        return {
            "type": "bullish" if bullish else "bearish",
            "status": "confirmed" if confirm_idx is not None else "forming",
            "neckline": round(float(v2), 4),
            "sweepLevel": round(float(v3), 4),
            "zone": zone,
        }
    return None


def _stop_buffer(price):
    """Small buffer placed just beyond a zone edge for the stop loss, so the
    stop isn't sitting exactly on the line that defines the zone."""
    return max(price * 0.0005, 0.0001)


def compute_smc_trade_setup(smc_result, ltp):
    """Combine unmitigated OB zones, unmitigated FVG zones, and a confirmed
    QML into ONE actionable entry/stop-loss/target for this timeframe.

    Logic: a bullish zone sitting at-or-below price is potential support (a
    long entry on a pullback into it); a bearish zone at-or-above price is
    potential resistance (a short entry on a pullback into it). Whichever
    zone's near edge is closest to the live price wins — that's the most
    immediately relevant setup. Target uses a 1:2 reward:risk off that zone,
    matching the convention already used by the EMA-based signal.
    Returns None when there's no zone to trade off on this timeframe.
    """
    bullish, bearish = [], []

    for ob in smc_result.get("orderBlocks", []):
        if not ob["mitigated"]:
            (bullish if ob["type"] == "bullish" else bearish).append(
                {"top": ob["top"], "bottom": ob["bottom"], "source": "OB"}
            )

    for fvg in smc_result.get("fvg", []):
        if not fvg["mitigated"]:
            (bullish if fvg["type"] == "bullish" else bearish).append(
                {"top": fvg["top"], "bottom": fvg["bottom"], "source": "FVG"}
            )

    qml_b = smc_result.get("qmlBullish")
    if qml_b and qml_b["status"] == "confirmed":
        bullish.append({"top": qml_b["zone"][1], "bottom": qml_b["zone"][0], "source": "QML"})

    qml_s = smc_result.get("qmlBearish")
    if qml_s and qml_s["status"] == "confirmed":
        bearish.append({"top": qml_s["zone"][1], "bottom": qml_s["zone"][0], "source": "QML"})

    # Only zones on the "right side" of price count: support must be at or
    # below price, resistance at or above (a 0.1% tolerance lets a zone the
    # price is currently sitting inside still qualify).
    supports = [z for z in bullish if z["top"] <= ltp * 1.001]
    resistances = [z for z in bearish if z["bottom"] >= ltp * 0.999]

    def nearest(zones):
        if not zones:
            return None
        return min(zones, key=lambda z: abs(ltp - (z["top"] + z["bottom"]) / 2))

    best_support = nearest(supports)
    best_resistance = nearest(resistances)
    buffer = _stop_buffer(ltp)

    support_dist = abs(ltp - best_support["top"]) if best_support else None
    resistance_dist = abs(ltp - best_resistance["bottom"]) if best_resistance else None

    if best_support and (resistance_dist is None or support_dist <= resistance_dist):
        entry = round(best_support["top"], 2)
        stop = round(best_support["bottom"] - buffer, 2)
        risk = entry - stop
        if risk <= 0:
            return None
        return {
            "type": "bullish",
            "source": best_support["source"],
            "entry": entry,
            "stopLoss": stop,
            "target": round(entry + risk * 2, 2),
        }

    if best_resistance:
        entry = round(best_resistance["bottom"], 2)
        stop = round(best_resistance["top"] + buffer, 2)
        risk = stop - entry
        if risk <= 0:
            return None
        return {
            "type": "bearish",
            "source": best_resistance["source"],
            "entry": entry,
            "stopLoss": stop,
            "target": round(entry - risk * 2, 2),
        }

    return None


def compute_smc_for_df(df, ltp):
    """Run OB/FVG/QML detection on a single-timeframe OHLCV dataframe, plus
    the combined trade setup derived from those zones."""
    min_bars = SWING_LEFT + SWING_RIGHT + 6
    empty = {"orderBlocks": [], "fvg": [], "qmlBullish": None, "qmlBearish": None, "tradeSetup": None}
    if len(df) < min_bars:
        return empty

    if len(df) > MAX_BARS_FOR_SMC:
        df = df.iloc[-MAX_BARS_FOR_SMC:]

    d = detect_swings(df.copy())
    swings = _swings_list(d)

    result = {
        "orderBlocks": detect_order_blocks(d, swings),
        "fvg": detect_fvg(d),
        "qmlBullish": _find_qml(d, swings, bullish=True),
        "qmlBearish": _find_qml(d, swings, bullish=False),
    }
    result["tradeSetup"] = compute_smc_trade_setup(result, ltp)
    return result


def compute_smc_all_timeframes(df_5m, ltp, symbol_name="?"):
    """Resample the base 5-minute dataframe to every timeframe in TF_RULES
    and run SMC detection (+ trade setup) on each, using the same live
    price for all of them. Never raises — a failure on one timeframe just
    yields empty results for that timeframe so one bad symbol/timeframe
    can't take down the whole refresh cycle."""
    result = {}
    base = df_5m[["Open", "High", "Low", "Close", "Volume"]]

    for tf, rule in TF_RULES.items():
        try:
            tf_df = base if rule is None else resample_ohlc(base, rule)
            result[tf] = compute_smc_for_df(tf_df, ltp)
        except Exception as e:
            log.warning("%s: SMC computation failed on tf=%s: %s", symbol_name, tf, e)
            result[tf] = {"orderBlocks": [], "fvg": [], "qmlBullish": None, "qmlBearish": None, "tradeSetup": None}

    return result
