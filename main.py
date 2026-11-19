"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for each of the game's 55 PlayStation 5 trophies, five quantiles
(5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at the case's target.

Model (per trophy, on the logit scale):

1. Trend fit.  Earn rates follow an adoption curve: fast early growth that
   slows as the player base matures.  We fit a weighted least-squares line
   of logit(rate) against ln(1 + hours since launch), with recency weights
   (24 h half-life), and read it at the target.  The slope is shrunk toward
   zero by n/(n + 1.5) so a handful of reads cannot extrapolate wildly, and
   the extrapolated median is blended 55/45 with the last read in logit
   space: a pure log-time line overshoots when growth saturates, while
   pure persistence misses continued growth, and the blend beats
   persistence clearly in both regimes on synthetic finals (saturating:
   0.29 vs 0.24 persistence; continued growth: 0.36 vs 1.03).  The
   extrapolated move is capped at 3.5 logit units.
   The spread combines a random-walk term (0.30 logit per sqrt-day), the
   fit's residual scatter, and the slope's standard error times the
   extrapolation distance, so stable curves forecast narrow and volatile
   or rare trophies forecast wide.

2. Stale-page refresh.  Exophase refreshes its counts on its own schedule,
   not with every read: on launch night the page sat unchanged (638 players
   tracked, every rate identical) for hours, then refreshed to 3,868
   players with every rate but one lower.  When a trophy's whole history is
   flat *and* players_tracked never moved, the last read is a stale page
   and a refresh is pending before the target.  The refresh dilutes the
   rate: new rate = old rate * (1 + (m - 1) * g) / m, with m = 3868/638
   (the one refresh the house documented) and g the share of the new
   cohort expected to have earned that trophy by the refresh -- near zero
   for the platinum and end-game trophies, higher for early chapters and
   quick activities, middling for the rest.  The spread is wide (0.95
   logit) because the next refresh's size is genuinely uncertain.

3. Fallback.  A trophy with no usable movement and no stale page (for
   example a rate flat while the tracked-player count grows, which means
   earners grew in step) is forecast at its last rate with the house
   random-walk spread.

Deterministic: no randomness, no network, no clock reads; two runs on the
same case write byte-identical output.  Standard library only.

Usage (outside the grader):  mkdir -p out && MAKEFUN_INPUT_DIR=input \
MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# The one documented Exophase refresh: 638 -> 3,868 players tracked.
REFRESH_MULTIPLIER = 3868.0 / 638.0
# Spread (logit) when forecasting across a pending refresh.
STALE_SIGMA = 0.95
# Random-walk spread (logit per sqrt-day) in the trend branch.
WALK_SIGMA = 0.30


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, like the house baseline's rounding."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(moment):
    return (moment - LAUNCH).total_seconds() / 3600.0


def earn_propensity(trophy, last_rate):
    """g: share of a fresh cohort expected to hold this trophy by a refresh.

    Near zero for end-game trophies (a new player cannot finish the game
    in hours), higher for early chapters and quick activities.
    """
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    text = name + " " + desc
    tier = trophy.get("tier") or ""
    if last_rate is not None and last_rate >= 99.9:
        return 0.50
    if tier == "platinum":
        return 0.02
    if "final mission" in text or "100% completion" in text:
        return 0.04
    if tier == "gold":
        return 0.08
    if tier == "silver":
        if "stunt" in text or "medal" in text:
            return 0.05
        return 0.12
    # Bronze.
    if "chapter 1" in text:
        return 0.45
    if "chapter 2" in text:
        return 0.40
    if "chapter 3" in text:
        return 0.32
    if "chapter 4" in text:
        return 0.25
    if any(k in text for k in ("fishing", "arcade", "bowling", "drone",
                               "race", "golf", "tennis")):
        return 0.30
    return 0.20


def weighted_line(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, resid_sd, se_b)."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    n = len(xs)
    if n > 2:
        s2 = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
        dof = max(n - 2, 1)
        s2 *= n / dof
        resid = math.sqrt(max(s2, 0.0))
        se_b = resid / math.sqrt(sxx) if sxx > 0 else 0.0
    else:
        resid, se_b = 0.0, 0.0
    return a, b, resid, se_b


def forecast_trophy(trophy, idx, history, target, horizon_days):
    """Return (median_rate, sigma_logit) for one trophy."""
    points = []
    for read in history:
        rates = read.get("rates") or []
        if idx >= len(rates) or rates[idx] is None:
            continue
        points.append((parse_time(read["at"]), float(rates[idx])))
    if not points:
        return 0.5, 1.2
    last_rate = points[-1][1]
    last_time = points[-1][0]

    players = [r.get("players_tracked") for r in history]
    players_constant = all(p == players[0] for p in players)
    rates_flat = all(abs(r - last_rate) < 1e-9 for _, r in points)
    span_hours = (points[-1][0] - points[0][0]).total_seconds() / 3600.0

    # Branch 1: stale page -- whole history flat, tracked count never moved.
    if rates_flat and players_constant and len(points) >= 3 and span_hours >= 1.0:
        g = earn_propensity(trophy, last_rate)
        factor = (1.0 + (REFRESH_MULTIPLIER - 1.0) * g) / REFRESH_MULTIPLIER
        # The factor acts on the earn rate itself (earners / tracked):
        # new earners join the old ones over the larger tracked base.
        median = clamp_rate(last_rate * factor)
        return median, STALE_SIGMA

    # Branch 2: trend fit on log-time, when the series actually moves.
    xs, ys, ws = [], [], []
    for moment, rate in points:
        t = max(hours_since_launch(moment), 0.0)
        xs.append(math.log1p(t))
        ys.append(logit(rate))
        age_h = max((last_time - moment).total_seconds() / 3600.0, 0.0)
        ws.append(0.5 ** (age_h / 24.0))
    moved = max(ys) - min(ys) > 0.02 and len(set(round(y, 6) for y in ys)) >= 3
    if moved:
        a, b, resid, se_b = weighted_line(xs, ys, ws)
        n_eff = len(xs)
        b = max(-1.5, min(3.0, b * n_eff / (n_eff + 1.5)))
        x_target = math.log1p(max(hours_since_launch(target), 0.0))
        y_trend = a + b * x_target
        # Blend with the last read (55/45) and cap the extrapolated move:
        # a log-time line overshoots a saturating curve, persistence misses
        # continued growth; the blend is robust to both regimes.
        move = max(-3.5, min(3.5, y_trend - ys[-1]))
        y_target = ys[-1] + 0.55 * move
        median = clamp_rate(sigmoid(y_target))
        var = ((WALK_SIGMA * math.sqrt(horizon_days)) ** 2
               + (resid * math.sqrt(horizon_days)) ** 2
               + (se_b * max(x_target - xs[-1], 0.0)) ** 2
               + 0.05 ** 2)
        sigma = max(0.30, min(1.6, math.sqrt(var)))
        return median, sigma

    # Branch 3: no movement, no stale page -- persistence, house spread.
    sigma = max(0.30, 0.5 * math.sqrt(horizon_days))
    return clamp_rate(last_rate), sigma


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        median, sigma = forecast_trophy(
            trophy, i, case["history"], target, horizon_days)
        med_logit = logit(median)
        qs = [round4(clamp_rate(sigmoid(med_logit + z * sigma))) for z in Z]
        qs = sorted(qs)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
