# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Absaa00/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Absaa00/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast how many GTA VI players earn each trophy in week one — for each of
the game's 55 PlayStation 5 trophies, five quantiles (5%, 25%, 50%, 75%,
95%) of its Exophase earn rate at 2026-11-26 00:00 UTC.

## How it works

`main.py` (Python 3.13, standard library only, offline, deterministic) reads
one case from `/input/<id>.in` and writes `/output/<id>.out`. Per trophy,
on the logit scale:

1. **Trend fit.** A weighted least-squares line of logit(earn rate)
   against ln(1 + hours since launch), with 24-hour recency half-life.
   The slope is shrunk toward zero and the extrapolated median is blended
   55/45 with the last read (a pure log-time line overshoots when growth
   saturates; pure persistence misses continued growth), with the move
   capped at 3.5 logit units.
2. **Stale-page refresh.** Exophase refreshes its counts on its own
   schedule: on launch night the page sat unchanged (638 players tracked)
   for hours, then refreshed to 3,868 players with every rate but one
   lower. When a trophy's whole history is flat *and* the tracked count
   never moved, the last read is a stale page and a refresh is pending:
   the median is diluted by `(1 + (m − 1)·g) / m` with m = 3868/638 and
   g the share of the fresh cohort expected to hold that trophy (near
   zero for the platinum/end-game, higher for early chapters and quick
   activities), with a wide 0.95-logit spread.
3. **Fallback.** No movement and no stale page → last rate with a
   random-walk spread.

The spread combines a random-walk term, the fit's residual scatter, and
the slope's standard error, so stable curves forecast narrow and rare or
volatile trophies forecast wide.

## Local validation

On the kit's practice cases (p00, p01), scored against simulated refresh
answers (the real refreshed rates are not public — the house only says
players went 638 → 3,868 and every rate but 100% fell), across five
refresh scenarios (multiplier 3–10×, new-cohort earn propensity 0–2×):

| scenario | this forecaster | persistence baseline |
|---|---|---|
| central (6.06×, g as modelled) | **0.191** | 1.225 |
| pure dilution (g = 0) | **0.370** | 1.883 |
| easy cohort (g ×2) | **0.284** | 0.828 |
| small refresh (3×) | **0.267** | 0.846 |
| big refresh (10×) | **0.231** | 1.428 |

Mean pinball loss on the logit scale; lower is better. The forecaster
beats persistence in every scenario (90% coverage 0.98–1.00). On a
synthetic final-like case (200 reads over 33 h, 108 h horizon) it ties
persistence when growth saturates (0.28 vs 0.27) and beats it clearly
when growth continues (0.36 vs 1.03).

Two runs on the same case write byte-identical output; runtime ≈ 0.1 s
per case (limit 8 s).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see [LICENSE](LICENSE).
