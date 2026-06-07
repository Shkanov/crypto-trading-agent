"""Driver: native price predictor — standalone skill + funding overlay.

    uv run --extra research python -m research.price_predict.run

Runs the ForecastAGLT capability (nonlinear univariate AR on weekly log-returns)
ported onto clean Binance perp klines, validated walk-forward, and answers BOTH
questions the user asked for:

  (1) STANDALONE — does the ~55% weekly directional pulse survive proper
      walk-forward n and beat costs, across two model classes (GBM + MLP)?
  (2) OVERLAY    — does requiring the predictor to agree with a Δfunding leg's
      direction improve that leg's after-cost economics, OOS?

The predictor is trained ONCE over the union of the liquid-majors study set and
the funding universe, so both readouts share the same OOS forecasts.
"""
from __future__ import annotations

import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")     # sklearn convergence + lgbm feature-name noise

from research.ml_meta.data import build_funding_panel, build_panel
from research.ml_meta.funding_events import DFundingParams
from research.price_predict import evaluate, overlay
from research.price_predict.series import WindowSpec, build_pooled_samples
from research.price_predict.walkforward import WalkForwardConfig, run_walkforward

# Liquid USDⓈ-M perps with multi-year history — the directional-skill study set.
# (Majors + established alts; weekly horizon needs long history, so newly-listed
# names are intentionally absent here and live only in the funding overlay.)
STUDY_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT", "BCHUSDT", "TRXUSDT",
    "DOTUSDT", "ATOMUSDT", "NEARUSDT", "FILUSDT", "UNIUSDT", "AAVEUSDT",
    "ETCUSDT", "XLMUSDT", "ALGOUSDT", "EOSUSDT", "AXSUSDT", "SANDUSDT",
    "MANAUSDT", "ICPUSDT", "APTUSDT", "INJUSDT", "ARBUSDT", "OPUSDT",
]

# Funding book's actual (validated-CPCV) universe — the overlay target. Many are
# short-history alts; they get whatever OOS predictions the walk-forward can make.
FUNDING_UNIVERSE = [
    "HYPEUSDT", "ZECUSDT", "LABUSDT", "XLMUSDT", "PORTALUSDT", "STGUSDT",
    "ALLOUSDT", "WLDUSDT", "NEARUSDT", "PLAYUSDT", "HUSDT", "DOGEUSDT",
    "SUIUSDT", "CLUSDT", "HOMEUSDT", "ASTERUSDT", "TONUSDT", "1000PEPEUSDT",
    "AIAUSDT", "ADAUSDT", "MUUSDT", "ONDOUSDT", "FETUSDT", "TAOUSDT",
    "HIVEUSDT", "GUNUSDT", "BCHUSDT",
]

STUDY_DAYS = 365 * 4            # long window: weekly horizon needs the history
FUNDING_DAYS = 400             # matches the validated-universe replay window
# Honest default: "last" = true close-to-close weekly returns. ForecastAGLT's
# "average" (block-mean) aggregation manufactures a momentum edge via MA
# autocorrelation and isn't tradable — we print it alongside ONLY to expose the
# artifact, never as the economic claim.
SPEC = WindowSpec(step_days=7, lookback=15, agg="last")
SPEC_AVG = WindowSpec(step_days=7, lookback=15, agg="average")


def _ms_window(days: int) -> tuple[int, int]:
    end = (int(time.time() * 1000) // 3_600_000) * 3_600_000 - 3_600_000 * 6
    return end - days * 24 * 3_600_000, end


def main() -> None:
    study_start, end = _ms_window(STUDY_DAYS)
    fund_start, _ = _ms_window(FUNDING_DAYS)

    union = list(dict.fromkeys(STUDY_UNIVERSE + FUNDING_UNIVERSE))
    print(f"=== price panel: {len(union)} coins, {STUDY_DAYS//365}y (1h) ===")
    price = build_panel(union, "1h", study_start, end)

    cfg = WalkForwardConfig()
    print(f"\nwalk-forward: {cfg.oos_blocks}/{cfg.n_blocks} OOS blocks, "
          f"embargo {cfg.embargo_steps} steps, min_train {cfg.min_train}")

    # ---- Readout 1: standalone directional skill + money ----
    # Run BOTH aggregations so the average_prices artifact is visible: the edge
    # only exists on non-tradable block-mean returns and collapses on close-to-
    # close. `preds` (the honest "last" set) is what the overlay then consumes.
    print("\n" + "=" * 64)
    print("READOUT 1 — STANDALONE weekly directional (after cost)")
    print("=" * 64)
    preds = None
    for spec, tag in ((SPEC_AVG, 'agg="average"  (ForecastAGLT — artifact, NOT tradable)'),
                      (SPEC, 'agg="last"     (close-to-close — HONEST / tradable)')):
        samples = build_pooled_samples(price, spec)
        p = run_walkforward(samples, spec, cfg)
        if p.empty:
            print(f"  {tag}: no OOS predictions")
            continue
        print(f"\n--- {tag} ---")
        print(f"    {len(samples)} samples / {p['coin'].nunique()} coins / "
              f"{len(p)//4} OOS pre- per model, "
              f"{p['t'].min().date()}..{p['t'].max().date()}")
        with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
            print(evaluate.summarize(p).to_string())
        if spec is SPEC:
            preds = p
            # The decisive check: does the best learned model beat the drift
            # baselines for a REAL reason, or is it just harvesting market drift?
            for mdl in ("gbm", "mlp"):
                dd = evaluate.dedrift(p, mdl)
                print(f'  de-drift [{mdl}] (cross-sectionally demeaned y): {dd}')
    if preds is None:
        raise SystemExit("no honest OOS predictions — relax n_blocks/min_train")

    # ---- Readout 2: funding-leg overlay ----
    print("\n" + "=" * 64)
    print("READOUT 2 — OVERLAY: predictor as Δfunding side-filter")
    print("=" * 64)
    fprice = {s: price[s] for s in FUNDING_UNIVERSE if s in price}
    # funding window only; reuse the cache (build_funding_panel is cache-first).
    funding = build_funding_panel(list(fprice), fund_start, end)
    legs = overlay.funding_legs(fprice, funding, DFundingParams())
    print(f"funding legs replayed: {len(legs)} over "
          f"{legs['sym'].nunique() if not legs.empty else 0} symbols")
    if legs.empty:
        print("no funding legs — overlay skipped")
        return
    for model in ("gbm", "mlp"):
        matched = overlay.match_predictions(legs, preds, model, SPEC.step_days)
        res = overlay.overlay_compare(matched, model)
        print(f"\n[{model}] coverage {res.get('coverage',0)}/{res['total_legs']} legs "
              f"have an OOS prediction")
        if res.get("coverage"):
            print(f"   kept {res['kept']}/{res['coverage']} (sign agrees with side)")
            print(f"   raw      : {res['raw']}")
            print(f"   filtered : {res['filtered']}")


if __name__ == "__main__":
    main()
