"""Measure sleeve expectancy + correlations + the combined allocator book.

Pure analysis over a dict of {sleeve: daily_return_pct_array} (all arrays the
same length, one entry per calendar day; 0.0 on a no-trade day). Reuses the live
`allocate()` so the combined book reflects what the system would actually do.

Key outputs per sleeve:
  * ann_sharpe    — mean/std of daily returns × √365 (0 if flat)
  * mean_bps_day  — average daily return in bps of reference equity
  * pct_pos_day   — fraction of active days that were positive
  * h1/h2 sharpe  — both-halves Sharpe (stability: a sleeve that only works in
                    one half is NOT stable, per the stable-max-PnL directive)

And for the basket:
  * the allocator weights (HRP by default, stability-gated),
  * the COMBINED daily series under those weights → its Sharpe / mean / both-halves,
  * diversification_ratio = combined_sharpe / best_single_sleeve_sharpe
    (> 1 means diversification genuinely helped; ≤ 1 means the basket is not
    better than just holding the best sleeve — no free lunch here).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.services.portfolio import allocate

ANN = np.sqrt(365.0)


def _sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / sd * ANN) if sd > 1e-12 else 0.0


@dataclass
class SleeveStat:
    name: str
    n_days: int
    n_active: int
    mean_bps_day: float
    ann_sharpe: float
    pct_pos_day: float
    h1_sharpe: float
    h2_sharpe: float
    total_pct: float

    @property
    def both_halves_positive(self) -> bool:
        return self.h1_sharpe > 0 and self.h2_sharpe > 0


def sleeve_stats(name: str, r: np.ndarray) -> SleeveStat:
    r = np.asarray(r, dtype=float)
    active = r[r != 0.0]
    half = len(r) // 2
    return SleeveStat(
        name=name,
        n_days=len(r),
        n_active=int(active.size),
        mean_bps_day=float(r.mean() * 100.0),          # r is in pct → ×100 = bps
        ann_sharpe=_sharpe(r),
        pct_pos_day=float((active > 0).mean()) if active.size else 0.0,
        h1_sharpe=_sharpe(r[:half]),
        h2_sharpe=_sharpe(r[half:]),
        total_pct=float(r.sum()),
    )


@dataclass
class PortfolioReport:
    sleeves: list[SleeveStat]
    corr: np.ndarray                # correlation matrix, sleeve order
    corr_labels: list[str]
    weights: dict[str, float]
    combined: SleeveStat
    diversification_ratio: float
    mean_pairwise_corr: float


def _corr_matrix(mat: np.ndarray) -> np.ndarray:
    """Pearson correlation across columns (sleeves), robust to flat columns."""
    n = mat.shape[1]
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = mat[:, i], mat[:, j]
            if a.std() < 1e-12 or b.std() < 1e-12:
                c = 0.0
            else:
                c = float(np.corrcoef(a, b)[0, 1])
            out[i, j] = out[j, i] = c
    return out


def measure(returns: dict[str, np.ndarray], *,
            method: str = "hrp",
            min_sharpe: float = 0.0,
            **allocate_kwargs) -> PortfolioReport:
    """Full portfolio diagnostic. `returns` values must be equal-length arrays.

    `method`/`min_sharpe`/kwargs pass through to the live `allocate()` so the
    combined book reflects the real allocator (including its stability gates)."""
    names = list(returns)
    mat = np.column_stack([np.asarray(returns[n], dtype=float) for n in names])
    stats = [sleeve_stats(n, mat[:, i]) for i, n in enumerate(names)]

    res = allocate(returns, method=method, min_sharpe=min_sharpe,
                   **allocate_kwargs)
    w = dict(res.weights)
    # Combined daily series under the allocator weights.
    wvec = np.array([w.get(n, 0.0) for n in names])
    combined_series = mat @ wvec
    combined = sleeve_stats("COMBINED", combined_series)

    corr = _corr_matrix(mat)
    iu = np.triu_indices(len(names), k=1)
    mean_pair = float(corr[iu].mean()) if len(names) > 1 else 0.0
    best_single = max((s.ann_sharpe for s in stats), default=0.0)
    div_ratio = (combined.ann_sharpe / best_single) if best_single > 1e-9 else 0.0

    return PortfolioReport(
        sleeves=stats, corr=corr, corr_labels=names, weights=w,
        combined=combined, diversification_ratio=div_ratio,
        mean_pairwise_corr=mean_pair,
    )


def format_report(rep: PortfolioReport) -> str:
    lines = ["=== per-sleeve (standalone) ==="]
    lines.append(f"{'sleeve':<16}{'nAct':>5}{'mean_bps':>9}{'Sharpe':>8}"
                 f"{'pos%':>6}{'h1_shp':>8}{'h2_shp':>8}{'stable':>7}")
    for s in sorted(rep.sleeves, key=lambda x: -x.ann_sharpe):
        lines.append(f"{s.name:<16}{s.n_active:>5}{s.mean_bps_day:>9.2f}"
                     f"{s.ann_sharpe:>8.2f}{s.pct_pos_day*100:>5.0f}%"
                     f"{s.h1_sharpe:>8.2f}{s.h2_sharpe:>8.2f}"
                     f"{'yes' if s.both_halves_positive else 'NO':>7}")
    lines.append("\n=== pairwise correlation ===")
    hdr = "".join(f"{n[:8]:>9}" for n in rep.corr_labels)
    lines.append(f"{'':<10}{hdr}")
    for i, n in enumerate(rep.corr_labels):
        row = "".join(f"{rep.corr[i, j]:>9.2f}" for j in range(len(rep.corr_labels)))
        lines.append(f"{n[:10]:<10}{row}")
    lines.append(f"mean pairwise corr: {rep.mean_pairwise_corr:+.2f}")
    lines.append("\n=== combined allocator book ===")
    lines.append(f"weights: " + ", ".join(f"{k}={v:.2f}" for k, v in rep.weights.items()))
    c = rep.combined
    lines.append(f"combined Sharpe {c.ann_sharpe:.2f}  mean {c.mean_bps_day:.2f}bps/day "
                 f"pos {c.pct_pos_day*100:.0f}%  h1 {c.h1_sharpe:.2f} h2 {c.h2_sharpe:.2f} "
                 f"({'STABLE' if c.both_halves_positive else 'not both-halves+'})")
    lines.append(f"diversification ratio (combined/best-single): "
                 f"{rep.diversification_ratio:.2f}  "
                 f"({'helps' if rep.diversification_ratio > 1.02 else 'NO benefit'})")
    return "\n".join(lines)
