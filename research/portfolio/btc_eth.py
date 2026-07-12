"""BTC-ETH stat-arb: is the cointegration/mean-reversion regime back?

Prior finding (project-pairs-decay): ETH/BTC cointegration BROKE ~Apr 2026,
0/99 CPCV positive on 2y, but "synthetic perp legs win when the regime returns."
It's now Jul 2026 — this re-tests on fresh 1h mainnet data:

  1. rolling Dickey-Fuller t-stat on the spread (regime ON when << -2.9),
  2. half-life of mean reversion + Hurst exponent (per month),
  3. a z-score pairs backtest (entry 2 / exit 0.5 / stop 3.5, 60d rolling
     hedge ratio) net of costs, broken down by month + both halves.

Reuses the repo's PairsParams convention. Writes to /tmp/btceth.txt (flushed).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

from research.portfolio import sleeves as S

OUT = open("/tmp/btceth.txt", "w")
def emit(*a):
    line = " ".join(str(x) for x in a)
    print(line); OUT.write(line + "\n"); OUT.flush()


def closes(sym: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    ks = S.fetch_klines(sym, "1h", start, end)
    t = np.array([int(r[0]) for r in ks])
    c = np.array([float(r[4]) for r in ks])
    return t, c


def df_tstat(resid: np.ndarray) -> float:
    """Dickey-Fuller t-stat: Δr_t = a + b*r_{t-1}; t-stat of b. More negative =
    more mean-reverting. 5% crit ≈ -2.9, 1% ≈ -3.4."""
    r = resid[:-1]
    dr = np.diff(resid)
    X = np.column_stack([np.ones_like(r), r])
    beta, *_ = np.linalg.lstsq(X, dr, rcond=None)
    pred = X @ beta
    dof = len(dr) - 2
    if dof <= 0:
        return 0.0
    s2 = ((dr - pred) ** 2).sum() / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_b = np.sqrt(s2 * xtx_inv[1, 1])
    return float(beta[1] / se_b) if se_b > 0 else 0.0


def half_life(spread: np.ndarray) -> float:
    r = spread[:-1]
    dr = np.diff(spread)
    X = np.column_stack([np.ones_like(r), r])
    beta, *_ = np.linalg.lstsq(X, dr, rcond=None)
    b = beta[1]
    return float(-np.log(2) / b) if b < 0 else float("inf")


def hurst(ts: np.ndarray) -> float:
    """Hurst via variance of lagged differences. <0.5 mean-reverting, >0.5 trending."""
    lags = range(2, 20)
    tau = [np.sqrt(np.std(ts[lag:] - ts[:-lag])) for lag in lags]
    m = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(m[0] * 2.0)


def rolling_hedge_z(a: np.ndarray, b: np.ndarray, win: int):
    """Rolling OLS hedge ratio (log a on log b) + z-score of the residual.
    Returns (z array, beta array), NaN for the warmup region. PIT: window ends
    at t, so z_t uses only data up to t."""
    la, lb = np.log(a), np.log(b)
    n = len(a)
    z = np.full(n, np.nan)
    betas = np.full(n, np.nan)
    for t in range(win, n):
        x = lb[t - win:t]
        y = la[t - win:t]
        X = np.column_stack([np.ones(win), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        mu, sd = resid.mean(), resid.std()
        if sd > 0:
            cur = la[t] - (coef[0] + coef[1] * lb[t])
            z[t] = (cur - mu) / sd
            betas[t] = coef[1]
    return z, betas


def backtest(a, b, ts, z, betas, win, *, z_entry=2.0, z_exit=0.5, z_stop=3.5,
             cost_bps=5.0, regime_gate=None, regime_win=24 * 20):
    """Trade the spread: when z>z_entry short-A/long-B (spread rich), z<-entry
    long-A/short-B. Exit |z|<z_exit, stop |z|>z_stop. PnL in return of a $1
    notional-per-leg book. Cost charged per leg on entry+exit.

    If `regime_gate` is set, only ENTER when the trailing-`regime_win` DF-t on
    the residual spread is < regime_gate (i.e. the pair is CURRENTLY mean-
    reverting) — PIT, uses only the trailing window. This is the regime-timing
    overlay: trade the pair only when the regime is on, sit out when it's off."""
    la, lb = np.log(a), np.log(b)
    resid_full = la - np.where(np.isnan(betas), 1.0, betas) * lb
    pos = 0            # +1 = long spread (long A short B), -1 = short spread
    entry_i = 0
    trades = []
    for t in range(win, len(a)):
        if np.isnan(z[t]):
            continue
        if pos == 0:
            if regime_gate is not None:
                seg = resid_full[t - regime_win:t]
                if len(seg) < 50 or df_tstat(seg) >= regime_gate:
                    continue      # regime not mean-reverting → sit out
            if z[t] > z_entry:
                pos, entry_i = -1, t
            elif z[t] < -z_entry:
                pos, entry_i = 1, t
        else:
            exit_now = abs(z[t]) < z_exit or abs(z[t]) > z_stop
            if exit_now:
                # spread return over hold = Δ(logA) - beta*Δ(logB), signed by pos
                be = betas[entry_i] if not np.isnan(betas[entry_i]) else 1.0
                sret = ((la[t] - la[entry_i]) - be * (lb[t] - lb[entry_i])) * pos
                net = sret - 2 * cost_bps / 1e4 * (1 + abs(be))
                trades.append((ts[t], net, "stop" if abs(z[t]) > z_stop else "exit"))
                pos = 0
    return trades


def main():
    end = (int(time.time() * 1000) // S.DAY_MS) * S.DAY_MS
    start = end - 365 * S.DAY_MS
    ta, a = closes("BTCUSDT", start, end)
    tb, b = closes("ETHUSDT", start, end)
    n = min(len(a), len(b))
    a, b, ts = a[:n], b[:n], ta[:n]
    emit(f"BTC-ETH 1h, {datetime.utcfromtimestamp(start/1000).date()} .. "
         f"{datetime.utcfromtimestamp(end/1000).date()}  ({n} bars)")

    # Monthly regime diagnostics on the log-ratio spread (simple beta=1 proxy
    # for the regime read; the backtest uses the full rolling hedge ratio).
    ratio = np.log(a) - np.log(b)
    emit("\n=== monthly regime (log ratio) ===")
    emit(f"{'month':<9}{'DF_t':>8}{'half_life_h':>12}{'hurst':>8}  regime")
    months = {}
    for i in range(n):
        mk = datetime.utcfromtimestamp(ts[i] / 1000).strftime("%Y-%m")
        months.setdefault(mk, []).append(i)
    for mk in sorted(months):
        idx = months[mk]
        if len(idx) < 200:
            continue
        seg = ratio[idx[0]:idx[-1] + 1]
        dft = df_tstat(seg)
        hl = half_life(seg)
        hu = hurst(seg)
        reg = "MEAN-REVERT" if dft < -2.9 and 0 < hl < 24 * 20 else \
              ("weak" if dft < -2.0 else "trending/broken")
        hls = f"{hl:.0f}" if hl < 1e5 else "inf"
        emit(f"{mk:<9}{dft:>8.2f}{hls:>12}{hu:>8.2f}  {reg}")

    # Full-sample backtest with rolling hedge ratio.
    win = 24 * 60      # 60d at 1h (matches PairsParams.lookback_bars)
    z, betas = rolling_hedge_z(a, b, win)
    trades = backtest(a, b, ts, z, betas, win)
    emit(f"\n=== z-score pairs backtest (60d roll, entry2/exit0.5/stop3.5, 5bps/leg) ===")
    if not trades:
        emit("no trades"); OUT.close(); return
    nets = np.array([t[1] for t in trades])
    mid = ts[win + (n - win) // 2]
    h1 = nets[[t[0] < mid for t in trades]]
    h2 = nets[[t[0] >= mid for t in trades]]
    def rep(name, x):
        if len(x) == 0:
            emit(f"  {name:<8} n=0"); return
        sh = x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else 0
        emit(f"  {name:<8} n={len(x):>3}  WR {(x>0).mean()*100:>3.0f}%  "
             f"mean {x.mean()*1e4:>+7.1f}bps  sum {x.sum()*100:>+6.2f}%  t~{sh:>+4.2f}")
    rep("all", nets)
    rep("half1", h1)
    rep("half2", h2)
    stops = sum(1 for t in trades if t[2] == "stop")
    emit(f"  stops (regime-break exits): {stops}/{len(trades)}")

    # Regime-gated variant: only enter when the pair is currently mean-reverting.
    emit("\n=== SAME strategy + regime gate (enter only when trailing DF-t < -2.5) ===")
    for gate in (-2.5, -3.0):
        gt = backtest(a, b, ts, z, betas, win, regime_gate=gate)
        if gt:
            gn = np.array([t[1] for t in gt])
            sh = gn.mean() / gn.std() * np.sqrt(len(gn)) if gn.std() > 0 else 0
            gstops = sum(1 for t in gt if t[2] == "stop")
            emit(f"  gate {gate}: n={len(gn):>3}  WR {(gn>0).mean()*100:>3.0f}%  "
                 f"mean {gn.mean()*1e4:>+7.1f}bps  sum {gn.sum()*100:>+6.2f}%  "
                 f"t~{sh:>+4.2f}  stops {gstops}/{len(gt)}")
        else:
            emit(f"  gate {gate}: n=0 (regime never qualified)")
    emit("DONE")
    OUT.close()


if __name__ == "__main__":
    main()
