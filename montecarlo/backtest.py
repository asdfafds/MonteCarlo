"""
Walk-forward backtest: the calibration check itself.

At every candidate day t we may only use prices[:t] (data "available then")
to calibrate the model, then we simulate `horizon` trading days forward and
ask whether the price that actually occurred at t + horizon fell inside the
model's `conf` prediction interval. This is what separates a real backtest
from a fitted-in-hindsight sanity check: the model at day t has never seen
day t+1, let alone day t+horizon.

If the model is well-calibrated, the hit rate across many such windows
should converge to `conf` (e.g. 90%). A hit rate well below `conf` means the
model's intervals are too narrow -- it is overconfident.
"""
import numpy as np
from scipy import stats


def walk_forward_backtest(
    prices, model_factory, window=252, horizon=21, n_sims=2000,
    conf=0.90, step=1, seed=42, event_flags=None,
):
    """
    `event_flags`, if given, is a boolean array aligned to `prices` marking
    days on which some structural event happened (here: a jump). Each
    record then also notes whether such an event fell inside its own
    forecast window (t, t+horizon] -- letting the caller stratify coverage
    by "was this a business-as-usual window or one containing a shock the
    model's trailing calibration couldn't have seen coming."
    """
    rng = np.random.default_rng(seed)
    log_returns = np.diff(np.log(prices))

    records = []
    t = window
    while t + horizon < len(prices):
        trailing = log_returns[t - window: t]
        model = model_factory()
        model.fit(trailing)

        s0 = prices[t]
        lo, hi, _ = model.interval(s0, horizon, n_sims, rng, conf)
        actual = prices[t + horizon]
        hit = bool(lo <= actual <= hi)

        record = {
            "t": t, "s0": float(s0), "actual": float(actual),
            "lo": float(lo), "hi": float(hi), "hit": hit,
        }
        if event_flags is not None:
            record["event_in_horizon"] = bool(event_flags[t + 1: t + horizon + 1].any())
        records.append(record)
        t += step
    return records


def stratify_by_event(records):
    """Split records into (event-in-horizon, no-event) groups for a
    conditional coverage comparison."""
    with_event = [r for r in records if r.get("event_in_horizon")]
    without_event = [r for r in records if not r.get("event_in_horizon")]
    return with_event, without_event


def coverage_stats(records, nominal=0.90):
    hits = np.array([r["hit"] for r in records])
    n = len(hits)
    k = int(hits.sum())
    p_hat = k / n

    # Wilson score interval -- more reliable than a normal approx near 0.90
    # with a few hundred samples.
    z = 1.959963985
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom

    # Two-sided exact binomial test: H0 = true coverage is `nominal`.
    binom = stats.binomtest(k, n, nominal, alternative="two-sided")

    return {
        "n": n,
        "hits": k,
        "coverage": p_hat,
        "ci_95": (center - half, center + half),
        "nominal": nominal,
        "p_value": binom.pvalue,
    }
