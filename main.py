"""
Backtest the Monte Carlo model's 90% prediction interval for calibration.

Runs two Monte Carlo forecasting models walk-forward over a (synthetic,
see montecarlo/market.py) daily price history, at three horizons, and
reports how often reality actually landed inside the model's predicted 90%
interval. Results and plots are written to output/. See VALIDATION.md for
the full write-up and interpretation.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from montecarlo.backtest import coverage_stats, stratify_by_event, walk_forward_backtest
from montecarlo.market import generate_synthetic_market
from montecarlo.models import BlockBootstrapModel, GBMModel

OUTPUT_DIR = "output"
WINDOW = 252         # trailing trading days used to calibrate the model
N_SIMS = 2000        # Monte Carlo simulations per forecast
CONF = 0.90
HORIZONS = [5, 21, 63]  # trading days ahead: ~1 week, ~1 month, ~1 quarter

PALETTE = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "blue": "#2a78d6",
    "aqua": "#1baf7a",
    "red": "#e34948",
    "good": "#0ca30c",
    "critical": "#d03b3b",
}


def style_axes(ax):
    ax.set_facecolor(PALETTE["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["ink2"], labelsize=9)
    ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(PALETTE["ink2"])
    ax.yaxis.label.set_color(PALETTE["ink2"])


def plot_fan_chart(prices, log_returns, t, window, horizon, out, n_sims=1000, n_draw=40):
    rng = np.random.default_rng(123)
    model = GBMModel()
    model.fit(log_returns[t - window: t])
    s0 = prices[t]
    paths = model.simulate_paths(s0, horizon, n_sims, rng)
    full_paths = np.hstack([np.full((n_sims, 1), s0), paths])

    days = np.arange(horizon + 1)
    lo = np.quantile(full_paths, 0.05, axis=0)
    hi = np.quantile(full_paths, 0.95, axis=0)
    median = np.quantile(full_paths, 0.5, axis=0)
    actual = prices[t: t + horizon + 1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(PALETTE["surface"])
    style_axes(ax)

    for i in range(n_draw):
        ax.plot(days, full_paths[i], color=PALETTE["blue"], alpha=0.10, linewidth=1, zorder=1)
    ax.fill_between(days, lo, hi, color=PALETTE["blue"], alpha=0.18, zorder=2,
                     label="90% predicted interval")
    ax.plot(days, median, color=PALETTE["blue"], linewidth=2, zorder=3, label="Median simulated path")
    ax.plot(days, actual, color=PALETTE["red"], linewidth=2.3, zorder=4, label="Actual realized price")

    miss = not (lo[-1] <= actual[-1] <= hi[-1])
    tag = "MISS: actual landed outside the 90% interval" if miss else "actual landed inside the 90% interval"
    ax.set_title(f"Monte Carlo fan chart — GBM model, {horizon}-trading-day horizon\n({tag})",
                 color=PALETTE["ink"], fontsize=11, loc="left")
    ax.set_xlabel("Trading days ahead")
    ax.set_ylabel("Price")
    ax.legend(frameon=False, loc="upper left", labelcolor=PALETTE["ink2"], fontsize=9)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_coverage_summary(stats_by_model, out):
    model_names = list(stats_by_model.keys())
    colors = [PALETTE["blue"], PALETTE["aqua"]]
    x = np.arange(len(HORIZONS))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(PALETTE["surface"])
    style_axes(ax)

    for i, name in enumerate(model_names):
        cov = [stats_by_model[name][h]["coverage"] * 100 for h in HORIZONS]
        ci = [stats_by_model[name][h]["ci_95"] for h in HORIZONS]
        yerr_lo = [cov[j] - ci[j][0] * 100 for j in range(len(HORIZONS))]
        yerr_hi = [ci[j][1] * 100 - cov[j] for j in range(len(HORIZONS))]
        offset = (i - (len(model_names) - 1) / 2) * bar_w
        ax.bar(x + offset, cov, width=bar_w, color=colors[i], label=name,
               yerr=[yerr_lo, yerr_hi], capsize=3,
               error_kw={"ecolor": PALETTE["ink2"], "linewidth": 1})

    ax.axhline(CONF * 100, color=PALETTE["muted"], linestyle="--", linewidth=1.3, zorder=0)
    ax.text(len(HORIZONS) - 0.5, CONF * 100 + 1.2, "90% nominal", color=PALETTE["muted"], fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in HORIZONS])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_xlabel("Forecast horizon")
    ax.set_title("Backtested coverage of the 90% prediction interval\n(walk-forward, non-overlapping windows, error bars = 95% Wilson CI)",
                 color=PALETTE["ink"], fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower right", labelcolor=PALETTE["ink2"], fontsize=9)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_stratified_coverage(stratified_all, out):
    """
    Coverage conditioned on whether an actual jump fell inside the forecast
    horizon, pooled across all three horizons per model. This is the check
    that an aggregate coverage number can hide: a model can look ~90%
    calibrated overall while being badly overconfident specifically on the
    windows that contained the event it can't see coming, simply because
    those windows are rare.
    """
    model_names = list(stratified_all.keys())
    colors = [PALETTE["blue"], PALETTE["aqua"]]
    groups = ["no_jump", "jump_in_horizon"]
    group_labels = ["No jump in horizon", "Jump in horizon"]
    x = np.arange(len(groups))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 5))
    fig.patch.set_facecolor(PALETTE["surface"])
    style_axes(ax)

    for i, name in enumerate(model_names):
        pooled = {"no_jump": {"hits": 0, "n": 0}, "jump_in_horizon": {"hits": 0, "n": 0}}
        for horizon_strat in stratified_all[name].values():
            for g in groups:
                if g in horizon_strat:
                    pooled[g]["hits"] += horizon_strat[g]["hits"]
                    pooled[g]["n"] += horizon_strat[g]["n"]

        cov, yerr_lo, yerr_hi = [], [], []
        for g in groups:
            n, k = pooled[g]["n"], pooled[g]["hits"]
            if n == 0:
                cov.append(0); yerr_lo.append(0); yerr_hi.append(0)
                continue
            p = k / n
            z = 1.959963985
            denom = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denom
            half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
            cov.append(p * 100)
            yerr_lo.append(max(0, p * 100 - (center - half) * 100))
            yerr_hi.append(max(0, (center + half) * 100 - p * 100))

        offset = (i - (len(model_names) - 1) / 2) * bar_w
        ax.bar(x + offset, cov, width=bar_w, color=colors[i], label=name,
               yerr=[yerr_lo, yerr_hi], capsize=3,
               error_kw={"ecolor": PALETTE["ink2"], "linewidth": 1})

    ax.axhline(CONF * 100, color=PALETTE["muted"], linestyle="--", linewidth=1.3, zorder=0)
    ax.text(0.98, CONF * 100 + 1.5, "90% nominal", color=PALETTE["muted"], fontsize=9,
            ha="right", transform=ax.get_yaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Coverage conditional on a jump falling inside the forecast horizon\n(pooled across all horizons, error bars = 95% Wilson CI)",
                 color=PALETTE["ink"], fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower left", labelcolor=PALETTE["ink2"], fontsize=9)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_rolling_coverage(records, variance, out, roll=60):
    t_vals = np.array([r["t"] for r in records])
    hits = np.array([r["hit"] for r in records], dtype=float)
    rolling_cov = pd.Series(hits).rolling(roll, min_periods=roll // 2).mean().to_numpy()
    ann_vol = np.sqrt(variance[t_vals]) * np.sqrt(252) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(PALETTE["surface"])
    for ax in (ax1, ax2):
        style_axes(ax)

    ax1.plot(t_vals, rolling_cov * 100, color=PALETTE["blue"], linewidth=1.6,
             label=f"Rolling {roll}-window coverage")
    ax1.axhline(CONF * 100, color=PALETTE["muted"], linestyle="--", linewidth=1.2, label="90% nominal")
    ax1.set_ylabel("Coverage (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("GBM model: rolling coverage tracks realized volatility\n(21-day horizon, overlapping windows — for diagnosis, not the headline statistic)",
                  color=PALETTE["ink"], fontsize=11, loc="left")
    ax1.legend(frameon=False, loc="lower left", labelcolor=PALETTE["ink2"], fontsize=9)

    ax2.plot(t_vals, ann_vol, color=PALETTE["aqua"], linewidth=1.3)
    ax2.fill_between(t_vals, 0, ann_vol, color=PALETTE["aqua"], alpha=0.15)
    ax2.set_ylabel("Realized ann. vol (%)")
    ax2.set_xlabel("Trading day index")

    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    prices, returns, variance, jump_occurred = generate_synthetic_market()
    log_returns = np.diff(np.log(prices))

    model_factories = {
        "GBM (iid normal)": lambda: GBMModel(),
        "Block bootstrap": lambda: BlockBootstrapModel(block_size=10),
    }

    records_all = {}
    stats_all = {}
    stratified_all = {}
    for name, factory in model_factories.items():
        records_all[name] = {}
        stats_all[name] = {}
        stratified_all[name] = {}
        for horizon in HORIZONS:
            # step=horizon -> non-overlapping windows: independent samples,
            # the honest basis for a coverage statistic.
            records = walk_forward_backtest(
                prices, factory, window=WINDOW, horizon=horizon,
                n_sims=N_SIMS, conf=CONF, step=horizon, seed=42,
                event_flags=jump_occurred,
            )
            s = coverage_stats(records, nominal=CONF)
            records_all[name][horizon] = records
            stats_all[name][horizon] = s
            print(
                f"{name:20s} horizon={horizon:3d}d  n={s['n']:4d}  "
                f"coverage={s['coverage']*100:5.1f}%  "
                f"95% CI=({s['ci_95'][0]*100:5.1f}%, {s['ci_95'][1]*100:5.1f}%)  "
                f"p(vs 90%)={s['p_value']:.4f}"
            )

            with_event, without_event = stratify_by_event(records)
            strat = {}
            for label, group in (("jump_in_horizon", with_event), ("no_jump", without_event)):
                if len(group) >= 5:
                    strat[label] = coverage_stats(group, nominal=CONF)
            stratified_all[name][horizon] = strat
            if "jump_in_horizon" in strat and "no_jump" in strat:
                j, nj = strat["jump_in_horizon"], strat["no_jump"]
                print(
                    f"{'':20s}   -> windows with a jump inside the horizon "
                    f"(n={j['n']}): coverage={j['coverage']*100:5.1f}%   "
                    f"windows without (n={nj['n']}): coverage={nj['coverage']*100:5.1f}%"
                )

    with open(f"{OUTPUT_DIR}/backtest_summary.json", "w") as f:
        json.dump({"aggregate": stats_all, "stratified_by_jump": stratified_all}, f, indent=2, default=float)

    # Rolling diagnostic: overlapping windows so we get a coverage estimate
    # at (almost) every day, to see *when* miscalibration happens.
    rolling_records = walk_forward_backtest(
        prices, lambda: GBMModel(), window=WINDOW, horizon=21,
        n_sims=N_SIMS, conf=CONF, step=1, seed=42,
    )
    plot_rolling_coverage(rolling_records, variance, out=f"{OUTPUT_DIR}/rolling_coverage.png")

    # Fan chart: illustrate on a window where the GBM model actually missed;
    # failing that, a window where a jump fell inside the horizon (still
    # instructive even if it happened to land inside the band).
    gbm_63 = records_all["GBM (iid normal)"][63]
    misses = [r["t"] for r in gbm_63 if not r["hit"]]
    jump_windows = [r["t"] for r in gbm_63 if r.get("event_in_horizon")]
    snapshot_t = (misses or jump_windows or [gbm_63[0]["t"]])[0]
    plot_fan_chart(prices, log_returns, snapshot_t, WINDOW, 63, out=f"{OUTPUT_DIR}/fan_chart.png")

    plot_coverage_summary(stats_all, out=f"{OUTPUT_DIR}/coverage_summary.png")
    plot_stratified_coverage(stratified_all, out=f"{OUTPUT_DIR}/stratified_coverage.png")

    print(f"\nWrote plots and backtest_summary.json to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
