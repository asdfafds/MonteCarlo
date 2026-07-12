"""
Historical price series used to backtest the Monte Carlo model against.

This sandboxed environment has no outbound access to a real market data feed
(Yahoo Finance, etc. are blocked at the network boundary), so we substitute a
synthetic-but-realistic "ground truth" series: a GARCH(1,1) volatility
process with Student-t shocks, plus rare jumps. Real daily equity/index
returns show three features a plain constant-volatility random walk does
not: volatility clustering (calm and turbulent periods persist), fat tails
(extreme daily moves are more common than a normal distribution predicts),
and occasional discontinuous jumps (crashes, earnings surprises, macro
shocks) that aren't part of the day-to-day diffusion at all. GARCH(1,1) +
Student-t + a compound Poisson jump term reproduces all three, which is
exactly what makes it a fair adversary for testing whether a naive Monte
Carlo model -- one that only ever sees smooth trailing-window volatility --
is overconfident.

Everything downstream (models.py, backtest.py) only ever sees a plain array
of daily closing prices, so swapping this generator for `pandas.read_csv` on
a real price history requires no other code changes.
"""
import numpy as np


def generate_synthetic_market(
    n_days=6048,
    s0=100.0,
    mu=0.00030,
    omega=1.5e-6,
    alpha=0.08,
    beta=0.90,
    t_df=5,
    jump_prob=0.003,
    jump_mean=-0.04,
    jump_std=0.05,
    seed=7,
):
    """
    Simulate `n_days` of daily closes via GARCH(1,1) with standardized
    Student-t innovations, plus a compound Poisson jump term.

    Diffusion part:
        r_t = mu + sqrt(h_t) * z_t
        h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}
    z_t is a Student-t draw scaled to unit variance, so `h_t` remains the
    conditional variance regardless of the tail-fatness parameter `t_df`.
    alpha + beta = 0.98 gives realistic, slowly-decaying volatility
    clustering (a shock takes a couple months to fade).

    Jump part: with probability `jump_prob` per day (~1.5x/year at the
    default), an extra jump ~ N(jump_mean, jump_std) is added to that day's
    log return -- a crash-skewed stand-in for the discontinuous, largely
    unforecastable moves (crashes, surprise macro prints) that a smooth
    diffusion process doesn't generate on its own, and that a trailing-vol
    Monte Carlo model has no way to see coming.
    """
    rng = np.random.default_rng(seed)
    returns = np.empty(n_days)
    h = np.empty(n_days)
    h[0] = omega / (1 - alpha - beta)
    t_scale = np.sqrt((t_df - 2) / t_df)

    jump_occurred = rng.random(n_days) < jump_prob
    jump_size = rng.normal(jump_mean, jump_std, n_days)

    for t in range(n_days):
        if t > 0:
            h[t] = omega + alpha * returns[t - 1] ** 2 + beta * h[t - 1]
        z = rng.standard_t(t_df) * t_scale
        returns[t] = mu + np.sqrt(h[t]) * z
        if jump_occurred[t]:
            returns[t] += jump_size[t]

    prices = s0 * np.exp(np.cumsum(returns))
    return prices, returns, h, jump_occurred
