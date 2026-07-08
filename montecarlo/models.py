"""
Monte Carlo forecasting models. Every model exposes the same three-method
interface (`fit`, `simulate_paths`, `interval`) so the backtest engine can
drive either one identically:

  fit(trailing_log_returns)         -- calibrate using ONLY past data
  simulate_paths(s0, horizon, n_sims, rng) -> (n_sims, horizon) price paths
  interval(s0, horizon, n_sims, rng, conf) -> (lo, hi, paths)

GBMModel is the "naive" model most Monte Carlo stock-path tutorials build:
geometric Brownian motion with drift/vol estimated from a trailing window,
and iid normal daily shocks. BlockBootstrapModel is a second, more honest
Monte Carlo model that resamples contiguous blocks of the *actual* historical
returns instead of assuming normality -- it inherits whatever fat tails and
short-run volatility clustering are really in the data, without needing to
fit a parametric model to them.
"""
import numpy as np


class GBMModel:
    """Geometric Brownian motion: iid N(mu, sigma^2) daily log returns."""

    name = "GBM (iid normal)"

    def fit(self, trailing_log_returns):
        self.mu = np.mean(trailing_log_returns)
        self.sigma = np.std(trailing_log_returns, ddof=1)

    def simulate_paths(self, s0, horizon, n_sims, rng):
        z = rng.standard_normal((n_sims, horizon))
        daily_log_return = (self.mu - 0.5 * self.sigma**2) + self.sigma * z
        log_paths = np.cumsum(daily_log_return, axis=1)
        return s0 * np.exp(log_paths)

    def interval(self, s0, horizon, n_sims, rng, conf=0.90):
        paths = self.simulate_paths(s0, horizon, n_sims, rng)
        terminal = paths[:, -1]
        tail = (1 - conf) / 2
        lo, hi = np.quantile(terminal, [tail, 1 - tail])
        return lo, hi, paths


class BlockBootstrapModel:
    """
    Stationary block bootstrap: builds each simulated path by splicing
    together randomly-chosen contiguous blocks of real historical daily log
    returns (with replacement). Preserves the empirical return distribution
    (fat tails, skew) and short-range autocorrelation/volatility clustering
    within a block, unlike an iid normal model.
    """

    name = "Block bootstrap"

    def __init__(self, block_size=10):
        self.block_size = block_size

    def fit(self, trailing_log_returns):
        self.history = np.asarray(trailing_log_returns)

    def simulate_paths(self, s0, horizon, n_sims, rng):
        n = len(self.history)
        b = self.block_size
        n_blocks = int(np.ceil(horizon / b))

        starts = rng.integers(0, n - b, size=(n_sims, n_blocks))
        offsets = np.arange(b)
        idx = starts[:, :, None] + offsets[None, None, :]
        idx = idx.reshape(n_sims, n_blocks * b)[:, :horizon]

        sampled_returns = self.history[idx]
        log_paths = np.cumsum(sampled_returns, axis=1)
        return s0 * np.exp(log_paths)

    def interval(self, s0, horizon, n_sims, rng, conf=0.90):
        paths = self.simulate_paths(s0, horizon, n_sims, rng)
        terminal = paths[:, -1]
        tail = (1 - conf) / 2
        lo, hi = np.quantile(terminal, [tail, 1 - tail])
        return lo, hi, paths
