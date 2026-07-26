"""Turn an ensemble into bucket probabilities, then into sized paper trades."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Config
from .forecast import EnsembleForecast
from .markets import Bucket, MarketQuote

#: Probabilities are never allowed all the way to 0 or 1. Thirty-one members
#: cannot justify certainty, and an unclamped estimate makes Kelly explode.
PROB_FLOOR = 0.01
PROB_CEIL = 0.99


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def empirical_probability(members: list[float], bucket: Bucket) -> float:
    """Share of ensemble members landing in the bucket."""
    low, high = bucket.continuous_bounds()
    hits = sum(1 for m in members if low <= m < high)
    return hits / len(members)


def gaussian_probability(members: list[float], bucket: Bucket, min_sigma: float) -> float:
    """Probability under a normal fitted to the ensemble.

    Smooths the histogram, which matters because 31 members give a granularity
    of ~3.2 percentage points and empty buckets that are not really impossible.
    """
    n = len(members)
    mu = sum(members) / n
    if n > 1:
        var = sum((m - mu) ** 2 for m in members) / (n - 1)
        sigma = math.sqrt(var)
    else:
        sigma = 0.0
    sigma = max(sigma, min_sigma)
    low, high = bucket.continuous_bounds()
    return _normal_cdf(high, mu, sigma) - _normal_cdf(low, mu, sigma)


def bucket_probability(forecast: EnsembleForecast, bucket: Bucket, cfg: Config) -> float:
    """Model probability that the settled high lands in `bucket`."""
    members = forecast.members
    if cfg.probability_model == "empirical":
        p = empirical_probability(members, bucket)
    elif cfg.probability_model == "gaussian":
        p = gaussian_probability(members, bucket, cfg.min_sigma_f)
    elif cfg.probability_model == "blend":
        w = cfg.empirical_weight
        p = w * empirical_probability(members, bucket) + (1 - w) * gaussian_probability(
            members, bucket, cfg.min_sigma_f
        )
    else:
        raise ValueError(f"unknown probability_model: {cfg.probability_model!r}")
    return min(max(p, PROB_FLOOR), PROB_CEIL)


def kelly_fraction(prob: float, price: float) -> float:
    """Full-Kelly stake as a fraction of bankroll for a binary at `price`.

    A contract costing `price` pays 1, so net odds are (1-price)/price and the
    Kelly optimum reduces to (p - price) / (1 - price).
    """
    if not 0.0 < price < 1.0:
        return 0.0
    return max(0.0, (prob - price) / (1.0 - price))


@dataclass
class Signal:
    quote: MarketQuote
    side: str  # "YES" or "NO"
    model_prob: float  # probability of the side paying out
    price: float  # cost of the side, in dollars per contract
    edge: float  # model_prob - price
    kelly: float  # full-Kelly fraction of bankroll
    stake: float  # dollars actually staked after fractional Kelly + caps
    contracts: float
    forecast_mean: float
    forecast_sigma: float

    @property
    def expected_value(self) -> float:
        """Expected profit in dollars on this stake, before fees and slippage."""
        if self.price <= 0:
            return 0.0
        return self.contracts * (self.model_prob - self.price)


def evaluate(quote: MarketQuote, forecast: EnsembleForecast, cfg: Config, bankroll: float) -> Signal | None:
    """Price one market against the ensemble and size it, or return None."""
    if not cfg.min_price <= quote.yes_price <= cfg.max_price:
        return None
    if quote.volume < cfg.min_volume:
        return None

    p_yes = bucket_probability(forecast, quote.bucket, cfg)

    yes_edge = p_yes - quote.yes_price
    no_price = 1.0 - quote.yes_price
    no_edge = (1.0 - p_yes) - no_price

    if yes_edge >= no_edge:
        side, prob, price, edge = "YES", p_yes, quote.yes_price, yes_edge
    else:
        side, prob, price, edge = "NO", 1.0 - p_yes, no_price, no_edge

    if edge < cfg.min_edge:
        return None

    kelly = kelly_fraction(prob, price)
    if kelly <= 0:
        return None

    fraction = min(kelly * cfg.kelly_fraction, cfg.max_position_fraction)
    stake = round(bankroll * fraction, 2)
    if stake < 1.0:
        return None

    return Signal(
        quote=quote,
        side=side,
        model_prob=prob,
        price=price,
        edge=edge,
        kelly=kelly,
        stake=stake,
        contracts=round(stake / price, 4),
        forecast_mean=forecast.mean,
        forecast_sigma=forecast.sigma,
    )


def find_signals(
    quotes: list[MarketQuote],
    forecasts: dict[tuple[str, str], EnsembleForecast],
    cfg: Config,
    bankroll: float,
) -> list[Signal]:
    """Evaluate every quote that has a matching forecast, best edge first."""
    signals: list[Signal] = []
    for quote in quotes:
        forecast = forecasts.get((quote.city_key, quote.target_date.isoformat()))
        if forecast is None:
            continue
        signal = evaluate(quote, forecast, cfg, bankroll)
        if signal is not None:
            signals.append(signal)
    signals.sort(key=lambda s: s.edge, reverse=True)
    return signals
