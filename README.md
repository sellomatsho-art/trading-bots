# weatherbot

A simulation-only research tool for Polymarket temperature markets. It prices
each market's bucket against the 31-member GFS ensemble, books paper positions
where the two disagree, and then — the part that matters — scores its own
forecasts against what the weather actually did, so it can learn how wrong it
tends to be in each city.

**It cannot trade.** There is no wallet, no key, no signer and no order path.
See [No keys, by construction](#no-keys-by-construction).

```
python -m weatherbot scan       # forecast, price, book paper fills
python -m weatherbot settle     # score past-dated positions against observations
python -m weatherbot calibrate  # learn per-city forecast bias
python -m weatherbot report     # paper PnL
python app.py                   # dashboard on http://localhost:5001
```

## Install

```bash
pip install -r requirements.txt
python -m weatherbot cities     # check it runs
```

Both data sources are public, free and unauthenticated:

| Source | Used for | Key needed |
| --- | --- | --- |
| [Open-Meteo ensemble API](https://open-meteo.com/en/docs/ensemble-api) | GFS `gfs025` — control run + 30 perturbed members | No |
| [Open-Meteo forecast API](https://open-meteo.com/en/docs) | Observed daily highs, for settlement | No |
| [Polymarket Gamma API](https://docs.polymarket.com/) | Open markets and current prices (read-only) | No |

## How it prices a market

**1. The ensemble is the distribution.** GFS is run 31 times from slightly
perturbed initial conditions. Where those members land for tomorrow's high
*is* the forecast uncertainty — which is exactly the thing a bucketed
temperature market prices. Each member's daily max becomes one sample.

**2. Buckets are widened before they're scored.** Markets settle on the
reported *integer* high, so the "90–91°F" bucket really covers `[89.5, 91.5)`
in the continuous temperature the model predicts. Scoring the literal bounds
understates narrow buckets systematically. Threshold markets are handled the
same way: `below 90` settles at 89 or cooler, `90 or below` includes 90 — a
one-degree difference that is the entire edge on a two-degree bucket.

**3. The histogram is smoothed.** 31 members give a granularity of ~3.2
percentage points and produce empty buckets that aren't really impossible.
The default `blend` model averages the raw member count with a normal fitted
to the ensemble, and floors the spread at `min_sigma_f` (1.5 °F) so a tightly
clustered run can't claim certainty. Probabilities are clamped to [0.01, 0.99].

**4. Sizing is fractional Kelly.** For a binary at price `c`, full Kelly is
`(p − c) / (1 − c)`. The bot stakes `kelly_fraction` of that (default ¼) and
caps any single position at `max_position_fraction` of bankroll (default 5%).
Full Kelly on a probability you estimated yourself is a good way to go broke.

**5. Both sides are considered.** If the model is *below* the market it buys
NO at `1 − price` rather than passing.

## The part worth having: bias calibration

The public GFS run is priced in the moment it publishes. Your record of how
that run has missed *at a specific station* is not.

Every scan logs its forecast. Every settle writes the observed high back onto
that log. `calibrate` turns the accumulated errors into a per-city offset:

```
$ python -m weatherbot calibrate
  chicago    n=41   bias=+2.90F  mae=3.10F  applied=+2.33F
  nyc        n=39   bias=-0.40F  mae=1.80F  applied=-0.32F

$ python -m weatherbot calibrate --apply    # writes it into config.json
```

If the ensemble has averaged 2.9 °F hotter than O'Hare actually got, the raw
forecast is not the best estimate — the forecast minus 2.9 °F is, and every
subsequent scan prices with that shift applied.

Measured bias is shrunk toward zero by `n / (n + 10)`, so a three-day fluke
doesn't move the whole book. Expect to need a few weeks of history before the
numbers mean anything.

## Configuration

`config.json`, all trading parameters:

| Key | Default | Notes |
| --- | --- | --- |
| `cities` | 5 US cities | `python -m weatherbot cities` lists all 10 |
| `bankroll` | 1000.0 | Virtual USD |
| `min_edge` | 0.08 | Minimum \|model − market\| to act |
| `min_price` / `max_price` | 0.05 / 0.95 | Tails are where thin books hurt most |
| `kelly_fraction` | 0.25 | Fraction of full Kelly staked |
| `max_position_fraction` | 0.05 | Hard cap per position |
| `max_open_positions` | 20 | |
| `min_volume` | 500.0 | Skip illiquid markets |
| `forecast_horizon_days` | 3 | GFS skill decays fast past this |
| `probability_model` | `blend` | or `empirical` / `gaussian` |
| `min_sigma_f` | 1.5 | Floor on ensemble spread, °F |
| `bias_correction` | `{}` | Written by `calibrate --apply` |

## No keys, by construction

The free weather-bot ecosystem has a real problem: several of these projects
ask for a hot wallet key in a plaintext `.env`, and at least one shipped a
dependency that npm later pulled from the registry and replaced with a
security-holding placeholder. A README promise is not a defence. So:

- No wallet, signer, or CLOB client, and none in `requirements.txt`.
- Every HTTP call is a `GET` against a public endpoint.
- The package never reads an environment variable, so a `.env` full of secrets
  has nothing to read it.

`tests/test_no_key_handling.py` enforces all of the above by walking the AST of
every source file. If someone adds `web3`, an identifier containing
`private_key`, an `os.environ` lookup, or a `POST`, the build fails. That test
is the actual guarantee.

The consequence is that this tool cannot execute. It tells you where it thinks
the market is wrong and keeps an honest scorecard; placing anything is a
separate, manual, deliberate act.

## Testing

```bash
python -m pytest -q     # 90 tests, no network
```

The HTTP layer is faked at the `requests.Session` boundary using the documented
response shapes, so the suite runs offline.

## What this does not do

- **Verify the settlement source.** Positions settle against Open-Meteo's
  observation blend for the city's coordinates, not the official station report
  Polymarket resolves on. Close enough to score a simulation; not exact.
- **Model the order book.** Fills are booked at the quoted price for the full
  size. Real books for these markets are thin, and that assumption is
  optimistic — often by more than the edge being chased.
- **Account for fees or slippage.** Expected value is gross.
- **Beat the market by default.** GFS is public. By the time a run is out, the
  obvious read is priced. Any real edge lives in the calibration history you
  build yourself, and it may turn out to be zero. Paper-trade it for a month and
  look at the scorecard before believing anything.
