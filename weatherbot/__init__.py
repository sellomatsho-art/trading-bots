"""weatherbot - a simulation-only Polymarket temperature-market scanner.

Design constraint, enforced by tests/test_no_key_handling.py: this package
never reads a private key, never signs anything and never places an order. It
is a research tool that scores its own forecasts against reality.
"""

SIMULATION_ONLY = True

__version__ = "0.1.0"
