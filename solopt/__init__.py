"""The walk-forward / Monte Carlo / crash-replay optimizer.

Invoked from two places (spec 5): directly, in-process, for the daily
incremental run on the trading droplet's own CPU, and via ``python -m
solopt run`` inside a RunPod GPU worker container for the monthly full
parameter-space retest. Neither caller changes what is in this package - only
where it runs and where the data and results come from and go.

Nothing in here imports :mod:`solbot`. The optimizer is deliberately generic
across asset classes - it knows about "symbols" and "bars", never about mints or
pools - so equities and forex can be added later by supplying a different
:class:`solopt.schema.Schema` rather than by rearchitecting the engine.
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
