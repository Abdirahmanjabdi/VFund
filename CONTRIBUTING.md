# Contributing to VFund

Thanks for your interest. VFund is early — the foundation is here and there's a
lot of high-value work to do.

## Principles

1. **Open microscope, closed discoveries.** We build shared *tools*. Strategies
   with real edge belong in your private fork, not here.
2. **Honesty over flattery.** A feature is good if it makes it *harder* to fool
   yourself about a strategy's performance (realistic costs, out-of-sample
   testing, look-ahead guards), not if it makes backtests look prettier.
3. **Local-first.** No feature should require sending a user's data or
   strategies to a third party.

## Getting started

```bash
pip install -e ".[dev]"
pytest
```

## Good first issues

- New indicators (RSI, ATR, Bollinger, z-score) as vectorised helpers
- Walk-forward / out-of-sample split utilities
- Additional baseline strategies (mean-reversion, momentum, breakout)
- More exchange data sources behind the canonical schema
- Property-based tests for the engine's accounting invariants

## Pull requests

- Keep PRs focused and add tests for new behaviour.
- Match the existing style: small modules, docstrings that explain *why*.
- `pytest` must pass.
