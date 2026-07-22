"""A zoo of published formulaic alphas, restricted to what crypto data supports.

Scope and honesty
-----------------
These are transcriptions of formulas from the public formulaic-alpha literature,
written against :mod:`vfund.factors.operators`. Two deliberate restrictions:

* **OHLCV only.** Formulas needing sector tags (``IndNeutralize``), market cap,
  or fundamentals are omitted rather than approximated. Crypto has no sector
  classification worth the name, and a fake one would quietly invent a result.
* **Daily bars.** ``vwap`` is the OHLC typical-price proxy — see
  :func:`vfund.factors.operators.vwap`.

Every alpha was designed for **equities**. Nothing here is expected to work on
crypto; that is precisely the open question ``examples/crypto_alpha_study.py``
exists to answer, and a mostly-negative answer is the likely one.

Provenance
----------
* ``alpha101_*`` — Z. Kakushadze, "101 Formulaic Alphas" (2015),
  arXiv:1601.00991. Formulas are mathematical content; transcribed from the
  paper's appendix, not ported from anyone's source.
* ``academic_*`` — standard price-based factor proxies from the asset-pricing
  literature (Jegadeesh reversal, Jegadeesh-Titman momentum, Amihud illiquidity,
  George-Hwang 52-week high, Ang et al. idiosyncratic volatility).

The operator surface these are written against follows HKUDS/Vibe-Trading's
Alpha Zoo base layer (MIT).
"""

from __future__ import annotations

import numpy as np

from vfund.factors.alpha import Panel, alpha
from vfund.factors.operators import (
    adv,
    decay_linear,
    delay,
    delta,
    rank,
    safe_div,
    scale,
    signed_power,
    ts_argmax,
    ts_argmin,
    ts_corr,
    ts_cov,
    ts_max,
    ts_mean,
    ts_min,
    ts_rank,
    ts_std,
    ts_sum,
)

K2015 = "Kakushadze (2015), 101 Formulaic Alphas, arXiv:1601.00991"
ACADEMIC = "Standard price-based academic factor proxy"


# --- alpha101 (OHLCV-only subset) -------------------------------------------


@alpha("alpha101_001", formula="rank(ts_argmax(signed_power(ret<0 ? stddev(ret,20) : close, 2), 5)) - 0.5",
       source=K2015, theme=("reversal", "volatility"), warmup=30)
def alpha101_001(p: Panel) -> np.ndarray:
    r = p.returns
    base = np.where(r < 0, ts_std(r, 20), p.close)
    return rank(ts_argmax(signed_power(base, 2.0), 5)) - 0.5


@alpha("alpha101_002", formula="-1 * corr(rank(delta(log(volume),2)), rank((close-open)/open), 6)",
       source=K2015, theme=("reversal", "volume"), warmup=12)
def alpha101_002(p: Panel) -> np.ndarray:
    logv = np.log(np.where(p.volume > 0, p.volume, np.nan))
    a = rank(delta(logv, 2))
    b = rank(safe_div(p.close - p.open, p.open))
    return -1.0 * ts_corr(a, b, 6)


@alpha("alpha101_003", formula="-1 * corr(rank(open), rank(volume), 10)",
       source=K2015, theme=("volume",), warmup=12)
def alpha101_003(p: Panel) -> np.ndarray:
    return -1.0 * ts_corr(rank(p.open), rank(p.volume), 10)


@alpha("alpha101_004", formula="-1 * ts_rank(rank(low), 9)",
       source=K2015, theme=("reversal",), warmup=12)
def alpha101_004(p: Panel) -> np.ndarray:
    return -1.0 * ts_rank(rank(p.low), 9)


@alpha("alpha101_006", formula="-1 * corr(open, volume, 10)",
       source=K2015, theme=("volume",), warmup=12)
def alpha101_006(p: Panel) -> np.ndarray:
    return -1.0 * ts_corr(p.open, p.volume, 10)


@alpha("alpha101_008", formula="-1*rank((sum(open,5)*sum(ret,5)) - delay(sum(open,5)*sum(ret,5),10))",
       source=K2015, theme=("reversal",), warmup=20)
def alpha101_008(p: Panel) -> np.ndarray:
    x = ts_sum(p.open, 5) * ts_sum(p.returns, 5)
    return -1.0 * rank(x - delay(x, 10))


@alpha("alpha101_009", formula="conditional momentum on delta(close,1) over 5-day min/max",
       source=K2015, theme=("momentum",), warmup=10)
def alpha101_009(p: Panel) -> np.ndarray:
    d = delta(p.close, 1)
    cond_up = ts_min(d, 5) > 0
    cond_dn = ts_max(d, 5) < 0
    return np.where(cond_up | cond_dn, d, -1.0 * d)


@alpha("alpha101_012", formula="sign(delta(volume,1)) * (-1 * delta(close,1))",
       source=K2015, theme=("reversal", "volume"), warmup=3)
def alpha101_012(p: Panel) -> np.ndarray:
    return np.sign(delta(p.volume, 1)) * (-1.0 * delta(p.close, 1))


@alpha("alpha101_013", formula="-1 * rank(cov(rank(close), rank(volume), 5))",
       source=K2015, theme=("volume",), warmup=8)
def alpha101_013(p: Panel) -> np.ndarray:
    return -1.0 * rank(ts_cov(rank(p.close), rank(p.volume), 5))


@alpha("alpha101_014", formula="(-1 * rank(delta(returns,3))) * corr(open, volume, 10)",
       source=K2015, theme=("reversal", "volume"), warmup=14)
def alpha101_014(p: Panel) -> np.ndarray:
    return (-1.0 * rank(delta(p.returns, 3))) * ts_corr(p.open, p.volume, 10)


@alpha("alpha101_016", formula="-1 * rank(cov(rank(high), rank(volume), 5))",
       source=K2015, theme=("volume",), warmup=8)
def alpha101_016(p: Panel) -> np.ndarray:
    return -1.0 * rank(ts_cov(rank(p.high), rank(p.volume), 5))


@alpha("alpha101_018", formula="-1*rank(stddev(abs(close-open),5)+(close-open)+corr(close,open,10))",
       source=K2015, theme=("reversal",), warmup=14)
def alpha101_018(p: Panel) -> np.ndarray:
    body = p.close - p.open
    return -1.0 * rank(ts_std(np.abs(body), 5) + body + ts_corr(p.close, p.open, 10))


@alpha("alpha101_019", formula="-sign((close-delay(close,7))+delta(close,7)) * (1+rank(1+sum(ret,250)))",
       source=K2015, theme=("momentum",), warmup=260)
def alpha101_019(p: Panel) -> np.ndarray:
    s = (p.close - delay(p.close, 7)) + delta(p.close, 7)
    return -1.0 * np.sign(s) * (1.0 + rank(1.0 + ts_sum(p.returns, 250)))


@alpha("alpha101_023", formula="if sum(high,20)/20 < high then -delta(high,2) else 0",
       source=K2015, theme=("reversal",), warmup=25)
def alpha101_023(p: Panel) -> np.ndarray:
    return np.where(ts_mean(p.high, 20) < p.high, -1.0 * delta(p.high, 2), 0.0)


@alpha("alpha101_024", formula="regime switch on 100-day mean close vs 100-day delta",
       source=K2015, theme=("momentum",), warmup=110)
def alpha101_024(p: Panel) -> np.ndarray:
    m = ts_mean(p.close, 100)
    cond = safe_div(delta(m, 100), delay(p.close, 100)) <= 0.05
    return np.where(cond, -1.0 * (p.close - ts_min(p.close, 100)),
                    -1.0 * delta(p.close, 3))


@alpha("alpha101_026", formula="-1 * ts_max(corr(ts_rank(volume,5), ts_rank(high,5), 5), 3)",
       source=K2015, theme=("volume",), warmup=16)
def alpha101_026(p: Panel) -> np.ndarray:
    c = ts_corr(ts_rank(p.volume, 5), ts_rank(p.high, 5), 5)
    return -1.0 * ts_max(c, 3)


@alpha("alpha101_028", formula="scale(corr(adv20, low, 5) + (high+low)/2 - close)",
       source=K2015, theme=("reversal", "volume"), warmup=28)
def alpha101_028(p: Panel) -> np.ndarray:
    c = ts_corr(adv(p.close, p.volume, 20), p.low, 5)
    return scale(c + (p.high + p.low) / 2.0 - p.close)


@alpha("alpha101_032", formula="scale(mean(close,7)-close) + 20*scale(corr(vwap, delay(close,5), 230))",
       source=K2015, theme=("reversal",), warmup=240)
def alpha101_032(p: Panel) -> np.ndarray:
    a = scale(ts_mean(p.close, 7) - p.close)
    b = scale(ts_corr(p.vwap, delay(p.close, 5), 230))
    return a + 20.0 * b


@alpha("alpha101_033", formula="rank(-1 * (1 - open/close))",
       source=K2015, theme=("reversal",), warmup=3)
def alpha101_033(p: Panel) -> np.ndarray:
    return rank(-1.0 * (1.0 - safe_div(p.open, p.close)))


@alpha("alpha101_034", formula="rank(2 - rank(stddev(ret,2)/stddev(ret,5)) - rank(delta(close,1)))",
       source=K2015, theme=("volatility", "reversal"), warmup=10)
def alpha101_034(p: Panel) -> np.ndarray:
    r = p.returns
    ratio = safe_div(ts_std(r, 2), ts_std(r, 5))
    return rank(2.0 - rank(ratio) - rank(delta(p.close, 1)))


@alpha("alpha101_035", formula="ts_rank(volume,32) * (1-ts_rank(close+high-low,16)) * (1-ts_rank(ret,32))",
       source=K2015, theme=("volume", "momentum"), warmup=36)
def alpha101_035(p: Panel) -> np.ndarray:
    return (ts_rank(p.volume, 32)
            * (1.0 - ts_rank(p.close + p.high - p.low, 16))
            * (1.0 - ts_rank(p.returns, 32)))


@alpha("alpha101_038", formula="-1 * rank(ts_rank(close,10)) * rank(close/open)",
       source=K2015, theme=("reversal",), warmup=14)
def alpha101_038(p: Panel) -> np.ndarray:
    return -1.0 * rank(ts_rank(p.close, 10)) * rank(safe_div(p.close, p.open))


@alpha("alpha101_040", formula="-1 * rank(stddev(high,10)) * corr(high, volume, 10)",
       source=K2015, theme=("volatility", "volume"), warmup=14)
def alpha101_040(p: Panel) -> np.ndarray:
    return -1.0 * rank(ts_std(p.high, 10)) * ts_corr(p.high, p.volume, 10)


@alpha("alpha101_041", formula="sqrt(high*low) - vwap",
       source=K2015, theme=("reversal",), warmup=3)
def alpha101_041(p: Panel) -> np.ndarray:
    prod = p.high * p.low
    return np.sqrt(np.where(prod > 0, prod, np.nan)) - p.vwap


@alpha("alpha101_043", formula="ts_rank(volume/adv20, 20) * ts_rank(-1*delta(close,7), 8)",
       source=K2015, theme=("volume", "reversal"), warmup=30)
def alpha101_043(p: Panel) -> np.ndarray:
    return (ts_rank(safe_div(p.volume, ts_mean(p.volume, 20)), 20)
            * ts_rank(-1.0 * delta(p.close, 7), 8))


@alpha("alpha101_044", formula="-1 * corr(high, rank(volume), 5)",
       source=K2015, theme=("volume",), warmup=8)
def alpha101_044(p: Panel) -> np.ndarray:
    return -1.0 * ts_corr(p.high, rank(p.volume), 5)


@alpha("alpha101_046", formula="regime switch on 20/10-day close slope difference",
       source=K2015, theme=("momentum",), warmup=25)
def alpha101_046(p: Panel) -> np.ndarray:
    a = (delay(p.close, 20) - delay(p.close, 10)) / 10.0
    b = (delay(p.close, 10) - p.close) / 10.0
    diff = a - b
    inner = np.where(diff < 0.0, 1.0, -1.0 * (p.close - delay(p.close, 1)))
    return np.where(diff > 0.25, -1.0, inner)


@alpha("alpha101_049", formula="if 20/10-day slope diff < -0.1 then 1 else -delta(close,1)",
       source=K2015, theme=("momentum",), warmup=25)
def alpha101_049(p: Panel) -> np.ndarray:
    a = (delay(p.close, 20) - delay(p.close, 10)) / 10.0
    b = (delay(p.close, 10) - p.close) / 10.0
    return np.where((a - b) < -0.1, 1.0, -1.0 * delta(p.close, 1))


@alpha("alpha101_051", formula="if 20/10-day slope diff < -0.05 then 1 else -delta(close,1)",
       source=K2015, theme=("momentum",), warmup=25)
def alpha101_051(p: Panel) -> np.ndarray:
    a = (delay(p.close, 20) - delay(p.close, 10)) / 10.0
    b = (delay(p.close, 10) - p.close) / 10.0
    return np.where((a - b) < -0.05, 1.0, -1.0 * delta(p.close, 1))


@alpha("alpha101_053", formula="-1 * delta(((close-low)-(high-close))/(close-low), 9)",
       source=K2015, theme=("reversal",), warmup=12)
def alpha101_053(p: Panel) -> np.ndarray:
    num = (p.close - p.low) - (p.high - p.close)
    return -1.0 * delta(safe_div(num, p.close - p.low), 9)


@alpha("alpha101_054", formula="-1*(low-close)*open^5 / ((low-high)*close^5)",
       source=K2015, theme=("reversal",), warmup=3)
def alpha101_054(p: Panel) -> np.ndarray:
    num = -1.0 * (p.low - p.close) * (p.open ** 5)
    den = (p.low - p.high) * (p.close ** 5)
    return safe_div(num, den)


@alpha("alpha101_101", formula="(close - open) / (high - low + 0.001)",
       source=K2015, theme=("reversal",), warmup=2)
def alpha101_101(p: Panel) -> np.ndarray:
    return safe_div(p.close - p.open, (p.high - p.low) + 0.001)


# --- academic price-based proxies --------------------------------------------


@alpha("academic_strev", formula="-1 * return over the last 5 bars",
       source=ACADEMIC + " (Jegadeesh 1990 short-term reversal)",
       theme=("reversal",), warmup=8)
def academic_strev(p: Panel) -> np.ndarray:
    return -1.0 * safe_div(p.close - delay(p.close, 5), delay(p.close, 5))


@alpha("academic_mom", formula="return from t-30 to t-7 (skip the last week)",
       source=ACADEMIC + " (Jegadeesh-Titman 1993 momentum, gap-adjusted)",
       theme=("momentum",), warmup=35)
def academic_mom(p: Panel) -> np.ndarray:
    return safe_div(delay(p.close, 7) - delay(p.close, 30), delay(p.close, 30))


@alpha("academic_illiq", formula="mean(|return| / dollar volume, 20)",
       source=ACADEMIC + " (Amihud 2002 illiquidity)",
       theme=("liquidity",), warmup=25)
def academic_illiq(p: Panel) -> np.ndarray:
    dv = p.close * p.volume
    return ts_mean(safe_div(np.abs(p.returns), dv), 20)


@alpha("academic_high52w", formula="close / max(high, 365)",
       source=ACADEMIC + " (George-Hwang 2004 52-week high)",
       theme=("momentum",), warmup=370)
def academic_high52w(p: Panel) -> np.ndarray:
    return safe_div(p.close, ts_max(p.high, 365))


@alpha("academic_ivol", formula="-1 * stddev(returns, 30)",
       source=ACADEMIC + " (Ang et al. 2006 idiosyncratic-volatility proxy)",
       theme=("volatility",), warmup=35)
def academic_ivol(p: Panel) -> np.ndarray:
    return -1.0 * ts_std(p.returns, 30)


@alpha("academic_maxret", formula="-1 * max(returns, 20) (lottery demand)",
       source=ACADEMIC + " (Bali-Cakici-Whitelaw 2011 MAX)",
       theme=("lottery",), warmup=25)
def academic_maxret(p: Panel) -> np.ndarray:
    return -1.0 * ts_max(p.returns, 20)


@alpha("academic_decay_mom", formula="decay_linear(returns, 20)",
       source=ACADEMIC + " (decay-weighted momentum)",
       theme=("momentum",), warmup=25)
def academic_decay_mom(p: Panel) -> np.ndarray:
    return decay_linear(p.returns, 20)


@alpha("academic_argmin_low", formula="ts_argmin(low, 30) - bars since the recent low",
       source=ACADEMIC + " (drawdown recency proxy)",
       theme=("reversal",), warmup=35)
def academic_argmin_low(p: Panel) -> np.ndarray:
    return ts_argmin(p.low, 30)


@alpha("academic_volume_trend", formula="rank(adv(5) / adv(30)) - volume acceleration",
       source=ACADEMIC + " (volume-trend proxy)",
       theme=("volume",), warmup=35)
def academic_volume_trend(p: Panel) -> np.ndarray:
    return rank(safe_div(adv(p.close, p.volume, 5), adv(p.close, p.volume, 30)))
