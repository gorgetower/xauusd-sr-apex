"""
XAUUSD Multi-Timeframe S/R Confluence + A+ Setup Engine  (v19 "APEX")
================================================================================

A production-grade rewrite of the original `Scalp_SR_AboveM5` scanner.

This version fixes every correctness/performance bug in the original and adds
the risk + execution layer required to grade setups (A+ / A / B / C):

CORRECTNESS FIXES
  * Levels are CLUSTERED into price bands before scoring (the original keyed a
    dict by an exact float, so multi-method / multi-timeframe confluence
    essentially never triggered - the core bug).
  * Live current price from the tick, with a closed-bar fallback; never crashes
    if one timeframe fails to load.
  * Session-anchored (daily) VWAP with volume-weighted std bands, instead of a
    meaningless 2000-bar cumulative VWAP.
  * Swing-based Fibonacci / Murrey levels (recent lookback) instead of the
    absolute 2000-bar extremes.
  * Standard Camarilla divisors (/12, /6, /4, /2).
  * Closed-bar analysis (drops the still-forming candle) to avoid repainting.
  * Round numbers generated once from live price on a configurable grid.

PERFORMANCE
  * Volume-profile POC computed ONCE per run, vectorised with numpy
    (was an O(levels x bars x bins) triple-loop per candidate).
  * Vectorised touch counting and swing detection.

A+ GRADE LAYER
  * ATR per timeframe -> volatility-aware stop / target / R:R.
  * Position sizing from account balance + risk % + broker contract specs.
  * DXY intermarket confluence (gold's inverse USD correlation).
  * News-window filter (avoid trading around high-impact releases).
  * Confluence score -> A+/A/B/C grade with a full trade plan per level.

v5.1 PATCH
  * Structure-based R:R (target = next opposing clustered level, ATR fallback).
  * Broker-aware lot sizing (rounds down to volume step; clamps to min/max).
  * De-duplicated confluence scoring (each method/timeframe weighted once).
  * Session VWAP restricted to intraday timeframes.

v5.2 SCORING (evidence-based, research-backed)
  * Normalised 0-100 confluence score with transparent per-component breakdown.
  * Method/timeframe confluence use DIMINISHING returns (3-4 factor sweet spot).
  * Touch strength is recency-weighted (half-life) and log-saturated - repeated
    testing no longer scales linearly, and stale touches decay.
  * ATR-qualified wick REJECTIONS (wick vs body + ATR) reward genuine reactions.
  * Volume acceptance: POC + 70% Value Area (VAH/VAL) added as first-class
    levels; a level's score scales with its share of profile volume.
  * Tiered gold round numbers ($100 > $50 > $25 > $10 > $5).
  * Over-tested levels with no recent rejection are penalised (liquidity used).

v5.3 SCORING (2026 research refresh)
  * KDE / density-based level finding: score scales with the *concentration* of
    confirmed swing pivots (Gaussian kernel), echoing the 2026 shift toward
    DBSCAN / Kernel-Density S/R detection over raw pivot counts.
  * Liquidity-structure scoring (Smart Money Concepts): equal highs/lows are
    treated as resting-liquidity magnets, and a recent sweep-and-reclaim
    (false breakout) adds a high-probability reversal bonus.
  * Confirmed fractal swing pivots now anchor both density and liquidity.

v9 STABILITY + CALIBRATION (2026)
  * Deterministic scans: the reference price defaults to the last CLOSED bar and
    is snapped to a coarse grid, so repeated runs on the same bar are identical
    (this is the root fix for "levels change every time I run it").
  * Absolute round-number grid + fixed-width volume-profile bins -> POC / value
    area / psychological levels no longer drift between runs.
  * Deterministic gap+width clustering with weight-based centroids (replaces the
    order-sensitive anchor clustering) -> stable, repeatable level prices.
  * Recalibrated 0..100 (and 0..10) scoring: structural strength is a weighted
    mean of confluence sub-scores with bonuses added on top, so a genuinely
    strong multi-timeframe, well-tested level can reach the 8+/10 (A+) band.
  * Deterministic tie-breaking when selecting the top levels.

v10 MULTI-TIMEFRAME SWING STRUCTURE (2026)
  * Confirmed fractal swing pivots for the KDE density + SMC liquidity-structure
    layer are now computed on EVERY timeframe (M15..D1), not just the lowest
    one - the v9 layer silently ignored H1/H4/D1 structure entirely. Pivots are
    pooled with HTF-weighted contribution via the existing `w_tf` map, so an
    H4/D1 swing cluster counts for more than an M15 one.
  * Recency decay is converted to real MINUTES before pooling, so ages from
    different timeframes are directly comparable (60 M15 bars is 15 hours;
    60 D1 bars is ~2 months - mixing raw bar counts across timeframes would be
    meaningless). The liquidity-sweep lookback window is time-normalised the
    same way.
  * The single LATEST confirmed swing high and swing low per timeframe is now
    also added as an explicit `swing_structure` level/method - the same "one
    color per timeframe, only the latest high/low label" rule used for the
    M3/M5/M15 chart overlay - so confirmed multi-timeframe market structure
    directly participates in method/timeframe confluence scoring, not just the
    density/liquidity sub-scores.

v11 SWING CLUSTER (5-bar, full MTF stack above M5)
  * Fractal swing confirmation widened from 3 bars/side to 5 bars/side
    (`swing_left`/`swing_right`/`swing_window`). A 3-bar fractal confirms on
    almost every wiggle; 5 bars each side is the classic "cluster swing"
    definition - materially fewer, cleaner, more reliable pivots - and it now
    applies uniformly to horizontal levels, the KDE density layer, the
    liquidity/sweep layer AND the new layer below.
  * NEW multi-timeframe SWING CLUSTER zones (`_swing_cluster_zones`): instead
    of only the single latest swing high/low per timeframe, the last
    `swing_cluster_lookback` confirmed 5-bar pivots from EVERY scanned
    timeframe (M15, M30, H1, H4, D1 - i.e. every timeframe above M5) are
    pooled onto one price axis, weighted by timeframe importance (`w_tf`) and
    real-time recency decay, then grouped into zones wherever >=2 pivots
    (often from different timeframes) print within `swing_cluster_tol_atr_mult`
    x ATR of each other. Zones are added as first-class `swing_cluster` raw
    levels - one row per contributing timeframe - so a genuine multi-TF swing
    cluster (e.g. an H4 swing low sitting on top of two M30 swing lows)
    directly boosts both the method and timeframe confluence sub-scores, not
    only the background density/liquidity components. This is distinct from
    `swing_structure` (exactly one high + one low per TF, no clustering) -
    the two are complementary, weighted separately in `w_type`.

v12 PATCH (correctness review)
  * FIXED ANCHOR for pivot-family levels: floor pivots, Camarilla, and DeMark
    are DEFINED on a fixed period (prior closed day, optionally prior week),
    never re-derived from the last bar of whatever intraday timeframe happens
    to be looping. The old code called `pivot_points(df)` / `camarilla(df)` /
    `demark_levels(df)` once per SCANNED timeframe (M15..D1), so on M15 it
    was computing "pivot points" from a single 15-minute candle - noise, not
    structure - and that noise could coincidentally cluster with genuine
    daily levels, inflating method/timeframe confluence with false agreement
    on exactly the setups being graded A+. Now computed once from
    `pivot_anchor` (default D1) plus an optional W1 anchor, each contributing
    a single well-defined `tf` tag ("D1"/"W1") to the confluence scoring.
  * Equal-high/low fractal detection no longer discards ties: the old
    strict-uniqueness check (`(window == h[i]).sum() == 1`) meant a genuine
    double top/bottom registered ZERO confirmed pivots at that price -
    starving the SMC equal-highs/lows liquidity layer of the exact pattern
    it exists to detect. Now confirms on the first bar reaching the window
    extreme, so tied swings are captured.
  * `datetime.utcnow()` (deprecated) replaced with timezone-aware UTC
    throughout the news-window filter.
  * Stop-loss distance now adds a configurable `spread_buffer_dollars` on
    top of the ATR multiple, so position sizing isn't optimistic about
    realised risk on wide-spread gold feeds.

v15 PATCH (H1+ only scan stack, tiered pivot scoring)
  * REMOVED all sub-H1 timeframes. `timeframes` is now ["H1", "H4", "D1"]
    (was M15/M30/H1/H4/D1), `lowest_tf` is "H1" (was M15), `atr_tf` is "H1"
    (was M30), `vwap_tfs` is ["H1"] (was M15/M30/H1), and M15/M30 were
    dropped from `_TF_MINUTES`, `w_tf`, and the MT5 timeframe map. Every
    downstream consumer (swing pivots, swing-cluster zones, KDE density,
    liquidity/sweep structure, horizontal levels, fib/MA/VWAP/Murrey) reads
    `cfg.timeframes` / `df_dict`, so this removes M15/M30 noise everywhere
    without any further code changes. Intent: only trade-relevant structure
    (H1/H4/D1(/W1)) feeds the scan - no sub-H1 micro-noise levels.
  * TIERED pivot-family scoring: floor pivots and Camarilla levels used to
    get one FLAT reason weight ("any pivot line present -> 0.65") regardless
    of which line it was or how many anchors it came from. A central PP is
    tested and respected far more often than an outer R3/S3 extension, and
    for Camarilla the R3/S3/R4/S4 lines are the actual reversal/breakout
    levels the method targets - R1/R2/S1/S2 are comparatively weak in-between
    lines. `_PIVOT_TIER_WEIGHT` / `_CAMARILLA_TIER_WEIGHT` now score the
    ACTUAL named line(s) present, and a level confirmed on BOTH the daily
    and weekly anchor earns an explicit multi-anchor confluence bonus on top
    of the line tier (see `build_reasons` step 8).

v16 PATCH (structural-validity review - see chat critique)
  * PER-TIMEFRAME ATR: `compute_atr()` now computes ATR for every scanned
    timeframe (`_atr_by_tf`), not just `cfg.atr_tf` (H1). KDE bandwidth
    (`_density_strength`), equal-high/low tolerance and sweep window
    (`_liquidity_structure`) now use EACH pivot's own timeframe ATR instead
    of stretching a single H1 ATR across H1/H4/D1 - the old shared H1 ATR
    was too tight for D1 structure and arbitrary for H4.
  * Swing-cluster zone grouping (`_swing_cluster_zones`) now tolerances on
    the HIGHEST-timeframe ATR available (`_reference_htf_atr`, D1 preferred)
    instead of H1 ATR, since these zones are explicitly meant to pool
    H1..D1 pivots into one daily-scale structural zone.
  * CORRELATED-METHOD GROUPING (`_METHOD_GROUP`): method/timeframe
    confluence used to count every raw `type` (Fibonacci, MA, Bollinger,
    Murrey Math, ...) as independent evidence, even though most are
    deterministic transforms of the SAME recent OHLC window and will
    naturally cluster together with no real market significance. Method
    confluence, method quality, the reasons text, and the "lone-method"
    penalty now all operate on distinct correlated-evidence GROUPS
    (price_math / anchor_pivot / volume / price_structure / round_number),
    not raw type counts.
  * OUTLIER-ANCHOR DETECTION for pivot/Camarilla/DeMark: `_get_anchor_ohlc`
    now also returns a 0..1 quality score that discounts the entire
    pivot-family level set when the anchor day/week's range is an abnormal
    multiple of its own recent median range (news gap, thin holiday
    liquidity) - previously a single outlier anchor bar silently
    contaminated ~17 levels with no way to detect it.
  * VOLATILITY-REGIME-ADAPTIVE CLUSTERING: `cluster_pct` (a fixed % of
    price) is now rescaled by the live volatility regime (expansion /
    compression) before both `horizontal_levels` and `cluster_levels` run,
    so grouping tolerance is no longer blind to how quiet or violent the
    current market is.
  * SESSION VWAP is now a fixed-real-hours rolling window
    (`cfg.vwap_session_hours`), not a calendar-day mask - the old
    `day == day[-1]` boundary assumed the MT5 server's own midnight was a
    meaningful session start for a 24-hour OTC gold market, which is not a
    safe assumption without knowing the server's timezone.
  NOT fixed here (data/infrastructure limitations, not code bugs): real
  exchange trade volume (MT5 `tick_volume` is tick count, not traded
  volume), and order-book/DOM access for genuine liquidity-sweep detection.
  These require a different data source, not a different algorithm.

v17 PATCH (structural-validity fixes, chat-reviewed)
  * TIMEFRAME-PROPORTIONAL RECENCY HALF-LIFE (`_halflife_minutes(tf)`): the
    old half-life was a single H1-tuned constant applied to every pooled
    pivot regardless of timeframe, which directly contradicted `w_tf`
    (which weights D1/W1 as structurally stronger/more durable) - a 40-day
    D1 pivot decayed to ~11% and a 90-day W1 pivot to <1% under the old
    constant. Half-life now scales with each pivot's OWN timeframe, so HTF
    structure keeps a real, proportionally longer memory.
  * TICK-VOLUME HONESTY (`cfg.volume_confidence`, default 0.5): every
    volume-derived weight/bonus (POC/VAH/VAL/naked-POC/HVN-LVN, and the
    "volume" method-group's contribution to quality/confluence) is now
    explicitly discounted, since MT5 `tick_volume` is a tick-count proxy,
    not real traded volume, and the old code weighted it as the single
    heaviest evidence source (`w_type["naked_poc"]=23`,
    `w_type["volume_profile"]=22`) with no such discount. A startup warning
    fires if `volume_confidence` is left high without a real volume feed.
  * METHOD-CONFLUENCE DE-SATURATION (`_GROUP_INDEPENDENCE`): grouping
    correlated methods (v16) wasn't enough on its own - `price_math`
    (Fib/MA/Bollinger/Murrey) is a continuous function of recent price and
    sits near almost ANY price to within tolerance, and `round_number`
    prints every $25, dense relative to the cluster tolerance. Both were
    still counted as full "independent groups", letting confluence
    saturate (~0.92) from three groups of mostly self-fulfilling agreement.
    Groups now carry fractional independence credit before the saturating
    exponential.
  * CENTROID-BASED LEVEL CLUSTERING: `cluster_levels` compared each new raw
    level only to the PREVIOUS point in sorted order (greedy single-linkage)
    - a classic chaining artifact where unrelated levels merge transitively
    through an intermediate point. Now compares each point to the cluster's
    running WEIGHTED CENTROID, plus a convergence pass that merges adjacent
    clusters whose settled centroids end up within tolerance.
  * HISTORICAL POLARITY CHECK (`_polarity_check`): Support/Resistance was
    assigned purely by `level > current_price`, with no check on which side
    the market has actually respected the level from. An "untested flip"
    (broken level, historically rejected mostly from the OPPOSITE side, not
    yet held from the new side) is now detected, penalised (-8 pts), and
    blocked from A+ regardless of other confluence.
  * PROVISIONAL (FAST-CONFIRMING) SWING LAYER (`_latest_provisional_swing`):
    the standard 5-bar/5-bar fractal only confirms a pivot 5 bars after it
    forms (5h on H1, 5 days on D1), leaving the engine blind to the
    developing high/low for that whole window. A low-weight
    `provisional_structure` source (2-bar right-confirmation) now
    participates in scoring without ever being able to outscore a fully
    confirmed swing.
  * DXY BIAS WIDENED + CORRELATION-GATED: slope lookback widened 20 -> 60
    bars, and the bias is now dropped to "neutral" unless gold and DXY are
    actually correlated (|corr| >= `dxy_min_abs_corr`) over that same
    window - the old code applied a fixed slope-sign bias unconditionally,
    even during regimes where the usual inverse relationship breaks down.
    The score effect itself is also halved (+/-3/4 instead of +6/-8).
  * ATR-FLOORED CLUSTER TOLERANCE (`cluster_atr_floor_mult`): the base
    clustering tolerance was a pure % of price with only a volatility-regime
    multiplier, inconsistent with the ATR-based touch tolerance used
    elsewhere. It now has an ATR-based floor so clustering never uses a
    tighter tolerance than the live volatility regime would justify.



v18 PATCH (S/R DETECTION REPAIR)
  * FIXED DEAD MAJOR-STRUCTURE ENGINE: `MajorMarketStructureEngine.detect()`
    previously ran but its output was never added to `raw`, so previous-day/
    week levels, HTF swing levels and equal-high/low zones had ZERO effect on
    the actual S/R results. Major structure is now a first-class source.
  * FIXED INITIALIZATION ORDER: major-structure detection previously ran
    BEFORE ATR was computed, forcing its ATR-aware equal-high/low tolerance to
    fall back to a hard-coded $3. ATR is now computed before structure logic.
  * FIXED PREVIOUS-PERIOD OFF-BY-ONE: scanned data already removes the forming
    bar, so `df.iloc[-2]` pointed to two completed periods ago. The last closed
    D1/H4 bar is now `df.iloc[-1]`; W1 is fetched explicitly when needed.
  * REBUILT EQUAL-HIGH/LOW DETECTION around CONFIRMED swing pivots instead of
    adjacent H4 candles. Repeated pivots within an ATR-scaled tolerance now
    produce durable liquidity zones rather than random neighboring-bar matches.
  * MAJOR LEVEL CLUSTERING is now ATR-scaled and centroid-based rather than a
    fixed $15 merge radius, preventing over-merging in quiet gold and
    under-merging in volatile regimes.
  * FIXED HORIZONTAL LEVEL CHAINING: horizontal pivot clusters now compare to
    their running centroid, not merely the previous point, eliminating
    single-linkage chains that could merge distinct S/R zones.
  * ADDED STRUCTURAL FALLBACK OUTPUT: the final table now guarantees nearby
    support/resistance candidates when structural evidence exists, even if a
    strict confluence score would otherwise remove every level on one side.

v19 PATCH (internal-consistency review - see chat critique)
  * GROUP-AWARE STRONG-REASON GATE (`_strong_count_deduped`,
    `_REASON_CORR_GROUP`): the A+/A "N strong reasons" gate counted every
    `Reason` independently, but several reason keys (mtf, htf,
    major_structure, swing_structure, swing_cluster, density) are different
    textual readings of the SAME underlying confirmed-swing-pivot data -
    `_METHOD_GROUP` already prevented this kind of correlated-evidence
    inflation in the numeric score, but the separate reason-count gate had
    no equivalent dedup, so one well-aligned HTF swing could rack up 5-6
    "independent" strong reasons on its own. Correlated reasons now
    contribute at most one strong credit each; `min_strong_reasons_aplus`/
    `min_strong_reasons_a` are rescaled (11->6, 7->4) to match honest
    counting.
  * VOLUME-CONFIDENCE APPLIED TO CLUSTERING WEIGHT, NOT JUST SCORING:
    `_item_weight` (which sets each raw level's pull on its cluster's
    weighted centroid) now discounts `naked_poc`/`volume_profile`/`vwap`/
    `vwap_band` by `cfg.volume_confidence`, the same discount already
    applied to the volume/acceptance SUB-SCORES. Previously naked_poc
    (w_type=23) and volume_profile (w_type=22) - the two heaviest weights
    in the whole table, built entirely on MT5 tick_volume - could out-pull
    a confirmed horizontal level (w_type=20) or swing_cluster zone
    (w_type=19) in deciding where a cluster's actual price sits, contrary
    to the engine's own stated tick-volume-honesty principle.
  * CONDITIONAL STRUCTURAL FALLBACK (`cfg.min_display_score`): the v18
    `ensure_nearest` fallback force-included the nearest candidate on every
    scan regardless of score, so the table could never honestly report "no
    defensible S/R on this side". It now only fires for candidates scoring
    >= `min_display_score` (default 35); below that floor the side is left
    empty instead of padded with a low-quality "closest" row.
  * UNIFIED HORIZONTAL-LEVEL CLUSTERING (`_centroid_cluster_1d`): the v17
    chaining fix (weighted centroid + width cap, replacing greedy
    single-linkage) was applied to `cluster_levels` but never to
    `horizontal_levels`, which still merged swing pivots against a plain
    running mean with no width cap - the same chaining defect the v17 fix
    was written to eliminate. Both callers now share one clustering
    algorithm.

Requires: MetaTrader5, pandas, numpy  (run on a Windows / MT5 host).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from datetime import time as dt_time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:  # allow import/lint on non-MT5 hosts
    mt5 = None

logging.basicConfig(
    level=logging.WARNING,   # set to logging.INFO for scan/diagnostic logs
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("apex")

# Minutes per bar, used to convert swing-pivot "age" from bar counts to real
# time so pivots from different timeframes can be pooled/decayed consistently
# (see `_get_swings_by_tf`, `_density_strength`, `_liquidity_structure`).
_TF_MINUTES: Dict[str, int] = {"H1": 60, "H4": 240, "D1": 1440, "W1": 10080}

# v15: per-level importance tiers for the pivot-family reason scorer (below).
# Not every named pivot line is equally reliable, so a flat "pivot alignment"
# bonus regardless of WHICH line it is (PP vs R3) is itself noise. Central
# pivots and the near tiers get tested/respected far more often than the
# outer extension tiers, and for Camarilla the R3/S3/R4/S4 lines are the
# actual reversal/breakout levels the method is built around - R1/R2/S1/S2
# are comparatively weak "in-between" lines. See `build_reasons` step 8.
_PIVOT_TIER_WEIGHT: Dict[str, float] = {
    "PP": 1.00,
    "R1": 0.70, "S1": 0.70,
    "R2": 0.55, "S2": 0.55,
    "R3": 0.40, "S3": 0.40,
}
_CAMARILLA_TIER_WEIGHT: Dict[str, float] = {
    "C_R4": 0.90, "C_S4": 0.90,   # breakout targets
    "C_R3": 0.85, "C_S3": 0.85,   # key intraday reversal levels
    "C_R2": 0.45, "C_S2": 0.45,
    "C_R1": 0.35, "C_S1": 0.35,
}

# v16 fix: "method confluence" previously counted every raw `type` as an
# independent piece of evidence. But Fibonacci / MA-EMA / Bollinger / Murrey
# Math are all deterministic transforms of the SAME recent OHLC window on the
# SAME timeframe - they are mathematically correlated, not independent
# observations, and will often print near-identical prices purely because
# they share an input, not because the market "agrees" on the level. Pivot /
# Camarilla / DeMark are similarly correlated (same fixed daily/weekly anchor
# bar). Grouping them means confluence scoring only rewards evidence coming
# from genuinely DIFFERENT sources (price-derived math vs. anchor pivots vs.
# volume vs. confirmed price structure), not the same math counted 4 times.
# v19: several `Reason` keys produced by `build_reasons` are independent
# TEXT but not independent EVIDENCE - they are different readings of the
# same underlying confirmed-swing-pivot data (mtf/htf timeframe-span,
# major_structure, swing_structure, swing_cluster and KDE density are all
# derived from the same `_get_swings_by_tf()` pivots). `_METHOD_GROUP` /
# `_GROUP_INDEPENDENCE` already stop this kind of correlated-evidence
# inflation inside the numeric `score_cluster` core, but the SEPARATE "N
# strong reasons" A+ gate had no equivalent dedup, so a single well-aligned
# HTF swing could count as 5-6 "independent strong reasons" on its own.
# `_REASON_CORR_GROUP` fixes that: reasons sharing a group contribute at
# most ONE strong count (their best weight), matching the score's own logic.
_REASON_CORR_GROUP: Dict[str, str] = {
    "mtf": "swing_evidence", "htf": "swing_evidence",
    "major_structure": "swing_evidence", "swing_structure": "swing_evidence",
    "swing_cluster": "swing_evidence", "density": "swing_evidence",
    "methods": "method_evidence", "method_quality": "method_evidence",
}

_METHOD_GROUP: Dict[str, str] = {
    "fibonacci": "price_math", "moving_average": "price_math",
    "bollinger": "price_math", "murrey_math": "price_math",
    "pivot": "anchor_pivot", "camarilla": "anchor_pivot", "demark": "anchor_pivot",
    "volume_profile": "volume", "naked_poc": "volume",
    "vwap": "volume", "vwap_band": "volume",
    "horizontal": "price_structure", "major_structure": "price_structure", "swing_structure": "price_structure",
    "swing_cluster": "price_structure", "provisional_structure": "price_structure",
    "round_number": "round_number",
}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class ApexConfig:
    symbol: str = "XAUUSD"
    dxy_symbol: str = "DXY"              # broker may call it USDX / DX / DXY
    max_range_dollars: float = 20.0     # ignore levels farther than this
    bars: int = 5000
    use_closed_bars: bool = True        # drop the still-forming candle

    # --- v9 stability controls (kill run-to-run jitter) ---
    reference_mode: str = "closed_bar"  # "closed_bar" = reproducible; "tick" = live mid
    price_quantum: float = 0.5          # snap reference price to this $ grid (0 = off)
    vp_bin_dollars: float = 0.0         # volume-profile bin width $ (0 = auto from ATR)
    cluster_max_width_mult: float = 2.0 # max cluster width = mult * cluster_pct * price

    # Cluster tolerance as a fraction of price (0.0007 ~= $2.4 @ $3400 gold).
    cluster_pct: float = 0.0007

    # v16 fix: cluster_pct alone is a FIXED % of price, blind to the current
    # volatility regime - the same $ tolerance is too wide in a compressed
    # market (merges genuinely distinct levels) and too narrow in an
    # expansion (splits one real level into several near-duplicate clusters).
    # These multipliers rescale the clustering tolerance by `self._vol_regime`
    # (already computed from ATR-vs-its-own-median, see compute_vol_regime).
    cluster_regime_adaptive: bool = True
    cluster_expansion_mult: float = 1.3
    cluster_compression_mult: float = 0.75
    # v17: ATR-based floor for the clustering tolerance (as a fraction of
    # H1 ATR) so clustering tolerance is never inconsistent with the
    # ATR-based touch tolerance used elsewhere in the engine.
    cluster_atr_floor_mult: float = 0.08

    # Horizontal swing detection (5-bar fractal = "cluster swing" strength).
    swing_window: int = 5
    horiz_min_cluster: int = 3
    swing_lookback: int = 200           # bars for fib / murrey / swing anchors

    # Round-number grid.
    round_step: float = 25.0
    round_span: int = 5

    # --- v12: fixed anchor for pivot-family levels ---
    # Floor pivots / Camarilla / DeMark are DEFINED on a fixed period (prior
    # closed day, optionally prior week) - NOT re-derived per scanned
    # intraday timeframe. pivot_anchor="D1" uses the prior closed daily bar;
    # "W1" additionally adds a weekly-anchored set (useful for H4/D1 bias).
    pivot_anchor: str = "D1"
    use_weekly_pivot: bool = True

    # Spread/commission buffer added on top of the ATR stop so sizing isn't
    # optimistic on wide-spread gold feeds.
    spread_buffer_dollars: float = 0.30

    # ATR.
    atr_period: int = 14
    atr_tf: str = "H1"                   # timeframe used for stop/target sizing
    sl_atr_mult: float = 1.5            # stop = entry +/- sl_atr_mult * ATR
    tp_atr_mult: float = 3.0            # fallback target when no structure target
    min_target_atr_mult: float = 1.0    # ignore structure targets closer than this * ATR

    # Risk / sizing.
    account_risk_pct: float = 0.5       # % of balance risked per trade
    fallback_balance: float = 10_000.0  # used if account info unavailable
    min_rr: float = 2.0                 # setups below this are penalised

    # --- Evidence-based scoring (v2) ---
    touch_tol_pct: float = 0.0006       # half-band for counting touches at a level
    recency_halflife_bars: int = 300    # recency half-life for touches / rejections
    wick_atr_mult: float = 0.5          # min wick vs ATR to qualify a rejection
    round_tol_dollars: float = 1.5      # tolerance ($) for round-number tiers
    over_test_touches: int = 45         # beyond this a level is treated as "used up"

    # --- v13: EVIDENCE-GATED A+ ("N strong reasons") ----------------------
    # A setup is only promoted to A+ when it can articulate at least
    # `min_strong_reasons_aplus` INDEPENDENT, high-evidence reasons on top of
    # the numeric score. Every reason produced by `build_reasons` carries a
    # 0..1 evidence weight; only reasons at/above `strong_reason_min_weight`
    # count as "strong". This stops a level from being graded A+ on a pile of
    # cosmetic observations (one fib + one MA + "round number").
    #
    # v19 fix: the count is now taken from `_strong_count_deduped`, which
    # collapses correlated swing-evidence reasons (mtf/htf/major_structure/
    # swing_structure/swing_cluster/density - all derived from the same
    # confirmed-pivot data) into a single credit, same as `_METHOD_GROUP`
    # already does for the numeric score. The old default of 11 was tuned
    # against the UN-deduped (inflatable) count and is no longer reachable
    # under honest counting; thresholds below are rescaled accordingly.
    min_strong_reasons_aplus: int = 6
    min_strong_reasons_a: int = 4
    strong_reason_min_weight: float = 0.60
    aplus_min_score: int = 80
    a_min_score: int = 68
    b_min_score: int = 52
    max_reasons_display: int = 20
    show_reasons: bool = True
    # v19: floor a "structural fallback" candidate must clear before it can be
    # force-included when the strict top-3 score cut misses it. Without this
    # floor, `ensure_nearest` guaranteed a row on every side unconditionally,
    # which meant the table could never honestly say "no defensible S/R here" -
    # directly undercutting the "strict scoring" premise of the whole engine.
    min_display_score: float = 35.0
    compact_output: bool = True    # short, friendly terminal report
    table_only: bool = True        # print ONLY the summary table

    # --- v14: 2026 market-data calibration -------------------------------
    # Gold traded ~$3,300-3,400 when the fixed dollar constants below were
    # chosen; through 2026 XAUUSD has run in the ~$4,000-4,700 band, so every
    # hard-coded dollar distance (scan range, round-number grid, tolerances,
    # spread buffer) is now auto-rescaled to live price + ATR instead of
    # silently getting tighter as price rises.
    auto_scale_params: bool = True
    price_baseline: float = 3400.0      # price the fixed $ constants were tuned at

    # Naked / virgin POC: a prior session's point of control that price has
    # never returned to. Untested POCs act as magnets and are among the
    # highest-quality institutional levels.
    use_naked_poc: bool = True
    naked_poc_days: int = 20            # prior sessions scanned for untested POCs
    naked_poc_max: int = 6              # keep the N nearest untested POCs

    # HVN / LVN acceptance. High-volume nodes hold (reversal levels); low-volume
    # nodes are traversed fast, so they are poor A+ reversal candidates.
    lvn_threshold: float = 0.15         # node volume < this fraction of POC = LVN
    hvn_threshold: float = 0.55         # node volume >= this fraction of POC = HVN

    # --- v18: durable HTF market-structure detector ------------------------
    use_major_structure: bool = True
    major_swing_left: int = 8
    major_swing_right: int = 8
    major_eq_min_pivots: int = 2
    major_eq_tol_atr_mult: float = 0.30
    major_cluster_atr_mult: float = 0.45
    major_range_htf_atr_mult: float = 1.5
    major_max_per_side: int = 12

    # --- v17: tick-volume honesty -----------------------------------------
    # MT5 `tick_volume` is a TICK COUNT (number of price changes), not real
    # traded volume - there is no consolidated tape for OTC spot gold/CFDs.
    # It correlates loosely with activity but can diverge a lot from a
    # broker's true liquidity picture, and every POC/VAH/VAL/naked-POC/HVN-
    # LVN level is built entirely on it. `volume_confidence` (0..1) discounts
    # every volume-derived weight/bonus accordingly instead of treating this
    # proxy data as if it were real exchange volume. Set to 1.0 only if
    # `df["tick_volume"]` is actually replaced with a real volume feed
    # upstream (e.g. a futures/exchange volume source).
    volume_confidence: float = 0.5

    # VWAP standard-deviation bands (institutional mean-reversion levels).
    vwap_sigma_bands: List[float] = field(default_factory=lambda: [1.0, 2.0])

    # v16 fix: session VWAP used to be masked by `df.index.normalize()` (the
    # broker server's own calendar-day boundary). For a 24-hour OTC gold
    # market, that boundary is an arbitrary server-timezone artifact, not a
    # real session start - it silently assumed "the feed's midnight" is a
    # meaningful VWAP anchor. Instead, VWAP now uses a fixed ROLLING window of
    # real hours (independent of server timezone), which is a well-defined
    # quantity regardless of where the broker's day happens to roll over.
    vwap_session_hours: float = 24.0

    # Volatility regime: ATR vs its own median. Expansion widens stops and
    # makes far levels reachable; compression favours tight, close levels.
    vol_regime_lookback: int = 100
    vol_expansion_mult: float = 1.25
    vol_compression_mult: float = 0.80

    # ICT-style killzones (UTC) used as a context reason.
    killzones_utc: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        ("London", "07:00", "10:00"),
        ("NY AM", "12:00", "15:00"),
    ])

    # --- 2026 upgrade: density (KDE/DBSCAN) + liquidity structure (SMC) ---
    swing_left: int = 5                 # bars left of a confirmed fractal swing pivot (5 = cluster-swing strength)
    swing_right: int = 5                # bars right of a confirmed fractal swing pivot
    provisional_swing_right: int = 2    # v17: shorter right window for the low-weight "fresh structure" layer
    kde_bw_atr_mult: float = 0.6        # KDE bandwidth = mult * ATR (pivot density)
    eqh_eql_tol_atr_mult: float = 0.35  # equal-high/low + sweep tolerance vs ATR
    sweep_lookback_bars: int = 60       # recent bars scanned for liquidity sweeps

    # --- v11: multi-timeframe SWING CLUSTER zones (pools confirmed 5-bar
    # pivots across every timeframe above M5 - M15..D1 - into first-class
    # confluence levels, distinct from the single "latest pivot" per TF). ---
    swing_cluster_lookback: int = 8         # most recent confirmed pivots per tf/side to pool
    swing_cluster_tol_atr_mult: float = 0.5 # group pivots within this * ATR into one zone
    swing_cluster_min_weight: float = 0.8   # min combined tf/recency weight to keep a zone
    swing_cluster_top_n: int = 3            # keep top-N swing-cluster zones per side (S and R)

    # DXY confluence.
    use_dxy: bool = True
    # v17 fix: 20 bars on H1 (~20 hours) is too short/noisy to represent a
    # stable macro relationship, and the old code applied a fixed slope-sign
    # bias without ever checking whether gold and DXY were even correlated
    # over that window (the inverse relationship regularly breaks down -
    # real-yield-driven or safe-haven regimes can move both together).
    # Lookback widened and a rolling correlation gate added (see
    # `compute_dxy_bias`); the score effect itself is also roughly halved
    # (below, in `score_cluster`) to reflect that this is a soft macro
    # tailwind/headwind, not a structural S/R fact.
    dxy_slope_lookback: int = 60        # bars to measure DXY slope (was 20)
    dxy_min_abs_corr: float = 0.30      # below this |corr| with gold, bias is ignored

    # News windows to avoid (UTC HH:MM). Populate from your economic calendar.
    # Each entry: ("YYYY-MM-DD", "HH:MM", minutes_buffer_each_side)
    news_windows_utc: List[Tuple[str, str, int]] = field(default_factory=list)

    timeframes: List[str] = field(
        default_factory=lambda: ["H1", "H4", "D1"]
    )
    lowest_tf: str = "H1"

    # Session VWAP is only meaningful intraday; skip on H4/D1.
    vwap_tfs: List[str] = field(default_factory=lambda: ["H1"])

    w_type: Dict[str, int] = field(default_factory=lambda: {
        "naked_poc": 23, "volume_profile": 22, "major_structure": 22, "horizontal": 20, "swing_cluster": 19,
        "vwap_band": 13,
        "swing_structure": 17, "fibonacci": 18, "pivot": 15, "vwap": 15,
        "murrey_math": 12, "camarilla": 10, "bollinger": 8,
        "moving_average": 7, "demark": 6, "round_number": 5,
        "provisional_structure": 4,   # v17: fast/unconfirmed - deliberately low
    })
    # HTF weighted higher than LTF (classic S/R: HTF levels are stronger).
    w_tf: Dict[str, int] = field(default_factory=lambda: {
        "W1": 35, "D1": 30, "H4": 25, "H1": 20,
    })


@dataclass
class TradePlan:
    direction: str          # "BUY" or "SELL"
    entry: float
    stop: float
    target: float
    rr: float
    lots: float
    risk_dollars: float


@dataclass
class Reason:
    """A single, independent piece of evidence supporting a level.

    weight   0..1 evidence strength (how hard the underlying proof is)
    strong   True when weight >= cfg.strong_reason_min_weight; only strong,
             group-deduped reasons count toward the A+ gate (see
             `_strong_count_deduped`; default: 6 required).
    """
    key: str
    text: str
    weight: float
    category: str           # structure | volume | liquidity | momentum | risk | context
    strong: bool = False

    def label(self) -> str:
        return f"{'*' if self.strong else '-'} {self.text}  [{self.weight:.2f}]"



@dataclass
class MajorLevel:
    """Institutional / HTF market structure level, separated from trade scoring."""
    price: float
    side: str
    strength: float
    reasons: List[str]
    timeframe: str


class MajorMarketStructureEngine:
    """Detect durable HTF support/resistance independently from setup scoring.

    v18 repairs the old implementation in three ways:
      1) uses confirmed pivots, not adjacent candles, for equal-high/low zones;
      2) scales all tolerances with the relevant timeframe ATR;
      3) returns levels that are actually injected into the scanner's raw pool.
    """

    def __init__(self, scanner):
        self.scanner = scanner

    def detect(self) -> List[MajorLevel]:
        if not self.scanner.cfg.use_major_structure:
            return []
        levels: List[MajorLevel] = []
        levels.extend(self.htf_swings())
        levels.extend(self.equal_high_low())
        levels.extend(self.previous_period_levels())
        return self.cluster(levels)

    def _confirmed_pivots(self, tf: str) -> Tuple[List[float], List[float]]:
        df = self.scanner.df_dict.get(tf)
        if df is None:
            return [], []
        l = max(2, int(self.scanner.cfg.major_swing_left))
        r = max(2, int(self.scanner.cfg.major_swing_right))
        if len(df) < l + r + 5:
            return [], []
        h = df["high"].to_numpy(dtype=float)
        lo = df["low"].to_numpy(dtype=float)
        ph: List[float] = []
        pl: List[float] = []
        for i in range(l, len(df) - r):
            hw = h[i-l:i+r+1]
            lw = lo[i-l:i+r+1]
            if h[i] >= np.max(hw) and h[i] > np.max(h[i-l:i]):
                ph.append(float(h[i]))
            if lo[i] <= np.min(lw) and lo[i] < np.min(lo[i-l:i]):
                pl.append(float(lo[i]))
        return ph, pl

    def htf_swings(self) -> List[MajorLevel]:
        out: List[MajorLevel] = []
        for tf in ("D1", "H4"):
            highs, lows = self._confirmed_pivots(tf)
            tf_strength = 94.0 if tf == "D1" else 88.0
            # Keep recent pivots only; old history still participates through
            # clustering/touches elsewhere, while this layer represents the
            # currently relevant HTF map.
            for price in highs[-8:]:
                side = "RESISTANCE" if price >= self.scanner.current_price else "SUPPORT"
                out.append(MajorLevel(price, side, tf_strength, [f"{tf} confirmed swing high"], tf))
            for price in lows[-8:]:
                side = "SUPPORT" if price <= self.scanner.current_price else "RESISTANCE"
                out.append(MajorLevel(price, side, tf_strength, [f"{tf} confirmed swing low"], tf))
        return out

    def equal_high_low(self) -> List[MajorLevel]:
        out: List[MajorLevel] = []
        cfg = self.scanner.cfg
        for tf in ("D1", "H4"):
            highs, lows = self._confirmed_pivots(tf)
            atr = max(self.scanner._atr_for_tf(tf), 1e-9)
            tol = max(cfg.major_eq_tol_atr_mult * atr, 0.25)

            def group(vals: List[float], side_hint: str) -> None:
                if len(vals) < cfg.major_eq_min_pivots:
                    return
                vals = sorted(vals)
                groups: List[List[float]] = [[vals[0]]]
                for v in vals[1:]:
                    c = float(np.mean(groups[-1]))
                    if abs(v - c) <= tol:
                        groups[-1].append(v)
                    else:
                        groups.append([v])
                for g in groups:
                    if len(g) < cfg.major_eq_min_pivots:
                        continue
                    price = float(np.mean(g))
                    side = side_hint
                    # Once decisively crossed, old resistance/support can flip.
                    if side_hint == "RESISTANCE" and price < self.scanner.current_price:
                        side = "SUPPORT"
                    elif side_hint == "SUPPORT" and price > self.scanner.current_price:
                        side = "RESISTANCE"
                    strength = min(99.0, 90.0 + 2.0 * (len(g) - 2) + (3.0 if tf == "D1" else 0.0))
                    label = "equal highs" if side_hint == "RESISTANCE" else "equal lows"
                    out.append(MajorLevel(price, side, strength, [f"{tf} {label} ({len(g)} pivots)"], tf))

            group(highs[-20:], "RESISTANCE")
            group(lows[-20:], "SUPPORT")
        return out

    def previous_period_levels(self) -> List[MajorLevel]:
        out: List[MajorLevel] = []
        # Scanned H4/D1 frames already have their forming bar removed, so the
        # last completed period is iloc[-1], not iloc[-2].
        for tf in ("D1",):
            df = self.scanner.df_dict.get(tf)
            if df is None or len(df) < 1:
                continue
            bar = df.iloc[-1]
            out.append(MajorLevel(float(bar.high), "RESISTANCE", 93, [f"Previous {tf} high"], tf))
            out.append(MajorLevel(float(bar.low), "SUPPORT", 93, [f"Previous {tf} low"], tf))

        # W1 is not part of cfg.timeframes by default, so fetch it explicitly.
        w1 = self.scanner.get_data("W1")
        if w1 is not None and len(w1):
            bar = w1.iloc[-1]
            out.append(MajorLevel(float(bar.high), "RESISTANCE", 97, ["Previous W1 high"], "W1"))
            out.append(MajorLevel(float(bar.low), "SUPPORT", 97, ["Previous W1 low"], "W1"))
        return out

    def cluster(self, levels: List[MajorLevel]) -> List[MajorLevel]:
        if not levels:
            return []
        zones: List[MajorLevel] = []
        for side in ("SUPPORT", "RESISTANCE"):
            side_levels = sorted((x for x in levels if x.side == side), key=lambda x: x.price)
            if not side_levels:
                continue
            groups: List[List[MajorLevel]] = [[side_levels[0]]]
            for lvl in side_levels[1:]:
                g = groups[-1]
                weights = np.array([x.strength for x in g], dtype=float)
                prices = np.array([x.price for x in g], dtype=float)
                center = float(np.average(prices, weights=weights))
                ref_tf = max((x.timeframe for x in g + [lvl]),
                             key=lambda t: self.scanner.cfg.w_tf.get(t, 0))
                atr = max(self.scanner._atr_for_tf(ref_tf), self.scanner._atr or 0.0, 1.0)
                tol = self.scanner.cfg.major_cluster_atr_mult * atr
                if abs(lvl.price - center) <= tol:
                    g.append(lvl)
                else:
                    groups.append([lvl])
            for g in groups:
                weights = np.array([x.strength for x in g], dtype=float)
                prices = np.array([x.price for x in g], dtype=float)
                center = float(np.average(prices, weights=weights))
                tf = max((x.timeframe for x in g), key=lambda t: self.scanner.cfg.w_tf.get(t, 0))
                strength = min(100.0, max(x.strength for x in g) + 2.0 * (len(g) - 1))
                reasons: List[str] = []
                for x in g:
                    for reason in x.reasons:
                        if reason not in reasons:
                            reasons.append(reason)
                zones.append(MajorLevel(center, side, strength, reasons, tf))
        # Keep a bounded, nearby map on each side.
        px = float(self.scanner.current_price or 0.0)
        final: List[MajorLevel] = []
        for side in ("SUPPORT", "RESISTANCE"):
            candidates = [z for z in zones if z.side == side]
            candidates.sort(key=lambda z: (abs(z.price - px), -z.strength))
            final.extend(candidates[: self.scanner.cfg.major_max_per_side])
        return final

# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #
class ApexScanner:
    def __init__(self, config: Optional[ApexConfig] = None) -> None:
        self.cfg = config or ApexConfig()
        self.current_price: Optional[float] = None
        self.major_levels: List[MajorLevel] = []
        self.df_dict: Dict[str, pd.DataFrame] = {}
        self._poc: Optional[float] = None
        self._vol_profile: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._vah: Optional[float] = None
        self._val: Optional[float] = None
        # Confirmed swing pivots PER TIMEFRAME (ages in minutes), keyed by tf.
        self._swings_by_tf: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None
        self._atr: Optional[float] = None
        self._atr_by_tf: Dict[str, float] = {}   # v16: ATR per scanned timeframe (see compute_atr)
        self._vwap_sd: Optional[float] = None
        self._naked_pocs: List[Tuple[float, int]] = []   # (price, days untested)
        self._vol_regime: str = "normal"         # "expansion" | "normal" | "compression"
        self._scaled: bool = False
        self._dxy_bias: str = "neutral"          # "bullish_usd" | "bearish_usd"
        self._account_balance: float = 0.0
        self._sym_info = None
        self._tf_map = {
            "H1": mt5.TIMEFRAME_H1 if mt5 else None,
            "H4": mt5.TIMEFRAME_H4 if mt5 else None,
            "D1": mt5.TIMEFRAME_D1 if mt5 else None,
            "W1": mt5.TIMEFRAME_W1 if mt5 else None,
        }

    # ---------------- MT5 plumbing ---------------- #
    def initialize_mt5(self) -> bool:
        if mt5 is None:
            log.error("MetaTrader5 package not available on this host.")
            return False
        if not mt5.initialize():
            log.error("MT5 init failed: %s", mt5.last_error())
            return False
        if not mt5.symbol_select(self.cfg.symbol, True):
            log.error("Symbol %s not found", self.cfg.symbol)
            mt5.shutdown()
            return False
        self._sym_info = mt5.symbol_info(self.cfg.symbol)
        acc = mt5.account_info()
        self._account_balance = acc.balance if acc else self.cfg.fallback_balance
        log.info("MT5 ready - %s | balance %.2f", self.cfg.symbol, self._account_balance)
        return True

    def get_data(self, timeframe: str, symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
        symbol = symbol or self.cfg.symbol
        rates = mt5.copy_rates_from_pos(symbol, self._tf_map[timeframe], 0, self.cfg.bars)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        if self.cfg.use_closed_bars and len(df) > 1:
            df = df.iloc[:-1]
        df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3
        if "tick_volume" not in df.columns:
            df["tick_volume"] = 1.0
        return df

    def _get_anchor_ohlc(self, timeframe: str) -> Tuple[Optional[pd.Series], float]:
        """Last FULLY CLOSED bar on `timeframe`, used as the fixed anchor for
        pivot/Camarilla/DeMark levels. Independent of `cfg.timeframes` /
        `use_closed_bars` bookkeeping for the scanned dataframes - this always
        fetches its own small window and drops the still-forming bar itself,
        so it stays correct even if daily/weekly aren't otherwise scanned.

        v16 fix: floor pivots, Camarilla, and DeMark are ENTIRELY defined by
        ONE anchor bar's OHLC. If that single day/week had an abnormal gap or
        volatility spike (news, thin holiday liquidity), all ~17 pivot-family
        levels derived from it were silently contaminated by one outlier bar
        with no way to detect it. Returns (anchor_bar, quality) where quality
        in (0, 1] shrinks when the anchor's range is a large multiple of its
        own recent median range - callers down-weight pivot-family levels
        built on a low-quality anchor instead of trusting them at face value.
        """
        if mt5 is None:
            return None, 1.0
        tf_const = self._tf_map.get(timeframe)
        if tf_const is None:
            return None, 1.0
        rates = mt5.copy_rates_from_pos(self.cfg.symbol, tf_const, 0, 25)
        if rates is None or len(rates) < 2:
            return None, 1.0
        df = pd.DataFrame(rates)
        anchor = df.iloc[-2]   # last closed bar (iloc[-1] may still be forming)
        history = df.iloc[:-2] if len(df) > 2 else df.iloc[:-1]
        quality = 1.0
        anchor_range = float(anchor["high"] - anchor["low"])
        if len(history) >= 5 and anchor_range > 0:
            med_range = float((history["high"] - history["low"]).median())
            if med_range > 0:
                ratio = anchor_range / med_range
                if ratio > 1.8:
                    # Discount linearly from 1.0 (at 1.8x median range) down
                    # to a 0.4 floor (at 3.5x+) - never fully zeroed, since an
                    # unusual day is still real information, just less trustworthy.
                    quality = max(0.4, 1.0 - (ratio - 1.8) / (3.5 - 1.8) * 0.6)
        return anchor, quality

    def _refresh_current_price(self) -> None:
        price: Optional[float] = None
        # Live tick only when explicitly requested; it is the #1 source of
        # run-to-run instability (every tick reshuffles proximity/range/grid).
        if self.cfg.reference_mode == "tick":
            tick = mt5.symbol_info_tick(self.cfg.symbol) if mt5 else None
            if tick and tick.bid and tick.ask:
                price = (tick.bid + tick.ask) / 2.0
        # Deterministic default: last CLOSED bar close (identical for the whole bar).
        if price is None and self.cfg.lowest_tf in self.df_dict:
            price = float(self.df_dict[self.cfg.lowest_tf]["close"].iloc[-1])
        # Snap to a coarse grid so sub-quantum wiggle cannot move levels/scores.
        if price is not None and self.cfg.price_quantum > 0:
            price = round(price / self.cfg.price_quantum) * self.cfg.price_quantum
        self.current_price = price

    # ---------------- Indicators ---------------- #
    @staticmethod
    def _atr_series(df: pd.DataFrame, period: int) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    def compute_atr(self) -> None:
        """v16 fix: compute ATR for EVERY scanned timeframe, not just
        `cfg.atr_tf` (H1). Pooling swing pivots from H1/H4/D1 onto one price
        axis (density/liquidity/swing-cluster) previously scaled every
        timeframe's kernel bandwidth and grouping tolerance off a single H1
        ATR - too tight for D1 structure, arbitrary for H4. `self._atr`
        (H1, used for stop/target sizing) is kept for backward compatibility."""
        self._atr_by_tf = {}
        for tf, df in self.df_dict.items():
            if df is not None and len(df) > self.cfg.atr_period:
                self._atr_by_tf[tf] = float(self._atr_series(df, self.cfg.atr_period).iloc[-1])
        df = self.df_dict.get(self.cfg.atr_tf)
        if df is not None and len(df) > self.cfg.atr_period:
            self._atr = float(self._atr_series(df, self.cfg.atr_period).iloc[-1])
        else:
            self._atr = self._atr_by_tf.get(self.cfg.atr_tf)

    def _atr_for_tf(self, tf: str) -> float:
        """Per-timeframe ATR with a safe fallback chain (own tf -> atr_tf ->
        largest available) so a missing/short-history timeframe never breaks
        the KDE/liquidity/cluster math that depends on a positive ATR."""
        v = self._atr_by_tf.get(tf)
        if v:
            return v
        if self._atr:
            return self._atr
        return max(self._atr_by_tf.values()) if self._atr_by_tf else 0.0

    def _reference_htf_atr(self) -> float:
        """ATR of the highest timeframe available (D1 > H4 > H1). Used as the
        grouping scale for MTF swing-cluster zones: those zones deliberately
        pool H1..D1 pivots into one daily-scale structural zone, so the
        tolerance that decides "do these pivots belong to the same zone"
        should be a daily-scale distance, not an H1 ATR distance (which was
        systematically too tight and under-grouped genuine HTF clusters)."""
        for tf in ("W1", "D1", "H4", "H1"):
            if tf in self._atr_by_tf:
                return self._atr_by_tf[tf]
        return self._atr or 0.0

    # ---------------- Level calculators ---------------- #
    @staticmethod
    def pivot_points(ohlc: pd.Series) -> Dict[str, float]:
        """Classic floor pivots from a FIXED anchor period's OHLC (e.g. prior
        closed day or week). Must be computed once from a period anchor, not
        re-derived from the last bar of an arbitrary intraday timeframe -
        doing the latter turns "pivot points" into noise from a single
        15/30/60-min candle and pollutes method/timeframe confluence with
        false agreement (see v12 patch notes)."""
        h, l, c = ohlc["high"], ohlc["low"], ohlc["close"]
        pp = (h + l + c) / 3
        return {"PP": pp, "R1": 2 * pp - l, "R2": pp + (h - l), "R3": h + 2 * (pp - l),
                "S1": 2 * pp - h, "S2": pp - (h - l), "S3": l - 2 * (h - pp)}

    def fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        window = df.tail(self.cfg.swing_lookback)
        h, l = window["high"].max(), window["low"].min()
        d = h - l
        if d == 0:
            return {}
        return {f"Fib_{int(p * 100)}": l + p * d
                for p in (0.236, 0.382, 0.5, 0.618, 0.786)}

    @staticmethod
    def moving_averages(df: pd.DataFrame) -> Dict[str, float]:
        mas: Dict[str, float] = {}
        for p in (20, 50, 100, 200):
            if len(df) >= p:
                mas[f"MA_{p}"] = df["close"].rolling(p).mean().iloc[-1]
        for p in (9, 21, 55, 89):
            if len(df) >= p:
                mas[f"EMA_{p}"] = df["close"].ewm(span=p, adjust=False).mean().iloc[-1]
        return mas

    def _cluster_tolerance_pct(self) -> float:
        """cfg.cluster_pct rescaled by the current volatility regime (v16
        fix) - shared by horizontal_levels and cluster_levels so both use the
        same volatility-aware tolerance instead of a fixed % of price."""
        mult = 1.0
        if self.cfg.cluster_regime_adaptive:
            if self._vol_regime == "expansion":
                mult = self.cfg.cluster_expansion_mult
            elif self._vol_regime == "compression":
                mult = self.cfg.cluster_compression_mult
        pct = self.cfg.cluster_pct * mult
        if self._atr and self.current_price:
            # v17 fix: a pure % tolerance conflates timeframes - the same
            # $2.4-ish band (0.0007 * $3400) is applied whether the two
            # points being compared are two H1 wiggles or an H1 point and a
            # D1 structural level. Floor the tolerance at a small ATR
            # fraction (consistent with the touch-tolerance logic in
            # `_touch_and_rejection`) so clustering tolerance never falls
            # BELOW what the live volatility regime would consider "the same
            # price" on the lowest scanned timeframe.
            atr_floor_pct = (self.cfg.cluster_atr_floor_mult * self._atr) / self.current_price
            pct = max(pct, atr_floor_pct)
        return pct

    @staticmethod
    def _centroid_cluster_1d(values: List[float], weights: List[float],
                             tol_pct: float, max_width_mult: float
                             ) -> List[List[int]]:
        """Shared weighted-centroid, width-capped 1D clustering (v19).

        v17 fixed exactly this chaining bug (greedy single-linkage: comparing
        each new point only to the previous raw point lets A and C merge
        transitively through an unrelated intermediate B, even when A and C
        are individually far apart) inside `cluster_levels`, but the fix was
        never applied to `horizontal_levels`, which kept comparing each new
        swing pivot to a plain running mean with NO width cap - so a chain of
        swing highs/lows could drift across an arbitrarily wide price range
        and still register as "one level". This helper is now shared by both
        callers so there is exactly one clustering algorithm in the file.
        Returns index groups (into `values`/`weights`), sorted by value.
        """
        order = sorted(range(len(values)), key=lambda i: values[i])
        clusters: List[List[int]] = [[order[0]]]
        wsums: List[float] = [weights[order[0]]]
        centroids: List[float] = [values[order[0]]]
        anchors: List[float] = [values[order[0]]]
        for idx in order[1:]:
            v, w = values[idx], weights[idx]
            centroid = centroids[-1]
            anchor = anchors[-1]
            gap_tol = tol_pct * max(v, 1e-9)
            width_tol = max_width_mult * tol_pct * max(anchor, 1e-9)
            if abs(v - centroid) <= gap_tol and (v - anchor) <= width_tol:
                clusters[-1].append(idx)
                wsums[-1] += w
                centroids[-1] = sum(weights[i] * values[i] for i in clusters[-1]) / wsums[-1]
            else:
                clusters.append([idx])
                wsums.append(w)
                centroids.append(v)
                anchors.append(v)
        return clusters

    def horizontal_levels(self, df: pd.DataFrame) -> List[float]:
        w = self.cfg.swing_window
        highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
        n = len(df)
        pts: List[float] = []
        for i in range(w, n - w):
            if highs[i] == highs[i - w:i + w + 1].max():
                pts.append(highs[i])
            if lows[i] == lows[i - w:i + w + 1].min():
                pts.append(lows[i])
        if not pts:
            return []
        tol_pct = self._cluster_tolerance_pct()
        weights = [1.0] * len(pts)  # every swing pivot counted equally here
        groups = self._centroid_cluster_1d(pts, weights, tol_pct,
                                           self.cfg.cluster_max_width_mult)
        out: List[float] = []
        for g in groups:
            if len(g) >= self.cfg.horiz_min_cluster:
                vals = [pts[i] for i in g]
                out.append(float(np.mean(vals)))
        return out

    @staticmethod
    def bollinger_levels(df: pd.DataFrame) -> Dict[str, float]:
        sma = df["close"].rolling(20).mean().iloc[-1]
        std = df["close"].rolling(20).std().iloc[-1]
        return {"BB_Upper": sma + 2 * std, "BB_Middle": sma, "BB_Lower": sma - 2 * std}

    @staticmethod
    def demark_levels(ohlc: pd.Series) -> Dict[str, float]:
        """DeMark reaction points from a FIXED anchor period's OHLC (prior
        closed day/week) - same anchoring fix as pivot_points/camarilla."""
        h, l, c, o = ohlc["high"], ohlc["low"], ohlc["close"], ohlc["open"]
        if c < o:
            x = (2 * h + l + c) / 2
        elif c > o:
            x = (h + 2 * l + c) / 2
        else:
            x = (h + l + 2 * c) / 2
        return {"TD_R1": x - l, "TD_S1": x - h}

    def round_numbers(self) -> List[float]:
        # Absolute, price-independent grid across the scan window so the same
        # round levels appear every run (no drift as price wiggles).
        step = self.cfg.round_step
        rng = self.cfg.max_range_dollars + step
        lo = math.floor((self.current_price - rng) / step) * step
        hi = math.ceil((self.current_price + rng) / step) * step
        n = int(round((hi - lo) / step)) + 1
        return [lo + i * step for i in range(n) if lo + i * step > 0]

    def session_vwap(self, df: pd.DataFrame, tf: str) -> Dict[str, float]:
        """Rolling-window VWAP anchored to a fixed real-time DURATION
        (`cfg.vwap_session_hours`), not the broker's calendar-day boundary.

        v16 fix: the old `day == day[-1]` mask assumed the feed's own
        midnight (in whatever timezone the MT5 server happens to report bars
        in - unknown/unverified here) is a meaningful trading-session
        boundary. For a 24-hour OTC gold market that assumption is not
        guaranteed to hold, and silently produces a wrong VWAP if the
        server's midnight doesn't line up with an actual session open. A
        fixed-hours rolling window sidesteps the timezone assumption
        entirely: it's always exactly `vwap_session_hours` of real trading,
        whatever timezone the bars are stamped in.
        """
        mins = _TF_MINUTES.get(tf, 60)
        n_bars = max(2, int(round(self.cfg.vwap_session_hours * 60 / mins)))
        window = df.tail(n_bars)
        vol = window["tick_volume"].to_numpy(dtype=float)
        tp = window["hlc3"].to_numpy(dtype=float)
        if vol.sum() == 0:
            return {}
        vwap = float((tp * vol).sum() / vol.sum())
        sd = (float((vol * (tp - vwap) ** 2).sum() / vol.sum())) ** 0.5
        self._vwap_sd = sd
        return {"VWAP": vwap}

    def session_vwap_bands(self, df: pd.DataFrame, tf: str) -> Dict[str, float]:
        """VWAP +/- k*sigma bands (1s ~ 68%, 2s ~ 95% of session volume).
        Price stretched to a band is statistically 'expensive'/'cheap' vs the
        session's volume-weighted fair value - a genuine mean-reversion edge."""
        base = self.session_vwap(df, tf)
        if not base or not self._vwap_sd:
            return {}
        vwap, sd = base["VWAP"], self._vwap_sd
        out: Dict[str, float] = {}
        for k in self.cfg.vwap_sigma_bands:
            out[f"VWAP_U{k:g}"] = vwap + k * sd
            out[f"VWAP_L{k:g}"] = vwap - k * sd
        return out

    def murrey_math(self, df: pd.DataFrame) -> Dict[str, float]:
        window = df.tail(self.cfg.swing_lookback)
        h, l = window["high"].max(), window["low"].min()
        step = (h - l) / 8
        return {f"MM_{i}/8": l + i * step for i in range(9)} if step else {}

    @staticmethod
    def camarilla(ohlc: pd.Series) -> Dict[str, float]:
        """Camarilla levels from a FIXED anchor period's OHLC (prior closed
        day/week) - same anchoring fix as pivot_points/demark_levels."""
        h, l, c = ohlc["high"], ohlc["low"], ohlc["close"]
        r = h - l
        divs = {1: 12, 2: 6, 3: 4, 4: 2}
        out = {f"C_R{i}": c + r * 1.1 / d for i, d in divs.items()}
        out.update({f"C_S{i}": c - r * 1.1 / d for i, d in divs.items()})
        return out

    # ---------------- DXY intermarket ---------------- #
    def compute_dxy_bias(self) -> None:
        if not self.cfg.use_dxy or mt5 is None:
            return
        if not mt5.symbol_select(self.cfg.dxy_symbol, True):
            log.warning("DXY symbol %s not found - skipping intermarket filter",
                        self.cfg.dxy_symbol)
            return
        df = self.get_data(self.cfg.atr_tf, symbol=self.cfg.dxy_symbol)
        gold_df = self.df_dict.get(self.cfg.atr_tf)
        if df is None or len(df) < self.cfg.dxy_slope_lookback or gold_df is None:
            return
        closes = df["close"].tail(self.cfg.dxy_slope_lookback).to_numpy()
        slope = np.polyfit(np.arange(len(closes)), closes, 1)[0]

        # v17 fix: don't assume the inverse gold/DXY relationship is always
        # live - check it over the SAME window before trusting the slope
        # sign. If gold and DXY aren't meaningfully (inversely) correlated
        # right now, the "bias" is macro noise, not a tradeable fact.
        n = min(len(closes), len(gold_df))
        if n >= 5:
            dxy_win = closes[-n:]
            gold_win = gold_df["close"].tail(n).to_numpy()
            if dxy_win.std() > 0 and gold_win.std() > 0:
                corr = float(np.corrcoef(dxy_win, gold_win)[0, 1])
            else:
                corr = 0.0
        else:
            corr = 0.0

        if abs(corr) < self.cfg.dxy_min_abs_corr:
            self._dxy_bias = "neutral"
            log.info("DXY bias: neutral (corr %.2f too weak over %d bars)",
                     corr, self.cfg.dxy_slope_lookback)
            return

        # Gold is inversely correlated to USD strength (when corr confirms it).
        if slope > 0:
            self._dxy_bias = "bullish_usd"   # headwind for gold longs
        elif slope < 0:
            self._dxy_bias = "bearish_usd"   # tailwind for gold longs
        log.info("DXY bias: %s (slope %.5f, corr %.2f)",
                 self._dxy_bias, slope, corr)

    # ---------------- News filter ---------------- #
    def in_news_window(self) -> Optional[str]:
        now = datetime.now(timezone.utc)
        for date_str, hm, buf in self.cfg.news_windows_utc:
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
                t = time.fromisoformat(hm)
            except ValueError:
                continue
            event = datetime.combine(d, t, tzinfo=timezone.utc)
            if abs((now - event).total_seconds()) <= buf * 60:
                return f"{date_str} {hm} UTC (+/-{buf}m)"
        return None

    # ---------------- Collection + clustering ---------------- #
    def collect_raw_levels(self) -> List[dict]:
        self._swings_by_tf = None   # rebuild multi-timeframe swing-pivot cache for this scan
        for tf in self.cfg.timeframes:
            df = self.get_data(tf)
            if df is None or len(df) < 50:
                log.warning("%s: insufficient data", tf)
                continue
            self.df_dict[tf] = df

        self._refresh_current_price()
        if self.current_price is None:
            raise RuntimeError("Could not determine current price.")

        # v18: ATR MUST be ready before major-structure detection because the
        # equal-high/low and zone clustering logic is ATR-scaled. v17 ran the
        # structure engine first, so it silently fell back to a hard-coded $3.
        self.compute_atr()
        # 2026 calibration: rescale $ constants to the live price/vol regime
        # BEFORE any range filtering or round-number grid is built.
        self._auto_scale_params()
        self.compute_vol_regime()
        self.major_levels = MajorMarketStructureEngine(self).detect()

        raw: List[dict] = []

        def add(mapping, typ, tf, weight_mult: float = 1.0):
            for name, lvl in mapping.items():
                if lvl and lvl > 0:
                    raw.append({"level": float(lvl), "type": typ, "name": name, "tf": tf,
                                "weight_mult": weight_mult})

        # v18: this engine used to be dead code: detected major HTF levels
        # were stored in self.major_levels and then never scored or displayed.
        # Inject them as first-class price-structure evidence.
        for i, lvl in enumerate(self.major_levels):
            raw.append({
                "level": float(lvl.price),
                "type": "major_structure",
                "name": f"Major_{lvl.side}_{i+1}",
                "tf": lvl.timeframe,
                "weight_mult": max(0.55, min(1.15, lvl.strength / 90.0)),
                "major_side": lvl.side,
                "major_reasons": list(lvl.reasons),
            })

        # Pivot-family levels (floor pivots / Camarilla / DeMark) are DEFINED
        # on a fixed anchor period, not per scanned intraday timeframe -
        # computing them from e.g. the last M15 bar turns them into noise
        # from a single 15-min candle and pollutes method/timeframe
        # confluence with false agreement. Compute once from the prior
        # closed daily bar (and optionally weekly) instead.
        daily_anchor, daily_q = self._get_anchor_ohlc(self.cfg.pivot_anchor)
        if daily_anchor is not None:
            add(self.pivot_points(daily_anchor), "pivot", self.cfg.pivot_anchor, daily_q)
            add(self.camarilla(daily_anchor), "camarilla", self.cfg.pivot_anchor, daily_q)
            add(self.demark_levels(daily_anchor), "demark", self.cfg.pivot_anchor, daily_q)
            if daily_q < 0.99:
                log.info("%s anchor bar range is an outlier (x%.2f median) - "
                         "pivot/camarilla/demark down-weighted", self.cfg.pivot_anchor, daily_q)
        else:
            log.warning("No %s anchor bar available - skipping pivot/camarilla/demark",
                        self.cfg.pivot_anchor)
        if self.cfg.use_weekly_pivot:
            weekly_anchor, weekly_q = self._get_anchor_ohlc("W1")
            if weekly_anchor is not None:
                add(self.pivot_points(weekly_anchor), "pivot", "W1", weekly_q)
                add(self.camarilla(weekly_anchor), "camarilla", "W1", weekly_q)

        for tf, df in self.df_dict.items():
            add(self.fibonacci_levels(df), "fibonacci", tf)
            add(self.moving_averages(df), "moving_average", tf)
            add(self.bollinger_levels(df), "bollinger", tf)
            if tf in self.cfg.vwap_tfs:
                add(self.session_vwap(df, tf), "vwap", tf)
                add(self.session_vwap_bands(df, tf), "vwap_band", tf)
            add(self.murrey_math(df), "murrey_math", tf)
            for i, lvl in enumerate(self.horizontal_levels(df)):
                raw.append({"level": float(lvl), "type": "horizontal",
                            "name": f"HL_{i + 1}", "tf": tf})
            # Latest confirmed swing high/low for this timeframe only (one high,
            # one low - "only show the latest" rule from the MTF chart overlay).
            add(self._latest_swing_levels(tf), "swing_structure", tf)
            # v17: fast-confirming, low-weight "developing structure" pivot -
            # see _latest_provisional_swing for why this exists.
            add(self._latest_provisional_swing(tf), "provisional_structure", tf)

        # Multi-timeframe SWING CLUSTER zones - pools several recent confirmed
        # 5-bar pivots across the whole H1..D1 stack and keeps only the zones
        # where pivots agree. One raw row per contributing timeframe, so a
        # zone spanning H4+H1 naturally lifts both the method AND timeframe
        # confluence sub-scores.
        for side, tag in (("Support", "Low"), ("Resistance", "High")):
            for i, zone in enumerate(self._swing_cluster_zones(side)):
                for tf in zone["tfs"]:
                    raw.append({"level": zone["level"], "type": "swing_cluster",
                                "name": f"SwingCluster{tag}_{i + 1}", "tf": tf})

        for i, lvl in enumerate(self.round_numbers()):
            raw.append({"level": float(lvl), "type": "round_number",
                        "name": f"RN_{i + 1}", "tf": self.cfg.lowest_tf})
        return raw

    # v19: volume-derived source types whose raw clustering weight must carry
    # the same `volume_confidence` discount already applied downstream in
    # score_cluster (s["volume"], s["acceptance"], naked/HVN bonuses). Before
    # this fix, naked_poc (w_type=23) and volume_profile (w_type=22) - the
    # two heaviest weights in the entire w_type table, built entirely on MT5
    # tick_volume (a tick-count proxy, not real traded volume) - could pull a
    # cluster's weighted centroid harder than a confirmed horizontal level
    # (w_type=20) or swing_cluster zone (w_type=19), contradicting the
    # engine's own stated "tick-volume honesty" principle.
    _VOLUME_ITEM_TYPES = {"naked_poc", "volume_profile", "vwap", "vwap_band"}

    @staticmethod
    def _item_weight(cfg: "ApexConfig", s: dict) -> float:
        w_type = max(cfg.w_type.get(s["type"], 1), 1)
        if s["type"] in ApexScanner._VOLUME_ITEM_TYPES:
            w_type *= cfg.volume_confidence
        return (w_type *
                max(cfg.w_tf.get(s["tf"], 1), 1) *
                float(s.get("weight_mult", 1.0)))

    def cluster_levels(self, raw: List[dict]) -> List[dict]:
        """Deterministic gap+width clustering with weight-based centroids.

        v17 fix (chaining artifact): the old version compared each new point
        only to the PREVIOUS raw point in sorted order. That is greedy
        single-linkage and is vulnerable to chaining - point A and point C
        end up in the same cluster just because some unrelated point B
        happened to sit exactly between them, even though A and C individually
        may be `2 * tol` apart (bounded only by `cluster_max_width_mult`,
        which still allows up to 2x the intended tolerance to merge). This
        version instead compares each new point to the cluster's running
        WEIGHTED CENTROID (a closer analogue of proper density-based
        grouping), then runs a second convergence pass that merges any
        adjacent clusters whose final centroids ended up within tolerance -
        fixing boundary points that were mis-split by the first, order-
        dependent pass. Still O(n log n) and fully deterministic (depends
        only on sorted prices + weights, not visitation order).
        """
        if not raw:
            return []
        # v16 fix: scale the fixed cluster_pct tolerance by the current
        # volatility regime so grouping doesn't over-merge in compression or
        # over-split in expansion (shared helper with horizontal_levels).
        cluster_pct = self._cluster_tolerance_pct()

        def build_clusters(items: List[dict]) -> List[List[dict]]:
            items = sorted(items, key=lambda r: r["level"])
            clusters: List[List[dict]] = [[items[0]]]
            weights: List[List[float]] = [[self._item_weight(self.cfg, items[0])]]
            centroids = [items[0]["level"]]
            for item in items[1:]:
                w = self._item_weight(self.cfg, item)
                anchor = clusters[-1][0]["level"]
                centroid = centroids[-1]
                gap_tol = cluster_pct * item["level"]
                width_tol = self.cfg.cluster_max_width_mult * cluster_pct * anchor
                if (abs(item["level"] - centroid) <= gap_tol) and \
                   (item["level"] - anchor <= width_tol):
                    clusters[-1].append(item)
                    weights[-1].append(w)
                    wsum = sum(weights[-1])
                    centroids[-1] = sum(wi * ci["level"] for wi, ci in
                                        zip(weights[-1], clusters[-1])) / wsum
                else:
                    clusters.append([item])
                    weights.append([w])
                    centroids.append(item["level"])
            return clusters

        clusters = build_clusters(raw)

        # Convergence pass: merge adjacent clusters whose centroids (after
        # the first pass settled) are themselves within tolerance - catches
        # cases where the first, order-dependent pass split a genuine single
        # level across a cluster boundary.
        for _ in range(2):
            merged: List[List[dict]] = []
            changed = False
            for c in clusters:
                if not merged:
                    merged.append(c)
                    continue
                prev_center = self._weighted_center(merged[-1])
                this_center = self._weighted_center(c)
                tol = cluster_pct * this_center
                if abs(this_center - prev_center) <= tol:
                    merged[-1] = merged[-1] + c
                    changed = True
                else:
                    merged.append(c)
            clusters = merged
            if not changed:
                break

        out: List[dict] = []
        for c in clusters:
            center = self._weighted_center(c)
            out.append({"level": float(center), "sources": c})
        return out

    def _weighted_center(self, sources: List[dict]) -> float:
        # v16: weight_mult discounts pivot-family levels built on an
        # outlier-range anchor bar (see _get_anchor_ohlc) - default 1.0
        # for every other source type.
        weights = [self._item_weight(self.cfg, s) for s in sources]
        wsum = float(sum(weights)) or 1.0
        return sum(w * s["level"] for w, s in zip(weights, sources)) / wsum

    # ---------------- Volume profile POC ---------------- #
    def compute_poc(self) -> None:
        df = self.df_dict.get(self.cfg.lowest_tf)
        if df is None:
            return
        lows, highs = df["low"].to_numpy(), df["high"].to_numpy()
        vols = df["tick_volume"].to_numpy(dtype=float)
        # Fixed absolute bin grid (snapped to bin width) -> stable POC/VA across runs.
        bin_w = self.cfg.vp_bin_dollars or max(0.20 * (self._atr or 1.0), 0.10)
        lo = math.floor(lows.min() / bin_w) * bin_w
        hi = math.ceil(highs.max() / bin_w) * bin_w
        bins = np.arange(lo, hi + bin_w, bin_w)
        if len(bins) < 3:
            bins = np.linspace(lows.min(), highs.max(), 100)
        centers = (bins[:-1] + bins[1:]) / 2
        lo_idx = np.clip(np.searchsorted(bins, lows) - 1, 0, len(centers) - 1)
        hi_idx = np.clip(np.searchsorted(bins, highs) - 1, 0, len(centers) - 1)
        profile = np.zeros(len(centers))
        for lo, hi, v in zip(lo_idx, hi_idx, vols):
            profile[lo:hi + 1] += v / (hi - lo + 1)
        if profile.sum() <= 0:
            return
        self._vol_profile = (centers, profile)
        poc_i = int(np.argmax(profile))
        self._poc = float(centers[poc_i])
        # Value area: expand around the POC until 70% of volume is enclosed.
        total = profile.sum()
        lo = hi = poc_i
        acc = float(profile[poc_i])
        while acc < 0.70 * total and (lo > 0 or hi < len(centers) - 1):
            left = float(profile[lo - 1]) if lo > 0 else -1.0
            right = float(profile[hi + 1]) if hi < len(centers) - 1 else -1.0
            if right >= left:
                hi += 1; acc += max(right, 0.0)
            else:
                lo -= 1; acc += max(left, 0.0)
        self._vah = float(centers[hi])
        self._val = float(centers[lo])

    # ---------------- v14: 2026 calibration + node quality ---------------- #
    def _auto_scale_params(self) -> None:
        """Rescale fixed dollar constants to the live price/ATR regime.

        The original constants were tuned when gold was ~$3,400. With XAUUSD
        trading in the $4,000-4,700 band during 2026 (and a proportionally
        larger daily range), a fixed $30 scan window and $25 round grid become
        far too tight, which silently starves the confluence engine.
        """
        cfg = self.cfg
        if not cfg.auto_scale_params or self._scaled or not self.current_price:
            return
        k = max(0.5, min(3.0, float(self.current_price) / max(cfg.price_baseline, 1.0)))
        atr = float(self._atr or 0.0)

        # Scan range: price-scaled, but never smaller than ~6 ATR of headroom.
        cfg.max_range_dollars = round(max(cfg.max_range_dollars * k, 6.0 * atr), 1)
        # Round-number grid: keep it on a clean 5$ multiple of the scaled step.
        cfg.round_step = float(max(5.0, round(cfg.round_step * k / 5.0) * 5.0))
        cfg.round_tol_dollars = round(max(cfg.round_tol_dollars * k, 0.12 * atr), 2)
        cfg.spread_buffer_dollars = round(cfg.spread_buffer_dollars * k, 2)
        self._scaled = True
        log.info("auto-scaled to price %.0f (x%.2f): range $%.0f | grid $%.0f | ATR %.2f",
                 self.current_price, k, cfg.max_range_dollars, cfg.round_step, atr)

    def compute_vol_regime(self) -> None:
        """Classify volatility as expansion / normal / compression.

        2026 gold alternates between long compressions and violent expansions,
        so the same $ distance means different things week to week. The regime
        rescales proximity and stop-room expectations instead of using fixed $.
        """
        # NOTE: never use `or` on DataFrames - pandas raises on truth-testing.
        df = self.df_dict.get(self.cfg.atr_tf)
        if df is None:
            df = self.df_dict.get(self.cfg.lowest_tf)
        if df is None or len(df) == 0 or self._atr is None:
            return
        series = self._atr_series(df, self.cfg.atr_period).tail(self.cfg.vol_regime_lookback)
        med = float(series.median()) if len(series) else 0.0
        if med <= 0:
            return
        ratio = self._atr / med
        if ratio >= self.cfg.vol_expansion_mult:
            self._vol_regime = "expansion"
        elif ratio <= self.cfg.vol_compression_mult:
            self._vol_regime = "compression"
        else:
            self._vol_regime = "normal"

    def compute_naked_pocs(self) -> None:
        """Find prior-session POCs that price has never traded back to.

        A 'naked' (virgin) POC is the highest-volume price of a finished
        session that has not been revisited since. Untested POCs act as
        magnets and are among the highest-quality institutional levels, so
        they enter the engine as first-class sources.
        """
        self._naked_pocs = []
        if not self.cfg.use_naked_poc:
            return
        df = self.df_dict.get(self.cfg.lowest_tf)
        if df is None or len(df) == 0 or not self.current_price:
            return
        bin_w = self.cfg.vp_bin_dollars or max(0.20 * (self._atr or 1.0), 0.10)
        days = sorted(set(df.index.normalize()))[-(self.cfg.naked_poc_days + 1):]
        out: List[Tuple[float, int]] = []
        for i, day in enumerate(days[:-1]):          # skip today's unfinished session
            sess = df[df.index.normalize() == day]
            if len(sess) < 5:
                continue
            prices = sess["hlc3"].to_numpy(dtype=float)
            vols = sess["tick_volume"].to_numpy(dtype=float)
            if vols.sum() <= 0:
                continue
            grid = np.round(prices / bin_w) * bin_w
            uniq, inv = np.unique(grid, return_inverse=True)
            agg = np.bincount(inv, weights=vols)
            poc = float(uniq[int(np.argmax(agg))])
            # Untested = no later session traded through it.
            later = df[df.index.normalize() > day]
            if len(later) and (later["low"].min() <= poc <= later["high"].max()):
                continue
            out.append((poc, len(days) - 1 - i))
        out.sort(key=lambda t: abs(t[0] - self.current_price))
        self._naked_pocs = out[: self.cfg.naked_poc_max]

    def _node_acceptance(self, level: float) -> Tuple[float, str]:
        """Classify a level as HVN / neutral / LVN and score its acceptance.

        Volume-profile logic: price *holds* at high-volume nodes (accepted,
        heavy resting interest) and *slices through* low-volume nodes. An LVN
        is therefore a poor A+ reversal level no matter how much geometric
        confluence sits on it.
        """
        frac = self._volume_strength(level)
        if frac >= self.cfg.hvn_threshold:
            return min(frac, 1.0), "HVN"
        if frac <= self.cfg.lvn_threshold:
            return 0.0, "LVN"
        return frac, "neutral"

    def _volume_strength(self, level: float) -> float:
        """Volume at a price relative to the POC (0..1), from the profile."""
        if self._vol_profile is None:
            return 0.0
        centers, profile = self._vol_profile
        vmax = float(profile.max())
        if vmax <= 0:
            return 0.0
        i = int(np.argmin(np.abs(centers - level)))
        return float(profile[i] / vmax)

    def _touch_and_rejection(self, level: float, side: str) -> Dict[str, float]:
        """Recency-weighted touches + ATR-qualified rejections at a level."""
        out = {"touches": 0.0, "recency": 0.0, "rejections": 0.0}
        df = self.df_dict.get(self.cfg.lowest_tf)
        if df is None or not self._atr:
            return out
        h = df["high"].to_numpy(); l = df["low"].to_numpy()
        o = df["open"].to_numpy(); c = df["close"].to_numpy()
        n = len(h)
        tol = max(self.cfg.touch_tol_pct * level, 0.10 * self._atr)
        touched = (l <= level + tol) & (h >= level - tol)
        if not touched.any():
            return out
        ages = (n - 1) - np.arange(n)
        decay = 0.5 ** (ages / max(self.cfg.recency_halflife_bars, 1))
        body = np.abs(c - o)
        thresh = np.maximum(body, self.cfg.wick_atr_mult * self._atr)
        if side == "Support":
            wick = np.minimum(o, c) - l
            rej = touched & (l <= level) & (c > level) & (wick >= thresh)
        else:
            wick = h - np.maximum(o, c)
            rej = touched & (h >= level) & (c < level) & (wick >= thresh)
        out["touches"] = float(touched.sum())
        out["recency"] = float(decay[touched].sum())
        out["rejections"] = float(decay[rej].sum())
        return out

    def _polarity_check(self, level: float, side: str) -> Dict[str, object]:
        """Was this level historically respected mostly from the OPPOSITE
        side of the label it's about to get?

        v17 fix: `side` in `score_cluster` was previously assigned purely by
        `level > current_price` - a level has no persistent identity in the
        old code, so a price crossing a former resistance instantly relabels
        it "Support" with zero check on whether the market has ever actually
        rejected from that new side. This mirrors real S/R behaviour more
        closely (broken resistance CAN become support - "polarity flip" -
        but only once it has actually been retested and held from the new
        side; before that first hold it's an unconfirmed/untested flip and
        should be graded lower confidence than an already-proven level).
        """
        opposite = "Resistance" if side == "Support" else "Support"
        same = self._touch_and_rejection(level, side)
        opp = self._touch_and_rejection(level, opposite)
        is_untested_flip = (opp["rejections"] >= 1.0 and
                            opp["rejections"] >= 2.0 * same["rejections"] and
                            same["rejections"] < 1.0)
        return {"flip": is_untested_flip,
                "same_rejections": same["rejections"],
                "opposite_rejections": opp["rejections"]}

    def _round_number_tier(self, level: float) -> int:
        """Tiered psychological weight for gold: $100 > $50 > $25 > $10 > $5."""
        tol = self.cfg.round_tol_dollars
        for step, pts in ((100, 15), (50, 11), (25, 7), (10, 4), (5, 2)):
            if abs(level - round(level / step) * step) <= tol:
                return pts
        return 0

    # ---------------- 2026: multi-timeframe density + liquidity structure ---------------- #
    def _swing_pivots(self, df: pd.DataFrame,
                      right_override: Optional[int] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Confirmed fractal swing highs/lows -> (high_px, high_age_bars, low_px, low_age_bars).

        Arrays are chronological (oldest -> newest), so index [-1] is always
        the MOST RECENT confirmed pivot on that timeframe - the same "latest
        swing high / latest swing low per timeframe" idea used for the
        M3/M5/M15 chart overlay, generalised here to every scan timeframe.

        `right_override` lets a caller ask for a SHORTER right-side
        confirmation window than the standard `swing_left`/`swing_right`
        (see `_latest_provisional_swing`) - used for a low-weight,
        faster-confirming "provisional" pivot layer so the engine isn't
        structurally blind to a developing high/low for the full 5-bar wait.
        """
        h = df["high"].to_numpy(); l = df["low"].to_numpy()
        n = len(h); L = self.cfg.swing_left
        R = self.cfg.swing_right if right_override is None else right_override
        hp: List[float] = []; ha: List[int] = []; lp: List[float] = []; la: List[int] = []
        for i in range(L, n - R):
            wh = h[i - L:i + R + 1]; wl = l[i - L:i + R + 1]
            # Confirm on ties too (first bar reaching the window extreme), so
            # genuine equal-highs/equal-lows (double tops/bottoms) still
            # register as confirmed pivots instead of being dropped entirely
            # - the old strict-uniqueness check silently starved the
            # equal-high/low liquidity layer of the exact pattern it looks for.
            if h[i] == wh.max() and h[i] == h[i - L:i + 1].max():
                hp.append(float(h[i])); ha.append((n - 1) - i)
            if l[i] == wl.min() and l[i] == l[i - L:i + 1].min():
                lp.append(float(l[i])); la.append((n - 1) - i)
        return (np.array(hp), np.array(ha, dtype=float),
                np.array(lp), np.array(la, dtype=float))

    def _get_swings_by_tf(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Confirmed swing pivots for EVERY loaded timeframe, ages converted to
        MINUTES (not bars) so recency decay is directly comparable once pivots
        from different timeframes are pooled together in `_density_strength`
        and `_liquidity_structure`. Cached once per scan.
        """
        if self._swings_by_tf is None:
            out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
            for tf, df in self.df_dict.items():
                if len(df) <= self.cfg.swing_left + self.cfg.swing_right:
                    continue
                hp, ha_bars, lp, la_bars = self._swing_pivots(df)
                mins = _TF_MINUTES.get(tf, 15)
                out[tf] = (hp, ha_bars * mins, lp, la_bars * mins)
            self._swings_by_tf = out
        return self._swings_by_tf

    def _latest_provisional_swing(self, tf: str) -> Dict[str, float]:
        """v17: fast-confirming, LOW-WEIGHT swing high/low using a shorter
        right-side confirmation window (`provisional_swing_right`, default 2
        bars instead of the standard 5).

        The standard `swing_left`/`swing_right`=5/5 fractal only confirms a
        pivot 5 bars AFTER it forms - 5 hours on H1, 5 days on D1 - so the
        engine is structurally blind to the developing high/low of the
        current move for that whole window. This provisional layer trades
        confirmation strength for recency: it is tagged as a distinct,
        low-weight source (`provisional_structure` in `w_type`) so it can
        never outscore a fully-confirmed swing, but it stops the freshest
        structure from being completely invisible for days.
        """
        df = self.df_dict.get(tf)
        if df is None or len(df) <= self.cfg.swing_left + self.cfg.provisional_swing_right:
            return {}
        hp, _ha, lp, _la = self._swing_pivots(
            df, right_override=self.cfg.provisional_swing_right)
        out: Dict[str, float] = {}
        if hp.size:
            out[f"ProvisionalHigh_{tf}"] = float(hp[-1])
        if lp.size:
            out[f"ProvisionalLow_{tf}"] = float(lp[-1])
        return out

    def _latest_swing_levels(self, tf: str) -> Dict[str, float]:
        """Latest confirmed swing high + swing low for ONE timeframe - the
        direct analogue of the MQL5 overlay's 'only show the latest label per
        timeframe' rule. Fed into `collect_raw_levels` as a first-class
        `swing_structure` method so confirmed market structure participates in
        method/timeframe confluence scoring like every other level type.
        """
        empty = np.array([])
        hp, _ha, lp, _la = self._get_swings_by_tf().get(tf, (empty, empty, empty, empty))
        out: Dict[str, float] = {}
        if hp.size:
            out[f"SwingHigh_{tf}"] = float(hp[-1])
        if lp.size:
            out[f"SwingLow_{tf}"] = float(lp[-1])
        return out

    def _halflife_minutes(self, tf: Optional[str] = None) -> float:
        """Recency half-life in minutes, scaled to the PIVOT'S OWN timeframe.

        v17 fix: this used to be a single constant derived from `lowest_tf`
        (H1) and applied uniformly to every pooled pivot regardless of which
        timeframe it came from. That silently contradicted `w_tf` (which
        weights D1/W1 pivots as structurally stronger/more durable): a D1
        pivot 40 days old decayed to ~11% and a W1 pivot 90 days old decayed
        to <1% under the old H1-tuned half-life, i.e. exactly the levels the
        engine claims are "more durable" were the ones getting crushed
        fastest. Half-life must scale with the pivot's own bar duration so a
        D1 swing gets a D1-scale memory and a W1 swing gets a W1-scale
        memory, consistent with `w_tf`.
        """
        ref_tf = tf or self.cfg.lowest_tf
        base = _TF_MINUTES.get(ref_tf, 60)
        return max(self.cfg.recency_halflife_bars * base, 1.0)

    def _bars_for_minutes(self, tf: str, minutes: float) -> int:
        """How many `tf` bars correspond to a duration in minutes - keeps the
        liquidity-sweep lookback window equal in real TIME across timeframes
        instead of equal in bar count (60 M15 bars != 60 D1 bars)."""
        m = _TF_MINUTES.get(tf, 15)
        return max(1, int(round(minutes / m)))

    def _density_strength(self, level: float, side: str) -> float:
        """Gaussian-kernel (KDE) density of confirmed swing pivots at a price (0..1).

        2026 research favours density-based methods (DBSCAN / KDE) over raw pivot
        counts: S/R is strongest where turning points *concentrate*, with recent
        pivots weighted more heavily. Pivots are pooled across ALL timeframes
        (not just the lowest one) - an H4/D1 pivot cluster is real structure and
        is weighted accordingly via `w_tf`, mirroring the multi-timeframe swing
        structure concept (each timeframe contributes its own confirmed swing
        highs/lows onto the same price axis).
        """
        if not self._atr:
            return 0.0
        total = 0.0
        for tf, (hp, ha_min, lp, la_min) in self._get_swings_by_tf().items():
            prices, ages_min = (lp, la_min) if side == "Support" else (hp, ha_min)
            if prices.size == 0:
                continue
            # v16 fix: bandwidth scaled to THIS pivot's own timeframe ATR, not
            # a single H1 ATR for every timeframe - a D1 pivot cluster has a
            # materially different natural "spread" than an H1 one, so a
            # shared H1-scaled bandwidth systematically under-counted D1
            # density and over-counted H1 density.
            bw = max(self.cfg.kde_bw_atr_mult * self._atr_for_tf(tf), 1e-6)
            tf_w = self.cfg.w_tf.get(tf, 10) / 30.0   # D1 (30) -> 1.0 baseline
            # v17 fix: half-life is now scaled to THIS pivot's own timeframe
            # (see `_halflife_minutes`), not a single H1-tuned constant, so
            # HTF pivots keep real structural memory instead of decaying at
            # the same rate as an H1 wiggle.
            halflife = self._halflife_minutes(tf)
            kernel = np.exp(-0.5 * ((prices - level) / bw) ** 2)
            decay = 0.5 ** (ages_min / halflife)
            total += tf_w * float((kernel * decay).sum())
        return min(total / 3.0, 1.0)   # ~3 recent, HTF-weighted, aligned pivots -> saturates

    def _liquidity_structure(self, level: float, side: str) -> Dict[str, float]:
        """Equal-highs/lows liquidity pools + sweep-and-reclaim near a level.

        2026 SMC: equal highs/lows are resting-liquidity magnets (stop clusters);
        a sweep that reclaims (false breakout) is a high-probability reversal.
        Pooled across ALL timeframes (weighted by `w_tf`) - an equal-highs pool
        on H4 is a materially stronger magnet than one on M15, and the sweep
        lookback window is held equal in real TIME (not bar count) across
        timeframes so H4/D1 aren't scanning an unrealistically short window.
        """
        out = {"equal": 0.0, "sweep": 0.0}
        if not self._atr:
            return out
        sweep_lookback_minutes = self.cfg.sweep_lookback_bars * _TF_MINUTES.get(self.cfg.lowest_tf, 15)
        equal_score = 0.0
        swept_weight = 0.0

        for tf, (hp, _ha, lp, _la) in self._get_swings_by_tf().items():
            prices = lp if side == "Support" else hp
            df = self.df_dict.get(tf)
            if df is None or prices.size == 0:
                continue
            tf_w = self.cfg.w_tf.get(tf, 10) / 30.0
            # v16 fix: tolerance from THIS timeframe's own ATR, not a shared H1
            # ATR - an "equal lows" pool on D1 needs a daily-scale tolerance to
            # be detected correctly, not an hourly one.
            tol = self.cfg.eqh_eql_tol_atr_mult * self._atr_for_tf(tf)

            near = prices[np.abs(prices - level) <= tol]
            equal_score += tf_w * min(max(near.size - 1, 0) / 2.0, 1.0)

            h = df["high"].to_numpy(); l = df["low"].to_numpy(); c = df["close"].to_numpy()
            w = self._bars_for_minutes(tf, sweep_lookback_minutes)
            hs, ls, cs = h[-w:], l[-w:], c[-w:]
            if side == "Support":
                swept = (ls <= level - tol) & (cs > level)
            else:
                swept = (hs >= level + tol) & (cs < level)
            if bool(swept.any()):
                swept_weight = max(swept_weight, tf_w)

        out["equal"] = min(equal_score, 1.0)
        out["sweep"] = 1.0 if swept_weight > 0 else 0.0
        return out

    def _swing_cluster_zones(self, side: str) -> List[Dict]:
        """Pool recent confirmed 5-bar swing pivots from EVERY timeframe into
        weighted price zones, keeping only zones where >=2 pivots agree.

        This is the "cluster swing" layer: every configured timeframe is H1
        or higher, so pooling here already spans the full MTF stack. Unlike
        `_latest_swing_levels` (exactly one high + one low per timeframe, no
        grouping), this looks at the last `swing_cluster_lookback` confirmed
        pivots per timeframe/side, weights each by timeframe importance
        (`w_tf`) and real-minute recency decay, then groups pivots sitting
        within `swing_cluster_tol_atr_mult` * ATR of one another. A zone only
        survives if its combined weight clears `swing_cluster_min_weight` -
        e.g. an H4 swing low sitting on top of two H1 swing lows is a real,
        repeated, multi-timeframe structure point; a single isolated H1
        pivot is not.
        """
        if not self._atr:
            return []
        # v16 fix: group on the HIGHEST-timeframe ATR available (D1 preferred),
        # not H1 ATR. These zones exist specifically to pool H1..D1 pivots into
        # one daily-scale structural zone; an H1-scaled tolerance was too tight
        # and silently under-grouped genuine multi-timeframe clusters.
        tol = max(self.cfg.swing_cluster_tol_atr_mult * self._reference_htf_atr(), 1e-6)

        points: List[Tuple[float, float, str]] = []   # (price, weight, tf)
        for tf, (hp, ha_min, lp, la_min) in self._get_swings_by_tf().items():
            prices, ages = (lp, la_min) if side == "Support" else (hp, ha_min)
            if prices.size == 0:
                continue
            prices = prices[-self.cfg.swing_cluster_lookback:]
            ages = ages[-self.cfg.swing_cluster_lookback:]
            tf_w = self.cfg.w_tf.get(tf, 10) / 30.0   # D1 (30) -> 1.0 baseline
            # v17 fix: per-timeframe half-life (see `_halflife_minutes`).
            halflife = self._halflife_minutes(tf)
            decay = 0.5 ** (ages / halflife)
            points.extend((float(p), tf_w * float(d), tf) for p, d in zip(prices, decay))

        if not points:
            return []

        points.sort(key=lambda t: t[0])
        groups: List[List[Tuple[float, float, str]]] = [[points[0]]]
        for p in points[1:]:
            if p[0] - groups[-1][-1][0] <= tol:
                groups[-1].append(p)
            else:
                groups.append([p])

        zones: List[Dict] = []
        for g in groups:
            if len(g) < 2:
                continue   # a "cluster" needs at least 2 agreeing pivots
            wsum = sum(w for _, w, _ in g)
            if wsum < self.cfg.swing_cluster_min_weight:
                continue
            center = sum(p * w for p, w, _ in g) / wsum
            zones.append({"level": center, "weight": wsum,
                          "tfs": sorted({tf for _, _, tf in g}), "n": len(g)})

        zones.sort(key=lambda z: -z["weight"])
        return zones[: self.cfg.swing_cluster_top_n]

    # ---------------- Reasons (v13: evidence-weighted) ---------------- #
    def _killzone_now(self) -> Optional[str]:
        """Return the active ICT killzone name (UTC), if any."""
        now = datetime.now(timezone.utc).time()
        for name, start, end in self.cfg.killzones_utc:
            try:
                t0 = dt_time(*map(int, start.split(":")))
                t1 = dt_time(*map(int, end.split(":")))
            except Exception:
                continue
            if t0 <= now <= t1:
                return name
        return None

    def build_reasons(
        self,
        level: float,
        sources: List[dict],
        dist_dollars: float,
        side: Optional[str] = None,
        ctx: Optional[Dict[str, object]] = None,
    ) -> List[Reason]:
        """Build INDEPENDENT, evidence-weighted reasons supporting a level.

        Each reason gets a 0..1 weight reflecting how hard its evidence is.
        Reasons at/above `cfg.strong_reason_min_weight` are flagged `strong`,
        and A+ requires at least `cfg.min_strong_reasons_aplus` (default 11).
        `ctx` lets `score_cluster` pass already-computed analytics so nothing
        is recomputed (touch/rejection, KDE density, liquidity, plan, DXY).
        """
        cfg = self.cfg
        ctx = dict(ctx or {})
        side = side or ("Resistance" if level > self.current_price else "Support")
        out: List[Reason] = []
        seen: set = set()

        def add(key: str, text: str, weight: float, category: str) -> None:
            if key in seen:
                return
            w = max(0.0, min(1.0, float(weight)))
            if w <= 0.0:
                return
            seen.add(key)
            out.append(Reason(key=key, text=text, weight=round(w, 3),
                              category=category,
                              strong=w >= cfg.strong_reason_min_weight))

        tfs = {s["tf"] for s in sources}
        types = {s["type"] for s in sources}
        low_tf = cfg.lowest_tf
        df_low = self.df_dict.get(low_tf)
        atr = float(self._atr or 0.0)
        near_tol = max(0.0005 * float(self.current_price or 0.0), 0.25 * atr)

        polarity = ctx.get("polarity") or self._polarity_check(level, side)
        tr = ctx.get("touch") or self._touch_and_rejection(level, side)
        liq = ctx.get("liquidity") or self._liquidity_structure(level, side)
        dens = float(ctx.get("density", self._density_strength(level, side)))
        vol_frac = float(ctx.get("volume", self._volume_strength(level)))
        round_pts = int(ctx.get("round_pts", self._round_number_tier(level)))
        plan = ctx.get("plan")
        dxy_note = str(ctx.get("dxy_note", ""))

        # --- 1. Structure: multi-timeframe confluence (HTF weighted) ---
        tf_raw = sum(cfg.w_tf.get(tf, 0) for tf in tfs)
        if len(tfs) >= 2:
            tf_list = ", ".join(sorted(tfs, key=lambda t: -cfg.w_tf.get(t, 0)))
            add("mtf", f"Confirmed on {len(tfs)} timeframes ({tf_list})",
                min(tf_raw / 70.0, 1.0), "structure")

        # --- 2. Structure: higher-timeframe anchor ---
        htf_tfs = sorted({s["tf"] for s in sources if s["tf"] in ("H4", "D1", "W1")})
        if htf_tfs:
            add("htf", f"Higher-timeframe anchor ({', '.join(htf_tfs)})",
                0.55 + 0.15 * len(htf_tfs), "structure")

        # --- 3. Structure: independent method confluence ---
        # v16 fix: count distinct correlated-evidence GROUPS, not raw method
        # types. Fibonacci/MA/Bollinger/Murrey are all deterministic
        # transforms of the same recent OHLC window, so "4 independent
        # methods" was often really "1 source of evidence, described 4 ways".
        groups = self._method_groups(types)
        n_groups = len(groups)
        # v17: fractional independence weighting (see _GROUP_INDEPENDENCE in
        # score_cluster) - keeps the reason text's implied confidence
        # consistent with the actual score contribution instead of
        # saturating near 1.0 on near-ubiquitous groups (price_math/round).
        n_groups_eff = sum(self._GROUP_INDEPENDENCE.get(g, 1.0) for g in groups)
        if n_groups >= 2:
            group_list = ", ".join(sorted(groups))
            add("methods", f"Confirmed by {n_groups} independent evidence groups ({group_list})",
                1.0 - math.exp(-0.6 * n_groups_eff), "structure")
        q_raw = self._group_weighted_quality(types)
        if q_raw:
            add("method_quality",
                f"High-grade method mix ({', '.join(sorted(types)[:4])})",
                min(q_raw / 55.0, 1.0), "structure")

        # --- v18. Structure: durable HTF market-structure map ---
        major_src = [s for s in sources if s["type"] == "major_structure"]
        if major_src:
            major_tfs = sorted({s["tf"] for s in major_src}, key=lambda t: -cfg.w_tf.get(t, 0))
            labels: List[str] = []
            for src in major_src:
                for txt in src.get("major_reasons", []):
                    if txt not in labels:
                        labels.append(txt)
            w = min(1.0, 0.72 + 0.08 * max(0, len(major_tfs)-1) + 0.04 * max(0, len(labels)-1))
            detail = "; ".join(labels[:3]) if labels else "HTF structural zone"
            add("major_structure", f"Major HTF S/R: {detail}", w, "structure")

        # --- 4. Structure: tested horizontal ---
        if "horizontal" in types and df_low is not None:
            touches = self.count_touches(df_low, level)
            if touches >= 3:
                add("horizontal",
                    f"Tested horizontal level ({touches} touches, {low_tf})",
                    min(touches / 8.0, 1.0), "structure")

        # --- 5. Structure: recency-weighted touch memory ---
        rec = float(tr.get("recency", 0.0))
        if rec > 0:
            add("touch_memory",
                f"Live touch memory (recency-weighted {rec:.1f})",
                min(math.log1p(rec) / math.log1p(12), 1.0), "structure")

        # --- 6. Momentum: ATR-qualified rejections ---
        rej_n = float(tr.get("rejections", 0.0))
        if rej_n > 0:
            add("rejections",
                f"Repeated ATR-qualified {side.lower()} rejections ({rej_n:.1f})",
                min(rej_n / 3.0, 1.0), "momentum")

        # --- 7. Momentum: fresh pinbar at the level ---
        if df_low is not None:
            pin = self._pinbar(df_low, level)
            if pin:
                add("pinbar", f"{pin} on last {low_tf} candles", 0.65, "momentum")

        # --- 8. Structure: pivot family (fixed-anchor, tiered by line + anchor) ---
        # v15: score the ACTUAL pivot line(s) present instead of a flat bonus
        # for "any pivot type at all". The central PP is respected far more
        # often than an outer R3/S3 extension, and a level that lines up on
        # BOTH the daily and weekly anchor (e.g. D1 PP sitting on W1 S1) is
        # meaningfully stronger evidence than a single-anchor hit, so that
        # earns an explicit multi-anchor bonus on top of the line tier.
        pivot_src = [s for s in sources if s["type"] == "pivot"]
        if pivot_src:
            names = {str(s["name"]) for s in pivot_src}
            anchors = sorted({s["tf"] for s in pivot_src}, key=lambda t: -cfg.w_tf.get(t, 0))
            tier_w = max((_PIVOT_TIER_WEIGHT.get(n, 0.5) for n in names), default=0.5)
            multi_anchor_bonus = 0.15 if len(anchors) >= 2 else 0.0
            # v16: discount when the anchor day/week was an outlier-range bar.
            anchor_q = min((float(s.get("weight_mult", 1.0)) for s in pivot_src), default=1.0)
            w = min((tier_w + multi_anchor_bonus) * anchor_q, 1.0)
            q_note = "" if anchor_q >= 0.99 else f" [outlier-range anchor x{anchor_q:.2f}]"
            add("pivot",
                f"Floor-pivot alignment [{', '.join(sorted(names))}] "
                f"({'/'.join(anchors)}-anchored){q_note}", w, "structure")

        camarilla_src = [s for s in sources if s["type"] == "camarilla"]
        if camarilla_src:
            names = {str(s["name"]) for s in camarilla_src}
            anchors = sorted({s["tf"] for s in camarilla_src}, key=lambda t: -cfg.w_tf.get(t, 0))
            tier_w = max((_CAMARILLA_TIER_WEIGHT.get(n, 0.35) for n in names), default=0.35)
            multi_anchor_bonus = 0.10 if len(anchors) >= 2 else 0.0
            anchor_q = min((float(s.get("weight_mult", 1.0)) for s in camarilla_src), default=1.0)
            w = min((tier_w + multi_anchor_bonus) * anchor_q, 1.0)
            q_note = "" if anchor_q >= 0.99 else f" [outlier-range anchor x{anchor_q:.2f}]"
            add("camarilla",
                f"Camarilla level alignment [{', '.join(sorted(names))}] "
                f"({'/'.join(anchors)}-anchored){q_note}", w, "structure")

        demark_src = [s for s in sources if s["type"] == "demark"]
        if demark_src:
            anchor_q = min((float(s.get("weight_mult", 1.0)) for s in demark_src), default=1.0)
            q_note = "" if anchor_q >= 0.99 else f" [outlier-range anchor x{anchor_q:.2f}]"
            add("demark", f"DeMark projection alignment{q_note}", 0.45 * anchor_q, "structure")
        if "murrey_math" in types:
            add("murrey", "Murrey Math octave line", 0.45, "structure")

        # --- 9. Structure: fibonacci (golden pocket weighted higher) ---
        fib_src = [s for s in sources if s["type"] == "fibonacci"]
        if fib_src:
            fib_names = {str(s["name"]) for s in fib_src}
            golden = any(k in n for n in fib_names for k in ("0.618", "0.5", "0.786"))
            add("fib",
                f"Fibonacci confluence ({len(fib_src)} TFs: {', '.join(sorted(fib_names)[:3])})",
                (0.55 if len(fib_src) >= 2 else 0.35) + (0.20 if golden else 0.0),
                "structure")

        # --- 10. Structure: dynamic MA/EMA + bands ---
        mas = [str(s["name"]) for s in sources if s["type"] == "moving_average"]
        if mas:
            add("ma", f"Dynamic MA/EMA confluence ({', '.join(mas[:3])})",
                min(0.30 + 0.15 * len(mas), 0.90), "structure")
        if "bollinger" in types:
            add("bollinger", "Bollinger band extreme touch", 0.35, "momentum")

        # --- 11. Volume: session VWAP ---
        if "vwap" in types:
            add("vwap", "Session VWAP alignment (institutional average price)",
                0.60, "volume")

        # --- 12. Structure: confirmed swing pivots + MTF swing clusters ---
        swing_tfs = sorted({s["tf"] for s in sources if s["type"] == "swing_structure"})
        if swing_tfs:
            add("swing_structure",
                f"Confirmed swing-structure pivot ({', '.join(swing_tfs)})",
                min(0.45 + 0.15 * len(swing_tfs), 1.0), "structure")
        cluster_tfs = sorted({s["tf"] for s in sources if s["type"] == "swing_cluster"})
        if cluster_tfs:
            add("swing_cluster",
                f"Multi-timeframe swing-cluster zone ({', '.join(cluster_tfs)})",
                min(0.50 + 0.20 * len(cluster_tfs), 1.0), "structure")

        # --- 13. Structure: KDE pivot density ---
        if dens > 0:
            add("density", f"Dense swing-pivot cluster (KDE {dens * 100:.0f}%)",
                dens, "structure")

        # --- 14. Volume: acceptance / POC / value area ---
        if vol_frac > 0:
            add("volume", f"Volume acceptance ({vol_frac * 100:.0f}% of POC node)",
                vol_frac, "volume")
        if self._poc is not None and abs(level - self._poc) <= near_tol:
            add("poc", "Sitting on the high-volume node (POC)", 0.80, "volume")

        # --- v14: naked / virgin POC (untested prior-session control price) ---
        naked = ctx.get("naked_hit")
        if naked:
            _, age = naked  # type: ignore[misc]
            add("naked_poc",
                f"Naked (untested) POC from {age} session(s) ago - price magnet",
                min(0.80 + 0.02 * min(int(age), 5), 0.95), "volume")

        # --- v14: HVN acceptance vs LVN void ---
        node = str(ctx.get("node", ""))
        if node == "HVN":
            add("hvn", "High-volume node: price is accepted and stalls here",
                0.60 + 0.30 * float(ctx.get("acceptance", 0.0)), "volume")
        elif node == "LVN":
            add("lvn_warn",
                "Low-volume node: thin liquidity, price tends to slice through",
                0.20, "risk")

        # --- v14: VWAP sigma band (institutional mean-reversion) ---
        band_names = sorted({s["name"] for s in sources if s["type"] == "vwap_band"})
        if band_names:
            two_sigma = any("2" in n for n in band_names)
            add("vwap_band",
                f"VWAP deviation band ({', '.join(band_names)}) - statistically stretched",
                0.72 if two_sigma else 0.62, "volume")

        # --- v14: volatility regime fit ---
        if atr > 0:
            atr_dist = dist_dollars / atr
            regime = self._vol_regime
            if regime == "expansion" and atr_dist >= 0.8:
                add("regime", "Volatility expansion and level sits beyond one ATR of noise",
                    0.62, "context")
            elif regime == "compression" and atr_dist <= 2.0:
                add("regime", "Volatility compression and level is within reach this session",
                    0.60, "context")
            elif regime == "normal" and 0.5 <= atr_dist <= 3.0:
                add("regime", "Normal volatility with the level in the tradeable ATR band",
                    0.55, "context")
        for nm, lvl in (("VAH", self._vah), ("VAL", self._val)):
            if lvl and abs(level - float(lvl)) <= near_tol:
                add(f"va_{nm}", f"Value-area edge ({nm})", 0.65, "volume")

        # --- 15. Liquidity (SMC): equal highs/lows + sweep-reclaim ---
        if float(liq.get("equal", 0.0)) > 0:
            pool = "highs" if side == "Resistance" else "lows"
            add("eqhl", f"Equal {pool} liquidity pool above/below the level",
                min(float(liq["equal"]), 1.0), "liquidity")
        if float(liq.get("sweep", 0.0)) > 0:
            add("sweep", "Recent liquidity sweep + reclaim (stop-run absorbed)",
                min(float(liq["sweep"]), 1.0), "liquidity")

        # --- 16. Context: psychological round number ---
        if round_pts > 0:
            add("round", f"Psychological round number (tier {round_pts}/15)",
                min(round_pts / 10.0, 1.0), "context")

        # --- 17. Risk: asymmetric R:R and stop placement ---
        if plan is not None:
            rr = float(getattr(plan, "rr", 0.0))
            if rr > 0:
                add("rr", f"Asymmetric R:R {rr:.1f} to structure target",
                    min(rr / (2.0 * max(cfg.min_rr, 0.1)), 1.0), "risk")
            if atr > 0:
                stop_dist = abs(float(plan.entry) - float(plan.stop))
                add("stop_room",
                    f"Stop {stop_dist / atr:.1f}x ATR beyond the level (noise-proof)",
                    min(stop_dist / atr / 2.0, 1.0), "risk")

        # --- 18. Context: DXY intermarket alignment ---
        if dxy_note == "DXY aligned":
            add("dxy", f"DXY intermarket alignment ({self._dxy_bias})", 0.75, "context")

        # --- 19. Context: HTF trend agreement ---
        df_d1 = self.df_dict.get("D1")
        if df_d1 is not None and len(df_d1) >= 50:
            sma50 = float(df_d1["close"].tail(50).mean())
            up = float(df_d1["close"].iloc[-1]) > sma50
            if (side == "Support" and up) or (side == "Resistance" and not up):
                add("trend",
                    f"Trades with the D1 trend ({'bullish' if up else 'bearish'} vs SMA50)",
                    0.70, "context")

        # --- 20. Context: tradeable proximity ---
        if dist_dollars <= 5.0:
            add("proximity", f"Actionable proximity ({dist_dollars:.1f}$ away)",
                1.0 - (dist_dollars / 5.0) * 0.5, "context")

        # --- 21. Context: killzone timing + clean news window ---
        kz = self._killzone_now()
        if kz:
            add("killzone", f"Inside the {kz} killzone (peak liquidity)", 0.65, "context")
        if not self.in_news_window():
            add("news_clear", "No high-impact news window active", 0.40, "context")

        out.sort(key=lambda r: -r.weight)
        return out

    def _strong_count_deduped(self, reason_objs: List[Reason]) -> int:
        """v19: group-aware strong-reason count for the A+/A gate.

        Reasons in the same `_REASON_CORR_GROUP` (see module-level comment)
        contribute at most one strong count, taken from their best weight in
        that group - mirroring how `_METHOD_GROUP`/`_GROUP_INDEPENDENCE`
        already prevent the numeric score from double-counting correlated
        method evidence. Ungrouped reasons (touches, rejections, volume,
        liquidity, round number, R:R, DXY, trend, killzone, ...) still count
        individually, since those genuinely are separate observations.
        """
        best_per_group: Dict[str, float] = {}
        standalone = 0
        for r in reason_objs:
            g = _REASON_CORR_GROUP.get(r.key)
            if g is None:
                if r.strong:
                    standalone += 1
            else:
                best_per_group[g] = max(best_per_group.get(g, 0.0), r.weight)
        strong_groups = sum(1 for wgt in best_per_group.values()
                            if wgt >= self.cfg.strong_reason_min_weight)
        return standalone + strong_groups

    def _pinbar(self, df: pd.DataFrame, level: float) -> Optional[str]:
        near = 0.0005
        for _, candle in df.tail(5).iterrows():
            body = abs(candle["close"] - candle["open"])
            up = candle["high"] - max(candle["close"], candle["open"])
            dn = min(candle["close"], candle["open"]) - candle["low"]
            if level < self.current_price and dn > body * 2 and abs(candle["low"] - level) / level <= near:
                return "Bullish pinbar rejection"
            if level > self.current_price and up > body * 2 and abs(candle["high"] - level) / level <= near:
                return "Bearish pinbar rejection"
        return None

    @staticmethod
    def count_touches(df: pd.DataFrame, level: float) -> int:
        return int(((df["low"] <= level) & (df["high"] >= level)).sum())

    @staticmethod
    def _method_groups(types: set) -> set:
        """Map raw method `type`s to their correlated-evidence group (v16 fix
        for the method-confluence double-counting problem - see _METHOD_GROUP)."""
        return {_METHOD_GROUP.get(t, t) for t in types}

    def _group_weighted_quality(self, types: set) -> float:
        """Sum the BEST (highest-weight) type per correlated method-group
        instead of summing every raw type's weight. Stops a level that only
        has Fib + MA + Bollinger + Murrey (all price-math on one timeframe -
        one real source of evidence) from scoring as if it had four
        independent confirmations."""
        best_per_group: Dict[str, float] = {}
        for t in types:
            g = _METHOD_GROUP.get(t, t)
            w = self.cfg.w_type.get(t, 0)
            # v17: the "volume" group (POC/VAH/VAL/naked-POC/VWAP) is built on
            # MT5 tick-volume, not real traded volume - discount its
            # contribution to method QUALITY the same way `volume_confidence`
            # discounts the volume/acceptance sub-scores, so tick-count noise
            # can't buy a level "high-quality confluence" credit on its own.
            if g == "volume":
                w *= self.cfg.volume_confidence
            best_per_group[g] = max(best_per_group.get(g, 0), w)
        return float(sum(best_per_group.values()))

    # ---------------- Risk / trade plan ---------------- #
    def _pip_value_per_lot(self) -> float:
        """Approx. $ move per 1.00 price move per 1 lot for the symbol."""
        if self._sym_info is not None:
            contract = getattr(self._sym_info, "trade_contract_size", 100.0)
            return float(contract)      # gold: 100 oz/lot -> $100 per $1 move
        return 100.0

    def _clamp_lots(self, lots: float) -> float:
        """Round DOWN to the broker's volume step and clamp to min/max.

        Rounding down keeps realised risk <= the configured risk budget.
        Returns 0.0 when the risk-based size is below the broker minimum.
        """
        vmin = float(getattr(self._sym_info, "volume_min", 0.01) or 0.01)
        vmax = float(getattr(self._sym_info, "volume_max", 100.0) or 100.0)
        vstep = float(getattr(self._sym_info, "volume_step", 0.01) or 0.01)
        if vstep <= 0:
            vstep = 0.01
        stepped = round(math.floor(lots / vstep + 1e-9) * vstep, 8)
        if stepped < vmin:
            return 0.0
        return min(stepped, vmax)

    def _next_structure_target(self, entry: float, direction: str,
                               all_levels: List[float]) -> Optional[float]:
        """Nearest opposing clustered level far enough to be a real target."""
        if not all_levels:
            return None
        min_dist = self.cfg.min_target_atr_mult * (self._atr or 0.0)
        if direction == "BUY":
            candidates = [lv for lv in all_levels if lv - entry >= min_dist]
            return min(candidates) if candidates else None
        candidates = [lv for lv in all_levels if entry - lv >= min_dist]
        return max(candidates) if candidates else None

    def build_trade_plan(self, level: float, side: str,
                         all_levels: Optional[List[float]] = None) -> Optional[TradePlan]:
        if self._atr is None or self._atr <= 0:
            return None
        # ATR-based stop plus a fixed spread/commission buffer, so realised
        # risk on a wide-spread gold feed isn't understated at low
        # sl_atr_mult (retail XAUUSD spreads are often $0.20-$0.50+).
        sl_dist = self.cfg.sl_atr_mult * self._atr + self.cfg.spread_buffer_dollars
        if sl_dist <= 0:
            return None

        if side == "Support":
            direction, entry = "BUY", level
            stop = entry - sl_dist
        else:
            direction, entry = "SELL", level
            stop = entry + sl_dist

        # Target the next opposing structure; fall back to an ATR multiple.
        target = self._next_structure_target(entry, direction, all_levels or [])
        if target is None:
            tp_dist = self.cfg.tp_atr_mult * self._atr
            target = entry + tp_dist if direction == "BUY" else entry - tp_dist

        tp_dist = abs(target - entry)
        rr = tp_dist / sl_dist if sl_dist else 0.0

        risk_dollars = self._account_balance * (self.cfg.account_risk_pct / 100.0)
        per_lot_risk = sl_dist * self._pip_value_per_lot()
        raw_lots = risk_dollars / per_lot_risk if per_lot_risk > 0 else 0.0
        lots = self._clamp_lots(raw_lots)
        return TradePlan(direction, entry, stop, target, rr, lots, risk_dollars)

    # ---------------- Scoring + grading ---------------- #
    # v9: structural strength weights (0..1 sub-scores -> weighted mean = core).
    # Bonuses (round / proximity / DXY) are added on top so a genuinely strong,
    # multi-timeframe, well-tested level can reach the 8+/10 (A+) band.
    _CORE_W = {
        "timeframe": 0.18, "method": 0.14, "quality": 0.09, "touch": 0.14,
        "volume": 0.11, "density": 0.12, "liquidity": 0.07, "rejection": 0.08,
        # v14: volume-profile ACCEPTANCE (HVN holds / LVN gets sliced through).
        "acceptance": 0.07,
    }

    # v17: fractional "independence" credit per method-group for the method-
    # confluence saturation fix above. 1.0 = fully independent evidence
    # source; lower values reflect groups that are almost always present
    # near any price (continuous price-math functions, dense round-number
    # grid) and therefore weak evidence of a genuine, distinct level on
    # their own.
    _GROUP_INDEPENDENCE = {
        "price_math": 0.5, "round_number": 0.4, "anchor_pivot": 0.8,
        "volume": 0.9, "price_structure": 1.0,
    }

    def score_cluster(self, cluster: dict,
                      all_levels: Optional[List[float]] = None) -> Optional[dict]:
        level, sources = cluster["level"], cluster["sources"]
        dist_dollars = abs(level - self.current_price)
        has_major_structure = any(s.get("type") == "major_structure" for s in sources)
        range_limit = self.cfg.max_range_dollars
        if has_major_structure:
            # Daily/weekly structural S/R can legitimately sit farther away
            # than an H1 execution window. Keep those levels visible within a
            # bounded HTF-ATR envelope instead of deleting them before score.
            range_limit = max(range_limit,
                              self.cfg.major_range_htf_atr_mult * self._reference_htf_atr())
        if dist_dollars > range_limit:
            return None

        tfs = {s["tf"] for s in sources}
        types = {s["type"] for s in sources}
        groups = self._method_groups(types)   # v16: correlated methods deduped
        side = "Resistance" if level > self.current_price else "Support"

        # ---- 0..1 structural strength sub-scores (each independently bounded) ----
        s: Dict[str, float] = {}

        # Timeframe confluence - HTF weighted, saturating.
        tf_raw = sum(self.cfg.w_tf.get(tf, 0) for tf in tfs)
        s["timeframe"] = min(tf_raw / 70.0, 1.0)

        # Method confluence - diminishing returns (sweet spot ~3-4 METHOD
        # GROUPS). v16 fix: counts distinct correlated-evidence groups, not
        # raw types - Fib+MA+Bollinger+Murrey together used to count as 4
        # "independent" methods when they're one correlated price-math family.
        #
        # v17 fix: even after grouping, not all groups are equally
        # informative. `price_math` (Fib/MA/Bollinger/Murrey) is a continuous
        # function of recent price and will sit near ALMOST ANY price to
        # within the cluster tolerance - so its mere presence is weak
        # evidence. `round_number` sits every `round_step` dollars (every $25
        # on gold) which is dense relative to the cluster tolerance, so it is
        # also nearly always "confirming" something nearby. Counting these as
        # full independent groups let the score saturate (n_groups=3 ->
        # 0.92) from noise-level agreement. Fractional independence weights
        # make the score reflect genuinely separate evidence sources.
        n_groups_eff = sum(self._GROUP_INDEPENDENCE.get(g, 1.0) for g in groups)
        s["method"] = 1.0 - math.exp(-0.6 * n_groups_eff)
        n_groups = len(groups)

        # Method quality - best weight per group, saturating (v16: same fix).
        q_raw = self._group_weighted_quality(types)
        s["quality"] = min(q_raw / 55.0, 1.0)

        # Touch strength - recency-weighted, log-saturated (over-testing != linear).
        tr = self._touch_and_rejection(level, side)
        s["touch"] = min(math.log1p(tr["recency"]) / math.log1p(12), 1.0)

        # v17: historical polarity - is this label an untested flip (level
        # was historically respected mostly from the OPPOSITE side)?
        polarity = self._polarity_check(level, side)

        # Volume acceptance - proximity to high-volume node / POC.
        # v17: discounted by `volume_confidence` since the underlying data is
        # MT5 tick-volume (a tick-count proxy), not real traded volume.
        vol_frac = self._volume_strength(level)
        s["volume"] = vol_frac * self.cfg.volume_confidence

        # Structural density - KDE of confirmed swing pivots (2026 DBSCAN/KDE).
        dens = self._density_strength(level, side)
        s["density"] = dens

        # Liquidity structure - equal highs/lows + sweep-reclaim (2026 SMC).
        liq = self._liquidity_structure(level, side)
        s["liquidity"] = 0.6 * liq["equal"] + 0.4 * liq["sweep"]

        # Rejection quality - ATR-qualified wick rejections, recency-weighted.
        s["rejection"] = min(tr["rejections"] / 4.0, 1.0)

        # v14 acceptance: is this price a high-volume node (accepted, holds)
        # or a low-volume node (rejected, price slices through it)?
        # v17: also discounted (see `volume_confidence` above).
        acc, node = self._node_acceptance(level)
        s["acceptance"] = acc * self.cfg.volume_confidence

        # v14: naked / virgin POC - an untested prior-session point of control
        # sitting on this cluster. Strongest single magnet in the volume model.
        naked_hit = None
        if self._naked_pocs and self._atr:
            tol = max(0.35 * self._atr, self.cfg.round_tol_dollars)
            for poc, age in self._naked_pocs:
                if abs(poc - level) <= tol:
                    naked_hit = (poc, age)
                    break

        # ---- Core strength = weighted mean of structural sub-scores (0..1) ----
        core = (sum(self._CORE_W[k] * s[k] for k in self._CORE_W)
                / sum(self._CORE_W.values()))
        score_pct = 100.0 * core

        # ---- Bonuses added on top (they do NOT dilute the core) ----
        round_pts = self._round_number_tier(level)          # 0..15
        s["round"] = round_pts / 15.0
        score_pct += 8.0 * s["round"]

        # Proximity aids tradeability; kept small vs. the structural core.
        if dist_dollars <= 1.0:
            prox = 1.00
        elif dist_dollars <= 2.0:
            prox = 0.80
        elif dist_dollars <= 5.0:
            prox = 0.55
        elif dist_dollars <= 10.0:
            prox = 0.30
        else:
            prox = 0.15
        s["proximity"] = prox
        score_pct += 6.0 * prox

        # v14 volume-model adjustments.
        # v17: scaled by `volume_confidence` (tick-volume proxy, not real
        # traded volume - see cfg docstring) so these bonuses/penalties don't
        # swing the grade as if they were confirmed institutional data.
        vconf = self.cfg.volume_confidence
        if naked_hit:
            score_pct += 7.0 * vconf    # untested POC magnet
        if node == "LVN":
            score_pct -= 7.0 * vconf    # low-volume void: price passes through
        elif node == "HVN":
            score_pct += 3.0 * vconf    # accepted value: price stalls here

        # v14 volatility regime: in expansion, close levels get run over; in
        # compression, far levels are unlikely to be reached at all.
        if self._atr:
            atr_dist = dist_dollars / self._atr
            if self._vol_regime == "expansion" and atr_dist < 0.5:
                score_pct -= 4.0
            elif self._vol_regime == "compression" and atr_dist > 3.0:
                score_pct -= 4.0

        # ---- Directional / risk adjustments on the 0..100 scale ----
        plan = self.build_trade_plan(level, side, all_levels)

        dxy_note = ""
        if self.cfg.use_dxy and self._dxy_bias != "neutral" and plan:
            aligned = (plan.direction == "BUY" and self._dxy_bias == "bearish_usd") or \
                      (plan.direction == "SELL" and self._dxy_bias == "bullish_usd")
            # v17: effect halved - a 60-bar correlation-gated slope is a soft
            # macro tailwind/headwind, not structural S/R evidence, and
            # shouldn't move the grade as much as genuine price confluence.
            if aligned:
                score_pct += 3.0
                dxy_note = "DXY aligned"
            else:
                score_pct -= 4.0
                dxy_note = "DXY against"

        if plan and plan.rr < self.cfg.min_rr:
            score_pct -= 6.0

        # Over-tested level with little recent rejection = liquidity consumed.
        if tr["touches"] > self.cfg.over_test_touches and tr["rejections"] < 1.0:
            score_pct -= 6.0

        # v17: untested polarity flip - the level was historically respected
        # mostly from the OPPOSITE side and has not yet held from this new
        # side. Not disqualifying (flips do happen), but it should not score
        # like an already-proven level of this type.
        if polarity["flip"]:
            score_pct -= 8.0

        # A lone-method-GROUP "level" is not real confluence - penalise hard.
        # (v16: was `n_types < 2`, which let Fib+MA+Bollinger - one correlated
        # price-math family - dodge this penalty while contributing no
        # genuinely independent evidence.)
        if n_groups < 2:
            score_pct -= 10.0

        score_pct = max(0.0, min(100.0, score_pct))
        score = int(round(score_pct))
        score10 = round(score_pct / 10.0, 1)
        comp = {k: round(100.0 * v, 1) for k, v in s.items()}

        # ---- v13: evidence-weighted reasons + "11 strong reasons" A+ gate ----
        reason_objs = self.build_reasons(
            level, sources, dist_dollars, side,
            ctx={
                "touch": tr,
                "liquidity": liq,
                "density": dens,
                "volume": vol_frac,
                "round_pts": round_pts,
                "acceptance": acc,
                "node": node,
                "naked_hit": naked_hit,
                "plan": plan,
                "dxy_note": dxy_note,
                "polarity": polarity,
            },
        )
        strong_reasons = [r for r in reason_objs if r.strong]
        # v19: gate on the group-deduped count, not the raw count - see
        # `_strong_count_deduped` and `_REASON_CORR_GROUP`. The raw count is
        # still surfaced in `evidence_score`/`reasons` for display purposes.
        strong_count = self._strong_count_deduped(reason_objs)
        evidence_score = round(sum(r.weight for r in reason_objs), 2)

        grade, gate_note = self._grade(score, plan, dxy_note, strong_count,
                                       polarity["flip"])

        return {
            "level": level,
            "type": side,
            "dist_dollars": dist_dollars,
            "score": score,
            "score10": score10,
            "grade": grade,
            "gate_note": gate_note,
            "tf_count": len(tfs),
            "type_count": len(types),
            "strong_count": strong_count,
            "evidence_score": evidence_score,
            "reasons": [r.label() for r in reason_objs[: self.cfg.max_reasons_display]],
            "strong_reasons": [r.text for r in strong_reasons],
            "reason_objects": reason_objs,
            "plan": plan,
            "dxy_note": dxy_note,
            "components": comp,
            "polarity_flip": polarity["flip"],
        }

    def _grade(self, score: int, plan: Optional[TradePlan], dxy_note: str,
               strong_count: int = 0, polarity_flip: bool = False) -> Tuple[str, str]:
        """Grade a setup on score AND independent evidence.

        A+ now demands `cfg.min_strong_reasons_aplus` (default 11) strong,
        independent reasons in addition to score, R:R and DXY gates, so the
        top grade always comes with a defensible 11-point justification.
        Returns (grade, gate_note) where gate_note explains any A+ block.

        v17: an untested polarity flip (see `_polarity_check`) also blocks
        A+ - a level the market has never actually held from its NEW side
        shouldn't get the top grade no matter how much other confluence
        piles on top of it.
        """
        cfg = self.cfg
        rr_ok = plan is not None and plan.rr >= cfg.min_rr
        dxy_ok = dxy_note != "DXY against"
        reasons_ok = strong_count >= cfg.min_strong_reasons_aplus
        flip_ok = not polarity_flip

        if score >= cfg.aplus_min_score and rr_ok and dxy_ok and reasons_ok and flip_ok:
            return "A+", f"A+ confirmed: {strong_count} strong reasons"

        missing: List[str] = []
        if score < cfg.aplus_min_score:
            missing.append(f"score {score}<{cfg.aplus_min_score}")
        if not rr_ok:
            rr_txt = f"{plan.rr:.1f}" if plan else "n/a"
            missing.append(f"R:R {rr_txt}<{cfg.min_rr:.1f}")
        if not flip_ok:
            missing.append("untested polarity flip")
        if not dxy_ok:
            missing.append("DXY against")
        if not reasons_ok:
            missing.append(
                f"{strong_count}/{cfg.min_strong_reasons_aplus} strong reasons")
        note = "A+ blocked: " + ", ".join(missing)

        if (score >= cfg.a_min_score and rr_ok
                and strong_count >= cfg.min_strong_reasons_a):
            return "A", note
        if score >= cfg.b_min_score:
            return "B", note
        return "C", note

    # ---------------- Run + display ---------------- #
    def run(self) -> None:
        log.info("APEX scan | %s | %s", self.cfg.symbol,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if self.cfg.volume_confidence > 0.8:
            log.warning("volume_confidence=%.2f: POC/VAH/VAL/naked-POC/HVN-LVN "
                        "are still built on MT5 tick_volume (tick count), not "
                        "real traded volume - confirm you have a real volume "
                        "feed before trusting these near full weight.",
                        self.cfg.volume_confidence)
        if not self.initialize_mt5():
            return
        try:
            news = self.in_news_window()
            if news:
                log.warning("NEWS WINDOW active: %s - trade with caution", news)

            raw = self.collect_raw_levels()  # loads data, sets current price + ATR,
                                              # and builds the MTF swing-cluster levels
            self.compute_poc()
            self.compute_dxy_bias()

            # Volume-profile levels (POC / value area) as first-class sources.
            for name, lvl in (("POC", self._poc), ("VAH", self._vah),
                              ("VAL", self._val)):
                if lvl:
                    raw.append({"level": float(lvl), "type": "volume_profile",
                                "name": name, "tf": self.cfg.lowest_tf})

            # v14: naked / virgin POCs from prior sessions (magnet levels).
            self.compute_naked_pocs()
            for i, (poc, age) in enumerate(self._naked_pocs):
                raw.append({"level": float(poc), "type": "naked_poc",
                            "name": f"nPOC-{age}d", "tf": "D1"})

            clusters = self.cluster_levels(raw)
            log.info("%d raw levels -> %d clustered bands | ATR(%s)=%.2f",
                     len(raw), len(clusters), self.cfg.atr_tf, self._atr or 0.0)

            all_levels = sorted(c["level"] for c in clusters)
            scored = [r for r in (self.score_cluster(c, all_levels)
                                  for c in clusters) if r]
            support_pool = [s for s in scored if s["type"] == "Support"]
            resistance_pool = [s for s in scored if s["type"] == "Resistance"]
            supports = sorted(support_pool,
                              key=lambda x: (-x["score"], x["dist_dollars"], x["level"]))[:3]
            resistances = sorted(resistance_pool,
                                 key=lambda x: (-x["score"], x["dist_dollars"], x["level"]))[:3]

            # v19 structural fallback (revised): the v18 version force-included
            # the nearest candidate on EVERY scan regardless of its score, so
            # the table could never honestly report "no defensible S/R on this
            # side" - the strict confluence model's own negative conclusion was
            # silently overridden. Now the fallback only fires for candidates
            # that clear `cfg.min_display_score`; below that floor the side is
            # left empty rather than padded with a low-quality "closest" row.
            def ensure_nearest(chosen: List[dict], pool: List[dict]) -> List[dict]:
                if not pool:
                    return chosen
                viable = [p for p in pool if p["score"] >= self.cfg.min_display_score]
                if not viable:
                    return chosen  # no candidate is defensible - don't manufacture one
                nearest = min(viable, key=lambda x: (x["dist_dollars"], -x["score"]))
                if all(abs(x["level"] - nearest["level"]) > 1e-9 for x in chosen):
                    if len(chosen) >= 3:
                        chosen = chosen[:2] + [nearest]
                    else:
                        chosen = chosen + [nearest]
                return sorted(chosen, key=lambda x: (-x["score"], x["dist_dollars"], x["level"]))

            supports = ensure_nearest(supports, support_pool)
            resistances = ensure_nearest(resistances, resistance_pool)
            self.display(supports, resistances, news)
        except Exception:
            log.exception("Scan failed")
        finally:
            if mt5:
                mt5.shutdown()

    def display(self, sups: List[dict], ress: List[dict], news: Optional[str]) -> None:
        line = "=" * 96
        print(f"\n{line}")
        print(f"APEX S/R  {self.cfg.symbol} @ {self.current_price:.2f}  "
              f"| range ${self.cfg.max_range_dollars:.0f} | DXY: {self._dxy_bias}"
              + (f" | NEWS: {news}" if news else ""))
        print(line)
        need = self.cfg.min_strong_reasons_aplus
        print(f"{'Grade':<7}{'Type':<7}{'Price':<10}{'$Away':<8}{'/10':<6}"
              f"{f'Str/{need}':<8}{'Entry':<10}{'SL':<10}{'TP':<10}{'RR':<6}{'Lots':<6}")
        print("-" * 96)
        for r in ress + sups:
            p = r["plan"]
            entry = f"{p.entry:.2f}" if p else "-"
            sl = f"{p.stop:.2f}" if p else "-"
            tp = f"{p.target:.2f}" if p else "-"
            rr = f"{p.rr:.1f}" if p else "-"
            lots = f"{p.lots:.2f}" if p else "-"
            s10 = f"{r.get('score10', r['score'] / 10):.1f}"
            strong = f"{r.get('strong_count', 0)}/{need}"
            print(f"{r['grade']:<7}{r['type'][:4]:<7}{r['level']:<10.2f}"
                  f"{r['dist_dollars']:<8.1f}{s10:<6}{strong:<8}"
                  f"{entry:<10}{sl:<10}{tp:<10}{rr:<6}{lots:<6}")
        print(line)

        if self.cfg.table_only:
            return

        if self.cfg.compact_output:
            for r in ress + sups:
                p = r["plan"]
                tag = {"A+": "[A+]", "A": "[A ]", "B": "[B ]"}.get(r["grade"], "[C ]")
                plan_txt = (f"{p.direction} {p.entry:.2f} | SL {p.stop:.2f} "
                            f"| TP {p.target:.2f} | RR {p.rr:.1f} | {p.lots:.2f} lots"
                            if p else "no plan")
                print(f"{tag} {r['type'][:4]} {r['level']:.2f}  "
                      f"{r.get('strong_count', 0)}/{need} strong  ->  {plan_txt}")
                top = r.get("strong_reasons", [])[:3]
                if top:
                    print("      why: " + " | ".join(top))
            print(line)
            return

        if not self.cfg.show_reasons:
            return

        # ---- v13: print the evidence file behind every level ----
        for r in ress + sups:
            plan = r["plan"]
            print(f"\n[{r['grade']}] {r['type']} {r['level']:.2f}  "
                  f"| score {r['score']} ({r.get('score10', 0):.1f}/10) "
                  f"| strong reasons {r.get('strong_count', 0)}/{need} "
                  f"| evidence {r.get('evidence_score', 0):.2f}")
            if plan:
                print(f"    Plan: {plan.direction} @ {plan.entry:.2f} "
                      f"SL {plan.stop:.2f}  TP {plan.target:.2f}  "
                      f"R:R {plan.rr:.1f}  lots {plan.lots:.2f} "
                      f"(risk ${plan.risk_dollars:.0f})")
            strong_list = r.get("strong_reasons", [])
            if strong_list:
                print(f"    STRONG EVIDENCE ({len(strong_list)}):")
                for i, txt in enumerate(strong_list, 1):
                    print(f"      {i:>2}. {txt}")
            supporting = [ro for ro in r.get("reason_objects", []) if not ro.strong]
            if supporting:
                print("    Supporting: "
                      + "; ".join(ro.text for ro in supporting[:6]))
            note = r.get("gate_note")
            if note and r["grade"] != "A+":
                print(f"    {note}")
        print(line)


if __name__ == "__main__":
    # Example: populate news windows from your economic calendar (UTC).
    cfg = ApexConfig(
        # news_windows_utc=[("2026-07-17", "12:30", 30)],  # e.g. US CPI
    )
    ApexScanner(cfg).run()