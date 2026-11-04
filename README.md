# XAUUSD Support and Resistance Scanner for MetaTrader 5 — S/R APEX v19

A free, open-source Python scanner that finds multi-timeframe support and resistance levels for **XAUUSD (gold) on MetaTrader 5**, scores how much evidence backs each level, and grades the resulting setups. It is built for traders who want clear gold S/R levels on MT5 — without an indicator that repaints on the forming candle.

If you searched for an XAUUSD support resistance tool, a gold levels scanner for MT5, or a Python script that calculates gold support and resistance from H1, H4 and D1, this is what this project does.

## Contents

- [What you get](#what-you-get)
- [How the XAUUSD S/R levels are built](#how-the-xauusd-sr-levels-are-built)
- [Quick start on MetaTrader 5](#quick-start-on-metatrader-5)
- [Who this is for](#who-this-is-for)
- [What this tool does not do](#what-this-tool-does-not-do)
- [Configuration notes and limits](#configuration-notes-and-limits)
- [FAQ](#faq)
- [Status](#status)

## What you get

- **Multi-timeframe gold support and resistance levels** from H1, H4 and D1, with W1 anchors — not a single-timeframe line.
- **Confluence scoring, 0–100**, with a transparent evidence breakdown, so you can see *why* a level scored high instead of trusting a single line.
- **Setup grades A+ / A / B / C**, with an evidence gate for A+ so a level needs several independent, strong reasons, not just one cosmetic signal.
- **A trade plan per level**: direction, entry, stop loss, target, risk/reward and position size from account risk.
- **Closed-bar analysis only**: the forming candle is dropped, so levels do not repaint as the current candle moves.

## How the XAUUSD S/R levels are built

Levels are clustered into price bands first, then scored. Sources include:

- Confirmed swing pivots and multi-timeframe swing-cluster zones
- Floor pivot points, Camarilla and DeMark levels, anchored to the prior closed day (and optionally week)
- Fibonacci and Murrey Math levels from recent swings
- Volume profile: POC and value area (VAH / VAL), plus untested (“naked”) POCs from prior sessions
- Session VWAP and VWAP standard-deviation bands
- Round-number gold levels ($100 / $50 / $25 / $10 / $5 tiers)
- KDE pivot-density and Smart Money Concepts liquidity structure: equal highs/lows and sweep-and-reclaim signals
- DXY intermarket context, gated by actual gold–DXY correlation over the lookback window

Correlated methods are grouped before scoring, so Fibonacci, moving averages, Bollinger Bands and Murrey Math — all transforms of the same price window — are not counted as four independent votes.

## Quick start on MetaTrader 5

Requirements: Windows with MetaTrader 5 installed and logged in (the script connects to your running MT5 terminal), Python 3.10+, and the packages in `requirements.txt` (`MetaTrader5`, `pandas`, `numpy`).

```bash
pip install -r requirements.txt
python xauusd_sr_apex_v19_MMSL.py
```

The main script is [xauusd_sr_apex_v19_MMSL.py](xauusd_sr_apex_v19_MMSL.py). Edit `ApexConfig` at the bottom of the file to change the symbol, risk percentage, ATR multiples, or news windows.

## Who this is for

- Gold traders on MT5 who plan around support and resistance rather than a single indicator line
- Traders who want H1 / H4 / D1 confluence for XAUUSD before choosing an entry zone
- Python users who want a readable, configurable S/R engine they can inspect and adapt — not a black box

## What this tool does NOT do

- It does **not** place orders. It is an analysis and scanner tool only.
- It is not financial advice and makes no profit guarantee.
- Backtest and demo-test it before using any plan with real money.

## Configuration notes and limits

- Analysis uses closed bars only (the forming candle is dropped).
- MT5 `tick_volume` is a tick count, not real traded volume. Volume-derived levels are discounted by `volume_confidence` (default 0.5) for that reason.
- The DXY symbol varies by broker (`DXY`, `USDX`, `DX` …). Set `dxy_symbol` in `ApexConfig` to match your broker, or set `use_dxy=False`.
- News windows are empty by default. Add your own events to `news_windows_utc` (UTC) before relying on the news filter.
- Spread, session times and contract specs depend on your broker. Check them on your own MT5 chart before acting on any level.

## FAQ

**Does this place trades on MetaTrader 5?**
No. It calculates levels, scores and plans. It sends no orders.

**Which timeframes does it use for XAUUSD?**
H1, H4 and D1 for the main scan, with W1 anchors for pivot-family levels.

**Does it repaint?**
No, by design: it analyses closed bars only and drops the forming candle.

**Does it work with any MT5 broker?**
It connects to your running MT5 terminal, so the symbol name, DXY symbol, spread and contract specs must match your broker. Check those settings first.

**Is this a gold trading robot or EA?**
No. It is a Python scanner for MT5, not an Expert Advisor and not an MQL5 indicator.

## Status

v19 “APEX”. Python syntax was checked, but a full live run on Windows/MT5 is still needed to confirm behaviour with a real broker feed.
