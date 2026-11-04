# XAUUSD S/R APEX v19

Multi-timeframe support/resistance confluence scanner and A+ setup grader for XAUUSD on MetaTrader 5.

## What it does

- Builds S/R levels from H1, H4 and D1 (plus W1 anchors) using clustered swing pivots, pivot/Camarilla/DeMark levels, Fibonacci, volume profile (POC / value area), VWAP bands, round numbers and KDE / liquidity-structure signals.
- Scores clustered levels 0–100 with a transparent evidence breakdown and grades setups A+ / A / B / C.
- Calculates a trade plan per level (direction, entry, stop, target, R:R and position size from account risk).

## What it does NOT do

- It does **not** place orders. It is an analysis/scanner tool only.
- It is not financial advice and makes no profit guarantee.
- Backtest / demo-test it before using any plan with real money.

## Requirements

- Windows with MetaTrader 5 installed and logged in (the script connects to your running MT5 terminal).
- Python 3.10+
- `MetaTrader5`, `pandas`, `numpy`

```bash
pip install -r requirements.txt
python xauusd_sr_apex_v19_MMSL.py
```

## Notes and limits

- Analysis uses closed bars only (the forming candle is dropped).
- MT5 `tick_volume` is a tick count, not real traded volume. Volume-derived levels are discounted by `volume_confidence` (default 0.5) for that reason.
- The DXY symbol varies by broker (`DXY`, `USDX`, `DX` …). Set `dxy_symbol` in `ApexConfig` to match your broker, or set `use_dxy=False`.
- News windows are empty by default. Add your own events to `news_windows_utc` (UTC) before relying on the news filter.
- Spread, session times and contract specs depend on your broker. Check them on your own MT5 chart before acting on any level.

## Status

v19 "APEX". Python syntax was checked, but a full live run on Windows/MT5 is still needed to confirm behaviour with a real broker feed.
