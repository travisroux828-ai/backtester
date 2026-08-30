# CLAUDE.md

Project notes and pointers for Claude Code.

## Project

Python/Streamlit backtester. Entry points: `app.py` (Streamlit UI) and
`main.py`. Supporting packages: `engine/`, `strategies/`, `indicators/`,
`data/`, `export/`, `ai/`.

## Reference docs

- **MAT Filter Settings** — `docs/mat-filter-settings/`. Captured Trillium KB
  documentation for the MAT (Price Moves) Filter Settings window: filter
  creation, Add Price Moves, Define Symbol Set, Conditions, Input Advanced
  Queries, and Set Results Preferences. Start at
  `docs/mat-filter-settings/README.md`, which also summarizes the filter
  evaluation order and shared input conventions. Consult this whenever work
  touches MAT filter semantics, scanner/filter modeling, or advanced-query
  expression syntax.
- **Spray / guy-high-low filter design** —
  `docs/mat-filter-settings/strategy-spray-and-guy-filters.md`. Working (not yet
  validated) MAT filter design for two desk setups: sprays through the book, and
  guy high / guy low. Contains the square-root price ladder used to size
  "noticeable" moves, and a calibration plan against tick data.
- **SHEL DataGateway market data API** — `docs/shel-datagateway/`. Trillium's
  trade/quote-level market data API and the TF-Server binary framing protocol
  beneath it: request shapes, subscriptions, every response message, the Python
  SDK, and a list of spec-vs-observed gotchas. Start at
  `docs/shel-datagateway/README.md`. Consult this for any work touching live or
  historical market data ingestion. Key constraint: NBBO is live-only, never
  historical.
