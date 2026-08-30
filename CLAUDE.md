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
- **Spray / guy-high-low designs** — two *separate, non-interoperating* builds
  for the same pair of desk setups (sprays through the book, and guy high / guy
  low). Both unvalidated drafts.
  - `docs/mat-filter-settings/strategy-spray-and-guy-filters.md` — configured
    inside the platform's MAT Filter Settings window.
  - `docs/shel-datagateway/detector-spray-and-guy.md` — a standalone application
    written against the DataGateway feed.
  - `docs/noticeable-move-ladder.md` — the square-root price ladder for sizing a
    "noticeable" move. The only piece shared by both, because it describes market
    structure rather than either system.

  **MAT and the DataGateway API are separate systems with no data path between
  them.** MAT is a filter engine configured through the platform UI; DataGateway
  is a market data feed requiring your own client. Nothing computed from one can
  be fed to the other. Do not describe either as extending, fixing, or feeding
  the other.
- **SHEL DataGateway market data API** — `docs/shel-datagateway/`. Trillium's
  trade/quote-level market data API and the TF-Server binary framing protocol
  beneath it: request shapes, subscriptions, every response message, the Python
  SDK, and a list of spec-vs-observed gotchas. Start at
  `docs/shel-datagateway/README.md`. Consult this for any work touching live or
  historical market data ingestion. Key constraint: NBBO is live-only, never
  historical.
