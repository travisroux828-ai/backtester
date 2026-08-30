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
