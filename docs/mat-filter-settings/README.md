# MAT Filter Settings — Reference

Reference documentation for the **Filter Settings** window used by the MAT
(Price Moves) window. Captured from the Trillium knowledge base so it can be
referenced offline and by Claude during development.

## Contents

| Page | File |
| --- | --- |
| Filter Settings Window (overview) | [`filter-settings-window.md`](filter-settings-window.md) |
| Add Price Moves (MAT) | [`add-price-moves.md`](add-price-moves.md) |
| Define Symbol Set (MAT) | [`define-symbol-set.md`](define-symbol-set.md) |
| Conditions (MAT) | [`conditions.md`](conditions.md) |
| Input Advanced Queries (MAT) | [`input-advanced-queries.md`](input-advanced-queries.md) |
| Set Results Preferences (MAT) | [`set-results-preferences.md`](set-results-preferences.md) |

### Working designs (not KB content)

| Doc | File |
| --- | --- |
| Filter design: sprays and guy high/low | [`strategy-spray-and-guy-filters.md`](strategy-spray-and-guy-filters.md) |

## Quick model of how a MAT filter evaluates

A symbol must satisfy **every** stage to appear in the MAT window:

1. **Symbol Set** — Stage 1: Exchanges + Product Types. Stage 2: Include/Exclude
   symbols and lists. Muted symbols and the Mute After Count are applied here.
2. **Price Moves** — Time Frame (rolling seconds) and Minimum Volume Percent
   are the primary gate; the per-range criteria (Price Range, Primary Move,
   Secondary Move, Move Volume, 20 Day Volume) are secondary filters applied
   after that gate.
3. **Conditions** — market-metric ranges (price, volume, market cap, shares
   outstanding, GICS, etc.).
4. **Advanced Queries** — logical expressions; all AQ conditions must pass.
5. **Results Preferences** — presentation only (flash, colors); does not filter.

## Screenshots and version drift

Each page carries a **Panel layout (observed UI)** section transcribed from
screenshots of the running application (captured 2026-08-30) — exact field
labels, control types, defaults, and layout. The image files themselves are not
in this repo; the transcriptions stand in for them. Drop the images into
`images/` and link them from the layout sections if you want the visuals too.

Two places where the observed UI disagrees with the KB text, both flagged inline
on their pages:

| Page | KB says | Screenshot shows |
| --- | --- | --- |
| Add Price Moves | `Filter Out Prints` dropdown with None / Odd Lots / Less Than 100 | a plain `Odd Lots` On/Off toggle |
| Define Symbol Set | a `Mute After Count` field | no such field in the panel |

Both look like the KB documenting a newer build than the screenshots. Confirm
against your target build before relying on either.

## Shared input conventions

- **Volume / size inputs:** `100`; `1,000` or `1K`; `1,000,000` or `1M`;
  `10,000,000,000` or `10B`.
- **Price moves:** accept dollar or percentage formats.
- **Percent expressions in AQ:** whole numbers — `10` = 10%, `0.10` = 0.1%.
- Most panes require clicking **Apply** to save.
