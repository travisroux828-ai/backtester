# Detector Design: Sprays and Guy High / Guy Low (DataGateway)

Design for a **standalone application** that detects two desk setups from the
SHEL DataGateway feed. Status: **design only** — nothing built, nothing validated.

> **Scope.** This is its own system: your process, your connection, your alerting.
> A separate design exists for approximating the same setups inside the platform's
> MAT filter window
> ([`../mat-filter-settings/strategy-spray-and-guy-filters.md`](../mat-filter-settings/strategy-spray-and-guy-filters.md)).
> **The two do not interoperate.** MAT is a filter engine configured through the
> platform UI; DataGateway is a market data feed you write a client against.
> Nothing computed here can be fed into a MAT filter, and MAT cannot consume
> anything this produces. Build one or the other — or both, independently, and
> reconcile them by eye.
>
> The one shared piece is the move-sizing logic in
> [`../noticeable-move-ladder.md`](../noticeable-move-ladder.md), which is a fact
> about markets rather than about either system.

## What the setups are

| | Spray | Guy high / guy low |
| --- | --- | --- |
| **Mechanism** | An aggressor sweeps through multiple book levels, printing a run of ticks in one direction. Can repeat in succession. | A bid rests noticeably *above* the book (guy high) or an offer noticeably *below* it (guy low), with size, holding price away from fair briefly. |
| **Your action** | Post offers into a spray up / bids into a spray down | Hit the elevated bid / take the depressed offer |
| **Data needed** | `trade` — with `mkt` and `flags` | `nbbo` |

Unlike in MAT, here the two setups are **genuinely different detectors** on
**different subscriptions**. There is no reason to merge them.

## Guy high / guy low — from `nbbo`

The setup lives in the quote, and `nbbo` reads the quote. A resting bid above the
recent market is visible **before anyone trades against it** — which is the moment
the edge exists.

```
guy_high  when  bid        > reference + threshold(price)
           and  bid-size   >= size_floor(price)
           and  that state persists >= N milliseconds

guy_low   when  ask-price  < reference - threshold(price)
           and  ask        >= size_floor(price)
           and  that state persists >= N milliseconds
```

> **Mind the field names.** `bid` is a price, `bid-size` a size — but `ask-price`
> is the price and **`ask` is the size**. See the gotcha in
> [`response-messages.md`](response-messages.md). Getting this wrong inverts the
> guy_low test silently.

**`reference`** — a short trailing mid or VWAP. Either subscribe to `vwma-1s`, or
compute a trailing mid from the `nbbo` stream itself. Computing locally is
probably better: it keeps the detector on one subscription and avoids depending on
the server's window semantics.

**`threshold(price)`** — the √price ladder. Normalize per symbol using
`hist-stat`'s `20D` high/low span, which arrives free on every subscription.

**`persists >= N ms`** — the "sticks above where it was trading" part. This is what
separates a guy from a momentary quote flicker, and it has no analogue in the MAT
build. Start around 200–500ms and calibrate.

### The recording problem

**`nbbo` has no history.** Trades go back; quotes do not.

So this detector **cannot be backtested at all** on data that exists today. It can
only be studied on quote data already recorded. Every day without a recorder
running is a day permanently unavailable.

If this setup matters, **the recorder is the first thing to build** — before the
detector, before any modeling. It's a small program: subscribe `nbbo` (plus
`trade` for context) across the symbol universe, write to disk, done. The
analysis can happen any time later; the data cannot.

## Sprays — from `trade`

With `mkt` and `flags` on every print, a spray can be defined as what it actually
is rather than inferred from net displacement:

```
spray  when, inside a rolling window W:
         prints, after excluding Drk / OffMkt* / OOS / Odd,
         span >= L distinct price levels, monotonically ordered,
         and aggregate size >= size_floor(price),
         and total displacement >= threshold(price)
```

The **distinct-levels** requirement is the piece MAT cannot express, and it's the
difference between "price moved" and "someone swept the book". A single large
print at one price satisfies a displacement test but is not a spray.

**Flag filtering is not marginal.** `Drk` and `OffMkt*` prints never touched the
displayed book, so they cannot be part of a sweep of it. In the AAPL sample output
in the SDK docs, the large majority of prints are `FINN`/`Drk`. Excluding them
changes the answer substantially, not slightly.

Also exclude `OOS` (out-of-sequence prints shouldn't drive sequential logic) and
`Odd` (excluded by default unless `trade.include-odd-lots` is set).

Successive sprays fall out naturally: emit each detection and let repeats stand,
rather than muting.

### This one *can* be backtested

Historical `trade` is available with nanosecond timestamps, so the spray detector
can be developed and calibrated entirely offline before it ever runs live.

## Calibration

`data/scanner.py` works off Polygon grouped **daily** bars and is irrelevant here
— these are sub-second events. `request_data(..., ['trade'])` supplies what's
needed directly.

1. Pull raw trades for ~20 symbols spanning the price ladder, several sessions.
   Filter out `Drk`, `OffMkt*`, `OOS`.
2. Resample to 1s; compute rolling 2s displacement, levels crossed, and volume.
3. Histogram displacement per price bucket. The threshold worth using is a **tail
   quantile** (~99.5th percentile of that symbol's own 2s moves), not a fixed cent
   value — self-normalizing, and it subsumes both the price ladder and the
   volatility correction in one number.
4. Label forward returns at +5s / +15s / +30s to test whether the fade actually
   pays, and over what horizon.
5. Compare empirical per-bucket thresholds against the √price ladder. Where they
   diverge, trust the data.

Step 3 is the real prize. The ladder is scaffolding to be discarded once there's
enough tape to replace it.

For the guy detector, none of this is possible until the recorder has run. That
asymmetry — one detector calibratable today, the other gated on data collection
starting — is the main scheduling fact for this work.

## Build order

1. **NBBO recorder.** Perishable data. Nothing else is time-sensitive.
2. **Spray detector, offline.** Calibrate against historical trades.
3. **Spray detector, live.** Same logic against the stream.
4. **Guy detector.** Once the recorder has accumulated enough quote data.
