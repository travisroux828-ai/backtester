# Filter Design: Sprays and Guy High / Guy Low

Working design for two MAT setups. Status: **unvalidated draft** — thresholds are
derived analytically, not fitted to data. See [Calibration](#calibration) before
trading these.

## The setups

| | Spray | Guy high / guy low |
| --- | --- | --- |
| **Mechanism** | An aggressor uses enough size to sweep through multiple book levels, printing a run of ticks in one direction. Can repeat in succession. | A bid rests noticeably *above* the book (guy high) or an offer rests noticeably *below* it (guy low), with size, holding price away from fair for a short time. |
| **Aggressor** | Taker — crosses the spread repeatedly | Maker — posts away from the inside |
| **Your action** | Post offers into a spray up / bids into a spray down | Hit the elevated bid (guy high) / take the depressed offer (guy low) |
| **Trade direction** | Fade | Fade |
| **Qualifier** | Move must be a noticeable distance from where price was trading | Same |

### The unifying insight

Both are **the same detection problem**: a fast, size-backed displacement away
from recent price, which you then fade. They differ in *mechanism* (aggressive
sweep vs. passive resting size) and therefore in *execution* (post vs. take) —
but the price footprint MAT can see is nearly identical.

That argues for **one detection filter**, not two. See
[Recommended build](#recommended-build-one-filter).

## Sizing the move: the √price ladder

The core question is "how far is noticeable?" Two anchor points from the desk:

- A stable **$10** stock spraying **$0.10** is interesting (1.00% of price)
- A **$900** stock needs **$1.00** or more (0.11% of price)

Fitting `move = c · price^k` through both gives **k = 0.51, c = 0.031** — i.e.
almost exactly square-root scaling:

```
move ≈ √price / 31.6
```

This is the expected relationship (per-share volatility scales roughly with the
square root of price), which is a good sign the anchors are sound rather than
arbitrary. Note the percentage threshold *falls* as price rises — a flat percent
rule would be badly wrong at both ends.

| Price | Move threshold | As % of price |
| ---: | ---: | ---: |
| $1 | $0.03 | 3.16% |
| $5 | $0.07 | 1.42% |
| $10 | $0.10 | 1.00% |
| $20 | $0.14 | 0.71% |
| $50 | $0.22 | 0.45% |
| $100 | $0.32 | 0.32% |
| $200 | $0.45 | 0.22% |
| $500 | $0.71 | 0.14% |
| $900 | $0.95 | 0.11% |

### Price Move ladder (paste into Add Price Moves)

Thresholds evaluated at each bucket's geometric mean. Move Volume floors target
roughly **$100–200k notional** per move, so size scales inversely with price.

| # | Price Range | Primary Move | Secondary Move | Move Vol MIN | 20d Vol MIN |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 0.01 – 1.00 | $0.03 | $0.01 | 100,000 | 1M |
| 2 | 1.00 – 5.00 | $0.05 | $0.02 | 40,000 | 1M |
| 3 | 5.00 – 15.00 | $0.10 | $0.05 | 10,000 | 500K |
| 4 | 15.00 – 40.00 | $0.15 | $0.07 | 4,000 | 500K |
| 5 | 40.00 – 100.00 | $0.25 | $0.10 | 2,000 | 500K |
| 6 | 100.00 – 250.00 | $0.40 | $0.18 | 600 | 300K |
| 7 | 250.00 – 600.00 | $0.60 | $0.25 | 300 | 200K |
| 8 | 600.00 – 1200.00 | $0.90 | $0.40 | 200 | 200K |
| 9 | 1200.00 – 99999 | $1.25 | $0.55 | 100 | 200K |

Secondary Move is set to ~45% of Primary throughout — enough to confirm a real
continuation leg without demanding a second full-size spray.

> Bucket 1 (sub-$1) is set by judgment, not the formula: the $0.01 tick floor and
> the extreme percentage volatility of sub-$1 names make the fit meaningless
> there. Drop this bucket entirely if you don't trade sub-$1.

> Keep Move Volume MIN at 100 or above — a lower floor is unreachable once
> Filter Out Prints is set to "Less Than 100".

## Recommended build: one filter

Name it something like **Displacement Fade**.

### Add Price Moves

| Setting | Value | Why |
| --- | --- | --- |
| Time Frame | **2 seconds** | Both setups are near-instant. A longer window admits grinding moves, which are not the trade. Try 3s if 2s is too sparse. |
| Minimum Volume Percent | **Off** | See the warning below — this will silently kill your morning alerts. |
| Filter Out Prints | **Less Than 100** | Sprays and institutional resting size are round-lot. Odd lots are noise here. |
| Ranges | the 9-row ladder above | |

> **Trap — Minimum Volume Percent.** It compares *cumulative day volume* to the
> 20-day average. At 9:35am even a very hot stock has only a small fraction of
> its daily average, so any threshold ≥100 suppresses nearly everything before
> midday — including the highest-quality opens. Leave it Off and use the
> `volume_frame` advanced query below as your "active right now" gate instead.

### Define Symbol Set

- **Exchanges / Product Type** — restrict to what you actually trade.
- **Mute After Count** — **leave off, or set high (5+).**

> **Trap — Mute After Count.** Setting this to 1 or 2 would hide exactly the
> repeat sprays you said are the point ("these can sometimes happen multiple
> times in succession"). This control is actively counterproductive for the
> spray setup.

### Conditions

| Condition | Suggested | Why |
| --- | --- | --- |
| 20 Day Average Volume | MIN 500,000 | You have to be able to get out of the fade. |
| 20 Day Average Price Range | **MAX — see below** | The most important refinement. |
| Current Price | MIN per your mandate | Exclude sub-$1 if you don't trade it. |
| Market Cap | optional MIN | Excludes micro-cap noise. |

> **The price ladder alone is not enough.** A $10 stock whose normal daily range
> is $2.00 prints $0.10 moves constantly — that is noise, not a spray. A $10
> stock whose daily range is $0.30 spraying $0.10 is a genuine event. Price
> tiering cannot tell these apart; **20 Day Average Price Range** can. This is
> the single highest-value addition to the filter, and it's the formal version of
> your own "depending on liquidity as well" caveat.
>
> Two practical options:
> - **One filter:** cap 20 Day Average Price Range so wild movers are excluded.
> - **Two variants:** a "calm names" copy (low ADR cap, ladder as specified) and
>   a "volatile names" copy (high ADR, thresholds raised ~2×). This is the more
>   accurate approach if you're willing to run two windows.

### Input Advanced Queries

Expressions confirmed to exist from the KB docs:

```
volume_frame(60, 0) > 50000
```
Previous minute's volume — a far better "something is happening now" gate than
Minimum Volume Percent, because it isn't cumulative-since-open.

```
&& pctChng(open,last) < 10 && pctChng(open,last) > -10
```
Optional: restricts to names not already extended on the day, matching your
"stable $10 stock" framing. A spray in a name already up 30% is a different
trade with different reversion odds.

> **Unverified.** The ideal query normalizes the move against the symbol's own
> typical range — conceptually `move / price_range_sma(20)`. `price_range_sma(20)`
> is documented, but I have **no confirmation that the current move size is
> exposed as an AQ variable**. Search the AQ metric box for "move", "spread",
> "tick", and "range" to find out. If such a variable exists it is strictly
> better than the price ladder and should replace most of it.

### Set Results Preferences

This is where one filter separates the two setups. MAT colors Primary and
Secondary moves independently, so:

| Alert | Reads as | Suggested color |
| --- | --- | --- |
| Primary only, no follow-through | **Guy candidate** — single displacement that holds | bright green / bright red |
| Primary **+** Secondary | **Spray** — sweeping, possibly in succession | dark green / dark red |

Flash New Entries **On**, duration **2 seconds** — matching the life of the
signal.

> **Caveat.** The classification is retrospective by a second or two: MAT fires on
> Primary immediately and cannot know whether a Secondary will follow. The colors
> tell you what happened, not what is about to. Treat every Primary alert as
> "look now", and let the tape decide which setup it is.

## If you'd rather run two filters

Clone the base filter and change only these:

| | Spray filter | Guy filter |
| --- | --- | --- |
| Time Frame | 2s | 3–5s (displacement plus the hold) |
| Secondary Move | as laddered — you *want* continuation | blank / 0 — you want the single displacement |
| Mute After Count | off | 1–2 |
| Primary Move | as laddered | ladder × ~0.8 (a guy displacement is often smaller than a full sweep) |

The cost of two filters is that the guy filter is strictly looser and will
re-catch every spray's first leg. There is no MAT setting that excludes it. This
is why the one-filter, two-color build above is the better default.

## Limitations

Read these before trusting the output.

> **These limitations apply to MAT itself.** The SHEL DataGateway API
> ([`../shel-datagateway/`](../shel-datagateway/README.md)) provides NBBO and can
> address several of them outside MAT — see
> [Building this outside MAT](#building-this-outside-mat) below.

**1. MAT sees prints, not quotes.** Both setups are defined by book behavior —
sweeping through levels, and resting size away from the inside. MAT's Price Moves
engine works on trades. Every filter here detects the *price footprint* the
behavior leaves, never the behavior itself. Confirmation requires L2 / time &
sales. The filter narrows the universe; it does not identify the setup.

**2. The guy setup has a structural blind spot.** Your edge is hitting the
elevated bid before others. But MAT can only fire once prints have occurred —
meaning someone already traded against it. Worse, if the guy's bid sits above the
last trade but still below the offer, it may generate *no prints at all*, and MAT
stays silent for as long as the opportunity is cleanest. **MAT is inherently
somewhat late to guy high/low, and blind to its best case.** A quote-driven alert
would be the right tool; this is the closest available proxy.

**3. Sprays revert unevenly.** Fading assumes the displacement is liquidity-driven
rather than information-driven. The same footprint appears ahead of real news. The
`pctChng` guard and an ADR cap help; neither is sufficient. Halts and news events
warrant a hard exclusion you'll have to apply manually.

**4. Thresholds are analytic, not empirical.** The ladder interpolates two anchor
points. It has never been tested against a print tape.

## Building this outside MAT

The SHEL DataGateway API changes what's feasible. See
[`../shel-datagateway/README.md`](../shel-datagateway/README.md).

### Guy high / guy low becomes directly detectable

Limitation 2 above — MAT's structural blind spot — exists because MAT reads
prints. The `nbbo` subscription reads the quote, which is exactly where the setup
lives. A resting bid above the recent market is visible **before** anyone trades
against it, which is the moment your edge actually exists.

Sketch of the detector, from `nbbo` messages alone:

```
guy_high  when  bid          > recent_reference + threshold(price)
           and  bid-size     >= size_floor
           and  the condition holds for >= N milliseconds
```

with `guy_low` the mirror on `ask-price` / `ask` (mind the asymmetric field
names — see the gotcha in `../shel-datagateway/response-messages.md`).

`recent_reference` wants to be a short trailing VWAP or mid — `vwma-1s` is
available as a subscription, or compute it from the trade stream. The
`threshold(price)` function is the √price ladder already derived above, and
`hist-stat`'s `20D` high/low gives the per-symbol volatility normalization for
free on every subscription.

This is a genuinely better instrument than the MAT proxy: it fires on the
resting order rather than on the prints that follow it.

### Sprays get a real definition

With trades carrying `mkt` and `flags`, a spray can be defined as what it
actually is — one aggressor crossing multiple price levels on the lit book —
rather than inferred from net displacement:

- exclude `Drk` and `OffMkt*` prints (they never touched the displayed book)
- exclude `Odd` (already excluded by default) and `OOS`
- require **N distinct price levels** in monotonic order inside the window
- require the aggregate size across those prints to clear the notional floor

The distinct-levels requirement is the piece MAT cannot express at all, and it is
the difference between "price moved" and "someone swept the book".

### The constraint that should drive scheduling

**`nbbo` is not available historically.** Trades go back; quotes do not.

So: sprays can be backtested today from historical trades, but **guy high/low can
only ever be studied on quote data you have already recorded.** Every day that
passes without a recorder running is a day permanently unavailable for this
research. If the guy setup matters, standing up an NBBO recorder is the
time-sensitive task here — ahead of any modeling work, which can be done later on
whatever has accumulated.

## Calibration

The repo's scanner (`data/scanner.py`) works off Polygon **grouped daily bars**,
so it cannot validate any of this — these are sub-second, trade-level events.
DataGateway's historical `trade` subscription supplies exactly what's needed,
with nanosecond timestamps.

A defensible calibration loop:

1. Pull raw trades for ~20 symbols spanning the price ladder, several sessions,
   via `request_data(..., ['trade'])`. Filter out `Drk`, `OffMkt*`, and `OOS`.
2. Resample to 1s; compute rolling 2s displacement and the volume behind it.
3. Histogram displacement per price bucket. The threshold worth using is a *tail
   quantile* (~99.5th percentile of 2s moves), not a fixed cent value — that
   automatically adapts per symbol and replaces the ladder.
4. Label forward returns at +5s / +15s / +30s to measure whether the fade
   actually pays, and at what horizon.
5. Compare the empirical per-bucket thresholds against the √price ladder. Where
   they diverge, trust the data.

Step 3 is the real prize: a per-symbol tail quantile is strictly better than any
static ladder, and it subsumes both the price tiering and the ADR normalization
in one number.
