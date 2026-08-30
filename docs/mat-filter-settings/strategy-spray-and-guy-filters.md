# MAT Filter Design: Sprays and Guy High / Guy Low

Working design for two desk setups, built **entirely within MAT's Filter Settings
window**. Status: **unvalidated draft** — thresholds are analytic, not fitted.

> **Scope.** This is a MAT filter, configured in the platform UI. It is a
> self-contained deliverable. A separate, independent design exists for detecting
> the same setups by building an application against the SHEL DataGateway API
> ([`../shel-datagateway/detector-spray-and-guy.md`](../shel-datagateway/detector-spray-and-guy.md)).
> **The two do not interoperate** — different systems, no shared runtime, no data
> path between them. You build one or the other. The only thing they have in
> common is the move-sizing logic in
> [`../noticeable-move-ladder.md`](../noticeable-move-ladder.md), which is a fact
> about markets rather than about either tool.

## The setups

| | Spray | Guy high / guy low |
| --- | --- | --- |
| **Mechanism** | An aggressor uses enough size to sweep through multiple book levels, printing a run of ticks in one direction. Can repeat in succession. | A bid rests noticeably *above* the book (guy high) or an offer rests noticeably *below* it (guy low), with size, holding price away from fair for a short time. |
| **Aggressor** | Taker — crosses the spread repeatedly | Maker — posts away from the inside |
| **Your action** | Post offers into a spray up / bids into a spray down | Hit the elevated bid (guy high) / take the depressed offer (guy low) |
| **Trade direction** | Fade | Fade |

### Why one filter, not two

Both are the same *detectable* event as far as MAT is concerned: a fast,
size-backed displacement away from recent price, which you then fade. They differ
in mechanism (aggressive sweep vs. passive resting size) and in execution (post
vs. take) — but MAT reads prints, and the print footprint of the two is nearly
identical.

MAT has no field that separates them. So the right build is **one filter**, with
the two setups distinguished visually at the point of alert rather than by
configuration. See [Set Results Preferences](#set-results-preferences).

## Move thresholds

The full derivation and the bucketed table live in
[`../noticeable-move-ladder.md`](../noticeable-move-ladder.md). Summary: the
desk's own two anchors ($10 → $0.10, $900 → $1.00) fit **square-root price
scaling** (k = 0.51), giving `move ≈ √price / 31.6`.

### Price Move ladder — paste into Add Price Moves

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

> Keep Move Volume MIN at 100 or above — anything lower is unreachable once
> Filter Out Prints is set to "Less Than 100".

## The build

Name it something like **Displacement Fade**.

### Add Price Moves

| Setting | Value | Why |
| --- | --- | --- |
| Time Frame | **2 seconds** | Both setups are near-instant. Longer windows admit grinding moves, which are not the trade. Try 3s if 2s is too sparse. |
| Minimum Volume Percent | **Off** | See the trap below. |
| Filter Out Prints | **Less Than 100** | Sprays and institutional resting size are round-lot. |
| Ranges | the 9-row ladder above | |

> **Trap — Minimum Volume Percent.** It compares *cumulative day volume* to the
> 20-day average. At 9:35am even a very hot stock has only a small fraction of its
> daily average, so any threshold ≥100 suppresses nearly everything before midday
> — including the best opens. Leave it Off and use the `volume_frame` advanced
> query below instead.

### Define Symbol Set

- **Exchanges / Product Type** — restrict to what you actually trade.
- **Mute After Count** — **leave off, or set high (5+).**

> **Trap — Mute After Count.** Set to 1 or 2 it hides exactly the repeat sprays
> that are the point of the setup. This control is actively counterproductive here.

### Conditions

| Condition | Suggested | Why |
| --- | --- | --- |
| 20 Day Average Volume | MIN 500,000 | You have to be able to get out of the fade. |
| 20 Day Average Price Range | **cap — see below** | The most important refinement. |
| Current Price | MIN per your mandate | Exclude sub-$1 if you don't trade it. |
| Market Cap | optional MIN | Excludes micro-cap noise. |

> **The price ladder alone is not enough.** A $10 stock whose normal daily range
> is $2.00 prints $0.10 moves constantly — noise. One whose range is $0.30 spraying
> $0.10 is an event. Price tiering can't separate them; **20 Day Average Price
> Range** can. This is the formal version of the "depending on liquidity" caveat.
>
> Either cap it in one filter, or run two variants — "calm names" (low ADR cap,
> ladder as specified) and "volatile names" (high ADR, thresholds ~2×). Two
> variants is more accurate if you'll run two windows.

### Input Advanced Queries

Expressions confirmed present in the KB docs:

```
volume_frame(60, 0) > 50000
```
Previous minute's volume — a better "something is happening now" gate than
Minimum Volume Percent, because it isn't cumulative since the open.

```
&& pctChng(open,last) < 10 && pctChng(open,last) > -10
```
Optional: restricts to names not already extended on the day. A spray in a name
already up 30% is a different trade with different reversion odds.

> **Unverified.** The ideal query normalizes the move against the symbol's own
> typical range — conceptually `move / price_range_sma(20)`. `price_range_sma(20)`
> is documented, but there is **no confirmation the current move size is exposed
> as an AQ variable**. Search the AQ metric box for "move", "spread", "tick",
> "range". If such a variable exists it beats the price ladder outright.

### Set Results Preferences

This is where one filter separates the two setups. MAT colors Primary and
Secondary moves independently:

| Alert | Reads as | Suggested color |
| --- | --- | --- |
| Primary only, no follow-through | **Guy candidate** — single displacement that holds | bright green / bright red |
| Primary **+** Secondary | **Spray** — sweeping, possibly in succession | dark green / dark red |

Flash New Entries **On**, duration **2 seconds**.

> **Caveat.** The classification is retrospective by a second or two: MAT fires on
> Primary immediately and cannot know whether a Secondary will follow. The colors
> tell you what happened, not what is about to. Treat every Primary alert as "look
> now" and let the tape decide.

## If you'd rather run two filters

Clone the base filter and change only these:

| | Spray filter | Guy filter |
| --- | --- | --- |
| Time Frame | 2s | 3–5s (displacement plus the hold) |
| Secondary Move | as laddered — you *want* continuation | blank / 0 |
| Mute After Count | off | 1–2 |
| Primary Move | as laddered | ladder × ~0.8 |

The cost is that the guy filter is strictly looser and re-catches every spray's
first leg. No MAT setting excludes it. That's why the one-filter, two-color build
is the better default.

## Limitations

These are **properties of MAT**, not gaps waiting to be filled. MAT's Price Moves
engine reads prints; that is what it is. Nothing configured in this window changes
any of the following.

**1. MAT sees prints, not quotes.** Both setups are defined by book behavior —
sweeping through levels, and resting size away from the inside. Every filter here
detects the *price footprint* the behavior leaves, never the behavior itself.
Confirmation requires L2 / time & sales. The filter narrows the universe; it does
not identify the setup.

**2. The guy setup has a structural blind spot.** Your edge is hitting the
elevated bid before others, but MAT can only fire once prints have occurred —
meaning someone already traded against it. Worse, if the guy's bid sits above the
last trade but below the offer, it may generate *no prints at all*, and MAT stays
silent for as long as the opportunity is cleanest. **MAT is inherently late to
guy high/low and blind to its best case.** Accept this as the cost of using MAT
for this setup; a quote-driven instrument is a different system entirely.

**3. MAT cannot count price levels.** A spray is one aggressor crossing several
levels. MAT measures net displacement over a window, which a single large print
at one price also produces. There is no field for "levels crossed".

**4. Sprays revert unevenly.** Fading assumes the displacement is liquidity-driven
rather than information-driven. The same footprint appears ahead of real news. The
`pctChng` guard and an ADR cap help; neither is sufficient. Halts and news warrant
a manual exclusion.

**5. Thresholds are analytic, not empirical.** The ladder interpolates two anchor
points and has never been tested against a print tape. There is no way to backtest
a MAT filter from inside MAT — validation means either running it live and keeping
a log of what it fires on, or reproducing its logic against historical trade data
in a separate system.
