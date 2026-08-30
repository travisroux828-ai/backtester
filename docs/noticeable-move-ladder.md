# Sizing a "Noticeable" Move: the √price Ladder

System-independent. This is a property of market structure, not of any
particular tool — it is used by both the MAT filter design
([`mat-filter-settings/strategy-spray-and-guy-filters.md`](mat-filter-settings/strategy-spray-and-guy-filters.md))
and the DataGateway detector design
([`shel-datagateway/detector-spray-and-guy.md`](shel-datagateway/detector-spray-and-guy.md)),
which are otherwise **separate builds that share no runtime**.

## The question

"How far does a stock have to move, over a couple of seconds, before it's worth
looking at?" A fixed cent value is wrong across prices. A fixed percentage is
also wrong — and wrong in a way that gets worse at both ends.

## Two anchor points

From the desk:

- A stable **$10** stock moving **$0.10** is interesting — 1.00% of price
- A **$900** stock needs **$1.00** or more — 0.11% of price

## The fit

Fitting `move = c · price^k` through both points:

```
k = 0.5117      c = 0.03078
```

k ≈ 0.5 — **square-root scaling**, to within measurement noise of two
eyeballed anchors:

```
move ≈ √price / 31.6
```

This matches the standard result that per-share volatility scales roughly with
the square root of price, which is a good sign the anchors reflect something
real rather than habit. Note the percentage threshold *falls* as price rises, so
a flat-percent rule is badly miscalibrated at both ends of the range.

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
| $1500 | $1.23 | 0.08% |

## Bucketed ladder

Thresholds evaluated at each bucket's geometric mean. Size floors target roughly
**$100–200k notional**, so share counts scale inversely with price.

| # | Price range | Primary move | Secondary move | Size floor (shares) |
| ---: | --- | ---: | ---: | ---: |
| 1 | 0.01 – 1.00 | $0.03 | $0.01 | 100,000 |
| 2 | 1.00 – 5.00 | $0.05 | $0.02 | 40,000 |
| 3 | 5.00 – 15.00 | $0.10 | $0.05 | 10,000 |
| 4 | 15.00 – 40.00 | $0.15 | $0.07 | 4,000 |
| 5 | 40.00 – 100.00 | $0.25 | $0.10 | 2,000 |
| 6 | 100.00 – 250.00 | $0.40 | $0.18 | 600 |
| 7 | 250.00 – 600.00 | $0.60 | $0.25 | 300 |
| 8 | 600.00 – 1200.00 | $0.90 | $0.40 | 200 |
| 9 | 1200.00 + | $1.25 | $0.55 | 100 |

"Secondary" is a continuation threshold at ~45% of primary — enough to confirm a
real follow-on leg without demanding a second full-size move.

> Bucket 1 (sub-$1) is judgment, not the formula: the $0.01 tick floor and the
> extreme percentage volatility of sub-$1 names make the fit meaningless there.
> Drop the bucket entirely if you don't trade sub-$1.

## What the ladder does not capture

Price is a proxy for volatility, and a coarse one. A $10 stock whose normal daily
range is $2.00 prints $0.10 moves constantly — noise. A $10 stock whose daily
range is $0.30 moving $0.10 is a genuine event. **The ladder cannot tell these
apart.**

Correcting for it needs a per-symbol volatility measure:

- In MAT: the **20 Day Average Price Range** condition, used as a cap or to split
  the filter into calm/volatile variants.
- Against the API: `hist-stat`'s `20D` high/low span, which arrives free on every
  subscription.

The better long-run answer is to drop the ladder entirely in favour of a
**per-symbol tail quantile** — the ~99.5th percentile of that symbol's own N-second
moves. That is self-normalizing and subsumes both the price tiering and the
volatility correction in one number. It requires tick data to compute, so the
ladder is the interim.

## Status

**Unvalidated.** Two anchor points, interpolated. Never tested against a tape.
Treat every number here as a starting point for calibration, not a result.
