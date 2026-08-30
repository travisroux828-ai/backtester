# SHEL DataGateway — Market Data API Reference

Captured documentation for the Trillium SHEL DataGateway market data API and the
TF-Server framing protocol beneath it.

## Contents

| Doc | Covers |
| --- | --- |
| [`framing-protocol.md`](framing-protocol.md) | TCP transport, binary frame header, flags, status codes, request lifetime, cancel, out-of-band messages |
| [`market-data-api.md`](market-data-api.md) | Live-stream and historical request shapes, subscriptions, modifiers, intervals, markets |
| [`response-messages.md`](response-messages.md) | Every message type — trade, eod, prev-eod, hist-stat, vwma, bar, nbbo — plus trade flags |
| [`python-sdk.md`](python-sdk.md) | `sheldatagateway` install, `Session` usage, CLI utilities |
| `TF-Server_Framing_Protocol_Specification.pdf` | Source PDF (16 Apr 2026) |

Confluence sources (access required):
- https://trillium.atlassian.net/wiki/spaces/TF/pages/1821147191

## The 30-second model

```
TCP connection
  └─ many multiplexed streams, one per request id
       ├─ Client→Server:  newline-delimited JSON requests
       └─ Server→Client:  binary frames (10B header, 14B if compressed)
                            └─ payload: NDJSON application messages
```

Two request verbs, both taking a `symbol`, a `subscriptions` array and a `token`:

- `shel-datagateway-stream` — live, current day, never closes
- `shel-datagateway-data` — historical date range, closes with *End of request*

The Python SDK hides all framing; callbacks receive parsed dicts.

## Designs built on this API

| Doc | Covers |
| --- | --- |
| [`detector-spray-and-guy.md`](detector-spray-and-guy.md) | Standalone detectors for sprays (`trade`) and guy high/low (`nbbo`) |

> **This API and the platform's MAT filters are separate systems.** MAT is a
> filter engine configured through the platform UI; DataGateway is a market data
> feed you write your own client against. There is no data path between them —
> nothing computed from this feed can be fed into a MAT filter, and MAT cannot
> consume anything a DataGateway client produces. Anything built here is a
> separate build from anything built in MAT, even when both target the same
> trading setup.

## What this API gives you

| Capability | Availability |
| --- | --- |
| Historical prints, nanosecond timestamps, with venue and condition flags | `trade`, any date range |
| Live quote — best bid/ask and sizes | `nbbo`, **live only** |
| Historical quote | **Not available at all** |
| Per-symbol volatility context (10D/20D/50D/200D/52Wk high, low, SMA, pct-chg) | `hist-stat`, free on every subscription |
| Bars and VWMA at intervals from 1s to 1day | `bar-*`, `vwma-*` |

### The one constraint that drives planning

**NBBO is live-only.** Trades go back historically; quotes do not. Quote-driven
research can only ever be done on data already recorded. If anything you want
depends on the book, **start recording now** — every day not recorded is a day
that can never be studied.

## Gotchas worth knowing before you write a parser

Each is detailed in the relevant doc:

- **NBBO field names are asymmetric** — `bid` is a price but `ask` is a *size*;
  the ask price is `ask-price`. Assuming symmetry silently corrupts every spread.
- **`bar-1day` returns `"type": "bar-1d"`** — request and response strings differ.
- **Trade flags can carry numeric suffixes** (`OffMkt1`) not present in the flag
  table. Match by prefix.
- **`eod` uses `pm-high`/`pm-low`** in practice, not the documented
  `premkt-high`/`premkt-low`, and adds an undocumented `vwap`.
- **Compression algorithm is contradicted** between the overview (ZSTD) and the
  formal spec (undefined).
- **A successful cancel reports Status Code 2 (error)** with body type
  `CANCELED`.
- **Bar previews flood the stream** at any interval; dedupe on `close-time`.
- **`eod` subscription is costly** — updates on nearly every trade including odd
  lots. Derive session stats from `trade` instead for multi-symbol work.

These come from cross-reading the spec against the SDK's own sample output, so
where they conflict, the samples are the observed behavior and the spec is the
intent. Verify against live data before depending on either.
