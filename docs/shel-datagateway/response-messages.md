# SHEL DataGateway — Response Message Specification

Messages are a newline-delimited stream of JSON objects (NDJSON / JSONLines).
Unknown or unset fields are `null`.

- The `symbol` field is on every message for convenience, but the spec
  recommends a **separate callback per request** so you never need to look it up.
- Timestamps are **Unix nanoseconds**: `datetime.fromtimestamp(timestamp / 1E9)`.
- Market identification codes are **ISO MIC codes** (`XNAS`, `ARCX`, `EDGX`,
  `IEXG`, `FINN`, `FINY`, `OOTC`, …).

## Trade

```json
{
  "type": "trade",
  "symbol": "AAPL",
  "time": 1702669702420156700,
  "price": 123.123,
  "size": 100,
  "mkt": "XNAS",
  "flags": "L HL V",
  "snapshot": true
}
```

`snapshot: true` marks the last trade, sent on initial subscription (omitted if
false).

### Flag definitions

Flags carry special information derived from trade conditions and other
calculations. Encoded as a **space-delimited string**. The first four can be used
to maintain EOD statistics on the fly.

| Flag | Description |
| --- | --- |
| `L` | Eligible Last |
| `V` | Eligible Volume |
| `HL` | Eligible High Low |
| `O` | Opening |
| `Rprt` | Official Price Report |
| `XTH` | Extended Hours |
| `Odd` | Odd Lot |
| `TTE` | Trade Through Exempt |
| `OOS` | Out Of Sequence |
| `Drk` | Dark Pool |
| `OffMkt` | Off Market |

> **Gotcha — numbered flag variants.** The SDK sample output contains
> `'flags': 'L V HL TTE OffMkt1'` — note `OffMkt1`, not `OffMkt`. The flag table
> documents no numeric suffixes. **Match flags by prefix or with a tolerant
> parser**, never by exact string equality against the table, or you will miss
> off-market prints.

> **Filtering guidance.** For any lit-book analysis, exclude `Drk` and `OffMkt*`
> — those prints did not interact with the displayed book. In the AAPL sample the
> large majority of prints are `FINN`/`Drk`, so this is not a marginal filter.
> `XTH` separates extended hours; `OOS` marks out-of-sequence prints that should
> not drive sequential logic.

## End Of Day Statistics

```json
{
  "type": "eod",
  "symbol": "AAPL",
  "last": 123.123,
  "open": 123.123,
  "open-time": 1702669702420156700,
  "close": 123.123,
  "close-time": 1702669702420156700,
  "high": 123.123,
  "low": 123.123,
  "premkt-high": 123.123,
  "premkt-low": 123.123,
  "postmkt-high": 123.123,
  "postmkt-low": 123.123,
  "volume": 123456,
  "snapshot": true
}
```

> **Spec vs. reality.** The SDK's actual output uses **`pm-high` / `pm-low`**,
> not `premkt-high` / `premkt-low`, and includes a **`vwap`** field the spec does
> not document. No `postmkt-*` fields appear in any sample. Code against the
> observed names, defensively.

## Previous End Of Day Statistics

```json
{
  "type": "prev-eod",
  "symbol": "AAPL",
  "last": 123.123,
  "open": 123.123,
  "open-time": 1702669702420156700,
  "close": 123.123,
  "close-time": 1702669702420156700,
  "high": 123.123,
  "low": 123.123,
  "volume": 123456
}
```

> **Spec vs. reality.** Observed `prev-eod` messages **omit `last` and
> `open-time`** and **add `vwap`**.

## Historical Statistics

```json
{
  "type": "hist-stat",
  "symbol": "AAPL",
  "10D":  { "high": 123.123, "low": 123.123, "SMA": 123.123, "pct-chg": 1.23 },
  "20D":  { "..." },
  "50D":  { "..." },
  "200D": { "..." },
  "52Wk": { "..." }
}
```

Sent first on every stream subscription. Note `SMA` is capitalized while every
other key is lowercase.

This message is the cheapest source of per-symbol normalization data — the `20D`
high/low span is an ADR proxy available at subscription time with no extra
request.

## VWMA Indicator

```json
{
  "type": "vwma-1s",
  "symbol": "AAPL",
  "time": 1702669702420156700,
  "value": 123.123
}
```

Documented types: `vwma-` + `1s`, `5s`, `15s`, `30s`, `45s`, `1min`, `2min`,
`3min`, `5min`, `7min`, `10min`, `15min`, `30min`, `1h`.

The interval is the **period parameter** (rolling window length). Updates are
disseminated at a fine interval (~1s) **regardless of the period chosen** — so a
`vwma-30min` still emits roughly every second.

## Bar

```json
{
  "type": "bar-1s",
  "symbol": "AAPL",
  "close-time": 1702669808000000000,
  "open": 123.123,
  "high": 123.123,
  "low": 123.123,
  "close": 123.123,
  "volume": 123456,
  "vwap": 123.123,
  "preview": true,
  "snapshot": true
}
```

Documented types: `bar-` + `1s`, `5s`, `15s`, `30s`, `45s`, `1min`, `2min`,
`3min`, `5min`, `7min`, `10min`, `15min`, `30min`, `1h`, `1d`.

- The bar is identified by **`close-time`, which is exclusive** — it is the
  opening timestamp of the *next* bar. There is no open-time field.
- `preview: true` means "what the unfinished bar looks like right now". When the
  bar completes, an update arrives with `preview` omitted.

> **Preview traffic is heavy.** Because previews fire on every update, *frequent
> messages arrive no matter how coarse the interval*. The spec says this feature
> "may be eliminated if not needed". Do not subscribe with
> `bar.include-previews` unless you genuinely need intra-bar state, and never
> treat a preview as a closed bar — dedupe on `close-time` and keep only the
> message with `preview` absent.

## NBBO

```json
{
  "type": "nbbo",
  "symbol": "AAPL",
  "time": 1702669702420156700,
  "bid": 123.123,
  "bid-size": 100,
  "bid-mkt": "XNAS",
  "ask-price": 123.123,
  "ask": 100,
  "ask-mkt": "XNAS",
  "snapshot": true
}
```

> **Gotcha — the bid and ask fields are not named symmetrically.** On the bid
> side, `bid` is the *price* and `bid-size` is the *size*. On the ask side,
> **`ask-price` is the price and `ask` is the size.** Any code that assumes
> symmetry (`bid`/`ask` both prices) will read a size as a price and produce
> nonsense spreads. This looks like a spec defect; confirm against live data
> before building on it.

Live subscription only — see the historical-availability warning in
[`market-data-api.md`](market-data-api.md).
