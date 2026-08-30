# SHEL DataGateway Market Data API — Request Specification

All fields are required unless otherwise specified. Symbology is whatever
traders currently use in their Trillium trading tools.

The `token` is acquired by authenticating against the **SHEL API for
DataGateway**.

## Request Live Stream

```json
{
  "id":               <integer>,
  "action":           "request",
  "what":             "shel-datagateway-stream",
  "symbol":           <string>,
  "subscriptions":    [],
  "token":            <string>
}
```

Returns a stream of real-time market data updates **for the current day**.

**Response order:**

1. Historical Statistics (`hist-stat`)
2. Previous End Of Day Statistics (`prev-eod`)
3. Per subscription: a **Snapshot** message if supported (last trade, last bar,
   etc.) giving context at subscription time
4. Live update messages

## Request Historical Data

```json
{
  "id":               <integer>,
  "action":           "request",
  "what":             "shel-datagateway-data",
  "symbol":           <string>,
  "start-date":       "yyyy-mm-dd",
  "end-date":         "yyyy-mm-dd",
  "split-adjust":     <boolean>,
  "subscriptions":    [],
  "token":            <string>
}
```

Static market data for a range of days. `end-date` is **inclusive** and may
include the current day — the endpoint returns whatever has been recorded so
far. `split-adjust` specifies whether prices are adjusted for splits occurring
in the range.

## Available Subscriptions

| Subscription | Description |
| --- | --- |
| `trade` | Prints. **Excludes odd lots by default** — add the `trade.include-odd-lots` modifier if needed. |
| `eod` | Current session statistics, resent every time they change. |
| `nbbo` | National Best Bid and Ask. **Live stream only — not available historically.** |
| `bar-<interval>` | Bars of the specified interval, e.g. `bar-1s`, `bar-1min`. |
| `vwma-<interval>` | Volume-weighted moving average price over the rolling window given by the interval. Same as the VWMA indicator on SHEL Charts. |

> **`eod` is expensive.** Statistics change after almost every trade, *including
> odd lots*. The spec advises against subscribing for more than a handful of
> symbols — and notes all of it can be computed from the `trade` stream yourself.
> For any multi-symbol scanner, derive these locally instead.

> **`nbbo` has no history.** This is the single most consequential limitation of
> the API for research: quote-driven behavior cannot be backtested, only recorded
> going forward. Plan accordingly.

### Modifiers

| Modifier | Requires | Effect |
| --- | --- | --- |
| `trade.include-odd-lots` | `trade` | Includes odd lot trades |
| `bar.include-previews` | `bar-<interval>` | Includes previews of the latest incomplete bar |

## Supported Intervals

```
1s   2s   3s   5s   7s   10s  15s  30s  45s
1min 2min 3min 4min 5min 7min 10min 15min 30min
1h   1day
```

> **Gotcha.** This list includes `2s`, `3s`, `7s`, `10s`, and `4min`, but the
> per-message `type` enumerations for bars and VWMA **omit** them. Either the
> type lists are incomplete or those intervals aren't really available for those
> subscriptions. Verify before relying on them.

> **Gotcha — `1day` vs `1d`.** You request `bar-1day`, but the response messages
> come back with `"type": "bar-1d"`. Confirmed in the SDK's own example 3. Code
> that echoes the requested string back when matching response types will
> silently drop every daily bar.

## Available Markets

| Market | Sample Symbols | Data Available |
| --- | --- | --- |
| US Equities | `AAPL`, `SPY` | Trade, NBBO (live) |
| Toronto Equities | `ABX.T`, `RY.T` | Trade, NBBO (live) |
| OTC Equities | `TCEHY`, `RHHBY` | Trade, NBBO (live) |
| US Options | `SPY260701P00744000`, `NVDA260701C00197500` | **(none)** |

> US Options symbols are accepted by the symbology but **no data is available**
> for them.
