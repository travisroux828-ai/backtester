# Input Advanced Queries (MAT)

> Source last updated: 14 October 2025 · 3 min read

## About

Input Advanced Query (MAT) allows traders to create custom filters for a MAT
(Price Moves) window using logical expressions, combining Boolean and Comparison
operators for greater precision. Symbols appear in the filter window only if they
meet **all** AQ and Price Moves conditions.

In the Input Advanced Queries field, each metric is written as a separate
expression in a pseudo-code format. For example, to find stocks with a market cap
over 5,000,000 you'd use `market_cap > 5,000,000`. Some metrics require more
advanced expressions, like `volume_frame(60, 0)` for the previous minute's volume
or `price_range_sma(20)` for the 20-day average daily price.

For more information on Boolean and Comparison Operators, see
[AQ Filter Settings](https://kb.trilliumtrading.com/space/SHL/2744778757/AQ+Filter+Settings).

## Building an Advanced Query Filter

### 1. Search for a Metric

Using the bottom-left search bar, enter the name of the metric you're interested
in. As you type, a dropdown menu of available Equivalent Expressions will appear.

### 2. Select the Desired Expression

If a desired expression is found, select it. Its corresponding expression will
appear in the Equivalent Expression box. If an expression is not prepopulated,
click **See More** to open the Add Expression window.

### 3. Insert the Expression into the Query

Select the blue up arrow next to the Equivalent Expression field. This populates
the Input Advanced Queries field with your selected expression.

### 4. Apply a Comparison Operator

In the Input Advanced Query box, apply a comparison operator (e.g. `=`, `>`,
`<`) to define your condition.

**Example:** After inserting the expression `pctChg`, a specific Comparison value
must be provided. A valid condition might be `pctChng(open,last)>=10` — filtering
for stocks with a percent change from open to last price greater than or equal to
10%.

> **Note:** When using percent (pct) expressions, input the percentage as a whole
> number (e.g. `10` for 10%, or `0.10` for 0.1%).

### 5. Add More Expressions (Optional)

To combine multiple expressions:

- Repeat the steps above to add additional expressions.
- Use Boolean operators `&&` (and), `||` (or), `NOT` (not), and parentheses to
  structure complex queries.

> **Note:** Though a filter supports just one expression, practical and effective
> queries generally incorporate multiple expressions in combination.

For more information, see
[AQ Filter Settings](https://kb.trilliumtrading.com/space/SHL/2744778757).

### 6. Finalize the Query

Review the query logic for accuracy, then click **Apply** to execute the filter.
Results will begin to render in the Advanced Query window based on the entered
criteria.

> **Note:** Errors, invalid expressions, and invalid parameters are highlighted
> in red and display an explanatory tooltip when hovered over. All errors must be
> resolved in order for the query to compile.

## Expression quick reference (from examples)

| Expression | Meaning |
| --- | --- |
| `market_cap` | Market capitalization |
| `volume_frame(60, 0)` | Volume over the previous minute (60s window, 0s offset) |
| `price_range_sma(20)` | 20-day average daily price range |
| `pctChng(open,last)` | Percent change from the open to the last price |
