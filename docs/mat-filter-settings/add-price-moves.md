# Add Price Moves (MAT)

> Source last updated: ~10 days before 2026-08-30 · 4 min read

## About

Add Price Moves allows the user to configure the selected filter's Time Frame,
Minimum Volume Percent, Odd Lots setting, Price Ranges, Primary/Secondary Moves,
and Move Volumes.

## Time Frame (in Seconds)

Time Frame defines the rolling window the MAT filter uses to evaluate symbols
against filter criteria. To set it, click in the text box and enter the time in
seconds.

**Example:** A Time Frame of 10 seconds sets a rolling 10-second period to
identify symbols satisfying the specified filter criteria.

## Minimum Volume Percent

Minimum Volume Percent compares the symbol's current volume to its 20-day
average volume. Toggle the button **On** and enter the percentage as a whole
number.

**Example:** A Minimum Volume Percent of 110 filters for symbols with volume
greater than 110% of their 20-day average volume.

## Filter Out Prints

> **Version note:** the live UI screenshot on hand shows this control as a
> simple **Odd Lots** On/Off toggle, not a three-option dropdown. The dropdown
> described below is the newer form documented in the KB. Confirm which build
> you are targeting before modeling this field.

The Filter Out Prints dropdown excludes certain types of prints from the
filter's results. Three options:

- **None** — no filtering applied
- **Odd Lots** — filters out odd lots, based on the latest Round Lot definition
- **Less Than 100** — filters out any result with fewer than 100 shares in size

## Covered Ranges

Covered Ranges visually displays the filter's price ranges and highlights
overlapping areas. Hover over the line to view the ranges and overlaps.

To add a price move using Covered Ranges, hover over an empty graph area and
click **New Price Move**.

> **Note:** Input Primary and Secondary Price Moves using dollar or percentage
> formats.

## Ranges

Ranges display in price order (lowest to highest) with options to add Price
Ranges, Primary and Secondary Moves, and Volume criteria.

> **Note:** Add multiple tiered ranges for a more focused filter.

To add a new Range, click the blue **+** icon and select **Add Price Move**.

### Configuring Range Options

Use the text boxes to enter the Price Range, Primary/Secondary Move, Move
Volume, and 20 Day Volume. Symbols meeting the Time Frame, Minimum Volume
Percent settings, and the specified price movement criteria will be displayed in
the MAT window.

> **Note:** Accepted Volume input syntax — `100`; `1,000` or `1K`; `1,000,000`
> or `1M`; `10,000,000,000` or `10B`.
>
> **Note:** Range options are secondary filters applied *after* the Time Frame
> and Minimum Volume Percent criteria are met.

### Range Options

| Option | Description | Example |
| --- | --- | --- |
| **Price Range** | Sets the Min and Max price limits for the range. | Min = 0, Max = 5 creates a price range for symbols between $0 and $5. |
| **Primary Move** | Sets the Primary Price Move criteria for the range (supports $ and % values). | Primary move = .25 filters for stocks with price movements of .25 cents or greater. |
| **Secondary Move** | Sets the Secondary Price Move criteria for the range (supports $ and % values). | Secondary move = .10 filters for stocks with an additional .10 cent price movement following the Primary Move criteria. |
| **Move Volume** | Sets the Min and Max Move Volume limits for the range. Enter contract or share quantity. | Min = 10,000, Max = 100,000 creates a move volume range of 10,000 to 100,000 shares. |
| **20 Day Volume** | Sets the Min and Max 20-Day Average Volume limits for the range. Enter contract or share quantity. | Min = 1M, Max = 10M creates a 20-Day Average Volume range of 1 million to 10 million shares. |

## Copying Price Moves

To copy a Price Move, expand a range and click the copy button at the
bottom-right.

> **Note:** When a Price Move is duplicated, its range is automatically aligned
> with adjacent Price Moves: the Minimum Price Range is set to the source's
> Maximum, and, if a subsequent Price Move exists, the Maximum Price Range is
> set to the next move's Minimum if one is available.

**Example** — configuration with:

- Price Move 1: $1 – $100
- Price Move 2: $500 – $1000

If Price Move 1 is duplicated:

- The new Price Move is inserted between Price Move 1 and Price Move 2.
- The system assigns a Minimum Price Range of $100 (from Price Move 1's maximum).
- The system assigns a Maximum Price Range of $500 (from Price Move 2's minimum).

## Deleting Price Moves

To delete a Price Move, expand a Price Move and click the trashcan icon at the
bottom-right.

## Panel layout (observed UI)

Collapsible section titled **Add Price Moves** with a chevron at the right.
Top-level fields, each label at left and control at right:

| Control | Type | Observed default |
| --- | --- | --- |
| `Time Frame (in seconds):` | text box | `10` |
| `Minimum Volume Percent` | toggle + text box | `Off`, `0` |
| `Odd Lots:` | toggle | `Off` |
| `Covered Ranges:` | horizontal bar, `MIN` … `MAX` | blue = covered, red/orange = overlap |

Below Covered Ranges, four column headers span the range list:
**Price Range · Price Moves · Move Volume · 20d Volume**.

Each range renders as a collapsible card (`Range 1`, `Range 2`, …) with a
chevron. Expanded, a card shows stacked rows:

| Row | Inputs | Example values |
| --- | --- | --- |
| `Price Range:` | two boxes (min, max) | `0.01`, `1.00` |
| `Primary Move:` | one box (right-aligned) | `$0.20` |
| `Secondary Move:` | one box (right-aligned) | `$0.05` |
| `Move Volume:` | two boxes | placeholders `MIN` / `MAX` |
| `20 Day Volume:` | two boxes | placeholders `MIN` / `MAX` |

Bottom-right of an expanded card: a blue **copy** icon and a red **trashcan**
icon. A round blue **+** floating button sits at the bottom-right of the panel.

Note the asymmetry: Price Range, Move Volume, and 20 Day Volume take min/max
pairs; Primary and Secondary Move each take a single threshold value.
