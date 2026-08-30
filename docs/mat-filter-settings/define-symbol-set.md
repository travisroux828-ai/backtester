# Define Symbol Set (MAT)

> Source last updated: 2 January 2026 · 3 min read

## About

Define Symbol Set allows the user to filter by Exchanges, Product Types, and
Symbols.

*(screenshot: symbol set about.jpg)*

## Define Symbol Set Output Rules

The Define Symbol Set feature works like a funnel, filtering symbols in 2 stages.

- **Stage 1** — Exchanges and Product Types filtering
- **Stage 2** — Symbol and List Include and Exclude filtering

**Example:**

- Stage 1 — Select only NYSE in the Exchange dropdown menu
- Stage 2 — Add S&P 500 List to the Include filter

The MAT filter displays S&P 500 symbols listed on the NYSE that meet the
filter's price, volume, and advanced query requirements. S&P 500 symbols listed
on the Nasdaq and other exchanges are filtered out in Stage 1.

i.e. IBM, JPM, V, and LLY will be displayed. AAPL, NVDA, and GOOGL will not be
displayed.

## Exchanges

Click the **Exchange** dropdown menu and select the Exchanges to include. Click
**Apply** to save.

## Product Type

Click the **Product Type** dropdown menu and select the Products to include.
Click **Apply** to save.

> **Version note:** the live UI screenshot on hand does not show a
> **Mute After Count** field in this panel. The KB page dates it to January
> 2026, so it is likely newer than the screenshot. Verify against your build.

## Mute After Count

To limit the number of times a symbol appears in the MAT results window, enter
the desired count in the **Mute After Count** field.

**Example:** Inputting 1 returns each symbol only once, whereas inputting 2
returns up to 2 of the same symbol before muting.

## MAT Include / Exclude

The MAT Include and Exclude feature filters the MAT window using individual
symbols and symbol lists.

- **Include** — the MAT filter will only display symbols added to the Include filter.
- **Exclude** — the MAT filter will exclude symbols added to the Exclude filter.

### Including / Excluding Lists

Click the **Include/Exclude** tab and select a List from the **Select List(s)**
dropdown menu. Check the List checkbox to include or exclude it, then click
**Apply** to save.

To edit the List in List Editor, click the List Name. Click **Apply** to save.

### Including / Excluding Symbols

Click the **Include/Exclude** tab, input the symbol in the **Symbols** text box,
press **Enter**, and click **Apply** to save.

### Searching Include / Exclude

Click the **Search Included** or **Search Excluded** bar, input the symbol, and
press **Enter**. To clear the search, click the **X** icon and press **Enter**.

### Deleting Symbols from Include / Exclude

Click on the symbol and select **Remove Selected** or **Remove All**. Confirm
the deletion and click **Apply** to save.

## Muted

The MAT Muted feature allows temporary symbol exclusion from the MAT window and
provides management of previously muted symbols.

> **Note:** Symbols added directly to the Muted list remain muted until the
> current day ends.

### Muting Symbols

Click the **Muted** tab, input the symbol in the **Symbols** text box, press
**Enter**, and click **Apply** to save.

### Deleting Symbols from Muted

Click on the symbol and select **Remove Selected** or **Remove All**. Confirm
the deletion and click **Apply** to save.

## Panel layout (observed UI)

Collapsible section titled **Define Symbol Set** with a chevron at the right.

Two dropdowns at the top, label left / control right, each summarizing its
selection count rather than listing values:

| Control | Type | Observed state |
| --- | --- | --- |
| `Exchanges:` | multi-select dropdown | `12 Items Selected` |
| `Product Type:` | multi-select dropdown | `11 Items Selected` |

Below them, three tabs: **Include · Exclude · Muted** (Include active by
default). The tab body is split into two columns:

- **Left column** — `List:` with a `Select List(s)` dropdown, and `Symbols:`
  with an `Enter symbol to include` text box.
- **Right column** — a `Search Included` search box above a table with columns
  **Symbol/List** and **Date Added**.

The search box placeholder tracks the active tab (`Search Included` /
`Search Excluded`). A round blue **+** floating button sits at the bottom-right.

The `Date Added` column means include/exclude entries carry a timestamp — worth
preserving if you model this state.
