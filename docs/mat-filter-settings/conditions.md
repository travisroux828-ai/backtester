# Conditions (MAT)

> Source last updated: 22 August 2025 · 4 min read

## About

Conditions allow the user to filter MAT window symbols using various market
metrics, including Price Range, Average Volume, Market Capitalization, Shares
Outstanding, and other key parameters.

## Available Conditions

### 20 Day Average Price Range

The average daily high–low price range over the past 20 trading sessions.

**Configuration:** Adjust the slider or manually enter the minimum and maximum
values. The histogram displays the distribution of prices and the number of
matching symbols within the configured range.

**Example:** min = 0.00, max = 5.00 returns stocks with a 20-day average price
range ≤ $5.

### 20 Day Average Volume

The average daily trading volume over the past 20 trading sessions.

**Configuration:** Adjust the slider or manually enter the minimum and maximum
values. The histogram displays the distribution of volume and the number of
matching symbols within the configured range.

**Example:** min = 0.00, max = 100,000 returns stocks with a 20-day average
volume ≤ 100,000 shares.

### 50 Day Average Volume

The average daily trading volume over the past 50 trading sessions.

**Configuration:** Adjust the slider or manually enter the minimum and maximum
values. The histogram displays the distribution of volume and the number of
matching symbols within the configured range.

**Example:** min = 0.00, max = 100,000 returns stocks with a 50-day average
volume ≤ 100,000 shares.

### Daily Price Range

The Daily Price Range (Previous Session) is the difference between the high and
low price for the previous trading day.

**Configuration:** Adjust the slider or manually enter the minimum and maximum
values. The histogram displays the distribution of prices and the number of
matching symbols within the configured range.

**Example:** min = 0.00, max = 5.00 returns stocks with a price range ≤ $5.

### GICS Level

GICS is a four-tiered, hierarchical industry classification system. The four
tiers are: Sectors, Industry Groups, Industries and Sub-Industries.

**Configuration:** Enter a symbol into the search bar to prepopulate the GICS
Levels, or manually select the Levels using the dropdown menus.

**Example:** Selecting "Consumer Staples" (Level 1) and "Food, Staples
Retailing" (Level 2) filters results to display only companies with this exact
hierarchical classification.

### Market Cap

A company's total value (shares outstanding × current price).

**Configuration:** Adjust the slider or manually enter the minimum and maximum
values. The histogram displays the market cap distribution and the number of
matching symbols within the configured range. Accepted inputs include `100`;
`1,000` or `1K`; `1,000,000` or `1M`; `10,000,000,000` or `10B`.

**Example:** min = 10B, max = 100B returns stocks with a market cap between $10
billion and $100 billion.

### Shares Outstanding

The total stock held by all shareholders, including institutional blocks and
restricted insider shares.

**Configuration:** Adjust the slider or manually enter the minimum and maximum
values. The histogram displays the shares outstanding distribution and the
number of matching symbols within the configured range. Accepted inputs include
`100`; `1,000` or `1K`; `1,000,000` or `1M`; `10,000,000,000` or `10B`.

**Example:** min = 10M, max = 100M returns stocks with shares outstanding
between 10 million and 100 million.

### Current Price

The price of the last execution.

**Configuration:** Adjust the slider or manually enter the minimum and maximum
values. The histogram displays the price distribution and the number of matching
symbols within the configured range.

**Example:** min = 5, max = 20 returns symbols with prices between $5 and $20.

### Current Volume

The symbol's total volume.

**Configuration:** Manually enter the minimum and maximum values. Accepted
inputs include `100`; `1,000` or `1K`; `1,000,000` or `1M`;
`10,000,000,000` or `10B`.

**Example:** min = 100000, max = 1000000 returns symbols with total volume
between 100,000 and 1,000,000 shares.

## Adding a New Condition

To add a new condition, click the blue **+** icon and select **Add Condition**.
Select the Condition and click **Add**.

## Configuring

To set the totality of available symbols, input the Min and Max values or adjust
the range using the slider. Click **Apply** to save.
