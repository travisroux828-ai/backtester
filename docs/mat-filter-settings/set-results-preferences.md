# Set Results Preferences (MAT)

> Source last updated: 7 August 2025 · 1 min read

## About

Set Results Preferences allows the user to customize how symbols appear in the
MAT window and configure personalized notifications.

## Flash New Entries

To temporarily highlight new results in the MAT window, toggle **Flash New
Entries** On. Select a color and enter the duration in seconds. Click **Apply**
to save.

## Up / Down Price Moves

- **Primary text color** — click the dropdown menus and select the colors for the
  MAT filter's Primary Up/Down Price Move text.
- **Primary background color** — click the dropdown menus and select the colors
  for the MAT filter's Primary Up/Down Price Move background.
- **Secondary text color** — click the dropdown menus and select the colors for
  the MAT filter's Secondary Up/Down Price Move text.
- **Secondary background color** — click the dropdown menus and select the colors
  for the MAT filter's Secondary Up/Down Price Move background.

These preferences affect presentation only — they do not filter results.

## Panel layout (observed UI)

Collapsible section titled **Set Results Preferences (Optional)** with a chevron
at the right — the panel is explicitly marked optional.

**Flash New Entries** row: a color swatch dropdown (observed yellow), a duration
text box (observed `3`, seconds), and an On/Off toggle (observed `Off`).

Below a divider, the price-move colors are laid out as a grid with three
right-hand columns headed **Preview · Text · Background**:

| Group | Row | Preview | Text | Background |
| --- | --- | --- | --- | --- |
| `Up Price Moves:` | `Primary:` | `AAPL` bright green | bright green | black |
| | `Secondary:` | `AAPL` dark green | dark green | black |
| `Down Price Moves:` | `Primary:` | `AAPL` bright red | bright red | black |
| | `Secondary:` | `AAPL` dark red | dark red | black |

The `Preview` column renders a live sample ticker in the chosen colors. The
default convention is bright for Primary moves and a darker shade of the same
hue for Secondary, on a black background — so Primary reads as louder than
Secondary at a glance.
