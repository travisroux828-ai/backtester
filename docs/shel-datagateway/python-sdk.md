# SHEL DataGateway — Python SDK

## Installation

Download both packages into the same directory:

- `labs_journal-1.1.3-py3-none-any.whl`
- `shel_datagateway-1.0.0-py3-none-any.whl`

Then, from that directory:

```bash
pip install shel_datagateway-1.0.0-py3-none-any.whl --find-links .
```

Installs for the current user.

> Note the import name is `sheldatagateway` (no underscores), while the wheel is
> `shel_datagateway`.

## Live stream

```python
#!/usr/bin/env python
import sheldatagateway
from sheldatagateway import environments
import getpass

with sheldatagateway.Session(environments.env_defs.Prod, 'username', getpass.getpass()) as session:
    def print_object(obj):
        print(obj)

    handle = session.request_stream(print_object, 'AAPL', ['bar-5s', 'trade'])
    handle.wait()
    handle.raise_on_error()   # raises if the request returned an error
```

## Historical data

```python
import sheldatagateway
from sheldatagateway import environments
import getpass, datetime

with sheldatagateway.Session(environments.env_defs.Prod, 'username', getpass.getpass()) as session:
    def print_object(obj):
        print(obj)

    handle = session.request_data(
        print_object, 'TCEHY',
        datetime.date(2025, 1, 16), datetime.date(2025, 1, 16),
        ['trade'],
    )
    handle.wait()
    handle.raise_on_error()
```

Daily bars since the start of 2024:

```python
handle = session.request_data(
    print_object, 'AAPL',
    datetime.date(2024, 1, 1), datetime.date.today(),
    ['bar-1day'],
)
```

## API surface

| Call | Purpose |
| --- | --- |
| `sheldatagateway.Session(env, username, password)` | Context manager holding the connection |
| `environments.env_defs.Prod` | Environment definition |
| `session.request_stream(callback, symbol, subscriptions)` | Live stream — returns a handle |
| `session.request_data(callback, symbol, start_date, end_date, subscriptions)` | Historical — returns a handle |
| `handle.wait()` | Block until the request finishes |
| `handle.raise_on_error()` | Raise if the request returned an error |

The callback receives **parsed Python dicts**, one per message — the SDK handles
framing, multiplexing, and decompression.

> `request_data` takes no `split-adjust` argument in any documented example, even
> though the wire protocol requires the field. Check the SDK signature for how it
> is defaulted or passed.

> `handle.wait()` blocks indefinitely on a live stream, since that stream never
> sets *End of request*. For a multi-symbol recorder, run requests on separate
> threads or don't call `wait()`.

## Command-line utilities

Two utilities are installed, both accepting **one or more symbols**:

```bash
# Live stream
sheldatagateway_stream -u <username> AAPL MSFT NVDA

# Historical, as JSON piped through jq
sheldatagateway_data -u <username> AAPL --subscriptions trade -d 2025-01-16 --json | jq .
```

| Flag | Meaning |
| --- | --- |
| `-u <username>` | Username |
| `--subscriptions <list>` | Subscription types |
| `-d <yyyy-mm-dd>` | Date |
| `--json` | Emit JSON (default output is Python-dict repr) |

> The default output is Python `repr`, **not** JSON — note the single quotes in
> the sample output. Pass `--json` for anything you intend to pipe or parse.

> The CLI accepts multiple symbols while the wire protocol's `symbol` field is a
> single string, so the utility is presumably issuing one request per symbol on
> the shared connection. Multiplexing is what makes that work.
