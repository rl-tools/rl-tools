# metra

Minimal metrics tracking for regression monitoring. One SQLite table (`commit`, `run`, `name`, `time`, `value` as arbitrary JSON, plus an `unreliable` flag and a `comment`), a stdlib-only Python server, a plain HTML table UI, and one-liner clients for C++, Python, and the shell.

## Server

```bash
python3 tools/metra/metra/server.py --ip 0.0.0.0 --port 13340   # db: METRA_DB env or ./metra.sqlite (or --db)
```

Install as a systemd service (user-level by default, no sudo required; `--system` for a system unit):

```bash
tools/metra/install_service.sh
```

The DB then lives in `~/.local/share/metra/metra.sqlite` and the server runs from the checkout (edit `server.py`, `systemctl --user restart metra`). User services stop at logout unless `loginctl enable-linger $USER` is set. Logs: `journalctl --user -u metra`.

## Logging metrics

C++ (`include/metra/metra.h`, header-only, no rl_tools dependency):

```cpp
#include <metra/metra.h>
metra::log("pendulum/return", 1.23);                       // scalar
metra::log("resnet/forward_us", std::vector<double>{...}); // list
metra::log_raw("run/config", "{\"lr\": 0.001}");           // arbitrary JSON
```

All calls of a process share one static run id. `METRA_URL` unset disables logging (a single stderr notice), so calls can stay in the code; the commit and its time are auto-detected via `git rev-parse HEAD` / `git show -s --format=%ct HEAD`, with the `METRA_COMMIT`/`METRA_COMMIT_TIME` compile definitions (set at configure time by `cmake/autodetect/git-hash.cmake` on `rl_tools_full` and the `metra` CLI) as fallbacks for binaries that run outside a git checkout. After 3 consecutive transport failures logging disables itself so a dead server cannot stall a training loop.

Shell (CLI target `metra`, values are JSON):

```bash
cmake --build build --target metra -j5
METRA_URL=http://localhost:13340 ./build/src/metra/metra pendulum/return 1.23
```

Python / notebooks (`tools/metra/metra/__init__.py`, zero dependencies):

```python
import sys; sys.path.insert(0, "tools/metra")   # or: pip install -e tools/metra
import metra
metra.log("pendulum/return", 1.23)
rows = metra.fetch("pendulum/return", since=time.time() - 30 * 86400)
# trend plot: rows are newest-first, each row a dict with commit_hash/run_id/time/value/value_scalar/comment
import matplotlib.pyplot as plt
rows = rows[::-1]
plt.plot([r["time"] for r in rows], [r["value_scalar"] for r in rows]); plt.show()
```

Raw HTTP:

```bash
curl -X POST -H 'Content-Type: application/json' --data '{"name":"pendulum/return","value":1.23,"commit":"abc123","commit_time":1700000000,"run":"my-run"}' http://localhost:13340/api/log
curl 'http://localhost:13340/api/metrics?name=pendulum/return&limit=10'          # filters: name, commit (prefix), run, since, until, include_unreliable
curl http://localhost:13340/api/names
curl -X POST --data '{"id":1,"unreliable":true}'  http://localhost:13340/api/flag
curl -X POST --data '{"id":1,"comment":"flaky machine"}' http://localhost:13340/api/comment
```

## Web UI

`http://localhost:13340/` — a plain table of the most recent entries. The checkbox marks a row unreliable (excluded from API queries unless `include_unreliable=1`; struck through in the UI), the comment column is editable inline.

## Environment

| Variable | Meaning |
|---|---|
| `METRA_URL` | server base URL; unset disables the C++ client, Python defaults to `http://127.0.0.1:13340` |
| `METRA_COMMIT` | overrides commit detection (`git rev-parse HEAD`, fallback `no-hash`) |
| `METRA_COMMIT_TIME` | overrides commit time detection (`git show -s --format=%ct HEAD`), unix seconds; setting `METRA_COMMIT` alone omits the auto-detected time |
| `METRA_RUN` | overrides the generated run id (`<timestamp>_<hostname>_<pid>[_<hex>]`) |
| `METRA_DB` | server: SQLite path (default `./metra.sqlite`) |

## Schema

```sql
CREATE TABLE metrics(
    id INTEGER PRIMARY KEY,
    time REAL NOT NULL,            -- unix seconds, server-assigned unless the client supplies "time"
    commit_hash TEXT NOT NULL,
    commit_time REAL,              -- committer timestamp in unix seconds (git %ct), NULL if unknown
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value TEXT NOT NULL,           -- JSON-encoded
    value_scalar REAL,             -- set when value is a number, for SQL aggregation
    unreliable INTEGER NOT NULL DEFAULT 0,
    comment TEXT NOT NULL DEFAULT ''
);
```

Direct SQL always works: `sqlite3 ~/.local/share/metra/metra.sqlite "SELECT name, AVG(value_scalar) FROM metrics WHERE unreliable=0 GROUP BY name"`.

## Hacking

The whole server is one file (`metra/server.py`): the UI is the `PAGE` string, endpoints are the `_handle_*` methods — add an endpoint by adding a branch in `do_GET`/`do_POST`. There is no auth (run it on a trusted network); for cross-origin browser access add `self.send_header("Access-Control-Allow-Origin", "*")` in `_send`. Tests: `python3 tools/metra/test_server.py` (server+Python client round trip) and the `test_utils_metra` ctest target (C++ client).
