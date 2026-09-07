# metra

Minimal metrics tracking for regression monitoring. One SQLite table (`commit`, `run`, `name`, `time`, `value` as arbitrary JSON, plus an `unreliable` flag and a `comment`), a dependency-free PHP server (`www/index.php`), a plain HTML table UI (`www/index.html`), and one-liner clients for C++, Python, and the shell.

## Server

PHP's built-in web server with `www/index.php` as the router script (needs `php` >= 8.1 with the bundled `pdo_sqlite`; Debian/Ubuntu: `apt install php-cli php-sqlite3`):

```bash
METRA_DB=metra.sqlite php -S 0.0.0.0:13340 -t tools/metra/www tools/metra/www/index.php   # db: METRA_DB env, default ./metra.sqlite
```

PHP re-reads the scripts on every request, so edits to `www/` are live on the next request - no restart. The deployed instance runs it via `docker-compose.yml`, see below.

### NAS / Docker (TrueNAS SCALE)

No image build: `docker-compose.yml` runs the stock `php:8.4-cli` image with an RLtools checkout and the DB bind-mounted from datasets. The infra setup keeps the checkout at `/mnt/fast/infra/metra/rl_tools` on the NAS, which is `/infra/metra/rl_tools` on the VMs (see `/infra/metra/README.md`), so the server can be edited from any VM and the edit is live on the next request:

```bash
git clone https://github.com/rl-tools/rl-tools rl_tools   # in /mnt/fast/infra/metra/
```

Then on TrueNAS SCALE: *Apps → Discover Apps → ⋮ → Install via YAML*, name `metra`, paste `docker-compose.yml` (host paths and the `user:` uid at the top of the file; the data directory must exist and be writable by that uid before the first start - see the header comment). On any plain Docker host: `docker compose up -d` in a directory containing the file. Updating the server = `git pull` in the checkout, nothing to restart; the DB migrates itself on the next request. Clients then use `METRA_URL=http://<nas>:13340` (or `http://<nas>/metra` behind the `proxy` app).

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

`http://localhost:13340/` — a plain table of the most recent entries. The checkbox marks a row unreliable (excluded from API queries unless `include_unreliable=1`; struck through in the UI), the comment column is editable inline. The page only references the API through relative URLs (`api/...`, never `/api/...`), so it also works under a path prefix behind a reverse proxy that strips the prefix (e.g. nginx `location /metra/ { proxy_pass http://<nas>:13340/; }` → `http://<nas>/metra/`); clients then use `METRA_URL=http://<nas>/metra`. Keep it that way when editing `www/index.html` (`test_html` checks it).

## Environment

| Variable | Meaning |
|---|---|
| `METRA_URL` | server base URL; unset disables the C++ client, Python defaults to `http://127.0.0.1:13340` |
| `METRA_COMMIT` | overrides commit detection (`git rev-parse HEAD`, fallback `no-hash`) |
| `METRA_COMMIT_TIME` | overrides commit time detection (`git show -s --format=%ct HEAD`), unix seconds; setting `METRA_COMMIT` alone omits the auto-detected time |
| `METRA_RUN` | overrides the generated run id (`<timestamp>_<hostname>_<pid>[_<hex>]`) |
| `METRA_DB` | server: SQLite path (default `./metra.sqlite`, relative to the working directory) |

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

Direct SQL always works: `sqlite3 metra.sqlite "SELECT name, AVG(value_scalar) FROM metrics WHERE unreliable=0 GROUP BY name"` (on the NAS: `/mnt/fast/apps/metra/data/metra.sqlite`; do not open it over NFS from a VM while the app runs, SQLite locking over NFS is unreliable).

## Hacking

The server is `www/index.php` (endpoints are the `handle_*` functions; add one by adding a branch in the dispatch at the bottom of the file) plus `www/index.html` (the UI). Edits are live on the next request. There is no auth (run it on a trusted network); for cross-origin browser access add `header('Access-Control-Allow-Origin: *')` in `respond`. Tests: `python3 tools/metra/test_server.py` starts the PHP server on a free port with a scratch DB and drives it through the Python client (`php` on the PATH); `METRA_TEST_URL=http://host:port python3 tools/metra/test_server.py` runs the same suite against a running server (it writes test rows, so not against the production DB). The `test_utils_metra` ctest target covers the C++ client.
