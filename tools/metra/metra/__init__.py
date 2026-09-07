"""metra: client for the minimal metrics tracking server (tools/metra/www/index.php).

Environment: METRA_URL (server base URL, default http://127.0.0.1:13340), METRA_COMMIT (override commit detection), METRA_COMMIT_TIME (override commit time detection, unix seconds), METRA_RUN (override the generated run id).
"""
from __future__ import annotations

import datetime, json, os, socket, subprocess, sys, urllib.parse, urllib.request

DEFAULT_URL = "http://127.0.0.1:13340"
_commit = None
_commit_time = False
_run = None


def _request(url, path, body=None, query=None):
    base = (url or os.environ.get("METRA_URL") or DEFAULT_URL).rstrip("/")
    if query:
        path += "?" + urllib.parse.urlencode({key: value for key, value in query.items() if value is not None})
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode())


def default_commit() -> str:
    global _commit
    if _commit is None:
        _commit = os.environ.get("METRA_COMMIT")
        if not _commit:
            try:
                _commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                _commit = "no-hash"
    return _commit


def default_commit_time():
    global _commit_time
    if _commit_time is False:
        try:
            _commit_time = float(os.environ["METRA_COMMIT_TIME"])
        except (KeyError, ValueError):
            try:
                _commit_time = float(subprocess.run(["git", "show", "-s", "--format=%ct", "HEAD"], capture_output=True, text=True, check=True).stdout.strip())
            except (OSError, subprocess.CalledProcessError, ValueError):
                _commit_time = None
    return _commit_time


def default_run() -> str:
    global _run
    if _run is None:
        _run = os.environ.get("METRA_RUN") or "{}_{}_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), socket.gethostname(), os.getpid())
    return _run


def log(name, value, commit=None, commit_time=None, run=None, time=None, url=None) -> int:
    entry = {"name": name, "value": value, "commit": commit or default_commit(), "run": run or default_run()}
    if commit_time is None and commit is None:  # an explicitly overridden commit must not be paired with the auto-detected HEAD time
        commit_time = default_commit_time()
    if commit_time is not None:
        entry["commit_time"] = commit_time
    if time is not None:
        entry["time"] = time
    return _request(url, "/api/log", body=entry)["ids"][0]


def fetch(name=None, commit=None, run=None, since=None, until=None, limit=1000, include_unreliable=False, url=None) -> list:
    query = {"name": name, "commit": commit, "run": run, "since": since, "until": until, "limit": limit, "include_unreliable": 1 if include_unreliable else None}
    return _request(url, "/api/metrics", query=query)


def names(url=None) -> list:
    return _request(url, "/api/names")


def flag(id, unreliable=True, url=None) -> None:
    _request(url, "/api/flag", body={"id": id, "unreliable": unreliable})


def comment(id, text, url=None) -> None:
    _request(url, "/api/comment", body={"id": id, "comment": text})


def main():
    arguments = sys.argv[1:]
    if len(arguments) >= 3 and arguments[0] == "log":
        for raw in arguments[2:]:
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            print(log(arguments[1], value))
        return
    if len(arguments) >= 1 and arguments[0] == "fetch":
        for row in fetch(name=arguments[1] if len(arguments) > 1 else None):
            print(json.dumps(row))
        return
    if arguments == ["names"]:
        for name in names():
            print(name)
        return
    print("Usage: metra log <name> <value> [<value> ...] | metra fetch [<name>] | metra names", file=sys.stderr)
    print("Environment: METRA_URL (server base URL, default {}), METRA_COMMIT, METRA_RUN.".format(DEFAULT_URL), file=sys.stderr)
    sys.exit(1)
