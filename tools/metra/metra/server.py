#!/usr/bin/env python3
"""Minimal metrics tracking server: commit/run/name/time/value rows in SQLite, plain HTML table UI at /."""
from __future__ import annotations

import argparse, http.server, json, os, sqlite3, time, urllib.parse

DB_FILE = os.environ.get("METRA_DB", "metra.sqlite")
DEFAULT_PORT = 13340

SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics(
    id INTEGER PRIMARY KEY,
    time REAL NOT NULL,
    commit_hash TEXT NOT NULL,
    commit_time REAL,
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value TEXT NOT NULL,
    value_scalar REAL,
    unreliable INTEGER NOT NULL DEFAULT 0,
    comment TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS metrics_name_time ON metrics(name, time);
"""


def init_db(path: str | None = None) -> None:
    path = os.path.abspath(path or DB_FILE)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path)
    except (OSError, sqlite3.OperationalError) as error:
        uid = os.getuid() if hasattr(os, "getuid") else "?"
        raise SystemExit(f"metra: cannot open database {path} ({error}); check that {os.path.dirname(path)} exists and is writable by uid {uid}") from error
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(metrics)")]
    if "commit_time" not in columns:
        conn.execute("ALTER TABLE metrics ADD COLUMN commit_time REAL")
    conn.commit()
    conn.close()


def with_connection(fn):
    def inner(self, *args, **kwargs):
        conn = sqlite3.connect(DB_FILE, timeout=30)
        try:
            return fn(self, conn, *args, **kwargs)
        finally:
            conn.close()
    return inner


class RequestHandler(http.server.BaseHTTPRequestHandler):
    server_version = "Metra/0.1"

    def _send(self, status: int, body=None, content_type: str = "application/json"):
        self.send_response(status)
        if body is not None:
            payload = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.end_headers()

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode() if length else "")

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = {key: values[-1] for key, values in urllib.parse.parse_qs(parsed.query).items()}
        try:
            if parsed.path in ("/", "/index.html"):
                return self._send(200, PAGE, content_type="text/html; charset=utf-8")
            if parsed.path == "/api/metrics":
                return self._handle_metrics(query)
            if parsed.path == "/api/names":
                return self._handle_names()
        except ValueError:
            return self._send(400, {"error": "Malformed query parameter"})
        return self._send(404, {"error": "Unknown endpoint"})

    def do_POST(self):  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        try:
            body = self._read_json()
        except json.JSONDecodeError:
            return self._send(400, {"error": "Body is not valid JSON"})
        if path == "/api/log":
            return self._handle_log(body)
        if path == "/api/flag":
            return self._handle_flag(body)
        if path == "/api/comment":
            return self._handle_comment(body)
        return self._send(404, {"error": "Unknown endpoint"})

    @with_connection
    def _handle_log(self, conn: sqlite3.Connection, body):
        entries = body if isinstance(body, list) else [body]
        rows = []
        for entry in entries:
            if not isinstance(entry, dict) or "value" not in entry or not isinstance(entry.get("name"), str) or entry["name"] == "":
                return self._send(400, {"error": 'Each entry needs a non-empty "name" and a "value"'})
            value = entry["value"]
            value_scalar = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
            try:
                entry_time = float(entry.get("time", time.time()))
                commit_time = float(entry["commit_time"]) if "commit_time" in entry and entry["commit_time"] is not None else None
            except (TypeError, ValueError):
                return self._send(400, {"error": '"time" and "commit_time" must be unix timestamps'})
            rows.append((entry_time, str(entry.get("commit", "no-hash")), commit_time, str(entry.get("run", "unknown")), entry["name"], json.dumps(value), value_scalar))
        cur = conn.cursor()
        ids = []
        for row in rows:
            cur.execute("INSERT INTO metrics(time,commit_hash,commit_time,run_id,name,value,value_scalar) VALUES (?,?,?,?,?,?,?)", row)
            ids.append(cur.lastrowid)
        conn.commit()
        return self._send(201, {"ids": ids})

    @with_connection
    def _handle_metrics(self, conn: sqlite3.Connection, query: dict):
        where, parameters = [], []
        if "name" in query:
            where.append("name=?")
            parameters.append(query["name"])
        if "commit" in query:
            where.append("commit_hash LIKE ?||'%'")
            parameters.append(query["commit"])
        if "run" in query:
            where.append("run_id=?")
            parameters.append(query["run"])
        if "since" in query:
            where.append("time>=?")
            parameters.append(float(query["since"]))
        if "until" in query:
            where.append("time<=?")
            parameters.append(float(query["until"]))
        if query.get("include_unreliable", "0") not in ("1", "true"):
            where.append("unreliable=0")
        sql = "SELECT id,time,commit_hash,commit_time,run_id,name,value,value_scalar,unreliable,comment FROM metrics"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY time DESC, id DESC LIMIT ?"
        parameters.append(int(query.get("limit", "1000")))
        rows = conn.execute(sql, parameters).fetchall()
        columns = ["id", "time", "commit_hash", "commit_time", "run_id", "name", "value", "value_scalar", "unreliable", "comment"]
        result = [dict(zip(columns, row)) for row in rows]
        for row in result:
            row["value"] = json.loads(row["value"])
        return self._send(200, result)

    @with_connection
    def _handle_names(self, conn: sqlite3.Connection):
        rows = conn.execute("SELECT DISTINCT name FROM metrics ORDER BY name").fetchall()
        return self._send(200, [row[0] for row in rows])

    @with_connection
    def _handle_flag(self, conn: sqlite3.Connection, body):
        if not isinstance(body, dict) or "id" not in body:
            return self._send(400, {"error": 'Expected {"id": ..., "unreliable": ...}'})
        cur = conn.cursor()
        cur.execute("UPDATE metrics SET unreliable=? WHERE id=?", (1 if body.get("unreliable", True) else 0, body["id"]))
        if cur.rowcount == 0:
            return self._send(404, {"error": "Metric not found"})
        conn.commit()
        return self._send(200, {"ok": True})

    @with_connection
    def _handle_comment(self, conn: sqlite3.Connection, body):
        if not isinstance(body, dict) or "id" not in body:
            return self._send(400, {"error": 'Expected {"id": ..., "comment": ...}'})
        cur = conn.cursor()
        cur.execute("UPDATE metrics SET comment=? WHERE id=?", (str(body.get("comment", "")), body["id"]))
        if cur.rowcount == 0:
            return self._send(404, {"error": "Metric not found"})
        conn.commit()
        return self._send(200, {"ok": True})

    def log_message(self, fmt, *args):
        return


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>metra</title>
<style>
body{font:14px sans-serif;margin:1em}
table{border-collapse:collapse;font:13px monospace;margin-top:1em}
td,th{border:1px solid #ccc;padding:2px 6px;text-align:left}
tr.unreliable td{opacity:.4;text-decoration:line-through}
td.value{max-width:30em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;cursor:copy}
td.value.copied{background:#cfc}
input.comment{width:16em;font:inherit;border:none;background:transparent}
</style></head><body>
<h1>metra</h1>
<select id="name"><option value="">all metrics</option></select>
<input id="limit" value="200" size="5">
<button onclick="refresh()">refresh</button>
<table><thead><tr><th title="unreliable">&#9888;</th><th>id</th><th>time</th><th>commit</th><th>commit time</th><th>run</th><th>name</th><th>value</th><th>comment</th></tr></thead>
<tbody id="rows"></tbody></table>
<script>
async function api(path, body){
    const response = await fetch(path, body === undefined ? {} : {method: "POST", body: JSON.stringify(body)});
    return response.json();
}
function copyToClipboard(text){
    if(navigator.clipboard && window.isSecureContext){ return navigator.clipboard.writeText(text); }
    const area = document.createElement("textarea"); // clipboard API is unavailable on plain-http origins like the NAS deployment
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
    return Promise.resolve();
}
function cell(row, content){
    const element = document.createElement("td");
    if(content instanceof Node){ element.appendChild(content); } else { element.textContent = content; }
    row.appendChild(element);
    return element;
}
async function refresh(){
    const name = document.getElementById("name").value;
    const limit = document.getElementById("limit").value;
    let query = `api/metrics?include_unreliable=1&limit=${encodeURIComponent(limit)}`;
    if(name !== ""){ query += `&name=${encodeURIComponent(name)}`; }
    const rows = await api(query);
    const container = document.getElementById("rows");
    container.textContent = "";
    for(const row of rows){
        const element = document.createElement("tr");
        element.classList.toggle("unreliable", row.unreliable !== 0);
        element.title = `run: ${row.run_id}\\ncommit: ${row.commit_hash}`;
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.title = "unreliable";
        checkbox.checked = row.unreliable !== 0;
        checkbox.onchange = () => api("api/flag", {id: row.id, unreliable: checkbox.checked}).then(refresh);
        cell(element, checkbox);
        cell(element, row.id);
        cell(element, new Date(row.time * 1000).toISOString().replace("T", " ").slice(0, 19));
        cell(element, row.commit_hash.slice(0, 7));
        cell(element, row.commit_time === null ? "" : new Date(row.commit_time * 1000).toISOString().replace("T", " ").slice(0, 19));
        cell(element, row.run_id);
        cell(element, row.name);
        const value = JSON.stringify(row.value);
        const valueCell = cell(element, value);
        valueCell.className = "value";
        valueCell.title = value;
        valueCell.onclick = () => copyToClipboard(value).then(() => {
            valueCell.classList.add("copied");
            setTimeout(() => valueCell.classList.remove("copied"), 500);
        });
        const comment = document.createElement("input");
        comment.className = "comment";
        comment.placeholder = "comment";
        comment.value = row.comment;
        comment.onchange = () => api("api/comment", {id: row.id, comment: comment.value});
        cell(element, comment);
        container.appendChild(element);
    }
}
async function init(){
    const select = document.getElementById("name");
    for(const name of await api("api/names")){
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
    }
    select.onchange = refresh;
    refresh();
}
init();
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description="Minimal metrics tracking server (SQLite + plain HTML table UI).")
    p.add_argument("--ip", default="0.0.0.0", help="Host to bind (default 0.0.0.0)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to listen on (default {DEFAULT_PORT})")
    p.add_argument("--db", default=None, help="SQLite database path (default: METRA_DB env or ./metra.sqlite)")
    args = p.parse_args()

    global DB_FILE
    if args.db is not None:
        DB_FILE = args.db
    init_db()
    httpd = http.server.ThreadingHTTPServer((args.ip, args.port), RequestHandler)
    print(f"Metra Server listening on http://{args.ip}:{args.port} (db: {os.path.abspath(DB_FILE)})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
