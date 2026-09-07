<?php
// metra server: commit/run/name/time/value rows in SQLite, a JSON API, and the table UI in index.html.
//
// Runs on PHP's built-in web server with this file as the router script (every request lands here):
//   php -S 0.0.0.0:13340 -t tools/metra/www tools/metra/www/index.php
// PHP re-reads the scripts on every request, so edits are live on the next request - no restart.
//
// Environment: METRA_DB (SQLite path, default ./metra.sqlite relative to the working directory;
// missing parent directories are created).
//
// GET  /              UI (index.html)
// GET  /api/metrics   rows, newest first. Filters: name, commit (prefix), run, since, until, include_unreliable, limit (default 1000)
// GET  /api/names     distinct metric names
// POST /api/log       {"name": ..., "value": <any JSON>, "commit"?: ..., "commit_time"?: ..., "run"?: ..., "time"?: ...}
//                     or a list of such entries -> 201 {"ids": [...]}
// POST /api/flag      {"id": ..., "unreliable": true|false} (default true)
// POST /api/comment   {"id": ..., "comment": "..."}
// Errors are {"error": "..."} with a 4xx/5xx status. No auth: run it on a trusted network.

declare(strict_types=1);

ini_set('display_errors', '0'); // PHP notices must not end up inside a JSON body; they go to stderr (docker logs) instead
ini_set('log_errors', '1');
ini_set('precision', '-1'); // float -> string conversion (PDO binds parameters as strings) keeps the full double, like json_encode

const DEFAULT_DB = 'metra.sqlite';
const JSON_FLAGS = JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE | JSON_PRESERVE_ZERO_FRACTION | JSON_INVALID_UTF8_SUBSTITUTE;
const SCHEMA = <<<'SQL'
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
SQL;

function respond(int $status, mixed $body): never {
    http_response_code($status);
    header('Content-Type: application/json');
    echo json_encode($body, JSON_FLAGS);
    exit;
}

function fail(int $status, string $message): never {
    respond($status, ['error' => $message]);
}

// PDO handle for the metrics DB. The schema is created (and migrated) on first use, so the server comes
// up against an empty data directory with no separate init step.
function db(): PDO {
    static $pdo = null;
    if ($pdo !== null) {
        return $pdo;
    }
    $path = getenv('METRA_DB') ?: DEFAULT_DB;
    $directory = dirname($path);
    if (!is_dir($directory)) {
        @mkdir($directory, 0777, true);
    }
    try {
        $pdo = new PDO('sqlite:' . $path);
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        $pdo->exec('PRAGMA journal_mode=WAL');
        $pdo->exec('PRAGMA busy_timeout=30000');
        $pdo->exec(SCHEMA);
        $columns = $pdo->query('PRAGMA table_info(metrics)')->fetchAll(PDO::FETCH_COLUMN, 1);
        if (!in_array('commit_time', $columns, true)) { // DBs from before the commit_time column
            $pdo->exec('ALTER TABLE metrics ADD COLUMN commit_time REAL');
        }
    } catch (PDOException $error) {
        $uid = function_exists('posix_getuid') ? posix_getuid() : '?';
        $message = "metra: cannot open database $path ({$error->getMessage()}); check that $directory exists and is writable by uid $uid";
        error_log($message);
        fail(500, $message);
    }
    return $pdo;
}

// Query parameter or null; "?key=" counts as absent (like Python's parse_qs, which the previous server used).
function param(array $query, string $key): ?string {
    $value = $query[$key] ?? null;
    return (is_string($value) && $value !== '') ? $value : null;
}

function as_text(mixed $value): string {
    return is_scalar($value) ? (string)$value : (string)json_encode($value, JSON_FLAGS);
}

function handle_metrics(array $query): never {
    $where = [];
    $parameters = [];
    if (($name = param($query, 'name')) !== null) {
        $where[] = 'name=?';
        $parameters[] = $name;
    }
    if (($commit = param($query, 'commit')) !== null) {
        $where[] = "commit_hash LIKE ?||'%'";
        $parameters[] = $commit;
    }
    if (($run = param($query, 'run')) !== null) {
        $where[] = 'run_id=?';
        $parameters[] = $run;
    }
    foreach (['since' => 'time>=?', 'until' => 'time<=?'] as $key => $condition) {
        if (($bound = param($query, $key)) !== null) {
            if (!is_numeric($bound)) {
                fail(400, 'Malformed query parameter');
            }
            $where[] = $condition;
            $parameters[] = (float)$bound;
        }
    }
    if (!in_array(param($query, 'include_unreliable') ?? '0', ['1', 'true'], true)) {
        $where[] = 'unreliable=0';
    }
    $limit = filter_var(param($query, 'limit') ?? '1000', FILTER_VALIDATE_INT);
    if ($limit === false) {
        fail(400, 'Malformed query parameter');
    }
    $sql = 'SELECT id,time,commit_hash,commit_time,run_id,name,value,value_scalar,unreliable,comment FROM metrics';
    if ($where) {
        $sql .= ' WHERE ' . implode(' AND ', $where);
    }
    $sql .= ' ORDER BY time DESC, id DESC LIMIT ' . $limit; // validated int; a negative limit means no limit (SQLite)
    $statement = db()->prepare($sql);
    $statement->execute($parameters);
    $result = [];
    while (($row = $statement->fetch(PDO::FETCH_ASSOC)) !== false) {
        $result[] = [ // explicit casts: the JSON types must not depend on the PDO driver's fetch typing
            'id' => (int)$row['id'],
            'time' => (float)$row['time'],
            'commit_hash' => (string)$row['commit_hash'],
            'commit_time' => $row['commit_time'] === null ? null : (float)$row['commit_time'],
            'run_id' => (string)$row['run_id'],
            'name' => (string)$row['name'],
            'value' => json_decode((string)$row['value']), // objects stay objects ({} must not turn into [])
            'value_scalar' => $row['value_scalar'] === null ? null : (float)$row['value_scalar'],
            'unreliable' => (int)$row['unreliable'],
            'comment' => (string)$row['comment'],
        ];
    }
    respond(200, $result);
}

function handle_names(): never {
    respond(200, db()->query('SELECT DISTINCT name FROM metrics ORDER BY name')->fetchAll(PDO::FETCH_COLUMN));
}

function handle_log(mixed $body): never {
    $entries = is_array($body) ? $body : [$body];
    $rows = [];
    foreach ($entries as $entry) {
        if (!is_object($entry) || !property_exists($entry, 'value') || !is_string($entry->name ?? null) || $entry->name === '') {
            fail(400, 'Each entry needs a non-empty "name" and a "value"');
        }
        $value = $entry->value;
        $time = property_exists($entry, 'time') ? $entry->time : microtime(true);
        $commit_time = $entry->commit_time ?? null;
        if (!is_numeric($time) || ($commit_time !== null && !is_numeric($commit_time))) {
            fail(400, '"time" and "commit_time" must be unix timestamps');
        }
        $encoded = json_encode($value, JSON_FLAGS);
        if ($encoded === false) {
            fail(400, '"value" is not encodable as JSON');
        }
        $rows[] = [
            (float)$time,
            as_text($entry->commit ?? 'no-hash'),
            $commit_time === null ? null : (float)$commit_time,
            as_text($entry->run ?? 'unknown'),
            $entry->name,
            $encoded,
            (is_int($value) || is_float($value)) ? (float)$value : null, // numbers only, not bools
        ];
    }
    $pdo = db();
    $pdo->beginTransaction(); // a batch lands atomically
    $statement = $pdo->prepare('INSERT INTO metrics(time,commit_hash,commit_time,run_id,name,value,value_scalar) VALUES (?,?,?,?,?,?,?)');
    $ids = [];
    foreach ($rows as $row) {
        $statement->execute($row);
        $ids[] = (int)$pdo->lastInsertId();
    }
    $pdo->commit();
    respond(201, ['ids' => $ids]);
}

function update_row(string $sql, mixed $value, mixed $id): never {
    $id = filter_var($id, FILTER_VALIDATE_INT);
    if ($id === false) {
        fail(404, 'Metric not found');
    }
    $statement = db()->prepare($sql);
    $statement->execute([$value, $id]);
    if ($statement->rowCount() === 0) {
        fail(404, 'Metric not found');
    }
    respond(200, ['ok' => true]);
}

function handle_flag(mixed $body): never {
    if (!is_object($body) || !property_exists($body, 'id')) {
        fail(400, 'Expected {"id": ..., "unreliable": ...}');
    }
    $unreliable = property_exists($body, 'unreliable') ? ($body->unreliable ? 1 : 0) : 1;
    update_row('UPDATE metrics SET unreliable=? WHERE id=?', $unreliable, $body->id);
}

function handle_comment(mixed $body): never {
    if (!is_object($body) || !property_exists($body, 'id')) {
        fail(400, 'Expected {"id": ..., "comment": ...}');
    }
    update_row('UPDATE metrics SET comment=? WHERE id=?', as_text($body->comment ?? ''), $body->id);
}

// Dispatch. Add an endpoint by adding a branch here and a handle_* function above.
try {
    $method = $_SERVER['REQUEST_METHOD'] ?? 'GET';
    $path = parse_url($_SERVER['REQUEST_URI'] ?? '/', PHP_URL_PATH);
    if (!is_string($path)) {
        $path = '/';
    }
    if ($method === 'GET' || $method === 'HEAD') {
        if ($path === '/' || $path === '/index.html') {
            header('Content-Type: text/html; charset=utf-8');
            readfile(__DIR__ . '/index.html');
            exit;
        }
        if ($path === '/api/metrics') {
            handle_metrics($_GET);
        }
        if ($path === '/api/names') {
            handle_names();
        }
        fail(404, 'Unknown endpoint');
    }
    if ($method === 'POST') {
        $body = json_decode((string)file_get_contents('php://input')); // objects as stdClass: {} and [] stay distinguishable
        if (json_last_error() !== JSON_ERROR_NONE) {
            fail(400, 'Body is not valid JSON');
        }
        if ($path === '/api/log') {
            handle_log($body);
        }
        if ($path === '/api/flag') {
            handle_flag($body);
        }
        if ($path === '/api/comment') {
            handle_comment($body);
        }
        fail(404, 'Unknown endpoint');
    }
    fail(405, 'Method not allowed');
} catch (Throwable $error) {
    error_log('metra: ' . $error);
    fail(500, 'Internal error: ' . $error->getMessage());
}
