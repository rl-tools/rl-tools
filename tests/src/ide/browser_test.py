#!/usr/bin/env python3
"""End-to-end test of the browser IDE in a real browser: serves the repository root, opens static/ide/test.html headless,
and waits for the page to post back what compile and run produced through the same workers the IDE uses.

usage: browser_test.py [--browser PATH] [--root DIR] [--timeout SECONDS]
Only the standard library is used. Chrome/Chromium and Firefox are recognised by executable name; RL_TOOLS_BROWSER_ARGS
appends extra command line flags (for example --no-sandbox in a container).
"""
import argparse
import http.server
import json
import os
import shlex
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlencode, urlsplit

BROWSER_CANDIDATES = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "firefox"]


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".js": "text/javascript", ".mjs": "text/javascript", ".tar": "application/x-tar", ".json": "application/json"}

    def translate_path(self, path):
        if urlsplit(path).path.startswith("/bundle/"):
            original = self.directory
            self.directory = str(self.server.bundle)
            try:
                return super().translate_path(path[len("/bundle"):])
            finally:
                self.directory = original
        return super().translate_path(path)

    def do_POST(self):
        if not self.path.endswith("/result"):
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.server.result = json.loads(body)
        self.server.result_ready.set()
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format, *args):
        if self.server.verbose:
            super().log_message(format, *args)


def browser_command(executable, url, profile):
    name = Path(executable).name
    if name.startswith("firefox"):
        command = [executable, "--headless", "--no-remote", "--new-instance", "--profile", profile, url]
    else:
        command = [executable, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--disable-background-networking", f"--user-data-dir={profile}", url]
    return command + shlex.split(os.environ.get("RL_TOOLS_BROWSER_ARGS", ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--browser", default=None, help="browser executable (default: first of %s on PATH)" % ", ".join(BROWSER_CANDIDATES))
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[3]), help="repository root to serve")
    parser.add_argument("--bundle", type=Path, help="candidate bundle root (default: static/ide/build)")
    parser.add_argument("--example", default="smoke")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--verbose", action="store_true", help="log every HTTP request")
    arguments = parser.parse_args()

    browser = arguments.browser or next((candidate for candidate in BROWSER_CANDIDATES if shutil.which(candidate)), None)
    if browser is None or shutil.which(browser) is None:
        print("no browser found; pass --browser or install google-chrome, chromium or firefox", file=sys.stderr)
        return 77 if arguments.allow_missing else 2
    bundle = (arguments.bundle or Path(arguments.root, "static/ide/build")).resolve()
    for artifact in ["toolchain/llvm.wasm", "toolchain/sysroot.tar", "rl_tools_include.tar", "examples.json", f"examples/{arguments.example}.cpp"]:
        if not (bundle / artifact).is_file():
            print(f"missing {bundle / artifact}", file=sys.stderr)
            return 77 if arguments.allow_missing else 2

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *handler_arguments: Handler(*handler_arguments, directory=arguments.root))
    server.bundle = bundle
    server.result = None
    server.result_ready = threading.Event()
    server.verbose = arguments.verbose
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/static/ide/test.html?" + urlencode({"bundle": "/bundle/", "example": arguments.example})

    with tempfile.TemporaryDirectory(prefix="rl-tools-ide-browser-") as profile:
        command = browser_command(shutil.which(browser), url, profile)
        print(f"serving {arguments.root} at {url}")
        print("launching: " + " ".join(shlex.quote(part) for part in command))
        started = time.monotonic()
        with tempfile.TemporaryFile(mode="w+") as stderr:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr, text=True, start_new_session=True)
            try:
                deadline = time.monotonic() + arguments.timeout
                while not server.result_ready.is_set() and process.poll() is None and time.monotonic() < deadline:
                    server.result_ready.wait(min(1.0, max(0, deadline - time.monotonic())))
                finished = server.result_ready.is_set()
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                stderr.seek(0)
                browser_stderr = stderr.read()
        server.shutdown()
    elapsed = time.monotonic() - started
    if not finished:
        print(f"FAIL: no result within {arguments.timeout:.0f} s")
        print(browser_stderr[-4000:], file=sys.stderr)
        return 1

    result = server.result
    for line in result.get("lines", []):
        print("  | " + line)
    evaluations = result.get("evaluations", [])
    print(f"user agent: {result.get('userAgent', '?')}")
    print(f"compile: exit {result.get('compileExitCode')} in {result.get('compileSeconds', 0):.1f} s, {result.get('programBytes', 0) / 1024:.0f} KB")
    print(f"run: exit {result.get('runExitCode')} in {result.get('runSeconds', 0):.1f} s, {len(evaluations)} evaluations (total {elapsed:.0f} s)")
    ok = result.get("ok") is True and result.get("compileExitCode") == 0 and result.get("runExitCode") == 0
    print("PASS: " + arguments.example if ok else "FAIL: " + str(result.get("error", result)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
