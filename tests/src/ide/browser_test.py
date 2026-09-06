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
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

MINIMUM_FINAL_MEAN_RETURN = -400
BROWSER_CANDIDATES = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "firefox"]


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".js": "text/javascript", ".mjs": "text/javascript", ".tar": "application/x-tar", ".json": "application/json"}

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
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--verbose", action="store_true", help="log every HTTP request")
    arguments = parser.parse_args()

    browser = arguments.browser or next((candidate for candidate in BROWSER_CANDIDATES if shutil.which(candidate)), None)
    if browser is None or shutil.which(browser) is None:
        print("no browser found; pass --browser or install google-chrome, chromium or firefox", file=sys.stderr)
        return 2
    for artifact in ["static/ide/build/toolchain/llvm.wasm", "static/ide/build/toolchain/sysroot.tar", "static/ide/build/rl_tools_include.tar", "static/ide/build/examples/pendulum_sac.cpp"]:
        if not Path(arguments.root, artifact).is_file():
            print(f"missing {artifact}: build the toolchain (tools/ide/toolchain) and run tools/ide/bundle.sh", file=sys.stderr)
            return 2

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *handler_arguments: Handler(*handler_arguments, directory=arguments.root))
    server.result = None
    server.result_ready = threading.Event()
    server.verbose = arguments.verbose
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/static/ide/test.html"

    with tempfile.TemporaryDirectory(prefix="rl-tools-ide-browser-") as profile:
        command = browser_command(shutil.which(browser), url, profile)
        print(f"serving {arguments.root} at {url}")
        print("launching: " + " ".join(shlex.quote(part) for part in command))
        started = time.monotonic()
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        try:
            finished = server.result_ready.wait(arguments.timeout)
        finally:
            process.kill()
            browser_stderr = process.communicate()[1] if process.stderr else ""
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
    last = evaluations[-1] if evaluations else None
    print(f"user agent: {result.get('userAgent', '?')}")
    print(f"compile: exit {result.get('compileExitCode')} in {result.get('compileSeconds', 0):.1f} s, {result.get('programBytes', 0) / 1024:.0f} KB")
    print(f"run: exit {result.get('runExitCode')} in {result.get('runSeconds', 0):.1f} s, {len(evaluations)} evaluations (total {elapsed:.0f} s)")
    ok = result.get("compileExitCode") == 0 and result.get("runExitCode") == 0 and last is not None and last["step"] == last["stepLimit"] and last["meanReturn"] > MINIMUM_FINAL_MEAN_RETURN
    print(f"PASS: final mean return {last['meanReturn']:.1f} > {MINIMUM_FINAL_MEAN_RETURN}" if ok else f"FAIL: {json.dumps(last)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
