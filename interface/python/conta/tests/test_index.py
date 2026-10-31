"""Tests for conta.index; the command line tests run "conta index" in a subprocess that imports the same conta as this process."""
import contextlib
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import conta
from conta import index as conta_index

REPOSITORY = Path(__file__).resolve().parents[4]
PACKAGE_PARENT = Path(conta.__file__).resolve().parents[1]


@contextlib.contextmanager
def working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class TerminalOutput(io.StringIO):
    def isatty(self):
        return True


class ScanProgressTests(unittest.TestCase):
    def test_terminal_progress_includes_initial_final_and_empty_counts(self):
        for total in (0, 3):
            with self.subTest(total=total):
                output = TerminalOutput()
                with contextlib.redirect_stderr(output):
                    with conta_index.ScanProgress(total) as progress:
                        for _ in range(total):
                            progress.advance()
                text = output.getvalue()
                self.assertIn(f"0/{total} blobs", text)
                self.assertIn(f"{total}/{total} blobs (100%)", text)
                self.assertTrue(text.endswith("\n"))

    def test_redirected_or_disabled_progress_has_no_output(self):
        for output, enabled in ((io.StringIO(), True), (TerminalOutput(), False)):
            with self.subTest(terminal=output.isatty(), enabled=enabled):
                with contextlib.redirect_stderr(output):
                    with conta_index.ScanProgress(1, enabled=enabled) as progress:
                        progress.advance()
                self.assertEqual(output.getvalue(), "")

    def test_interrupted_progress_ends_line_without_claiming_completion(self):
        output = TerminalOutput()
        with contextlib.redirect_stderr(output):
            with self.assertRaises(KeyboardInterrupt):
                with conta_index.ScanProgress(2) as progress:
                    progress.advance()
                    raise KeyboardInterrupt
        self.assertIn("1/2 blobs", output.getvalue())
        self.assertNotIn("100%", output.getvalue())
        self.assertTrue(output.getvalue().endswith("\n"))


class IngestionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "shards").mkdir()
        self.version = self.root / "version" / "1"
        self.version.mkdir(parents=True)
        (self.version / "meta.json").write_text(json.dumps({
            "format": 1, "index": {"location": "generated/index.json"},
        }))
        (self.version / "shards.json").write_text(json.dumps({
            name: {"type": "http", "url": f"https://example.test/{name}/{{sha1}}"}
            for name in ("primary", "mirror")
        }))
        self.checkout = self.root / "checkout"
        self.data = self.checkout / "data"
        self.input = self.root / "example.bin"
        self.payload = b"a seekable blob\x00\xff" * 1000
        self.input.write_bytes(self.payload)
        self.sha1 = hashlib.sha1(self.payload).hexdigest()
        self.raw = self.data / self.sha1
        self.zipped = self.data / f"{self.sha1}.gz"

    def run_cli(self, *args, success=True):
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, (str(PACKAGE_PARENT), os.environ.get("PYTHONPATH")))))
        result = subprocess.run(
            [sys.executable, "-c", "import sys, conta; sys.exit(conta.main())", "index", *map(str, args)],
            capture_output=True, text=True, cwd=self.root, env=environment,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def ingest(self, *options):
        return self.run_cli("ingest", *options, "primary", self.checkout, self.input)

    def index(self):
        return json.loads((self.version / "generated" / "index.json").read_bytes())

    def snapshot(self):
        return {path: path.read_bytes() for path in (
            self.root / "shards" / "primary.json",
            self.version / "generated" / "index.json",
            self.version / "meta.json",
        )}

    def assert_scan_fails_without_publishing(self, message):
        before = self.snapshot()
        result = self.run_cli("scan", "primary", self.checkout, success=False)
        self.assertIn(message, result.stderr)
        self.assertEqual(before, self.snapshot())

    def set_timestamp(self, shard, timestamp):
        path = self.root / "shards" / f"{shard}.json"
        entries = json.loads(path.read_bytes())
        for entry in entries:
            entry.pop("first_indexed_at", None)
            if timestamp is not None:
                entry["first_indexed_at"] = timestamp
        path.write_text(json.dumps(entries))

    def test_ingest_and_scan_timestamp_new_entries_in_utc(self):
        before = datetime.now(timezone.utc).replace(microsecond=0)
        self.ingest()
        mirror = self.root / "mirror-checkout"
        (mirror / "data").mkdir(parents=True)
        (mirror / "data" / self.sha1).write_bytes(self.payload)
        self.run_cli("scan", "mirror", mirror)
        after = datetime.now(timezone.utc).replace(microsecond=0)
        timestamps = []
        for shard in ("primary", "mirror"):
            entry = json.loads((self.root / "shards" / f"{shard}.json").read_bytes())[0]
            timestamp = entry["first_indexed_at"]
            self.assertRegex(timestamp, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
            instant = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            self.assertLessEqual(before, instant)
            self.assertLessEqual(instant, after)
            timestamps.append(timestamp)
        entry = self.index()["entries"][0]
        self.assertEqual(entry["first_indexed_at"], min(timestamps))
        self.run_cli("check")

    def test_scans_metadata_updates_and_gzip_preserve_known_and_unknown_dates(self):
        self.ingest("--no-gzip")
        path = self.root / "shards" / "primary.json"
        for timestamp in ("2020-02-29T12:34:56Z", None):
            with self.subTest(timestamp=timestamp):
                if self.zipped.exists():
                    self.zipped.unlink()
                entries = json.loads(path.read_bytes())
                entries[0]["size"] = {"raw": len(self.payload)}
                path.write_text(json.dumps(entries))
                self.set_timestamp("primary", timestamp)
                self.run_cli("build")
                self.run_cli("scan", "primary", self.checkout)
                description = f"updated metadata with date {timestamp}"
                (self.checkout / "metadata.json").write_text(json.dumps([
                    {"hash": self.sha1, "description": description},
                ]))
                self.run_cli("scan", "primary", self.checkout)
                self.ingest()
                self.ingest()
                entry = json.loads(path.read_bytes())[0]
                self.assertEqual(entry["description"], description)
                self.assertEqual(entry["size"], {
                    "raw": len(self.payload), "gz": self.zipped.stat().st_size,
                })
                generated = self.index()["entries"][0]
                if timestamp is None:
                    self.assertNotIn("first_indexed_at", entry)
                    self.assertNotIn("first_indexed_at", generated)
                else:
                    self.assertEqual(entry["first_indexed_at"], timestamp)
                    self.assertEqual(generated["first_indexed_at"], timestamp)
                self.run_cli("check")

    def test_first_indexed_date_uses_all_shared_shards_and_not_preference_order(self):
        self.ingest()
        self.run_cli("ingest", "mirror", self.root / "mirror-checkout", self.input)
        self.set_timestamp("primary", "2024-01-01T00:00:00Z")
        self.set_timestamp("mirror", "2020-01-01T00:00:00Z")
        self.run_cli("build")
        self.assertEqual(self.index()["entries"][0]["first_indexed_at"], "2020-01-01T00:00:00Z")
        archive = self.root / "shards" / "archive.json"
        shutil.copyfile(self.root / "shards" / "mirror.json", archive)
        self.set_timestamp("archive", "2018-01-01T00:00:00Z")
        table_path = self.version / "shards.json"
        table = json.loads(table_path.read_bytes())
        for order in (("primary", "mirror"), ("mirror", "primary"), ("primary",)):
            with self.subTest(order=order):
                table_path.write_text(json.dumps({shard: table[shard] for shard in order}))
                self.run_cli("build")
                entry = self.index()["entries"][0]
                self.assertEqual([location["id"] for location in entry["shards"]], list(order))
                self.assertEqual(entry["first_indexed_at"], "2018-01-01T00:00:00Z")
                self.run_cli("check")
        self.set_timestamp("primary", "2017-01-01T00:00:00Z")
        self.run_cli("build")
        entry = self.index()["entries"][0]
        self.assertEqual(entry["first_indexed_at"], "2017-01-01T00:00:00Z")

    def test_unknown_copy_prevents_a_false_first_indexed_date(self):
        self.ingest()
        self.run_cli("ingest", "mirror", self.root / "mirror-checkout", self.input)
        table_path = self.version / "shards.json"
        table = json.loads(table_path.read_bytes())
        for unknown_shard in ("primary", "mirror"):
            for visible_shards in (("primary", "mirror"), ("primary",)):
                with self.subTest(unknown_shard=unknown_shard, visible_shards=visible_shards):
                    table_path.write_text(json.dumps({shard: table[shard] for shard in visible_shards}))
                    for shard in ("primary", "mirror"):
                        self.set_timestamp(shard, "2020-01-01T00:00:00Z")
                    self.set_timestamp(unknown_shard, None)
                    self.run_cli("build")
                    entry = self.index()["entries"][0]
                    self.assertNotIn("first_indexed_at", entry)
                    self.run_cli("check")

    def test_invalid_timestamp_values_are_rejected(self):
        self.ingest()
        path = self.root / "shards" / "primary.json"
        original = json.loads(path.read_bytes())
        invalid_timestamps = (
            None, True, 123, [], {}, "", "2020-01-01", "2020-01-01T00:00:00+00:00",
            "2020-01-01T00:00:00.000Z", "2020-01-01T00:00:00z", "2020-1-01T00:00:00Z",
            "2020-01-01T00:00:00Z\n", "2021-02-29T00:00:00Z", "2020-13-01T00:00:00Z",
            "2020-01-01T24:00:00Z", "2020-01-01T00:00:60Z",
        )
        for timestamp in invalid_timestamps:
            with self.subTest(timestamp=timestamp):
                original[0]["first_indexed_at"] = timestamp
                path.write_text(json.dumps(original))
                result = self.run_cli("check", success=False)
                self.assertIn("first_indexed_at", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_failed_scan_does_not_publish_new_entry_or_timestamp(self):
        self.data.mkdir(parents=True)
        self.raw.write_bytes(self.payload)
        self.zipped.write_bytes(b"corrupt")
        result = self.run_cli("scan", "primary", self.checkout, success=False)
        self.assertIn("not a gzip file", result.stderr)
        self.assertFalse((self.root / "shards" / "primary.json").exists())
        self.assertFalse((self.version / "generated" / "index.json").exists())

    def test_build_and_check_preserve_stored_timestamps_without_reading_clock(self):
        self.ingest()
        self.set_timestamp("primary", "2020-02-29T12:34:56Z")
        self.run_cli("build")
        before = self.snapshot()
        with mock.patch.object(conta_index, "datetime", wraps=datetime) as clock:
            clock.now.side_effect = AssertionError("build/check must not read the clock")
            with contextlib.redirect_stdout(io.StringIO()):
                conta_index.build(self.root)
                conta_index.check(self.root)
                conta_index.build(self.root)
        self.assertEqual(before, self.snapshot())

    def test_default_ingestion_round_trips_empty_binary_and_large_files(self):
        empty = self.root / "empty.bin"
        empty.write_bytes(b"")
        large = self.root / "large.bin"
        large.write_bytes(bytes(range(256)) * 10000)
        self.run_cli("ingest", "primary", self.checkout, self.input, empty, large)
        for source in (self.input, empty, large):
            payload = source.read_bytes()
            sha1 = hashlib.sha1(payload).hexdigest()
            self.assertEqual((self.data / sha1).read_bytes(), payload)
            compressed = (self.data / f"{sha1}.gz").read_bytes()
            self.assertEqual(gzip.decompress(compressed), payload)
            self.assertEqual(compressed[3] & 8, 0)  # No original filename.
            self.assertEqual(compressed[4:8], b"\x00" * 4)  # Fixed timestamp.
        self.assertEqual(len(list(self.data.iterdir())), 6)
        sources = {entry["sha1"]: entry for entry in json.loads(
            (self.root / "shards" / "primary.json").read_bytes()
        )}
        for entry in self.index()["entries"]:
            sizes = {
                "raw": (self.data / entry["sha1"]).stat().st_size,
                "gz": (self.data / f"{entry['sha1']}.gz").stat().st_size,
            }
            self.assertEqual(entry["shards"], [{"id": "primary", "size": sizes}])
            self.assertEqual(sources[entry["sha1"]]["size"], sizes)
            self.assertNotIn("compressed", sources[entry["sha1"]])
        self.run_cli("check")

    def test_opt_out_then_reingest_backfills_without_changing_description(self):
        self.ingest("--no-gzip")
        self.assertFalse(self.zipped.exists())
        self.assertEqual(self.index()["entries"][0]["shards"], [
            {"id": "primary", "size": {"raw": len(self.payload)}},
        ])
        metadata = (self.checkout / "metadata.json").read_bytes()
        raw_mtime = self.raw.stat().st_mtime_ns
        renamed = self.root / "renamed.bin"
        renamed.write_bytes(self.payload)
        self.run_cli("ingest", "primary", self.checkout, renamed)
        self.assertEqual(gzip.decompress(self.zipped.read_bytes()), self.payload)
        self.assertEqual((self.checkout / "metadata.json").read_bytes(), metadata)
        self.assertEqual(self.raw.stat().st_mtime_ns, raw_mtime)
        before = self.snapshot()
        zipped_mtime = self.zipped.stat().st_mtime_ns
        self.ingest()
        self.ingest("--no-gzip")
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.zipped.stat().st_mtime_ns, zipped_mtime)

    def test_parallel_ingest_deduplicates_and_preserves_serial_output(self):
        duplicate = self.root / "duplicate.bin"
        duplicate.write_bytes(self.payload)
        empty = self.root / "empty.bin"
        empty.write_bytes(b"")
        other = self.root / "other.bin"
        other.write_bytes(bytes(range(256)) * 10000)
        inputs = (self.input, duplicate, empty, other)
        self.run_cli("ingest", "--jobs", "4", "primary", self.checkout, *inputs)
        metadata_path = self.checkout / "metadata.json"
        metadata = metadata_path.read_bytes()
        self.assertEqual([entry["description"] for entry in json.loads(metadata)],
                         [self.input.name, empty.name, other.name])
        self.assertEqual(len(list(self.data.iterdir())), 6)
        before = self.snapshot()
        mtimes = {path: path.stat().st_mtime_ns for path in self.data.iterdir()}
        self.run_cli("ingest", "-j", "2", "primary", self.checkout, *inputs)
        self.assertEqual(metadata_path.read_bytes(), metadata)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(mtimes, {path: path.stat().st_mtime_ns for path in self.data.iterdir()})
        for jobs in (1, 4):
            self.run_cli("scan", "--jobs", str(jobs), "primary", self.checkout)
            self.assertEqual(before, self.snapshot())
        mirror = self.root / "mirror"
        self.run_cli("ingest", "mirror", mirror, *inputs)
        self.assertEqual((mirror / "metadata.json").read_bytes(), metadata)
        for path in self.data.iterdir():
            self.assertEqual(path.read_bytes(), (mirror / "data" / path.name).read_bytes())
        self.run_cli("check")

    def test_ingestion_and_final_scan_actually_overlap_file_work(self):
        other = self.root / "other.bin"
        other.write_bytes(b"another blob")
        other_raw = self.data / hashlib.sha1(other.read_bytes()).hexdigest()
        concurrent_paths = {self.input, other, self.raw, other_raw}
        barrier = threading.Barrier(2, timeout=10)
        digests = conta_index.digests

        def synchronized_digests(path, **kwargs):
            if Path(path) in concurrent_paths:
                # Both the initial hashing and final scan must run two files
                # concurrently; a serial implementation times out at the barrier.
                barrier.wait()
            return digests(path, **kwargs)

        with mock.patch.object(conta_index, "digests", side_effect=synchronized_digests):
            with contextlib.redirect_stdout(io.StringIO()):
                conta_index.ingest(self.root, "primary", self.checkout, [self.input, other], jobs=2)
        self.assertEqual(len(self.index()["entries"]), 2)
        self.run_cli("check")

    def test_parallel_worker_failure_preserves_metadata_and_index(self):
        self.ingest()
        before = self.snapshot()
        metadata = (self.checkout / "metadata.json").read_bytes()
        self.zipped.write_bytes(b"corrupt")
        other = self.root / "other.bin"
        other.write_bytes(b"another blob")
        for command, inputs in (("ingest", (self.input, other)), ("scan", ())):
            with self.subTest(command=command):
                result = self.run_cli(command, "--jobs", "2", "primary", self.checkout,
                                      *inputs, success=False)
                self.assertIn("not a gzip file", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(before, self.snapshot())
                self.assertEqual((self.checkout / "metadata.json").read_bytes(), metadata)
                self.assertEqual(self.zipped.read_bytes(), b"corrupt")
                self.assertEqual(list(self.data.glob(".ingest-*")), [])

    def test_invalid_jobs_are_rejected_before_ingestion_or_scan(self):
        for command, inputs in (("ingest", (self.input,)), ("scan", ())):
            for jobs in ("0", "-1", "invalid"):
                with self.subTest(command=command, jobs=jobs):
                    result = self.run_cli(command, "--jobs", jobs, "primary", self.checkout,
                                          *inputs, success=False)
                    self.assertIn("jobs must be a positive integer", result.stderr)
                    self.assertFalse(self.checkout.exists())

    def test_scan_progress_cli_serial_parallel_and_opt_out(self):
        self.ingest()
        before = self.snapshot()
        mtimes = {path: path.stat().st_mtime_ns for path in self.data.iterdir()}
        for command in ("scan", "ingest"):
            for jobs in (1, 4):
                for enabled in (True, False):
                    with self.subTest(command=command, jobs=jobs, enabled=enabled):
                        args = [command, "--jobs", str(jobs), "primary", str(self.checkout)]
                        if command == "ingest":
                            args.append(str(self.input))
                        if not enabled:
                            args.append("--no-progress")
                        output = TerminalOutput()
                        stdout = io.StringIO()
                        with working_directory(self.root), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(output):
                            self.assertEqual(conta_index.main(args), 0)
                        if enabled:
                            self.assertIn("0/1 blobs (  0%)", output.getvalue())
                            self.assertIn("1/1 blobs (100%)", output.getvalue())
                            self.assertNotIn("2/2", output.getvalue())  # Gzip belongs to the same blob.
                        else:
                            self.assertEqual(output.getvalue(), "")
                        self.assertNotIn("Scan [", stdout.getvalue())
                        self.assertEqual(before, self.snapshot())
                        self.assertEqual(mtimes, {path: path.stat().st_mtime_ns for path in self.data.iterdir()})
        result = self.run_cli("scan", "primary", self.checkout)
        self.assertEqual(result.stderr, "")  # Captured stderr is not a terminal.

    def test_parallel_scan_progress_does_not_wait_for_first_input(self):
        other = self.root / "other.bin"
        other.write_bytes(b"another blob")
        self.run_cli("ingest", "primary", self.checkout, self.input, other)
        before = self.snapshot()
        first = min(path for path in self.data.iterdir() if conta._is_hex(path.name, 40))
        updated = threading.Event()
        first_finished = threading.Event()
        inspect_blob = conta_index.inspect_blob
        writers = []

        class ProgressOutput(TerminalOutput):
            def write(stream, text):
                writers.append(threading.get_ident())
                if "1/2 blobs" in text:
                    self.assertFalse(first_finished.is_set())
                    updated.set()
                return super().write(text)

        def delayed_inspect(path):
            result = inspect_blob(path)
            if path == first:
                if not updated.wait(timeout=5):
                    raise AssertionError("Progress waited for the first input instead of completed workers")
                first_finished.set()
            return result

        output = ProgressOutput()
        with mock.patch.object(conta_index, "inspect_blob", side_effect=delayed_inspect), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(output):
            conta_index.scan(self.root, "primary", self.checkout, jobs=2)
        self.assertTrue(updated.is_set())
        self.assertEqual(set(writers), {threading.get_ident()})
        self.assertIn("2/2 blobs (100%)", output.getvalue())
        self.assertEqual(before, self.snapshot())

    def test_scan_progress_failure_keeps_errors_legible_and_does_not_publish(self):
        self.ingest()
        before = self.snapshot()
        self.zipped.write_bytes(b"corrupt")
        for jobs in (1, 4):
            with self.subTest(jobs=jobs):
                output = TerminalOutput()
                with working_directory(self.root), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(output):
                    self.assertEqual(conta_index.main(["scan", "--jobs", str(jobs), "primary", str(self.checkout)]), 1)
                text = output.getvalue()
                self.assertIn("\nerror:", text)
                self.assertIn("not a gzip file", text)
                self.assertNotIn("100%", text)
                self.assertTrue(text.endswith("\n"))
                self.assertEqual(before, self.snapshot())

    def test_mixed_availability_uses_shard_preference_not_ingestion_order(self):
        mirror = self.root / "mirror-checkout"
        self.run_cli("ingest", "--no-gzip", "mirror", mirror, self.input)
        self.ingest()
        entry = self.index()["entries"][0]
        self.assertEqual(entry["shards"], [
            {"id": "primary", "size": {"raw": len(self.payload), "gz": self.zipped.stat().st_size}},
            {"id": "mirror", "size": {"raw": len(self.payload)}},
        ])
        self.assertEqual(entry["sha256"], hashlib.sha256(self.payload).hexdigest())
        self.run_cli("check")

    def test_gzip_output_does_not_depend_on_input_name_or_checkout(self):
        self.ingest()
        renamed = self.root / "renamed.bin"
        renamed.write_bytes(self.payload)
        mirror = self.root / "mirror-checkout"
        self.run_cli("ingest", "mirror", mirror, renamed)
        self.assertEqual(self.zipped.read_bytes(), (mirror / "data" / self.zipped.name).read_bytes())

    def test_each_shard_retains_its_own_gzip_size(self):
        self.ingest()
        mirror = self.root / "mirror-checkout"
        mirror_data = mirror / "data"
        mirror_data.mkdir(parents=True)
        (mirror_data / self.sha1).write_bytes(self.payload)
        mirror_gzip = gzip.compress(self.payload, compresslevel=1, mtime=123)
        self.assertNotEqual(len(mirror_gzip), self.zipped.stat().st_size)
        (mirror_data / self.zipped.name).write_bytes(mirror_gzip)
        self.run_cli("scan", "mirror", mirror)
        self.assertEqual(self.index()["entries"][0]["shards"], [
            {"id": "primary", "size": {"raw": len(self.payload), "gz": self.zipped.stat().st_size}},
            {"id": "mirror", "size": {"raw": len(self.payload), "gz": len(mirror_gzip)}},
        ])
        self.run_cli("check")

    def test_conflicting_raw_sizes_are_rejected_even_for_shards_outside_the_version(self):
        self.ingest()
        archive = self.root / "shards" / "archive.json"
        entries = json.loads((self.root / "shards" / "primary.json").read_bytes())
        entries[0]["size"]["raw"] += 1
        archive.write_text(json.dumps(entries))
        table_path = self.version / "shards.json"
        table = json.loads(table_path.read_bytes())
        table["archive"] = {"type": "http", "url": "https://example.test/archive/{sha1}"}
        for include_archive in (False, True):
            with self.subTest(include_archive=include_archive):
                table_path.write_text(json.dumps({
                    shard: spec for shard, spec in table.items()
                    if shard != "archive" or include_archive
                }))
                before = self.snapshot()
                for command in ("build", "check"):
                    result = self.run_cli(command, success=False)
                    self.assertIn("raw size", result.stderr)
                    self.assertIn("primary", result.stderr)
                    self.assertIn("archive", result.stderr)
                    self.assertEqual(before, self.snapshot())
                self.assert_scan_fails_without_publishing("raw size")

    def test_build_and_check_use_stored_sizes_without_the_checkout(self):
        self.ingest()
        before = self.snapshot()
        shutil.rmtree(self.checkout)
        self.run_cli("build")
        self.run_cli("check")
        self.assertEqual(before, self.snapshot())

    def test_scan_discovers_existing_gzip_and_ingestion_preserves_it(self):
        self.ingest("--no-gzip")
        compressed = gzip.compress(self.payload, compresslevel=1, mtime=123)
        self.zipped.write_bytes(compressed)
        self.run_cli("scan", "primary", self.checkout)
        self.assertEqual(self.index()["entries"][0]["shards"][0]["size"], {
            "raw": len(self.payload), "gz": len(compressed),
        })
        self.ingest()
        self.assertEqual(self.zipped.read_bytes(), compressed)

    def test_scan_rejects_incorrect_stored_sizes_without_publishing(self):
        self.ingest()
        source = self.root / "shards" / "primary.json"
        original = source.read_bytes()
        for representation in ("raw", "gz"):
            with self.subTest(representation=representation):
                entries = json.loads(original)
                entries[0]["size"][representation] += 1
                source.write_text(json.dumps(entries))
                self.assert_scan_fails_without_publishing("size")

    def test_scan_rejects_a_valid_recompressed_blob_with_a_changed_size(self):
        self.ingest()
        recompressed = gzip.compress(self.payload, compresslevel=1, mtime=123)
        self.assertNotEqual(len(recompressed), self.zipped.stat().st_size)
        self.zipped.write_bytes(recompressed)
        self.assert_scan_fails_without_publishing("size")

    def test_failed_scan_does_not_publish_discovered_sizes_or_new_entries(self):
        self.ingest("--no-gzip")
        self.zipped.write_bytes(gzip.compress(self.payload))
        added = b"new blob discovered before scan fails"
        added_sha1 = hashlib.sha1(added).hexdigest()
        (self.data / added_sha1).write_bytes(added)
        (self.data / f"{added_sha1}.gz").write_bytes(b"corrupt")
        self.assert_scan_fails_without_publishing("not a gzip file")

    def test_scan_rejects_corrupt_or_mismatched_gzip_without_publishing(self):
        self.ingest()
        valid = self.zipped.read_bytes()
        wrong_crc = bytearray(valid)
        wrong_crc[-8] ^= 1
        for compressed in (b"", b"not gzip", valid[:-4], bytes(wrong_crc), gzip.compress(b"wrong content")):
            with self.subTest(compressed=compressed[:10]):
                self.zipped.write_bytes(compressed)
                self.assert_scan_fails_without_publishing(str(self.zipped))

    def test_empty_file_is_not_a_valid_gzip_companion_for_an_empty_blob(self):
        self.input.write_bytes(b"")
        self.ingest()
        empty_sha1 = hashlib.sha1(b"").hexdigest()
        (self.data / f"{empty_sha1}.gz").write_bytes(b"")
        self.assert_scan_fails_without_publishing("not a gzip file")

    def test_missing_advertised_gzip_is_rejected(self):
        self.ingest()
        self.zipped.unlink()
        self.assert_scan_fails_without_publishing("advertised gzip is missing")

    def test_orphan_gzip_is_rejected(self):
        self.ingest()
        self.raw.unlink()
        self.assert_scan_fails_without_publishing("matching raw blob")

    def test_missing_raw_only_blob_is_rejected(self):
        self.ingest("--no-gzip")
        self.raw.unlink()
        self.assert_scan_fails_without_publishing("shards are append-only")

    def test_known_raw_content_is_reverified(self):
        self.ingest()
        self.raw.write_bytes(b"changed")
        self.assert_scan_fails_without_publishing("content hashes to")

    def test_ingestion_rejects_corrupt_existing_raw_before_compressing(self):
        self.ingest("--no-gzip")
        self.raw.write_bytes(b"changed")
        before = self.snapshot()
        result = self.run_cli("ingest", "primary", self.checkout, self.input, success=False)
        self.assertIn("does not match expected", result.stderr)
        self.assertEqual(before, self.snapshot())
        self.assertFalse(self.zipped.exists())
        self.assertEqual(self.raw.read_bytes(), b"changed")

    def test_ingestion_does_not_rewrite_corrupt_existing_gzip(self):
        self.ingest()
        self.zipped.write_bytes(b"corrupt")
        before = self.snapshot()
        result = self.run_cli("ingest", "primary", self.checkout, self.input, success=False)
        self.assertIn("not a gzip file", result.stderr)
        self.assertEqual(self.zipped.read_bytes(), b"corrupt")
        self.assertEqual(before, self.snapshot())

    def test_lfs_pointers_are_rejected_for_both_representations(self):
        self.ingest()
        pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n"
        for path in (self.raw, self.zipped):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(pointer)
                self.assert_scan_fails_without_publishing("git lfs pull")
                path.write_bytes(original)

    def test_invalid_source_size_maps_are_rejected(self):
        self.ingest()
        source = self.root / "shards" / "primary.json"
        entries = json.loads(source.read_bytes())
        invalid_sizes = (
            True, 123, "gz", None, [], {}, {"gz": 1},
            {"raw": -1}, {"raw": True}, {"raw": False}, {"raw": 1.5},
            {"raw": "123"}, {"raw": None}, {"raw": []}, {"raw": {}},
            {"raw": len(self.payload), "gz": -1},
            {"raw": len(self.payload), "gz": True},
            {"raw": len(self.payload), "gz": 1.5},
            {"raw": len(self.payload), "gz": "123"},
            {"raw": len(self.payload), "gz": None},
            *({"raw": len(self.payload), extension: 123} for extension in (
                "", ".gz", "../gz", "GZ", "gz?x", "gz\n", "raw.gz", "with-dash",
            )),
        )
        for size in invalid_sizes:
            with self.subTest(size=size):
                entries[0]["size"] = size
                source.write_text(json.dumps(entries))
                result = self.run_cli("check", success=False)
                self.assertIn("size", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_algorithm_name_requires_the_extension_token(self):
        self.ingest()
        source = self.root / "shards" / "primary.json"
        entries = json.loads(source.read_bytes())
        entries[0]["size"]["gzip"] = entries[0]["size"].pop("gz")
        source.write_text(json.dumps(entries))
        result = self.run_cli("check", success=False)
        self.assertIn("use gz instead of gzip", result.stderr)

    def test_missing_source_size_is_rejected(self):
        self.ingest("--no-gzip")
        source = self.root / "shards" / "primary.json"
        entries = json.loads(source.read_bytes())
        entries[0].pop("size")
        source.write_text(json.dumps(entries))
        result = self.run_cli("check", success=False)
        self.assertIn("size", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_scan_preserves_future_compression_sizes_when_adding_gzip(self):
        self.ingest("--no-gzip")
        source = self.root / "shards" / "primary.json"
        entries = json.loads(source.read_bytes())
        entries[0]["size"]["zst"] = 42
        entries[0]["size"]["lz4"] = 0
        source.write_text(json.dumps(entries))
        self.run_cli("build")
        self.assertEqual(self.index()["entries"][0]["shards"][0]["size"], {
            "raw": len(self.payload), "zst": 42, "lz4": 0,
        })
        self.zipped.write_bytes(gzip.compress(self.payload))
        self.run_cli("scan", "primary", self.checkout)
        before = self.snapshot()
        self.run_cli("scan", "primary", self.checkout)
        self.assertEqual(before, self.snapshot())
        sizes = {"raw": len(self.payload), "zst": 42, "lz4": 0, "gz": self.zipped.stat().st_size}
        self.assertEqual(json.loads(source.read_bytes())[0]["size"], sizes)
        self.assertEqual(self.index()["entries"][0]["shards"][0]["size"], sizes)
        self.run_cli("check")
        self.zipped.unlink()
        self.assert_scan_fails_without_publishing("advertised gzip is missing")

    def test_legacy_compression_fields_are_not_silently_dropped(self):
        self.ingest()
        source = self.root / "shards" / "primary.json"
        entries = json.loads(source.read_bytes())
        for field, value in (
            ("gzip", True), ("compression", ["gz"]), ("compressed", ["gz"]),
        ):
            with self.subTest(field=field):
                entries[0][field] = value
                source.write_text(json.dumps(entries))
                result = self.run_cli("check", success=False)
                self.assertIn(field, result.stderr)
                self.assertIn("size", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                entries[0].pop(field)

    def test_interrupted_compression_never_publishes_partial_gzip(self):
        self.ingest("--no-gzip")

        def interrupt(source, output, **kwargs):
            output.write(b"partial")
            raise OSError("interrupted")

        with mock.patch.object(conta_index.shutil, "copyfileobj", side_effect=interrupt):
            with self.assertRaisesRegex(OSError, "interrupted"):
                conta_index.store_blob(self.raw, self.zipped, conta_index.digests(self.raw), compressed=True)
        self.assertFalse(self.zipped.exists())
        self.assertEqual(list(self.data.iterdir()), [self.raw])

    def test_publication_survives_nfs_deferred_unlink(self):
        self.data.mkdir(parents=True)
        expected = conta_index.digests(self.input)

        def defer_unlink(path, *, dir_fd=None):
            # NFS can keep an unlinked file under a hidden name until the
            # last filesystem reference closes, leaving its directory nonempty.
            hidden = Path(path).with_name(".nfs-" + Path(path).name)
            conta_index.os.rename(path, hidden, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)

        for compressed, destination in ((False, self.raw), (True, self.zipped)):
            with self.subTest(compressed=compressed):
                with mock.patch.object(conta_index.os, "unlink", side_effect=defer_unlink):
                    conta_index.store_blob(self.input, destination, expected, compressed=compressed)
                self.assertEqual(conta_index.digests(destination, compressed=compressed), expected)

    def test_concurrent_publication_never_overwrites_existing_gzip(self):
        self.ingest("--no-gzip")
        for winner in (gzip.compress(self.payload, mtime=123), b"corrupt"):
            with self.subTest(valid=winner != b"corrupt"):
                def publish_first(source, destination):
                    destination.write_bytes(winner)
                    raise FileExistsError(destination)

                with mock.patch.object(conta_index.os, "link", side_effect=publish_first):
                    if winner == b"corrupt":
                        with self.assertRaises(conta.ContaError):
                            conta_index.store_blob(self.raw, self.zipped, conta_index.digests(self.raw), compressed=True)
                    else:
                        conta_index.store_blob(self.raw, self.zipped, conta_index.digests(self.raw), compressed=True)
                self.assertEqual(self.zipped.read_bytes(), winner)
                self.assertEqual(len(list(self.data.iterdir())), 2)
                self.zipped.unlink()


def conta_cli():
    if os.environ.get("CONTA_TEST_CLI"):
        return Path(os.environ["CONTA_TEST_CLI"])
    candidates = sorted(REPOSITORY.glob("build*/src/conta/conta"))
    return candidates[0] if candidates else None


class ClientRoundTripTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        checkout = self.root / "checkout"
        (self.root / "shards").mkdir()
        self.version = self.root / "version" / "1"
        self.version.mkdir(parents=True)
        (self.version / "meta.json").write_text(json.dumps({"format": 1, "index": {"location": "generated/index.json"}}))
        (self.version / "shards.json").write_text(json.dumps({"local": {"type": "http", "url": (checkout / "data").as_uri() + "/{sha1}"}}))
        self.payload = b"a blob served through a generated index" * 1000
        source = self.root / "blob.bin"
        source.write_bytes(self.payload)
        self.sha1 = hashlib.sha1(self.payload).hexdigest()
        with contextlib.redirect_stdout(io.StringIO()):
            conta_index.ingest(self.root, "local", checkout, [source])
        self.cache = self.root / "cache"
        self.gzip_url = (checkout / "data").as_uri() + f"/{self.sha1}.gz"

    def test_python_client_downloads_the_advertised_gzip(self):
        settings = conta.Config(cache=str(self.cache), entry_points=[(self.version / "meta.json").as_uri()])
        log = io.StringIO()
        with contextlib.redirect_stderr(log):
            self.assertEqual(conta.load(self.sha1, settings), self.payload)
        self.assertIn(f"from {self.gzip_url}\n", log.getvalue())
        self.assertTrue((self.cache / f"{self.sha1}.gz").exists())
        self.assertFalse((self.cache / self.sha1).exists())

    def test_cpp_client_downloads_the_advertised_gzip(self):
        cli = conta_cli()
        if cli is None:
            self.skipTest("conta CLI not built (cmake --build build --target conta)")
        self.cache.mkdir()
        shutil.copyfile(self.version / "generated" / "index.json", self.cache / conta.CACHED_INDEX_NAME)
        environment = {key: value for key, value in os.environ.items() if not key.startswith("CONTA_")}
        environment["CONTA_CACHE"] = str(self.cache)
        result = subprocess.run([str(cli), self.sha1], capture_output=True, text=True, env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()), self.cache / self.sha1)
        self.assertEqual((self.cache / self.sha1).read_bytes(), self.payload)
        self.assertIn(f"from {self.gzip_url}\n", result.stderr)


if __name__ == "__main__":
    unittest.main()
