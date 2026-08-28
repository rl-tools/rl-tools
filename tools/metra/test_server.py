#!/usr/bin/env python3
import http.server, json, os, sys, tempfile, threading, time, unittest, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metra
from metra import server


class MetraServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        server.DB_FILE = os.path.join(cls.tmp.name, "test.sqlite")
        server.init_db()
        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.RequestHandler)
        cls.url = "http://127.0.0.1:{}".format(cls.httpd.server_address[1])
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.tmp.cleanup()

    def _post(self, path, data):
        request = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(request)

    def _assert_http_error(self, code, fn, *args, **kwargs):
        with self.assertRaises(urllib.error.HTTPError) as context:
            fn(*args, **kwargs)
        self.assertEqual(context.exception.code, code)
        context.exception.close()

    def test_log_fetch_roundtrip(self):
        before = time.time()
        row_id = metra.log("roundtrip/scalar", 1.5, commit="a" * 40, run="run-a", url=self.url)
        metra.log("roundtrip/list", [1, 2, 3], commit="a" * 40, run="run-a", url=self.url)
        metra.log("roundtrip/struct", {"lr": 0.001}, commit="a" * 40, run="run-a", time=123.0, url=self.url)
        scalar = metra.fetch(name="roundtrip/scalar", url=self.url)
        self.assertEqual(len(scalar), 1)
        self.assertEqual(scalar[0]["id"], row_id)
        self.assertEqual(scalar[0]["value"], 1.5)
        self.assertEqual(scalar[0]["value_scalar"], 1.5)
        self.assertGreaterEqual(scalar[0]["time"], before)
        list_rows = metra.fetch(name="roundtrip/list", url=self.url)
        self.assertEqual(list_rows[0]["value"], [1, 2, 3])
        self.assertIsNone(list_rows[0]["value_scalar"])
        struct_rows = metra.fetch(name="roundtrip/struct", url=self.url)
        self.assertEqual(struct_rows[0]["value"], {"lr": 0.001})
        self.assertEqual(struct_rows[0]["time"], 123.0)

    def test_batch_and_filters(self):
        body = [
            {"name": "batch/a", "value": 1, "commit": "b" * 40, "run": "run-b", "time": 100.0},
            {"name": "batch/a", "value": 2, "commit": "c" * 40, "run": "run-c", "time": 200.0},
        ]
        with self._post("/api/log", json.dumps(body).encode()) as response:
            self.assertEqual(response.status, 201)
            ids = json.loads(response.read())["ids"]
        self.assertEqual(len(ids), 2)
        rows = metra.fetch(name="batch/a", url=self.url)
        self.assertEqual([row["value"] for row in rows], [2, 1])
        self.assertEqual(len(metra.fetch(name="batch/a", run="run-b", url=self.url)), 1)
        self.assertEqual(len(metra.fetch(name="batch/a", since=150.0, url=self.url)), 1)
        self.assertEqual(len(metra.fetch(name="batch/a", until=150.0, url=self.url)), 1)
        self.assertEqual(len(metra.fetch(name="batch/a", commit="ccccccc", url=self.url)), 1)
        self.assertEqual(len(metra.fetch(name="batch/a", limit=1, url=self.url)), 1)
        self.assertIn("batch/a", metra.names(url=self.url))

    def test_flag_comment(self):
        row_id = metra.log("flagging/metric", 1.0, commit="d" * 40, run="run-d", url=self.url)
        metra.flag(row_id, url=self.url)
        self.assertEqual(metra.fetch(name="flagging/metric", url=self.url), [])
        rows = metra.fetch(name="flagging/metric", include_unreliable=True, url=self.url)
        self.assertEqual(rows[0]["unreliable"], 1)
        metra.flag(row_id, unreliable=False, url=self.url)
        self.assertEqual(len(metra.fetch(name="flagging/metric", url=self.url)), 1)
        metra.comment(row_id, "known flaky machine", url=self.url)
        self.assertEqual(metra.fetch(name="flagging/metric", url=self.url)[0]["comment"], "known flaky machine")
        self._assert_http_error(404, metra.flag, 999999, url=self.url)

    def test_bad_requests(self):
        self._assert_http_error(400, self._post, "/api/log", b"not json")
        self._assert_http_error(400, self._post, "/api/log", json.dumps({"value": 1}).encode())
        self._assert_http_error(400, self._post, "/api/log", json.dumps({"name": "bad/time", "value": 1, "time": "yesterday"}).encode())
        self._assert_http_error(404, urllib.request.urlopen, self.url + "/api/unknown")

    def test_html(self):
        with urllib.request.urlopen(self.url + "/") as response:
            self.assertEqual(response.status, 200)
            self.assertIn("text/html", response.headers.get("Content-Type"))
            self.assertIn(b"<table", response.read())


if __name__ == "__main__":
    unittest.main()
