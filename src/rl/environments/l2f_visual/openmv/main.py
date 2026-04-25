# MicroPython file-manager web UI.
# Serves a listing at /, uploads via multipart POST /upload,
# downloads via /dl/<name>, deletes via POST /rm.
# Streams uploads to flash so large files don't blow up RAM.

import socket
import os
import gc

AP_SSID   = "mpy-files"
# CYW43 (modern OpenMV, Pico W) needs a country code set or the AP
# comes up in a half-broken state ("network activation failed" on the
# client). Set to your ISO country, e.g. "US", "DE", "GB".
AP_COUNTRY = "DE"
# With CYW43 use a normal 8-63 char WPA2 passphrase, or "" for open.
AP_PASS   = "uploadme123"
AP_CHANNEL = 6
PORT      = 80
UPLOAD_DIR = "/flash/"    # OpenMV-AE3 (alif port): only /flash/ is writable
CHUNK     = 1024

SHUTDOWN = False


# ---------- wifi ----------
def wifi_ap():
    import network, time
    # CYW43 needs regulatory domain set before the radio is useful
    try:
        network.country(AP_COUNTRY)
    except Exception as e:
        print("country() not supported:", e)
    # kill any stale STA config first
    try:
        network.WLAN(network.STA_IF).active(False)
    except Exception:
        pass
    ap = network.WLAN(network.AP_IF)

    # config param names differ across ports:
    #   OpenMV:   ssid / key / security (with ap.WPA_WPA2)
    #   ESP32:    essid / password / authmode
    #   CYW43:    ssid / key / security
    # so try aliases and keep the first that sticks.
    def try_cfg(**aliases):
        for k, v in aliases.items():
            try:
                ap.config(**{k: v})
                return True
            except (ValueError, OSError, TypeError):
                continue
        return False

    try_cfg(ssid=AP_SSID, essid=AP_SSID)
    if AP_PASS:
        try_cfg(key=AP_PASS, password=AP_PASS)
        # CYW43: 3 = WPA2-PSK
        sec = getattr(ap, "WPA_WPA2",
                      getattr(ap, "SEC_WPA_WPA2",
                              getattr(network, "AUTH_WPA2_PSK", 3)))
        try_cfg(security=sec, authmode=sec)
    else:
        sec = getattr(ap, "OPEN",
                      getattr(ap, "SEC_OPEN",
                              getattr(network, "AUTH_OPEN", 0)))
        try_cfg(security=sec, authmode=sec)
    try_cfg(channel=AP_CHANNEL)   # silently skipped if unsupported

    ap.active(True)
    for _ in range(50):
        if ap.active():
            break
        time.sleep_ms(100)
    ip = ap.ifconfig()[0]
    print("AP up: ssid=%s  ip=%s  ifconfig=%s" %
          (AP_SSID, ip, ap.ifconfig()))
    try:
        print("status:", ap.status())
    except Exception:
        pass
    return ap, ip


def wifi_down(ap):
    try: ap.active(False)
    except Exception as e: print("ap.active(False):", e)
    try: ap.deinit()
    except Exception as e: print("ap.deinit():", e)
    try:
        import network
        network.WLAN(network.STA_IF).active(False)
    except Exception as e: print("sta off:", e)
    print("wifi down.")


# ---------- fs ----------
def list_files(path=UPLOAD_DIR):
    out = []
    for name in os.listdir(path):
        try:
            st = os.stat(path + name)
            if st[0] & 0x4000:   # skip dirs
                continue
            out.append((name, st[6]))
        except OSError:
            pass
    out.sort()
    return out

def free_bytes():
    try:
        s = os.statvfs(UPLOAD_DIR)
        return s[0] * s[3]
    except Exception:
        return -1

def _is_alnum(c):
    return ("a" <= c <= "z") or ("A" <= c <= "Z") or ("0" <= c <= "9")

def safe_name(n):
    n = n.replace("\\", "/").rsplit("/", 1)[-1]
    # keep it simple: strip anything weird
    return "".join(c for c in n if _is_alnum(c) or c in "._-") or "file"


# ---------- http helpers ----------
def url_unquote(s):
    s = s.replace("+", " ")
    out = ""
    i = 0
    while i < len(s):
        if s[i] == "%" and i + 2 < len(s):
            try:
                out += chr(int(s[i+1:i+3], 16)); i += 3; continue
            except ValueError:
                pass
        out += s[i]; i += 1
    return out

def send_status(conn, status, headers=None, body=b""):
    headers = headers or {}
    if "Content-Length" not in headers:
        headers["Content-Length"] = len(body)
    headers["Connection"] = "close"
    out = b"HTTP/1.0 " + status.encode() + b"\r\n"
    for k, v in headers.items():
        out += ("%s: %s\r\n" % (k, v)).encode()
    out += b"\r\n"
    conn.write(out)
    if body:
        conn.write(body)

def parse_request(conn):
    first = conn.readline()
    if not first:
        return None
    try:
        method, path, _ = first.decode().split(" ", 2)
    except ValueError:
        return None
    headers = {}
    while True:
        line = conn.readline()
        if not line or line == b"\r\n":
            break
        k, _, v = line.decode().partition(":")
        headers[k.strip().lower()] = v.strip()
    return method, path.strip(), headers


# ---------- multipart streaming ----------
def stream_upload(conn, boundary, content_length):
    """Parse ONE file field from a multipart body, stream to disk."""
    delim      = b"--" + boundary.encode()
    end_marker = b"\r\n" + delim
    remaining  = content_length

    # 1. skip to first boundary line
    while remaining > 0:
        line = conn.readline()
        if not line:
            raise OSError("eof before boundary")
        remaining -= len(line)
        if line.rstrip(b"\r\n") == delim:
            break

    # 2. part headers -> extract filename
    filename = None
    while remaining > 0:
        line = conn.readline()
        remaining -= len(line)
        if line == b"\r\n":
            break
        if line.lower().startswith(b"content-disposition"):
            i = line.find(b'filename="')
            if i >= 0:
                j = line.find(b'"', i + 10)
                filename = line[i+10:j].decode()
    if not filename:
        raise OSError("no filename in upload")
    filename = safe_name(filename)

    # 3. stream body until \r\n--boundary
    path = UPLOAD_DIR + filename
    size = 0
    buf  = b""
    keep = len(end_marker) - 1
    f = open(path, "wb")
    try:
        while remaining > 0:
            chunk = conn.read(CHUNK if remaining >= CHUNK else remaining)
            if not chunk:
                break
            remaining -= len(chunk)
            buf += chunk
            idx = buf.find(end_marker)
            if idx >= 0:
                f.write(buf[:idx])
                size += idx
                buf = buf[idx:]
                break
            if len(buf) > keep:
                f.write(buf[:-keep])
                size += len(buf) - keep
                buf = buf[-keep:]
    finally:
        f.close()
    return filename, size


# ---------- pages ----------
PAGE = """<!doctype html>
<meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>files</title>
<style>
body{font-family:monospace;max-width:36em;margin:2em auto;padding:0 1em;color:#222}
h1{font-size:1.1em;margin:0 0 1em}
table{width:100%%;border-collapse:collapse}
td,th{text-align:left;padding:.2em .5em;border-bottom:1px solid #ddd}
th:nth-child(2),td:nth-child(2){text-align:right}
form.inline{display:inline;margin:0}
button{font-family:inherit;border:1px solid #888;background:#f4f4f4;padding:.1em .5em;cursor:pointer}
.bar{margin:1em 0;padding:.5em;border:1px solid #ccc;background:#fafafa}
small{color:#666}
</style>
<h1>files @ %s</h1>
<div class=bar>
<form method=post action=/upload enctype=multipart/form-data>
<input type=file name=f required>
<button>upload</button>
</form>
<form class=inline method=post action=/reboot onsubmit="return confirm('reboot the board?')">
<button>reboot</button>
</form>
<form class=inline method=post action=/exit onsubmit="return confirm('stop server and start inference?')">
<button>exit server</button>
</form>
</div>
<table>
<tr><th>name<th>size<th></tr>
%s
</table>
<p><small>free: %s bytes</small></p>
"""

ROW = '<tr><td><a href="/dl/%s">%s</a><td>%d<td><form class=inline method=post action=/rm><input type=hidden name=f value="%s"><button>delete</button></form></tr>'

def page_index(host):
    rows = "\n".join(ROW % (n, n, s, n) for n, s in list_files())
    return (PAGE % (host, rows, free_bytes())).encode()


# ---------- handlers ----------
def h_index(conn, headers):
    host = headers.get("host", "")
    body = page_index(host)
    send_status(conn, "200 OK",
                {"Content-Type": "text/html; charset=utf-8"}, body)

def h_download(conn, name):
    name = safe_name(url_unquote(name))
    path = UPLOAD_DIR + name
    try:
        st = os.stat(path)
    except OSError:
        send_status(conn, "404 Not Found"); return
    send_status(conn, "200 OK", {
        "Content-Type": "application/octet-stream",
        "Content-Length": st[6],
        "Content-Disposition": 'attachment; filename="%s"' % name,
    })
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b: break
            conn.write(b)

def h_upload(conn, headers):
    ctype = headers.get("content-type", "")
    if "multipart/form-data" not in ctype or "boundary=" not in ctype:
        send_status(conn, "400 Bad Request"); return
    boundary = ctype.split("boundary=", 1)[1].strip()
    if boundary.startswith('"'):
        boundary = boundary[1:-1]
    length = int(headers.get("content-length", "0"))
    try:
        name, size = stream_upload(conn, boundary, length)
        print("up:", name, size)
    except Exception as e:
        print("upload failed:", e)
        send_status(conn, "400 Bad Request"); return
    send_status(conn, "303 See Other", {"Location": "/"})

def h_exit(conn):
    global SHUTDOWN
    SHUTDOWN = True
    body = b"<!doctype html><meta charset=utf-8><title>exiting</title>" \
           b"<p>server stopping. starting inference on main thread.</p>"
    send_status(conn, "200 OK",
                {"Content-Type": "text/html; charset=utf-8"}, body)

def h_reboot(conn):
    body = b"<!doctype html><meta charset=utf-8><title>rebooting</title>" \
           b"<meta http-equiv=refresh content=5;url=/>" \
           b"<p>rebooting\xe2\x80\xa6 page will reload in 5s.</p>"
    send_status(conn, "200 OK",
                {"Content-Type": "text/html; charset=utf-8"}, body)
    try: conn.close()
    except: pass
    import machine, time
    time.sleep_ms(200)
    machine.reset()

def h_remove(conn, headers):
    length = int(headers.get("content-length", "0"))
    body = conn.read(length).decode() if length else ""
    params = {}
    for pair in body.split("&"):
        k, _, v = pair.partition("=")
        params[url_unquote(k)] = url_unquote(v)
    name = safe_name(params.get("f", ""))
    if name:
        try: os.remove(UPLOAD_DIR + name)
        except OSError as e: print("rm:", e)
    send_status(conn, "303 See Other", {"Location": "/"})


# ---------- server ----------
def serve(ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", PORT))
    s.listen(2)
    print("http://%s:%d/" % (ip, PORT))
    while not SHUTDOWN:
        conn, addr = s.accept()
        try:
            req = parse_request(conn)
            if not req:
                continue
            method, path, headers = req
            print(method, path)
            if method == "GET" and path == "/":
                h_index(conn, headers)
            elif method == "GET" and path.startswith("/dl/"):
                h_download(conn, path[4:])
            elif method == "POST" and path == "/upload":
                h_upload(conn, headers)
            elif method == "POST" and path == "/rm":
                h_remove(conn, headers)
            elif method == "POST" and path == "/reboot":
                h_reboot(conn)
            elif method == "POST" and path == "/exit":
                h_exit(conn)
            else:
                send_status(conn, "404 Not Found")
        except Exception as e:
            print("err:", e)
        finally:
            try: conn.close()
            except: pass
            gc.collect()
    try: s.close()
    except: pass
    print("server stopped.")


INFERENCE_PATH = UPLOAD_DIR + "inference.py"

def run_inference():
    try:
        src = open(INFERENCE_PATH).read()
    except OSError as e:
        print("inference: %s not found (%s) — upload via web UI and reboot" %
              (INFERENCE_PATH, e))
        return
    # Run inference.py as if it were __main__ so its `while True:` drives the
    # foreground; the server runs on the background thread alongside it.
    exec(compile(src, INFERENCE_PATH, "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    ap, ip = wifi_ap()
    serve(ip)
    wifi_down(ap)
    run_inference()

