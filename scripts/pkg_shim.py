"""Serve accentuk.Accentor at /v1/stress with the Go API's JSON shape, so the existing benchmark scripts can
measure the package exactly as they measure the service.

    pkg_shim.py port mode device [asset_dir]
"""
import os
import dataclasses, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from accentuk import Accentor

PORT, MODE, DEVICE = int(sys.argv[1]), sys.argv[2], sys.argv[3]
ASSETS = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("ACCENTUK_ASSETS", os.path.expanduser("~/.cache/accentuk"))
engine = Accentor(asset_dir=ASSETS, mode=MODE, device=DEVICE)
lock = threading.Lock()   # one engine, one request at a time
engine.stress("Мої сестри прийшли додому.")   # warm the models before the first measured request

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        self._send(200, {"status": "ok"})
    def do_POST(self):
        if not self.path.startswith("/v1/stress"):
            return self._send(404, {"error": "not found"})
        req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        with lock:
            res = engine.stress(req.get("text", ""), combiner=bool(req.get("combiner", False))) \
                if "combiner" in engine.stress.__code__.co_varnames else engine.stress(req.get("text", ""))
        self._send(200, {"text": res.text, "tokens": [dataclasses.asdict(t) for t in res.tokens], "warnings": res.warnings})

print(f"accentuk {MODE} on {DEVICE} at :{PORT}", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
