"""Serve the car-labelling page for a folder made by `prepare`. Labels are saved as you click.

Usage: python3 -m cardet.labeller data/cars/newtown   (then open http://127.0.0.1:8790)
"""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("labeller.html")
NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def make_handler(root: Path):
    labels = root / "labels"
    labels.mkdir(exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, kind: str, status=200):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _label_file(self, prefix):
            name = self.path[len(prefix):]
            return labels / f"{name}.json" if NAME.match(name) else None

        def do_GET(self):
            if self.path == "/":
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/tiles.json":
                data = json.loads((root / "tiles.json").read_text())
                for t in data["tiles"]:
                    f = labels / f"{t['name']}.json"
                    t["saved"] = json.loads(f.read_text()) if f.exists() else None
                self._send(json.dumps(data).encode(), "application/json")
            elif self.path.startswith("/tiles/") and (root / self.path[1:]).resolve().parent == (root / "tiles").resolve():
                self._send((root / self.path[1:]).read_bytes(), "image/jpeg")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            f = self._label_file("/labels/")
            if f is None:
                return self._send(b"bad name", "text/plain", 400)
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            f.write_text(json.dumps(json.loads(body)))
            self._send(b"{}", "application/json")

        def log_message(self, *args):
            pass

    return Handler


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    print(f"labelling {a.folder} at http://127.0.0.1:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(Path(a.folder))).serve_forever()
