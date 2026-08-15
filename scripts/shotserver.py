#!/usr/bin/env python3
"""Static server + a POST endpoint pages can push their canvas to.

Why this exists
---------------
Verifying that the garment LOOKS right needs an actual rendered frame. The two
obvious routes both fail here:

  - headless Chrome/Edge: gets a WebGPU adapter on this machine but
    requestDevice() hangs, so the cloth solver never starts
  - screen capture: fragile (Windows refuses foreground raises from a
    background process, so you photograph whatever is underneath) and it
    photographs the user's desktop, which is nobody's business

So the page renders normally in a real window and POSTs its own canvas back.
The pixels are exact, and nothing outside the page is ever captured.

Usage:
    python scripts/shotserver.py [port]
    # then open  http://localhost:8010/web/stage6/tryon-demo.html?shot=demo.png
    # the PNG lands in docs/media/
"""

import base64
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOT_DIR = os.path.join(ROOT, "docs", "media")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def do_POST(self):  # noqa: N802  (stdlib naming)
        if not self.path.startswith("/_shot"):
            self.send_error(404)
            return

        name = "shot.png"
        if "?" in self.path:
            for part in self.path.split("?", 1)[1].split("&"):
                if part.startswith("name="):
                    name = part[5:]

        # Refuse anything that could escape docs/media.
        name = os.path.basename(name)
        if not name.endswith(".png"):
            name += ".png"

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8", "replace")
        if "," in body:
            body = body.split(",", 1)[1]

        os.makedirs(SHOT_DIR, exist_ok=True)
        path = os.path.join(SHOT_DIR, name)
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(body))

        print(f"saved {path} ({os.path.getsize(path)} bytes)", flush=True)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(path)))
        self.end_headers()
        self.wfile.write(path.encode())

    def log_message(self, fmt, *args):
        if "_shot" in (args[0] if args else ""):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8010
    print(f"serving {ROOT} on http://localhost:{port}  (shots -> docs/media/)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
