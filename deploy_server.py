#!/usr/bin/env python3
"""
Private Render entry point for CollegeOS.

Keeps the existing collegeos.py, dashboard_server.py and
dashboard.html unchanged (all files are in the project root). A Basic Auth gate protects every
dashboard/API request; the original dashboard server remains bound to
127.0.0.1 and is not exposed publicly.
"""

from __future__ import annotations

import base64
import hmac
import http.client
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT

# Set this before importing collegeos.py because it reads COLLEGEOS_STATE
# when the module is imported. Render can override this with a persistent
# disk path later, e.g. /var/data/collegeos_state.json.
os.environ.setdefault("COLLEGEOS_STATE", str(ROOT / "collegeos_state.json"))
os.chdir(ROOT)
sys.path.insert(0, str(FRONTEND))

import dashboard_server  # noqa: E402

PUBLIC_PORT = int(os.environ.get("PORT", "10000"))
INTERNAL_PORT = int(os.environ.get("COLLEGEOS_INTERNAL_PORT", "8765"))
USERNAME = os.environ.get("COLLEGEOS_USERNAME", "")
PASSWORD = os.environ.get("COLLEGEOS_PASSWORD", "")

if not USERNAME or not PASSWORD:
    raise SystemExit(
        "Set COLLEGEOS_USERNAME and COLLEGEOS_PASSWORD as environment secrets."
    )
if len(PASSWORD) < 16:
    raise SystemExit("COLLEGEOS_PASSWORD must be at least 16 characters long.")

EXPECTED_AUTH = "Basic " + base64.b64encode(
    f"{USERNAME}:{PASSWORD}".encode("utf-8")
).decode("ascii")


class PrivateProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        # Avoid writing request details or credentials to application logs.
        pass

    def send_bytes(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        if hmac.compare_digest(supplied, EXPECTED_AUTH):
            return True
        body = b"Authentication required."
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="CollegeOS", charset="UTF-8"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        return False

    def do_GET(self):
        if self.path == "/healthz":
            return self.send_bytes(200, b"ok", "text/plain; charset=utf-8")
        if not self.authorized():
            return
        self.forward()

    def do_POST(self):
        if not self.authorized():
            return
        self.forward()

    def do_HEAD(self):
        if not self.authorized():
            return
        self.forward()

    def forward(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 20000:
            return self.send_bytes(413, b"Request too large.", "text/plain; charset=utf-8")

        body = self.rfile.read(length) if length else None
        headers = {}
        for key, value in self.headers.items():
            if key.lower() in {
                "host", "connection", "authorization", "proxy-authorization",
                "transfer-encoding", "accept-encoding",
            }:
                continue
            headers[key] = value
        headers["Host"] = f"127.0.0.1:{INTERNAL_PORT}"
        headers["Connection"] = "close"

        try:
            conn = http.client.HTTPConnection("127.0.0.1", INTERNAL_PORT, timeout=60)
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse()
            payload = response.read()
            self.send_response(response.status, response.reason)

            for key, value in response.getheaders():
                if key.lower() not in {
                    "connection", "transfer-encoding", "content-length",
                    "keep-alive", "proxy-authenticate", "proxy-authorization",
                    "te", "trailers", "upgrade", "server", "date",
                }:
                    self.send_header(key, value)

            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)
            conn.close()
            self.close_connection = True
        except (OSError, http.client.HTTPException):
            self.send_bytes(
                502, b"CollegeOS dashboard service is not available.",
                "text/plain; charset=utf-8"
            )


def start_internal_dashboard():
    thread = threading.Thread(
        target=dashboard_server.serve_dashboard,
        kwargs={
            "port": INTERNAL_PORT,
            "open_browser": False,
            "html_path": str(ROOT / "dashboard.html"),
        },
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if not thread.is_alive():
            raise RuntimeError("The internal CollegeOS dashboard server stopped during startup.")
        try:
            with socket.create_connection(("127.0.0.1", INTERNAL_PORT), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError("Timed out waiting for the internal CollegeOS dashboard server.")


if __name__ == "__main__":
    start_internal_dashboard()
    server = ThreadingHTTPServer(("0.0.0.0", PUBLIC_PORT), PrivateProxyHandler)
    print(f"Private CollegeOS gateway listening on port {PUBLIC_PORT}")
    server.serve_forever()
