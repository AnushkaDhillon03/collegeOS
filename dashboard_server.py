#!/usr/bin/env python3
"""
CollegeOS dashboard server — the web frontend for collegeos.py.

Layout (either works):
    Agentic/collegeos.py                      Agentic/collegeos.py
    Agentic/dashboard_server.py               Agentic/frontend/dashboard_server.py
    Agentic/dashboard.html                    Agentic/frontend/dashboard.html

dashboard.html must be next to this file (or pass --html). Run the command from
the folder that holds collegeos_state.json (the Agentic folder).

Usage:
    python3 dashboard_server.py                 # opens http://localhost:8765
    python3 dashboard_server.py --port 9000
    python3 dashboard_server.py --html path/to/dashboard.html
    python3 dashboard_server.py --export snapshot.json   # for the hosted page

It reads and writes the same collegeos_state.json as collegeos.py, so run it
from the same folder you run collegeos.py from.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from typing import Optional

# collegeos.py may sit next to this file or one folder up (e.g. Agentic/frontend/).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.append(_p)

try:
    from collegeos import (DEFAULT_EFFORT, DEFAULT_IMPORTANCE, TASK_TYPES, Agent,
                           State, Task, default_gmail_senders, new_id, run_cycle)
except ImportError as e:
    raise SystemExit(f"Couldn't import collegeos.py ({e}). Keep dashboard_server.py "
                     "in the same folder as collegeos.py.")


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------
# A local-only page (127.0.0.1) over the same state file the CLI uses.
# Safety: it only listens on localhost, rejects requests whose Host header
# isn't localhost (blocks DNS-rebinding), and requires a random per-session
# token on every call. It never writes to Google Calendar; use gcal-sync.
# The page itself lives in dashboard.html next to this file (re-read on every
# refresh, so you can edit the design without restarting the server).

DASHBOARD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


def dashboard_snapshot(today: date) -> dict:
    state = State.load()
    agent = Agent(state)
    by_id = {t.id: t for t in state.tasks}
    keep_done_after = today - timedelta(days=60)

    tasks = []
    for t in state.tasks:
        if t.done and t.done_on and t.done_on < keep_done_after:
            continue
        tasks.append({
            "id": t.id, "title": t.title, "course": t.course, "type": t.type,
            "due": t.due.isoformat() if t.due else None,
            "days_left": t.days_left(today) if t.due else None,
            "remaining": t.remaining(), "est": t.est_effort, "spent": t.spent,
            "importance": t.importance, "done": t.done,
            "done_on": t.done_on.isoformat() if t.done_on else None,
            "source": t.source, "score": t.score, "points_possible": t.points_possible,
        })

    plan = state.plan
    days = []
    for i in range(14):
        d = today + timedelta(days=i)
        blocks = []
        if plan:
            for b in sorted(plan.for_day(d), key=lambda x: -x.minutes):
                if b.task_id in by_id and not by_id[b.task_id].done:
                    blocks.append({"task_id": b.task_id, "minutes": b.minutes})
        classes = [{"code": c.course_code, "name": c.course_name, "start": c.start_min,
                    "end": c.end_min, "room": c.room}
                   for c in sorted(state.timetable, key=lambda c: c.start_min) if c.day == d]
        days.append({"day": d.isoformat(), "available": state.capacity.available(d),
                     "planned": sum(b["minutes"] for b in blocks),
                     "blocks": blocks, "classes": classes})

    at_risk = []
    if plan:
        for tid, short in plan.at_risk.items():
            if tid in by_id and not by_id[tid].done:
                at_risk.append({"id": tid, "title": by_id[tid].title, "short": short})

    attendance = [{"course": a.course, "held": a.held, "attended": a.attended,
                   "ratio": a.ratio, "threshold": a.threshold,
                   "low": bool(a.held and a.ratio < a.threshold),
                   "need": a.classes_needed()} for a in state.attendance]
    alerts = [{"level": a.level, "kind": a.kind, "message": a.message}
              for a in agent.alerts(today) if a.kind != "overrun"]
    return {"today": today.isoformat(), "generated_at": today.isoformat(),
            "plan_made": plan.made_on.isoformat() if plan else None,
            "tasks": tasks, "plan_days": days, "at_risk": at_risk,
            "attendance": attendance, "alerts": alerts,
            "grades": state.grades, "announcements": state.announcements}


def serve_dashboard(port: int = 8765, open_browser: bool = True,
                    html_path: Optional[str] = None) -> None:
    import contextlib
    import http.server
    import io
    import secrets
    import threading
    import webbrowser

    token = secrets.token_urlsafe(16)
    lock = threading.Lock()
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    html_file = html_path or DASHBOARD_FILE
    if not os.path.exists(html_file):
        raise RuntimeError(f"{html_file} not found. Keep dashboard.html in the same "
                           "folder as collegeos.py (or pass --html PATH).")

    def load_page() -> bytes:
        with open(html_file, encoding="utf-8") as f:
            return f.read().replace("__TOKEN__", token).encode("utf-8")

    def run_opts() -> argparse.Namespace:
        return argparse.Namespace(
            no_canvas=False, canvas_config="canvas_token.json", no_erp=False,
            erp_user_id="", day_budget=480, senders=default_gmail_senders(),
            gmail_days=14, gmail_token="gmail_token.json", llm_extract=False,
            use_llm=False, days=21, notify="desktop", cooldown=24.0, gcal=False,
            approve_gcal=False, gcal_token="google_token.json", tz="Asia/Kolkata")

    def find(state: State, body: dict) -> Task:
        tid = str(body.get("id", ""))
        t = next((x for x in state.tasks if x.id == tid), None)
        if t is None:
            raise ValueError("That task no longer exists. Refresh the page.")
        return t

    def replan(state: State, today: date) -> None:
        Agent(state).think(today, 21, "repair")
        state.save()

    def act(path: str, body: dict) -> dict:
        today = date.today()
        with lock:
            if path == "/api/sync":
                state = State.load()
                agent = Agent(state)
                buf = io.StringIO()
                try:
                    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                        run_cycle(state, agent, run_opts(), today)
                except Exception as e:                    # noqa: BLE001
                    buf.write(f"\n[error] {e}\n")
                return {"log": buf.getvalue()}

            state = State.load()
            if path == "/api/done":
                t = find(state, body)
                t.done, t.done_on = True, today
            elif path == "/api/reopen":
                t = find(state, body)
                t.done, t.done_on = False, None
            elif path == "/api/log":
                t = find(state, body)
                m = int(body.get("minutes", 0))
                if not 1 <= m <= 600:
                    raise ValueError("Minutes must be between 1 and 600.")
                t.spent += m
                if t.spent > t.est_effort:
                    t.est_effort = int(max(t.est_effort * 1.3, t.spent + 30))
            elif path == "/api/add":
                title = " ".join(str(body.get("title", "")).split())[:120]
                if not title:
                    raise ValueError("Give the task a title.")
                try:
                    due = date.fromisoformat(str(body.get("due", "")))
                except ValueError:
                    raise ValueError("Pick a due date.")
                ttype = str(body.get("type") or "assignment")
                if ttype not in TASK_TYPES:
                    ttype = "assignment"
                try:
                    effort = int(body.get("effort") or DEFAULT_EFFORT[ttype])
                    imp = int(body.get("importance") or DEFAULT_IMPORTANCE[ttype])
                except ValueError:
                    raise ValueError("Effort and importance must be numbers.")
                state.tasks.append(Task(
                    id=new_id(), title=title, due=due,
                    course=(str(body.get("course") or "GEN").strip().upper()[:40] or "GEN"),
                    type=ttype, est_effort=max(5, min(2400, effort)),
                    importance=max(1, min(5, imp)), source="manual", created=today))
            else:
                raise LookupError("Unknown action.")
            replan(state, today)
            return {"ok": True}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):                       # keep Terminal quiet
            pass

        def _send(self, code: int, payload, ctype="application/json; charset=utf-8"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; script-src 'unsafe-inline'; "
                             "style-src 'unsafe-inline' https://fonts.googleapis.com; "
                             "font-src https://fonts.gstatic.com; connect-src 'self'; "
                             "form-action 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def _guard(self) -> bool:
            if self.headers.get("Host", "") not in hosts:
                self._send(403, {"error": "Forbidden host."})
                return False
            return True

        def do_GET(self):
            if not self._guard():
                return
            if self.path in ("/", "/index.html"):
                try:
                    self._send(200, load_page(), "text/html; charset=utf-8")
                except OSError as e:
                    self._send(500, {"error": f"Couldn't read dashboard.html: {e}"})
            elif self.path == "/api/state":
                if self.headers.get("X-CollegeOS-Token") != token:
                    return self._send(403, {"error": "Bad session token. Reload the page."})
                try:
                    with lock:
                        snap = dashboard_snapshot(date.today())
                    self._send(200, snap)
                except Exception as e:                    # noqa: BLE001
                    self._send(500, {"error": f"Couldn't read your data: {e}"})
            else:
                self._send(404, {"error": "Not found."})

        def do_POST(self):
            if not self._guard():
                return
            if self.headers.get("X-CollegeOS-Token") != token:
                return self._send(403, {"error": "Bad session token. Reload the page."})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > 20000:
                    raise ValueError("Request too large.")
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("Bad request.")
                self._send(200, act(self.path, body))
            except LookupError as e:
                self._send(404, {"error": str(e)})
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:                        # noqa: BLE001
                self._send(500, {"error": f"Something went wrong: {e}"})

    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        raise RuntimeError(f"Couldn't start on port {port} ({e}). "
                           f"Try: python3 dashboard_server.py --port {port + 1}")
    url = f"http://localhost:{port}/"
    print(f"CollegeOS dashboard running at {url}\nPress Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()


def main() -> int:
    p = argparse.ArgumentParser(description="CollegeOS web dashboard")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true", help="don't open the browser")
    p.add_argument("--html", default=None, help="path to dashboard.html (default: next to this file)")
    p.add_argument("--export", metavar="FILE", default=None,
                   help="write a read-only snapshot of your plan to FILE and exit "
                        "(open it in the hosted dashboard)")
    args = p.parse_args()
    if args.export:
        snap = dashboard_snapshot(date.today())
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2)
        print(f"Saved {args.export}: {len(snap['tasks'])} task(s), "
              f"{len(snap['attendance'])} course(s) of attendance. "
              "Open it with 'Open my snapshot' in the hosted dashboard.")
        return 0
    try:
        serve_dashboard(args.port, not args.no_open, args.html)
    except RuntimeError as e:
        print(f"[error] {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
