#!/usr/bin/env python3
"""
CollegeOS — agentic academic planning core.

Single-file, stdlib-only prototype of the defensible core of the project:
task model -> extraction -> prioritization -> scheduling -> monitoring ->
incremental replanning, plus an evaluation harness that compares the agent
against an earliest-deadline-first baseline on simulated semesters.

External integrations are kept behind connector classes. CanPlus/Canvas
uses a read-only personal access token stored locally in canvas_token.json;
Google Calendar uses its own OAuth client and token files.

Usage
-----
    python collegeos.py demo                 # seed sample data, plan, show
    python collegeos.py add "Cleaning Bot Report" --due 2026-10-04 \
        --course CS401 --type assignment --effort 240 --importance 4
    python collegeos.py extract --file inbox.txt --course CS401
    python collegeos.py plan [--days 14]
    python collegeos.py status
    python collegeos.py done <task_id> [--minutes 90]
    python collegeos.py log <task_id> --minutes 45     # record work, re-estimate
    python collegeos.py attendance CS401 --held 40 --attended 28
    python collegeos.py replan [--mode repair|rebuild]
    python collegeos.py lms-sync                      # CanPlus assignments + grades + announcements
    python collegeos.py grades                         # show saved course grades / scores
    python collegeos.py run [--every 60]               # sync all -> monitor -> replan -> notify
    python collegeos.py notify-test [--notify both]    # send a test pop-up / email
    python collegeos.py schedule                       # how to run `run` automatically
    python collegeos.py gcal-sync [--approve]          # study plan -> Google Calendar
    python collegeos.py gmail-sync --from a@x.com --from b@y.com   # deadlines from 2 senders
    python collegeos.py plan --llm                    # advisor-weighted plan
    python collegeos.py explain                        # why, without replanning
    python collegeos.py evaluate [--runs 40]           # agent vs EDF baseline

LLM advisor (extraction --llm and plan/replan/explain --llm) works with either
ANTHROPIC_API_KEY or GEMINI_API_KEY in the environment — set whichever you
have (or LLM_PROVIDER=anthropic|gemini to force one if both are set). Without
either, every --llm flag and the explain command degrade to a clean no-op —
the deterministic core is never blocked on the model being available.

State lives in ./collegeos_state.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import uuid
import shutil
import smtplib
import subprocess
import platform
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

STATE_PATH = os.environ.get("COLLEGEOS_STATE", "collegeos_state.json")

# --------------------------------------------------------------------------
# Domain model
# --------------------------------------------------------------------------

TASK_TYPES = ("assignment", "quiz", "exam", "reading", "project", "admin")

# Default effort in minutes when nothing better is known.
DEFAULT_EFFORT = {
    "assignment": 180,
    "quiz": 60,
    "exam": 480,
    "reading": 60,
    "project": 360,
    "admin": 20,
}

# Default importance (1-5) by type; feeds the priority score.
DEFAULT_IMPORTANCE = {
    "assignment": 3,
    "quiz": 3,
    "exam": 5,
    "reading": 2,
    "project": 4,
    "admin": 1,
}


@dataclass
class Task:
    id: str
    title: str
    due: Optional[date]
    course: str = "GEN"
    type: str = "assignment"
    est_effort: int = 180          # minutes, the planner's current belief
    spent: int = 0                 # minutes actually logged
    importance: int = 3            # 1-5
    done: bool = False
    done_on: Optional[date] = None
    source: str = "manual"         # manual | email | canvas | erp
    created: date = field(default_factory=date.today)
    score: Optional[float] = None            # points earned (from Canvas)
    points_possible: Optional[float] = None

    def remaining(self) -> int:
        return 0 if self.done else max(0, self.est_effort - self.spent)

    def days_left(self, today: date) -> int:
        return (self.due - today).days if self.due else 10**6

    def to_json(self) -> dict:
        d = asdict(self)
        d["due"] = self.due.isoformat() if self.due else None
        d["created"] = self.created.isoformat()
        d["done_on"] = self.done_on.isoformat() if self.done_on else None
        return d

    @staticmethod
    def from_json(d: dict) -> "Task":
        d = dict(d)
        d["due"] = date.fromisoformat(d["due"]) if d.get("due") else None
        d["created"] = date.fromisoformat(d["created"])
        d["done_on"] = date.fromisoformat(d["done_on"]) if d.get("done_on") else None
        return Task(**d)


@dataclass
class Attendance:
    course: str
    held: int = 0
    attended: int = 0
    threshold: float = 0.75

    @property
    def ratio(self) -> float:
        return self.attended / self.held if self.held else 1.0

    def classes_needed(self) -> int:
        """Consecutive future classes needed to climb back to threshold."""
        if self.ratio >= self.threshold or self.threshold >= 1:
            return 0
        n = 0
        a, h = self.attended, self.held
        while a / h < self.threshold and n < 500:
            a += 1
            h += 1
            n += 1
        return n


@dataclass
class Block:
    """One scheduled study session."""
    task_id: str
    day: date
    minutes: int

    def key(self) -> Tuple[str, str]:
        return (self.task_id, self.day.isoformat())

    def to_json(self) -> dict:
        return {"task_id": self.task_id, "day": self.day.isoformat(), "minutes": self.minutes}

    @staticmethod
    def from_json(d: dict) -> "Block":
        return Block(d["task_id"], date.fromisoformat(d["day"]), d["minutes"])


@dataclass
class Plan:
    made_on: date
    blocks: List[Block] = field(default_factory=list)
    at_risk: Dict[str, int] = field(default_factory=dict)   # task_id -> shortfall minutes

    def for_day(self, day: date) -> List[Block]:
        return [b for b in self.blocks if b.day == day]

    def minutes_for(self, task_id: str) -> int:
        return sum(b.minutes for b in self.blocks if b.task_id == task_id)

    def to_json(self) -> dict:
        return {
            "made_on": self.made_on.isoformat(),
            "blocks": [b.to_json() for b in self.blocks],
            "at_risk": self.at_risk,
        }

    @staticmethod
    def from_json(d: dict) -> "Plan":
        return Plan(
            made_on=date.fromisoformat(d["made_on"]),
            blocks=[Block.from_json(b) for b in d["blocks"]],
            at_risk=d.get("at_risk", {}),
        )


# --------------------------------------------------------------------------
# Capacity model — how much study time exists on a given day
# --------------------------------------------------------------------------

@dataclass
class ClassSession:
    """One timetable entry. Deliberately excludes faculty names/photos:
    the planner only needs when and where the student is occupied."""
    course_code: str
    course_name: str
    day: date
    start_min: int          # minutes after midnight
    end_min: int
    room: str = ""

    @property
    def minutes(self) -> int:
        return max(0, self.end_min - self.start_min)

    def to_json(self) -> dict:
        d = asdict(self)
        d["day"] = self.day.isoformat()
        return d

    @staticmethod
    def from_json(d: dict) -> "ClassSession":
        d = dict(d)
        d["day"] = date.fromisoformat(d["day"])
        return ClassSession(**d)


@dataclass
class Capacity:
    weekday_minutes: int = 180
    weekend_minutes: int = 300
    committed: Dict[str, int] = field(default_factory=dict)  # iso date -> busy minutes

    def available(self, day: date) -> int:
        base = self.weekend_minutes if day.weekday() >= 5 else self.weekday_minutes
        return max(0, base - self.committed.get(day.isoformat(), 0))

    def apply_classes(self, sessions: List["ClassSession"], day_budget: int = 480) -> None:
        """
        Turn timetable classes into reduced study capacity.

        `day_budget` is the total academic minutes a student realistically has
        in a day (classes + study). Study time that day is what's left after
        classes, capped at the normal base for that weekday. A light class
        day therefore changes nothing; a packed one shrinks the study budget.
        """
        per_day: Dict[date, int] = {}
        for c in sessions:
            per_day[c.day] = per_day.get(c.day, 0) + c.minutes
        for d, class_min in per_day.items():
            base = self.weekend_minutes if d.weekday() >= 5 else self.weekday_minutes
            free = min(base, max(0, day_budget - class_min))
            self.committed[d.isoformat()] = base - free

    def cumulative(self, start: date, end: date) -> int:
        """Total available minutes in [start, end] inclusive."""
        total, d = 0, start
        while d <= end:
            total += self.available(d)
            d += timedelta(days=1)
        return total


# --------------------------------------------------------------------------
# Prioritizer
# --------------------------------------------------------------------------

class Prioritizer:
    """
    Scores a task for a specific day. The score blends four signals:

      urgency   — how close the deadline is
      weight    — declared importance, normalised to 0-1
      pressure  — remaining work / time available before the deadline
                  (this is what makes it more than a sorted to-do list)
      boost     — attendance-driven nudge for courses in trouble

    Weights are exposed so they can be tuned, and so the report can show an
    ablation: drop `pressure` and the policy collapses to near-EDF.
    """

    def __init__(self, w_urgency=0.35, w_weight=0.20, w_pressure=0.40, w_boost=0.05):
        self.w_urgency = w_urgency
        self.w_weight = w_weight
        self.w_pressure = w_pressure
        self.w_boost = w_boost

    def score(self, task: Task, day: date, cap: Capacity,
              boosts: Optional[Dict[str, float]] = None) -> float:
        boosts = boosts or {}
        days_left = max(0, (task.due - day).days) if task.due else 14
        urgency = 1.0 / (1.0 + days_left)

        weight = (task.importance - 1) / 4.0

        avail = cap.cumulative(day, task.due or (day + timedelta(days=14)))
        pressure = 1.0 if avail <= 0 else min(1.5, task.remaining() / avail)

        boost = boosts.get(task.course, 0.0)

        return (self.w_urgency * urgency
                + self.w_weight * weight
                + self.w_pressure * pressure
                + self.w_boost * boost)


# --------------------------------------------------------------------------
# Scheduler
# --------------------------------------------------------------------------

class Scheduler:
    """
    Greedy day-by-day packer. For each day in the horizon it repeatedly picks
    the highest-scoring eligible task and allots it a block, subject to:

      * min_block  — slices smaller than this are not worth a context switch
      * max_block  — no single sitting longer than this
      * max_per_task_per_day — spacing, so one task cannot eat a whole day

    Any task left with work after the horizon is reported in `at_risk` with
    its shortfall, which is the signal the monitor escalates on.
    """

    def __init__(self, cap: Capacity, prioritizer: Optional[Prioritizer] = None,
                 min_block=30, max_block=90, max_per_task_per_day=180):
        self.cap = cap
        self.pri = prioritizer or Prioritizer()
        self.min_block = min_block
        self.max_block = max_block
        self.max_per_task_per_day = max_per_task_per_day

    def build(self, tasks: List[Task], today: date, horizon: int = 21,
              boosts: Optional[Dict[str, float]] = None,
              pinned: Optional[List[Block]] = None) -> Plan:
        pinned = pinned or []
        remaining = {t.id: t.remaining() for t in tasks if not t.done}
        by_id = {t.id: t for t in tasks}

        blocks: List[Block] = []
        day_used: Dict[date, int] = {}

        # Honour pinned blocks first (used by repair-mode replanning).
        for b in pinned:
            if b.task_id not in remaining or remaining[b.task_id] <= 0:
                continue
            take = min(b.minutes, remaining[b.task_id],
                       self.cap.available(b.day) - day_used.get(b.day, 0))
            if take <= 0:
                continue
            blocks.append(Block(b.task_id, b.day, take))
            remaining[b.task_id] -= take
            day_used[b.day] = day_used.get(b.day, 0) + take

        for i in range(horizon):
            day = today + timedelta(days=i)
            free = self.cap.available(day) - day_used.get(day, 0)
            while free >= self.min_block:
                eligible = [t for t in tasks
                            if not t.done and remaining.get(t.id, 0) > 0 and (t.due is None or t.due >= day)]
                if not eligible:
                    break
                eligible.sort(key=lambda t: -self.pri.score(
                    _snapshot(t, remaining[t.id]), day, self.cap, boosts))

                placed = False
                for t in eligible:
                    used = sum(b.minutes for b in blocks
                               if b.task_id == t.id and b.day == day)
                    room = min(free, self.max_block, remaining[t.id],
                               self.max_per_task_per_day - used)
                    if room <= 0:
                        continue
                    # A slice below min_block is only allowed if it finishes the task.
                    if room < self.min_block and room < remaining[t.id]:
                        continue
                    blocks.append(Block(t.id, day, room))
                    remaining[t.id] -= room
                    free -= room
                    day_used[day] = day_used.get(day, 0) + room
                    placed = True
                    break
                if not placed:
                    break

        at_risk = {tid: m for tid, m in remaining.items() if m > 0 and tid in by_id}
        # Merge same-task-same-day blocks for a tidier plan.
        merged: Dict[Tuple[str, date], int] = {}
        for b in blocks:
            merged[(b.task_id, b.day)] = merged.get((b.task_id, b.day), 0) + b.minutes
        out = [Block(tid, d, m) for (tid, d), m in merged.items()]
        out.sort(key=lambda b: (b.day, -m_of(b)))
        return Plan(made_on=today, blocks=out, at_risk=at_risk)


def m_of(b: Block) -> int:
    return b.minutes


def _snapshot(task: Task, remaining_override: int) -> Task:
    """Copy of a task whose remaining() reflects in-progress scheduling."""
    t = Task(**{**task.to_json(),
                "due": task.due,
                "created": task.created,
                "done_on": task.done_on})
    t.est_effort = task.spent + remaining_override
    return t


# --------------------------------------------------------------------------
# Monitor
# --------------------------------------------------------------------------

@dataclass
class Alert:
    level: str      # info | warn | critical
    kind: str       # overdue | at_risk | attendance | overrun | sync
    message: str
    key: str = ""   # stable id, so the same alert isn't sent again and again


class Monitor:
    """Turns state + plan into alerts. Alerts are what trigger replanning."""

    def check(self, tasks: List[Task], plan: Optional[Plan], today: date,
              attendance: Iterable[Attendance]) -> List[Alert]:
        alerts: List[Alert] = []
        by_id = {t.id: t for t in tasks}

        for t in tasks:
            if t.done:
                continue
            if t.due and t.due < today:
                alerts.append(Alert("critical", "overdue",
                                    f"OVERDUE {t.days_left(today)*-1}d: {t.title} ({t.course})",
                                    key=f"overdue:{t.id}"))
            elif t.spent > t.est_effort * 1.25:
                alerts.append(Alert("warn", "overrun",
                                    f"Effort overrun on {t.title}: "
                                    f"{t.spent}m spent vs {t.est_effort}m estimated",
                                    key=f"overrun:{t.id}"))

        if plan:
            for tid, short in plan.at_risk.items():
                t = by_id.get(tid)
                if t:
                    deadline_note = (f"before {t.due.isoformat()} — deadline at risk"
                                     if t.due else "with no deadline set — work remains unscheduled")
                    alerts.append(Alert("critical", "at_risk",
                                        f"Cannot fit {short}m of {t.title} {deadline_note}",
                                        key=f"at_risk:{tid}"))

        for a in attendance:
            if a.held and a.ratio < a.threshold:
                need = a.classes_needed()
                alerts.append(Alert("warn", "attendance",
                                    f"{a.course} attendance {a.ratio*100:.1f}% "
                                    f"(below {a.threshold*100:.0f}%) — "
                                    f"{need} consecutive classes to recover",
                                    key=f"attendance:{a.course}"))
        return alerts

    def attendance_boosts(self, attendance: Iterable[Attendance]) -> Dict[str, float]:
        """Courses below threshold get a scheduling nudge, scaled by the gap."""
        out = {}
        for a in attendance:
            if a.held and a.ratio < a.threshold:
                out[a.course] = min(1.0, (a.threshold - a.ratio) / max(a.threshold, 1e-6))
        return out


# --------------------------------------------------------------------------
# Replanner
# --------------------------------------------------------------------------

class Replanner:
    """
    Two strategies, so the report has something to compare:

      rebuild — throw the plan away and schedule from scratch. Optimal-ish
                for the new state, but churns everything the student had
                already mentally committed to.
      repair  — keep blocks that are still valid (task alive, day in future,
                deadline not passed) and only reschedule what broke. Higher
                cost, far lower churn.

    Churn is measured as the fraction of (task, day) assignments that changed.
    """

    def __init__(self, scheduler: Scheduler):
        self.sched = scheduler

    def replan(self, tasks: List[Task], old: Optional[Plan], today: date,
               horizon: int = 21, mode: str = "repair",
               boosts: Optional[Dict[str, float]] = None) -> Plan:
        if mode == "rebuild" or old is None:
            return self.sched.build(tasks, today, horizon, boosts)

        by_id = {t.id: t for t in tasks}
        pinned = []
        for b in old.blocks:
            t = by_id.get(b.task_id)
            if not t or t.done:
                continue
            if b.day < today or (t.due is not None and b.day > t.due):
                continue
            pinned.append(b)
        return self.sched.build(tasks, today, horizon, boosts, pinned=pinned)

    @staticmethod
    def churn(old: Optional[Plan], new: Plan) -> float:
        if not old or not old.blocks:
            return 0.0
        o = {b.key(): b.minutes for b in old.blocks if True}
        n = {b.key(): b.minutes for b in new.blocks}
        keys = set(o) | set(n)
        if not keys:
            return 0.0
        changed = sum(1 for k in keys if o.get(k) != n.get(k))
        return changed / len(keys)


# --------------------------------------------------------------------------
# Extraction — deadlines out of unstructured text
# --------------------------------------------------------------------------

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}

DATE_PATTERNS = [
    # 2026-10-04
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"),
     lambda m, today: date(int(m.group(1)), int(m.group(2)), int(m.group(3)))),
    # 04/10/2026 or 04-10-26  (day first, Indian convention)
    (re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b"),
     lambda m, today: date(_year(m.group(3)), int(m.group(2)), int(m.group(1)))),
    # 4 October 2026 / 4 Oct
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
                r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
                r"(?:\s+(\d{4}))?\b", re.I),
     lambda m, today: _md(int(m.group(1)), MONTHS[m.group(2).lower()[:3]],
                          m.group(3), today)),
    # October 4 / Oct 4th
    (re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+"
                r"(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b", re.I),
     lambda m, today: _md(int(m.group(2)), MONTHS[m.group(1).lower()[:3]],
                          m.group(3), today)),
]

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday"]

TYPE_HINTS = [
    ("exam", ("exam", "end sem", "endsem", "mid sem", "midsem", "sessional", "viva")),
    ("quiz", ("quiz", "test", "mcq")),
    ("project", ("project", "capstone", "synopsis", "prototype")),
    ("reading", ("read", "chapter", "paper", "textbook")),
    ("assignment", ("assignment", "submit", "submission", "homework", "lab record",
                    "report", "due")),
]


def _year(s: str) -> int:
    y = int(s)
    return y if y > 99 else 2000 + y


def _md(day: int, month: int, year: Optional[str], today: date) -> date:
    if year:
        return date(_year(year), month, day)
    y = today.year
    try:
        cand = date(y, month, day)
    except ValueError:
        return today
    # Undated month/day in the past almost always means next year.
    if (cand - today).days < -60:
        cand = date(y + 1, month, day)
    return cand


def find_date(text: str, today: date) -> Optional[date]:
    for pat, fn in DATE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return fn(m, today)
            except (ValueError, KeyError):
                continue
    low = text.lower()
    if "tomorrow" in low:
        return today + timedelta(days=1)
    if "today" in low or "tonight" in low:
        return today
    for i, wd in enumerate(WEEKDAYS):
        if re.search(rf"\b(next\s+)?{wd}\b", low):
            delta = (i - today.weekday()) % 7
            delta = delta or 7
            if "next" in low:
                delta += 7 if delta < 7 else 0
            return today + timedelta(days=delta)
    return None


def guess_type(text: str) -> str:
    low = text.lower()
    for ttype, words in TYPE_HINTS:
        if any(w in low for w in words):
            return ttype
    return "assignment"


def looks_actionable(text: str) -> bool:
    low = text.lower()
    triggers = ("due", "deadline", "submit", "submission", "exam", "quiz",
                "test", "assignment", "by ", "last date", "viva", "presentation")
    return any(t in low for t in triggers)


# --------------------------------------------------------------------------
# LLM provider layer — one call site, works with whichever key is set
# --------------------------------------------------------------------------

ANTHROPIC_MODEL = "claude-sonnet-4-6"
# Comma-separated fallback chain: an overloaded model returns 503 rather than
# queueing, and capacity is tracked per model, so trying the next one usually
# succeeds immediately. GEMINI_MODEL overrides with a single forced model.
_GEMINI_DEFAULT_CHAIN = "gemini-flash-latest,gemini-2.5-flash,gemini-flash-lite-latest"
GEMINI_MODEL_CANDIDATES = [
    m.strip() for m in
    os.environ.get("GEMINI_MODEL", _GEMINI_DEFAULT_CHAIN).split(",") if m.strip()
]


def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def call_llm(prompt: str, max_tokens: int = 1000) -> Optional[str]:
    """
    Provider-agnostic text completion. Picks a provider from whichever
    key is set: ANTHROPIC_API_KEY, GEMINI_API_KEY, or LLM_PROVIDER to force
    one when both are present ("anthropic" | "gemini"). Returns the raw
    text response, or None if no key is set or every attempt fails —
    callers must treat None as "fall back to the deterministic path",
    never as an error to surface mid-plan.

    Gemini specifically: a 503 means Google's shared capacity for that
    model is temporarily saturated, not a bad key or a bad request. We
    retry once with a short backoff, then move to the next model in
    GEMINI_MODEL_CANDIDATES. Auth errors (401/403) are not retried —
    retrying won't fix a bad key.
    """
    forced = os.environ.get("LLM_PROVIDER", "").strip().lower()
    anth_key = os.environ.get("ANTHROPIC_API_KEY")
    gem_key = os.environ.get("GEMINI_API_KEY")

    provider = forced or ("anthropic" if anth_key else "gemini" if gem_key else "")
    if provider == "anthropic" and not anth_key:
        provider = "gemini" if gem_key else ""
    if provider == "gemini" and not gem_key:
        provider = "anthropic" if anth_key else ""
    if not provider:
        return None

    import time
    import urllib.error
    import urllib.request

    if provider == "anthropic":
        try:
            body = json.dumps({
                "model": ANTHROPIC_MODEL,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=body,
                headers={"content-type": "application/json",
                         "x-api-key": anth_key,
                         "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            return "".join(c.get("text", "") for c in data.get("content", []))
        except Exception as e:                       # noqa: BLE001
            print(f"[warn] LLM call failed (anthropic): {e}", file=sys.stderr)
            return None

    # provider == "gemini" — try each candidate model, with one retry each
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            # Gemini 2.5+ "thinking" is on by default and consumes the
            # output-token budget on internal reasoning before writing the
            # answer — with a small max_tokens that can leave zero tokens
            # for the actual JSON response. We want fast structured output
            # here, not reasoning, so disable it explicitly.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }).encode()
    last_err = None
    for model in GEMINI_MODEL_CANDIDATES:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent")
        for attempt in range(2):
            try:
                req = urllib.request.Request(
                    url, data=body,
                    headers={"content-type": "application/json",
                             "x-goog-api-key": gem_key})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = json.loads(r.read())
                cands = data.get("candidates") or []
                if not cands:
                    last_err = "empty response"
                    break
                parts = cands[0].get("content", {}).get("parts", [])
                return "".join(p.get("text", "") for p in parts)
            except urllib.error.HTTPError as e:
                last_err = f"{model}: HTTP {e.code}"
                if e.code == 503 and attempt == 0:
                    time.sleep(1.5)
                    continue           # one retry on this model
                if e.code in (401, 403):
                    print(f"[warn] Gemini auth rejected ({e.code}) — "
                         f"check GEMINI_API_KEY, retrying won't help",
                         file=sys.stderr)
                    return None        # don't burn time retrying a bad key
                break                  # try next model
            except Exception as e:      # noqa: BLE001
                last_err = f"{model}: {e}"
                break
    print(f"[warn] LLM call failed (gemini, all models tried): {last_err}", file=sys.stderr)
    return None


def strip_json_fence(raw: str) -> str:
    return raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


class RuleExtractor:
    """
    Deterministic baseline extractor. In the report this is the thing the LLM
    extractor has to beat — label ~200 real messages, report precision/recall
    for both. Without that comparison the LLM is an unjustified dependency.
    """

    name = "rules"

    def extract(self, text: str, today: date, course: str = "GEN") -> List[Task]:
        found: List[Task] = []
        for chunk in re.split(r"\n{2,}|(?<=[.!?])\s+(?=[A-Z])", text):
            chunk = chunk.strip()
            if len(chunk) < 12 or not looks_actionable(chunk):
                continue
            due = find_date(chunk, today)
            if not due or due < today - timedelta(days=1):
                continue
            ttype = guess_type(chunk)
            title = re.sub(r"\s+", " ", chunk)[:80]
            found.append(Task(
                id=new_id(),
                title=title,
                due=due,
                course=course,
                type=ttype,
                est_effort=DEFAULT_EFFORT[ttype],
                importance=DEFAULT_IMPORTANCE[ttype],
                source="email",
                created=today,
            ))
        return dedupe(found)


class LLMExtractor:
    """
    Optional LLM extractor. Kept behind the same interface as RuleExtractor so
    the evaluation can swap them. Falls back to rules if no key is present or
    the call fails — an extraction agent that hard-fails on a network blip is
    not usable. Works with either ANTHROPIC_API_KEY or GEMINI_API_KEY.
    """

    name = "llm"

    def __init__(self, fallback: Optional[RuleExtractor] = None):
        self.fallback = fallback or RuleExtractor()

    def extract(self, text: str, today: date, course: str = "GEN") -> List[Task]:
        if not llm_available():
            return self.fallback.extract(text, today, course)
        prompt = (
            "Extract academic tasks from the text below. Return ONLY a JSON "
            "array, no prose, no code fences. Each element: "
            '{"title": str, "due": "YYYY-MM-DD", "type": one of '
            f'{list(TASK_TYPES)}, "est_effort_minutes": int, '
            '"importance": 1-5}. '
            f"Today is {today.isoformat()}. Resolve relative dates against it. "
            "Skip anything without a concrete deadline.\n\n---\n" + text
        )
        raw = call_llm(prompt, max_tokens=1000)
        if raw is None:
            return self.fallback.extract(text, today, course)
        try:
            items = json.loads(strip_json_fence(raw))
            out = []
            for it in items:
                ttype = it.get("type", "assignment")
                if ttype not in TASK_TYPES:
                    ttype = "assignment"
                out.append(Task(
                    id=new_id(),
                    title=str(it["title"])[:80],
                    due=date.fromisoformat(it["due"]),
                    course=course,
                    type=ttype,
                    est_effort=int(it.get("est_effort_minutes") or DEFAULT_EFFORT[ttype]),
                    importance=int(it.get("importance") or DEFAULT_IMPORTANCE[ttype]),
                    source="email",
                    created=today,
                ))
            return dedupe(out)
        except Exception as e:                      # noqa: BLE001
            print(f"[warn] LLM extraction failed ({e}); using rules", file=sys.stderr)
            return self.fallback.extract(text, today, course)


def dedupe(tasks: List[Task]) -> List[Task]:
    """
    Same course + same due date + similar title == same task.

    Two rules count as "similar":
      1. More than half the title words are shared (good for near-identical titles).
      2. The tasks come from DIFFERENT sources (e.g. Canvas vs email) and every
         word of the shorter title appears in the longer one. Emails wrap the
         same assignment in extra words ("Assignment 3 report for CSL448 is due
         on ..."), which rule 1 alone misses. Numbers must match too, so
         "Lab 3" is never merged with "Lab 4".
    The first task in the list wins, so existing tasks beat new ones.
    """
    out: List[Task] = []
    for t in tasks:
        words = set(re.findall(r"\w+", t.title.lower()))
        dup = False
        for s in out:
            if s.course != t.course or s.due != t.due:
                continue
            sw = set(re.findall(r"\w+", s.title.lower()))
            if words and len(words & sw) / len(words | sw) > 0.5:
                dup = True
                break
            if s.source != t.source and words and sw:
                small, big = (words, sw) if len(words) <= len(sw) else (sw, words)
                if len(small) >= 2 and small <= big:
                    dup = True
                    break
        if not dup:
            out.append(t)
    return out


def new_id() -> str:
    return uuid.uuid4().hex[:8]


# --------------------------------------------------------------------------
# LLM Advisor — model-in-the-loop reasoning, kept off the critical path
# --------------------------------------------------------------------------

class LLMAdvisor:
    """
    Where the model actually reasons, as opposed to where deterministic rules
    are enough. Deliberately scoped to judgment calls rules can't make well:

      - conflicting deadline signals (two sources disagree on a due date)
      - explaining *why* today's plan looks the way it does, in plain language
      - flagging a re-estimate that looks structurally off (not just "over
        budget", but "this pattern of overruns suggests the original
        estimate was wrong by more than logged minutes account for")

    This class NEVER touches capacity, deadlines, or the schedule directly —
    it returns soft hints (0..1 boosts) that the Prioritizer blends in like
    the attendance boost, and a natural-language rationale for display. The
    Scheduler's hard constraints (available minutes, due dates) are enforced
    exactly as before, so a bad or missing LLM response can change *emphasis*
    but never produce an infeasible or crashing plan. If no API key is set or
    the call fails for any reason, advise() returns an empty, no-op result —
    the rest of the pipeline runs exactly as it does without an LLM at all.
    """

    name = "llm-advisor"

    def advise(self, tasks: List[Task], alerts: List[Alert], plan: Optional[Plan],
               today: date) -> dict:
        empty = {"boosts": {}, "rationale": "", "conflicts": []}
        if not llm_available():
            return empty

        open_tasks = [t for t in tasks if not t.done]
        if not open_tasks:
            return empty

        task_lines = "\n".join(
            f"- id={t.id} title={t.title!r} course={t.course} type={t.type} "
            f"due={t.due.isoformat() if t.due else 'none'} remaining_min={t.remaining()} "
            f"importance={t.importance} spent_min={t.spent} est_min={t.est_effort} "
            f"source={t.source}"
            for t in open_tasks
        )
        alert_lines = "\n".join(f"- [{a.level}] {a.kind}: {a.message}" for a in alerts) or "none"
        plan_summary = ""
        if plan:
            for tid, short in plan.at_risk.items():
                plan_summary += f"- at risk: task {tid} short by {short} minutes\n"
            plan_summary = plan_summary or "- no tasks currently at risk\n"

        prompt = f"""You are the reasoning layer of a student academic planning agent.
The scheduler and prioritizer below you are deterministic and already enforce
all hard constraints (available time, deadlines) — you cannot change those.
Your job is only the judgment calls rules can't make well.

Today is {today.isoformat()}.

Open tasks:
{task_lines}

Current alerts:
{alert_lines}

Plan feasibility:
{plan_summary}

Return ONLY a JSON object, no prose, no code fences, with exactly these keys:
{{
  "boosts": {{"<task_id>": <float 0.0-1.0>, ...}},   // soft priority nudges,
      // only for tasks that deserve extra attention beyond what deadline/
      // importance/workload pressure already capture (e.g. an estimate that
      // looks structurally wrong given spent_min vs est_min, or a task
      // whose type historically needs more buffer). Omit tasks needing no
      // adjustment. Do not use this to override deadlines.
  "rationale": "<2-3 sentences, plain language, explaining today's priorities
      to the student — this is shown directly to them>",
  "conflicts": ["<any task pair or signal that looks contradictory or
      suspicious, e.g. same course+week with wildly different effort
      estimates for similar task types>"]
}}"""

        raw = call_llm(prompt, max_tokens=700)
        if raw is None:
            return empty
        try:
            parsed = json.loads(strip_json_fence(raw))

            boosts = {}
            valid_ids = {t.id for t in open_tasks}
            for tid, v in (parsed.get("boosts") or {}).items():
                if tid in valid_ids:
                    try:
                        boosts[tid] = max(0.0, min(1.0, float(v)))
                    except (TypeError, ValueError):
                        continue
            return {
                "boosts": boosts,
                "rationale": str(parsed.get("rationale", ""))[:600],
                "conflicts": [str(c)[:200] for c in (parsed.get("conflicts") or [])][:5],
            }
        except Exception as e:                      # noqa: BLE001
            print(f"[warn] LLM advisor response unusable ({e}); continuing without it",
                 file=sys.stderr)
            return empty


# --------------------------------------------------------------------------
# Connectors — the plumbing seam
# --------------------------------------------------------------------------

class Connector:
    """
    Every external source implements this. Gmail/Canvas/Calendar/ERP clients
    slot in here without touching the planning core, which is the whole point
    of the separation.

    Write operations are gated: propose() returns an action for user approval,
    commit() is only called after approval. Nothing reaches an external system
    unattended.
    """
    name = "base"

    def fetch(self, since: date) -> List[Task]:
        raise NotImplementedError

    def propose(self, plan: Plan) -> List[dict]:
        return []

    def commit(self, actions: List[dict]) -> None:
        raise NotImplementedError


class FileInboxConnector(Connector):
    """Reads a plain-text dump of messages. Stand-in for Gmail during dev."""
    name = "file-inbox"

    def __init__(self, path: str, course: str = "GEN", extractor=None):
        self.path = path
        self.course = course
        self.extractor = extractor or RuleExtractor()

    def fetch(self, since: date) -> List[Task]:
        with open(self.path, encoding="utf-8") as f:
            return self.extractor.extract(f.read(), since, self.course)


def _num(x) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


class CanvasConnector(Connector):
    """Read-only CanPlus/Canvas connector using a locally stored API token.

    The token file is deliberately separate from Google's OAuth credentials.
    Only course, assignment, enrollment (grades) and announcement endpoints are
    requested; this connector never
    submits, edits, or deletes anything in the LMS.
    """
    name = "canplus-canvas"

    def __init__(self, config_path: str = "canvas_token.json"):
        self.config_path = config_path
        self.courses: List[dict] = []
        self.assignment_count = 0
        self.skipped_without_due = 0
        self.skipped_without_id = 0
        self.submitted_count = 0
        self.missing_count = 0
        self.assignment_rows: List[dict] = []

        if not os.path.exists(config_path):
            raise RuntimeError(
                f"{config_path} not found. Create it with base_url and access_token.")
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"{config_path} is not valid JSON. Check the file in VS Code and save it.") from e
        self.base_url = str(config.get("base_url", "")).rstrip("/")
        self.token = str(config.get("access_token", "")).strip()
        if not self.base_url or not self.token:
            raise RuntimeError(
                f"{config_path} must contain non-empty 'base_url' and 'access_token' values.")
        if not self.base_url.startswith("https://"):
            raise RuntimeError("CanPlus base_url must start with https://")
        try:
            os.chmod(config_path, 0o600)
        except OSError:
            pass

    def _get_pages(self, url: str) -> List[dict]:
        import urllib.error
        import urllib.parse
        import urllib.request

        results: List[dict] = []
        next_url = url
        while next_url:
            request = urllib.request.Request(
                next_url,
                headers={"Authorization": f"Bearer {self.token}",
                         "Accept": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=25) as response:
                    payload = json.loads(response.read())
                    link_header = response.headers.get("Link", "")
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise RuntimeError(
                        "CanPlus rejected the token (HTTP " + str(e.code) + "). "
                        "Check that the token is valid and has not expired.") from e
                raise RuntimeError(f"CanPlus API request failed: HTTP {e.code}") from e
            except Exception as e:
                raise RuntimeError(f"Could not read CanPlus API response: {e}") from e

            if not isinstance(payload, list):
                raise RuntimeError(
                    "CanPlus returned an unexpected response (expected a JSON list). "
                    "The API endpoint or permissions may have changed.")
            results.extend(item for item in payload if isinstance(item, dict))

            # Canvas pagination supplies a Link header with rel=next. Stop
            # cleanly when there is no next page.
            match = re.search(r'<([^>]+)>\s*;\s*rel="?next"?', link_header)
            if match:
                candidate = urllib.parse.urljoin(next_url, match.group(1))
                if urllib.parse.urlsplit(candidate).netloc != urllib.parse.urlsplit(self.base_url).netloc:
                    raise RuntimeError("CanPlus returned a pagination link to an unexpected host; stopping safely.")
                next_url = candidate
            else:
                next_url = None
        return results

    def fetch(self, since: date) -> List[Task]:
        from urllib.parse import urlencode

        courses_url = self.base_url + "/api/v1/courses?" + urlencode({"per_page": 100})
        self.courses = self._get_pages(courses_url)
        imported: List[Task] = []
        today = date.today()
        self.assignment_count = 0
        self.skipped_without_due = 0
        self.skipped_without_id = 0
        self.submitted_count = 0
        self.missing_count = 0
        self.assignment_rows = []

        for course in self.courses:
            course_id = course.get("id")
            if course_id is None:
                continue
            course_name = str(course.get("name") or "Unnamed course")
            course_code = str(course.get("course_code") or course_name or f"COURSE-{course_id}")
            query = urlencode({"per_page": 100, "include[]": "submission"})
            assignments_url = (
                f"{self.base_url}/api/v1/courses/{course_id}/assignments?{query}")
            assignments = self._get_pages(assignments_url)
            self.assignment_count += len(assignments)

            for assignment in assignments:
                if assignment.get("id") is None:
                    self.skipped_without_id += 1
                    continue
                due_at = assignment.get("due_at")
                due = None
                if due_at:
                    try:
                        due = date.fromisoformat(str(due_at)[:10])
                    except ValueError:
                        self.skipped_without_due += 1
                        continue

                title = str(assignment.get("name") or "Untitled Canvas assignment").strip()
                if not title:
                    title = "Untitled Canvas assignment"
                task_type = guess_type(title)
                submission = assignment.get("submission") or {}
                submitted_at = submission.get("submitted_at")
                is_submitted = bool(submitted_at) or submission.get("workflow_state") == "submitted"
                is_missing = bool(submission.get("missing"))
                if is_submitted:
                    self.submitted_count += 1
                if is_missing:
                    self.missing_count += 1
                status_label = "Submitted" if is_submitted else ("Missing" if is_missing else "Pending")
                self.assignment_rows.append({
                    "course": course_code,
                    "title": title,
                    "due": due.isoformat() if due else "No due date",
                    "status": status_label,
                })

                done_on = None
                if submitted_at:
                    try:
                        done_on = date.fromisoformat(str(submitted_at)[:10])
                    except ValueError:
                        done_on = None

                # Stable source ID means running lms-sync repeatedly updates
                # the same task instead of creating duplicates.
                stable_id = f"canvas_{course_id}_{assignment['id']}"
                imported.append(Task(
                    id=stable_id,
                    title=title,
                    due=due,
                    course=course_code[:40],
                    type=task_type,
                    est_effort=DEFAULT_EFFORT[task_type],
                    importance=DEFAULT_IMPORTANCE[task_type],
                    done=is_submitted,
                    done_on=done_on,
                    source="canvas",
                    created=today,
                    score=_num(submission.get("score")),
                    points_possible=_num(assignment.get("points_possible")),
                ))
        return imported

    def fetch_grades(self) -> List[dict]:
        """Current course grades from the enrollments endpoint (read-only).
        Some institutions hide grades; those fields then come back as None."""
        from urllib.parse import urlencode
        if not self.courses:
            self.courses = self._get_pages(
                self.base_url + "/api/v1/courses?" + urlencode({"per_page": 100}))
        meta = {c.get("id"): c for c in self.courses if c.get("id") is not None}
        query = urlencode([("type[]", "StudentEnrollment"), ("state[]", "active"),
                           ("per_page", 100)])
        rows = self._get_pages(f"{self.base_url}/api/v1/users/self/enrollments?{query}")
        out: List[dict] = []
        for r in rows:
            g = r.get("grades") or {}
            cid = r.get("course_id")
            c = meta.get(cid, {})
            out.append({
                "course_id": cid,
                "course": str(c.get("course_code") or c.get("name") or f"COURSE-{cid}"),
                "name": str(c.get("name") or ""),
                "current_score": _num(g.get("current_score")),
                "current_grade": g.get("current_grade"),
                "final_score": _num(g.get("final_score")),
                "final_grade": g.get("final_grade"),
            })
        return out

    def fetch_announcements(self, days: int = 14) -> List[dict]:
        """Recent course announcements, HTML stripped to plain text."""
        from urllib.parse import urlencode
        ids = [c["id"] for c in self.courses if c.get("id") is not None]
        names = {f"course_{c['id']}": str(c.get("course_code") or c.get("name") or c["id"])
                 for c in self.courses if c.get("id") is not None}
        start = (date.today() - timedelta(days=days)).isoformat()
        end = (date.today() + timedelta(days=1)).isoformat()
        out: List[dict] = []
        for i in range(0, len(ids), 10):          # Canvas allows 10 context codes per call
            params = [("context_codes[]", f"course_{x}") for x in ids[i:i + 10]]
            params += [("start_date", start), ("end_date", end), ("per_page", 100)]
            rows = self._get_pages(f"{self.base_url}/api/v1/announcements?{urlencode(params)}")
            for a in rows:
                out.append({
                    "id": str(a.get("id")),
                    "course": names.get(a.get("context_code"), "GEN"),
                    "title": str(a.get("title") or "Announcement"),
                    "posted": str(a.get("posted_at") or "")[:10],
                    "text": GmailConnector._html_to_text(str(a.get("message") or "")),
                })
        return out


class DryRunCalendarConnector(Connector):
    """Prints the events it would create. Replace with the Calendar API client."""
    name = "calendar-dryrun"

    def fetch(self, since: date) -> List[Task]:
        return []

    def propose(self, plan: Plan) -> List[dict]:
        return [{"action": "create_event", "day": b.day.isoformat(),
                 "minutes": b.minutes, "task_id": b.task_id} for b in plan.blocks]

    def commit(self, actions: List[dict]) -> None:
        for a in actions:
            print(f"  [calendar] would create {a['minutes']}m block on {a['day']}")


def google_credentials(scopes: List[str], credentials_path: str = "credentials.json",
                       token_path: str = "google_token.json"):
    """
    One Google login helper, shared by Calendar now and Gmail later.

    First run opens your browser so you can approve access, then saves a
    token file so you don't have to log in every time. credentials.json is
    the OAuth "Desktop app" file you download from Google Cloud Console.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        raise RuntimeError(
            "Google libraries are missing. Install them with:\n"
            "  pip3 install google-auth-oauthlib google-api-python-client")

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, scopes)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(token_path, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
            return creds
        except Exception:                            # noqa: BLE001
            pass                                     # fall through to a fresh login
    if not os.path.exists(credentials_path):
        raise RuntimeError(
            f"{credentials_path} not found. Download the OAuth client file "
            "(type: Desktop app) from Google Cloud Console and put it in this folder.")
    flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)
    creds = flow.run_local_server(port=0)
    with open(token_path, "w", encoding="utf-8") as f:
        f.write(creds.to_json())
    try:
        os.chmod(token_path, 0o600)                  # token = login, keep it private
    except OSError:
        pass
    return creds


class GoogleCalendarConnector(Connector):
    """
    Puts the study plan on Google Calendar as events.

    Safe by design:
      - propose() only READS your calendar and returns a list of changes;
        nothing is written until commit() is called (the CLI needs --approve).
      - It only ever deletes events that CollegeOS itself created (they carry
        a private tag), never your own events.
      - New study blocks are placed around your existing events, inside a
        study window (default 18:00-23:00 on weekdays).
    """
    name = "google-calendar"
    SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
    TAG = "collegeos"

    def __init__(self, tasks: Dict[str, Task], tz: str = "Asia/Kolkata",
                 study_start: str = "18:00", study_end: str = "23:00",
                 weekend_start: str = "10:00", gap_min: int = 10,
                 calendar_id: str = "primary", days: int = 14,
                 credentials_path: str = "credentials.json",
                 token_path: str = "google_token.json"):
        self.tasks = tasks
        self.tz_name = tz
        self.study_start, self.study_end = study_start, study_end
        self.weekend_start, self.gap_min = weekend_start, gap_min
        self.calendar_id, self.days = calendar_id, days
        self.credentials_path, self.token_path = credentials_path, token_path
        self.last_skipped: List[Tuple[Block, str]] = []
        self._svc = None

    def _tz(self):
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(self.tz_name)
        except Exception:                            # noqa: BLE001
            raise RuntimeError(f"Unknown time zone '{self.tz_name}'. "
                               "Use a name like Asia/Kolkata.")

    def _service(self):
        if self._svc is None:
            creds = google_credentials(self.SCOPES, self.credentials_path, self.token_path)
            from googleapiclient.discovery import build
            self._svc = build("calendar", "v3", credentials=creds)
        return self._svc

    def fetch(self, since: date) -> List[Task]:
        return []                                    # write-only connector

    # --- pure placement logic (easy to test without Google) -----------------
    @staticmethod
    def place_blocks(blocks: List[Block], busy: list, tz, study_start: str,
                     study_end: str, weekend_start: str, gap_min: int):
        """
        busy: list of (start, end) timezone-aware datetimes.
        Returns (placed, skipped); placed = [(Block, start, end)].
        Each block goes in the first free gap in the study window.
        """
        from datetime import datetime, time as dtime

        def hm(x: str):
            h, m = x.split(":")
            return dtime(int(h), int(m))

        busy = sorted(busy)
        by_day: Dict[date, List[Block]] = {}
        for b in blocks:
            by_day.setdefault(b.day, []).append(b)

        placed, skipped = [], []
        for day in sorted(by_day):
            first = hm(weekend_start) if day.weekday() >= 5 else hm(study_start)
            cursor = datetime.combine(day, first, tzinfo=tz)
            limit = datetime.combine(day, hm(study_end), tzinfo=tz)
            for b in by_day[day]:
                dur = timedelta(minutes=b.minutes)
                while True:
                    end = cursor + dur
                    clash = next((x for x in busy if x[0] < end and x[1] > cursor), None)
                    if clash is None:
                        break
                    cursor = clash[1]
                if end > limit:
                    skipped.append((b, f"no free slot before {study_end}"))
                    continue
                placed.append((b, cursor, end))
                cursor = end + timedelta(minutes=gap_min)
        return placed, skipped

    def _list_events(self, first: date, last: date) -> list:
        from datetime import datetime, time as dtime
        tz = self._tz()
        t_min = datetime.combine(first, dtime(0, 0), tzinfo=tz).isoformat()
        t_max = datetime.combine(last + timedelta(days=1), dtime(0, 0), tzinfo=tz).isoformat()
        svc, items, token = self._service(), [], None
        while True:
            resp = svc.events().list(
                calendarId=self.calendar_id, timeMin=t_min, timeMax=t_max,
                singleEvents=True, orderBy="startTime", maxResults=250,
                pageToken=token).execute()
            items.extend(resp.get("items", []))
            token = resp.get("nextPageToken")
            if not token:
                return items

    @staticmethod
    def _parse_dt(x: str):
        from datetime import datetime
        return datetime.fromisoformat(x.replace("Z", "+00:00"))

    def propose(self, plan: Plan) -> List[dict]:
        from datetime import datetime
        tz = self._tz()
        today = date.today()
        cutoff = today + timedelta(days=self.days)
        blocks = [b for b in plan.blocks if today <= b.day <= cutoff]
        self.last_skipped = []
        if not blocks:
            return []

        first, last = min(b.day for b in blocks), max(b.day for b in blocks)
        busy, ours = [], []
        now = datetime.now(tz)
        for ev in self._list_events(first, last):
            if ev.get("status") == "cancelled":
                continue
            start = ev.get("start", {}).get("dateTime")
            end = ev.get("end", {}).get("dateTime")
            if ev.get("extendedProperties", {}).get("private", {}).get(self.TAG) == "1":
                if start and self._parse_dt(start) >= now:   # only future ones
                    ours.append(ev)
                continue
            if ev.get("transparency") == "transparent" or not (start and end):
                continue                                     # free / all-day: not blocking
            busy.append((self._parse_dt(start), self._parse_dt(end)))

        placed, self.last_skipped = self.place_blocks(
            blocks, busy, tz, self.study_start, self.study_end,
            self.weekend_start, self.gap_min)

        actions: List[dict] = []
        for ev in ours:
            actions.append({"action": "delete_event", "event_id": ev["id"],
                            "label": ev.get("summary", "CollegeOS block"),
                            "when": ev["start"]["dateTime"]})
        for b, s, e in placed:
            t = self.tasks.get(b.task_id)
            title = t.title if t else b.task_id
            course = t.course if t else ""
            actions.append({
                "action": "create_event", "task_id": b.task_id,
                "label": f"Study: {title}", "when": s.isoformat(), "minutes": b.minutes,
                "body": {
                    "summary": f"Study: {title}",
                    "description": f"{course} \u2014 {b.minutes} min, planned by CollegeOS",
                    "start": {"dateTime": s.isoformat(), "timeZone": self.tz_name},
                    "end": {"dateTime": e.isoformat(), "timeZone": self.tz_name},
                    "reminders": {"useDefault": False,
                                  "overrides": [{"method": "popup", "minutes": 10}]},
                    "extendedProperties": {"private": {self.TAG: "1", "task_id": b.task_id}},
                }})
        return actions

    def commit(self, actions: List[dict]) -> None:
        svc, done = self._service(), 0
        try:
            for a in actions:
                if a["action"] == "delete_event":
                    svc.events().delete(calendarId=self.calendar_id,
                                        eventId=a["event_id"]).execute()
                elif a["action"] == "create_event":
                    svc.events().insert(calendarId=self.calendar_id, body=a["body"]).execute()
                done += 1
        except Exception as e:                       # noqa: BLE001
            raise RuntimeError(f"Google Calendar failed after {done} of "
                               f"{len(actions)} changes: {e}")


class GmailConnector(Connector):
    """
    Reads deadlines from emails sent by a short list of senders you choose.

    Safe by design:
      - Read-only Gmail permission. It can't send, delete or change anything.
      - Only looks at mail from the allowed addresses. Gmail's search is loose
        (it also matches display names), so every message is re-checked and
        thrown away unless the real sender address matches exactly.
      - Mail that failed Gmail's own spoofing checks (DMARC) is thrown away.
      - Email text is never saved. Only the tasks pulled out of it are kept.

    Relative dates like "tomorrow" are worked out from the day the email was
    received, not from today.
    """
    name = "gmail"
    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

    def __init__(self, senders: List[str], course: str = "GEN", extractor=None,
                 known_courses: Optional[List[str]] = None, days: int = 14,
                 max_messages: int = 25, credentials_path: str = "credentials.json",
                 token_path: str = "gmail_token.json"):
        self.senders = sorted({x.strip().lower() for x in senders if x and x.strip()})
        if not self.senders:
            raise RuntimeError("Give at least one sender address, e.g. "
                               "--from teacher@ncuindia.edu")
        self.course = course
        self.extractor = extractor or RuleExtractor()
        self.known_courses = known_courses or []
        self.days, self.max_messages = days, max_messages
        self.credentials_path, self.token_path = credentials_path, token_path
        self.stats = {"found": 0, "wrong_sender": 0, "failed_auth": 0,
                      "read": 0, "tasks": 0}
        self.log: List[dict] = []        # what was looked at; kept in memory only
        self._svc = None

    def _service(self):
        if self._svc is None:
            creds = google_credentials(self.SCOPES, self.credentials_path, self.token_path)
            from googleapiclient.discovery import build
            self._svc = build("gmail", "v1", credentials=creds)
        return self._svc

    # --- small pure helpers (easy to test without Google) -------------------
    @staticmethod
    def sender_ok(from_header: str, allowed: List[str]) -> bool:
        from email.utils import parseaddr
        return parseaddr(from_header or "")[1].strip().lower() in allowed

    @staticmethod
    def failed_auth(auth_results: str) -> bool:
        a = (auth_results or "").lower()
        return "dmarc=fail" in a or ("spf=fail" in a and "dkim=fail" in a)

    @staticmethod
    def _html_to_text(h: str) -> str:
        import html as htmllib
        h = re.sub(r"(?is)<(script|style).*?</\1>", " ", h)
        h = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h\d)>", "\n", h)
        h = re.sub(r"(?s)<[^>]+>", " ", h)
        h = htmllib.unescape(h)
        h = re.sub(r"[ \t\xa0]+", " ", h)
        return re.sub(r"\n\s*\n+", "\n\n", h).strip()

    @classmethod
    def message_text(cls, msg: dict) -> str:
        import base64

        plain: List[str] = []
        htmls: List[str] = []

        def walk(part: dict) -> None:
            data = part.get("body", {}).get("data")
            if data:
                try:
                    txt = base64.urlsafe_b64decode(
                        data + "=" * (-len(data) % 4)).decode("utf-8", "ignore")
                except Exception:                    # noqa: BLE001
                    txt = ""
                mt = part.get("mimeType", "")
                if mt == "text/plain":
                    plain.append(txt)
                elif mt == "text/html":
                    htmls.append(txt)
            for sub in part.get("parts") or []:
                walk(sub)

        walk(msg.get("payload", {}))
        body = "\n".join(plain).strip()
        if not body and htmls:
            body = cls._html_to_text("\n".join(htmls))
        return body or msg.get("snippet", "")

    @staticmethod
    def course_from_text(text: str, known: List[str], default: str) -> str:
        """Use a course code you actually have (from ERP) if the email names one."""
        for code in known:
            m = re.match(r"([A-Za-z]+)(\d+)$", code)
            if m and re.search(rf"\b{m.group(1)}\s?-?{m.group(2)}\b", text, re.I):
                return code
        if not known:
            m = re.search(r"\b([A-Z]{3})\s?-?(\d{3})\b", text)
            if m:
                return m.group(1) + m.group(2)
        return default

    # --- the actual fetch ----------------------------------------------------
    def fetch(self, since: date) -> List[Task]:
        from datetime import datetime
        query = f"from:({' OR '.join(self.senders)}) newer_than:{self.days}d"
        svc = self._service()
        try:
            ids, token = [], None
            while len(ids) < self.max_messages:
                resp = svc.users().messages().list(
                    userId="me", q=query, pageToken=token,
                    maxResults=min(self.max_messages, 100)).execute()
                ids += [m["id"] for m in resp.get("messages", [])]
                token = resp.get("nextPageToken")
                if not token:
                    break
            ids = ids[: self.max_messages]
            msgs = [svc.users().messages().get(userId="me", id=i, format="full").execute()
                    for i in ids]
        except Exception as e:                       # noqa: BLE001
            hint = ""
            if "accessNotConfigured" in str(e) or "has not been used" in str(e):
                hint = " (Enable the Gmail API in Google Cloud Console first.)"
            raise RuntimeError(f"Gmail request failed: {e}{hint}")

        self.stats["found"] = len(msgs)
        self.log = []
        out: List[Task] = []
        for msg in msgs:
            hdr = {}
            for h in msg.get("payload", {}).get("headers", []):
                hdr.setdefault(h["name"].lower(), []).append(h["value"])
            frm = (hdr.get("from") or [""])[0]
            subject = (hdr.get("subject") or [""])[0]
            received = datetime.fromtimestamp(int(msg.get("internalDate", "0")) / 1000).date()
            entry = {"from": frm, "subject": subject, "date": received.isoformat(),
                     "status": "read", "preview": "", "tasks": []}
            self.log.append(entry)

            # Mail from anyone else is never opened: no body is read or shown.
            if not self.sender_ok(frm, self.senders):
                self.stats["wrong_sender"] += 1
                entry["status"] = "SKIPPED - sender is not on your list"
                continue
            if self.failed_auth(" ".join(hdr.get("authentication-results", []))):
                self.stats["failed_auth"] += 1
                entry["status"] = "SKIPPED - failed Gmail's spoofing check"
                continue
            self.stats["read"] += 1

            body = self.message_text(msg)
            entry["preview"] = " ".join(body.split())[:400]
            text = f"{subject}\n\n{body}"
            course = self.course_from_text(text, self.known_courses, self.course)
            for t in self.extractor.extract(text, received, course):
                if t.due is None or t.due >= since:  # skip deadlines already gone
                    t.course = course
                    out.append(t)
                    entry["tasks"].append((t.due.isoformat() if t.due else "no date", t.title))
        out = dedupe(out)
        self.stats["tasks"] = len(out)
        return out


class ERPConnector:
    """
    Talks to the NorthCap ERP directly, against an undocumented
    endpoint found via the portal's own Network tab — there is no published
    public API, so this is intentionally narrow and defensive:

      - it only calls the one endpoint that has actually been confirmed
        working (attendance summary as of testing);
      - it never guesses at a response shape — if the JSON doesn't have the
        fields expected, it raises rather than silently returning nothing;
      - it requires a Bearer token the student copies from their own
        logged-in browser session (there is no automated login here), and
        fails with a clear, actionable message when that token has expired,
        rather than a confusing stack trace.

    Note: the confirmed endpoint lives on uatapi.ncuindia.edu — a "UAT"
    (User Acceptance Testing / staging) hostname, not necessarily the
    permanent production URL. Confirm with institutional IT whether this is
    the long-term address before relying on it past the current semester.

    Does NOT implement the Connector interface above (fetch() -> List[Task]),
    because attendance data isn't task-shaped — it feeds State.attendance
    directly instead, via fetch_attendance().
    """
    name = "erp"
    BASE = "https://uatapi.ncuindia.edu/api/myapp"

    def __init__(self, user_id: Optional[str] = None, token: Optional[str] = None,
                 config_path: str = "erp_token.json"):
        """Credentials are looked up in this order: arguments, environment
        variables (ERP_USER_ID / ERP_TOKEN), then erp_token.json, e.g.
            {"user_id": "12345", "token": "eyJ..."}
        Keep that file private (it is chmod 600 here) and never commit it."""
        cfg: dict = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, encoding="utf-8") as f:
                    cfg = json.load(f)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"{config_path} is not valid JSON. Expected "
                    '{"user_id": "...", "token": "..."}') from e
            try:
                os.chmod(config_path, 0o600)
            except OSError:
                pass
        self.user_id = str(user_id or os.environ.get("ERP_USER_ID")
                           or cfg.get("user_id") or "").strip()
        tok = token or os.environ.get("ERP_TOKEN") or cfg.get("token") or ""
        tok = str(tok).strip()
        if tok.lower().startswith("bearer "):
            tok = tok[7:].strip()
        self.token = tok
        if not self.user_id:
            raise RuntimeError(
                f"No ERP user id. Put \"user_id\" in {config_path}, or pass --user-id.")

    def _get(self, path: str) -> dict:
        if not self.token:
            raise RuntimeError(
                "No ERP token. Log into the ERP in your browser, open "
                "DevTools \u2192 Network \u2192 find a request to uatapi.ncuindia.edu, "
                "copy the value after 'Bearer ' in its Authorization header, and "
                "save it as \"token\" in erp_token.json. These tokens expire "
                "periodically, so you'll need to refresh this occasionally.")
        import urllib.error
        import urllib.request
        url = f"{self.BASE}/{path}"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.token}"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                outer = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise RuntimeError(
                    "ERP rejected the token (401/403) \u2014 it has almost "
                    "certainly expired. Grab a fresh one from DevTools and "
                    "update \"token\" in erp_token.json.")
            raise RuntimeError(f"ERP request failed: HTTP {e.code}")
        except Exception as e:                       # noqa: BLE001
            raise RuntimeError(f"ERP request failed: {e}")

        if not outer.get("Succeeded"):
            raise RuntimeError(f"ERP API reported failure: {outer.get('Message')}")
        try:
            return json.loads(outer["Data"])
        except (KeyError, TypeError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"ERP response didn't have the expected shape ({e}) \u2014 "
                "the endpoint may have changed since this connector was written.")

    def fetch_attendance(self, threshold: float = 0.75) -> List[Attendance]:
        inner = self._get(f"Registration/GetAttendanceSummary?userId={self.user_id}")
        rows = inner.get("Table")
        if rows is None:
            raise RuntimeError("ERP attendance response had no 'Table' field \u2014 "
                               "the endpoint may have changed since this was written.")
        out = []
        for row in rows:
            out.append(Attendance(
                course=row.get("CourseCode", "GEN"),
                held=int(row.get("TotalLectures", 0)),
                attended=int(row.get("TotalPresent", 0)),
                threshold=threshold,
            ))
        return out

    @staticmethod
    def _hhmm_to_min(s: str) -> int:
        h, m = s.split(":")[:2]
        return int(h) * 60 + int(m)

    def fetch_schedule(self) -> List[ClassSession]:
        inner = self._get(f"Student/GetSchedule?userId={self.user_id}")
        rows = inner.get("Table1")
        if rows is None:
            raise RuntimeError("ERP schedule response had no 'Table1' field \u2014 "
                               "the endpoint may have changed since this was written.")
        out: List[ClassSession] = []
        for r in rows:
            try:
                code, _, name = str(r.get("CourseDetails", "")).partition(" -")
                out.append(ClassSession(
                    course_code=code.strip() or "GEN",
                    course_name=name.strip(),
                    day=date.fromisoformat(str(r["LectureDate"])[:10]),
                    start_min=self._hhmm_to_min(r["StartTime"]),
                    end_min=self._hhmm_to_min(r["endtime"]),
                    room=str(r.get("ClassRoomNo", "")).strip(),
                ))
            except (KeyError, ValueError) as e:
                raise RuntimeError(
                    f"ERP schedule row didn't have the expected shape ({e}) \u2014 "
                    "the endpoint may have changed since this was written.")
        out.sort(key=lambda c: (c.day, c.start_min))
        return out


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------

@dataclass
class State:
    tasks: List[Task] = field(default_factory=list)
    attendance: List[Attendance] = field(default_factory=list)
    plan: Optional[Plan] = None
    capacity: Capacity = field(default_factory=Capacity)
    timetable: List[ClassSession] = field(default_factory=list)
    grades: List[dict] = field(default_factory=list)
    announcements: List[dict] = field(default_factory=list)
    notified: Dict[str, str] = field(default_factory=dict)   # alert key -> last sent (ISO)

    def save(self, path: str = STATE_PATH) -> None:
        payload = {
            "tasks": [t.to_json() for t in self.tasks],
            "attendance": [asdict(a) for a in self.attendance],
            "plan": self.plan.to_json() if self.plan else None,
            "timetable": [c.to_json() for c in self.timetable],
            "grades": self.grades,
            "announcements": self.announcements,
            "notified": self.notified,
            "capacity": {
                "weekday_minutes": self.capacity.weekday_minutes,
                "weekend_minutes": self.capacity.weekend_minutes,
                "committed": self.capacity.committed,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def load(path: str = STATE_PATH) -> "State":
        if not os.path.exists(path):
            return State()
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return State(
            tasks=[Task.from_json(t) for t in d.get("tasks", [])],
            attendance=[Attendance(**a) for a in d.get("attendance", [])],
            plan=Plan.from_json(d["plan"]) if d.get("plan") else None,
            capacity=Capacity(**d.get("capacity", {})),
            timetable=[ClassSession.from_json(c) for c in d.get("timetable", [])],
            grades=d.get("grades", []),
            announcements=d.get("announcements", []),
            notified=d.get("notified", {}),
        )

    def task(self, prefix: str) -> Optional[Task]:
        hits = [t for t in self.tasks if t.id.startswith(prefix)]
        return hits[0] if len(hits) == 1 else None


# --------------------------------------------------------------------------
# Agent — ties the loop together
# --------------------------------------------------------------------------

class Agent:
    def __init__(self, state: State):
        self.state = state
        self.sched = Scheduler(state.capacity)
        self.monitor = Monitor()
        self.replanner = Replanner(self.sched)
        self.advisor = LLMAdvisor()
        self.last_advice: dict = {}

    def sense(self, connectors: List[Connector], today: date) -> List[Task]:
        new: List[Task] = []
        for c in connectors:
            try:
                new.extend(c.fetch(today))
            except Exception as e:                  # noqa: BLE001
                print(f"[warn] connector {c.name} failed: {e}", file=sys.stderr)
        merged = dedupe(self.state.tasks + new)
        added = [t for t in merged if t not in self.state.tasks]
        self.state.tasks = merged
        return added

    def think(self, today: date, horizon: int = 21, mode: str = "repair",
              use_llm: bool = False) -> Tuple[Plan, float]:
        boosts = self.monitor.attendance_boosts(self.state.attendance)

        if use_llm:
            # Advisory pass runs against *current* alerts/plan (pre-replan),
            # since that's what a student would actually be looking at when
            # asking "why". Its boosts merge additively with attendance
            # boosts; the scheduler's hard constraints are unaffected either
            # way — see LLMAdvisor's docstring for why that's safe.
            pre_alerts = self.monitor.check(self.state.tasks, self.state.plan,
                                            today, self.state.attendance)
            advice = self.advisor.advise(self.state.tasks, pre_alerts,
                                         self.state.plan, today)
            self.last_advice = advice
            for tid, b in advice.get("boosts", {}).items():
                boosts[tid] = min(1.0, boosts.get(tid, 0.0) + b)
        else:
            self.last_advice = {}

        old = self.state.plan
        new = self.replanner.replan(self.state.tasks, old, today, horizon, mode, boosts)
        churn = Replanner.churn(old, new)
        self.state.plan = new
        return new, churn

    def alerts(self, today: date) -> List[Alert]:
        return self.monitor.check(self.state.tasks, self.state.plan,
                                  today, self.state.attendance)

    def act(self, connectors: List[Connector], approve: bool = False) -> None:
        if not self.state.plan:
            return
        for c in connectors:
            actions = c.propose(self.state.plan)
            if not actions:
                continue
            print(f"{c.name}: {len(actions)} proposed change(s)")
            if approve:
                c.commit(actions)
            else:
                print("  (not committed — approval required)")


# --------------------------------------------------------------------------
# Notifier — real alerts, not just terminal text
# --------------------------------------------------------------------------

class Notifier:
    """
    Sends alerts to you through channels outside the terminal:

      desktop — macOS pop-up (osascript) or Linux notify-send
      email   — SMTP to yourself. Configure with environment variables:
                  SMTP_USER, SMTP_PASSWORD (a Gmail *App Password*, not your
                  normal password), NOTIFY_EMAIL (defaults to SMTP_USER),
                  and optionally SMTP_HOST (smtp.gmail.com), SMTP_PORT (587).

    Each alert has a stable key and is re-sent at most once per cooldown
    (24h by default), so a course that stays below 75% reminds you daily
    instead of on every run. If no channel works, nothing is marked as sent,
    so it is retried on the next run.
    """
    KINDS = ("attendance", "overdue", "at_risk", "sync")

    def __init__(self, channels: Iterable[str] = ("desktop",), cooldown_hours: float = 24.0):
        self.channels = [c for c in channels if c in ("desktop", "email")]
        self.cooldown = timedelta(hours=cooldown_hours)
        self._warned_email = False

    @staticmethod
    def _clean(s: str, limit: int) -> str:
        s = " ".join(str(s).split())
        return s if len(s) <= limit else s[: limit - 1] + "…"

    def desktop(self, title: str, body: str) -> bool:
        system = platform.system()
        try:
            if system == "Darwin":
                esc = lambda x: x.replace("\\", "\\\\").replace('"', '\\"')
                msg = esc(self._clean(body, 230))
                ttl = esc(self._clean(title, 60))
                script = 'display notification "' + msg + '" with title "' + ttl + '"'
                subprocess.run(["osascript", "-e", script], check=True,
                               timeout=10, capture_output=True)
                return True
            if system == "Linux" and shutil.which("notify-send"):
                subprocess.run(["notify-send", self._clean(title, 60),
                                self._clean(body, 230)], check=True, timeout=10)
                return True
        except Exception as e:                       # noqa: BLE001
            print(f"[warn] desktop notification failed: {e}", file=sys.stderr)
        return False

    def email(self, subject: str, body: str) -> bool:
        host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        port = int(os.environ.get("SMTP_PORT", "587"))
        user = os.environ.get("SMTP_USER", "")
        pw = os.environ.get("SMTP_PASSWORD", "")
        to = os.environ.get("NOTIFY_EMAIL", "") or user
        if not (user and pw and to):
            if not self._warned_email:
                print("[warn] email not configured: set SMTP_USER and SMTP_PASSWORD "
                      "(Gmail app password) and optionally NOTIFY_EMAIL.", file=sys.stderr)
                self._warned_email = True
            return False
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = user, to, subject
        msg.set_content(body)
        try:
            if port == 465:
                with smtplib.SMTP_SSL(host, port, timeout=20) as s:
                    s.login(user, pw)
                    s.send_message(msg)
            else:
                with smtplib.SMTP(host, port, timeout=20) as s:
                    s.starttls()
                    s.login(user, pw)
                    s.send_message(msg)
            return True
        except Exception as e:                       # noqa: BLE001
            print(f"[warn] email failed: {e}", file=sys.stderr)
            return False

    def send(self, title: str, lines: List[str]) -> List[str]:
        """Deliver on every configured channel. Returns the channels that worked."""
        ok: List[str] = []
        if "desktop" in self.channels and self.desktop(title, " | ".join(lines)):
            ok.append("desktop")
        if "email" in self.channels and self.email(title, "\n".join(lines) + "\n\n— CollegeOS"):
            ok.append("email")
        return ok

    def notify_alerts(self, alerts: List[Alert], notified: Dict[str, str],
                      now: datetime) -> int:
        """Send one digest of alerts that are new or past their cooldown."""
        fresh: List[Alert] = []
        for a in alerts:
            if a.kind not in self.KINDS or not a.key:
                continue
            last = notified.get(a.key)
            if last:
                try:
                    if now - datetime.fromisoformat(last) < self.cooldown:
                        continue
                except ValueError:
                    pass
            fresh.append(a)
        if not fresh:
            return 0
        if not self.channels:
            return 0
        if {a.kind for a in fresh} == {"attendance"}:
            title = "CollegeOS: attendance is low"
        else:
            title = f"CollegeOS: {len(fresh)} alert(s) need attention"
        ok = self.send(title, [a.message for a in fresh])
        if not ok:
            print("[warn] no notification channel delivered; will retry next run",
                  file=sys.stderr)
            return 0
        stamp = now.isoformat(timespec="seconds")
        for a in fresh:
            notified[a.key] = stamp
        print(f"[notify] sent {len(fresh)} alert(s) via {', '.join(ok)}")
        return len(fresh)


# --------------------------------------------------------------------------
# Canvas pull — assignments + grades + announcements in one step
# --------------------------------------------------------------------------

def canvas_pull(state: State, config_path: str, today: date, ann_days: int = 14) -> dict:
    """
    Read-only sync from CanPlus/Canvas into `state`. Assignments are upserted by
    stable ID (locally logged work is preserved). Grades and announcements are
    best-effort: if the institution blocks those endpoints the assignment sync
    still succeeds and the problem is reported in res["errors"].
    Deadlines found in announcements become tasks (source="announcement").
    """
    conn = CanvasConnector(config_path)
    incoming = conn.fetch(today)

    existing = {t.id: t for t in state.tasks}
    added = updated = 0
    for it in incoming:
        cur = existing.get(it.id)
        if cur is None:
            state.tasks.append(it)
            existing[it.id] = it
            added += 1
        else:
            cur.title, cur.due, cur.course, cur.type = it.title, it.due, it.course, it.type
            cur.source = "canvas"
            cur.score, cur.points_possible = it.score, it.points_possible
            if cur.spent == 0:
                cur.est_effort = it.est_effort
            if it.done:
                cur.done = True
                cur.done_on = it.done_on or cur.done_on
            updated += 1

    res = {"conn": conn, "added": added, "updated": updated, "grades": [],
           "grade_changes": [], "ann_tasks": 0, "announcements": [], "errors": []}

    try:
        grades = conn.fetch_grades()
        old = {g.get("course_id"): g for g in state.grades}
        for g in grades:
            prev = old.get(g["course_id"])
            if prev and prev.get("current_score") != g["current_score"] \
                    and g["current_score"] is not None:
                before = prev.get("current_score")
                res["grade_changes"].append(
                    f"{g['course']}: {before if before is not None else '-'}% -> "
                    f"{g['current_score']}%")
        state.grades = grades
        res["grades"] = grades
    except RuntimeError as e:
        res["errors"].append(f"grades: {e}")

    try:
        anns = conn.fetch_announcements(ann_days)
        res["announcements"] = anns
        extractor = RuleExtractor()
        found: List[Task] = []
        for a in anns:
            try:
                posted = date.fromisoformat(a["posted"]) if a["posted"] else today
            except ValueError:
                posted = today
            for t in extractor.extract(f"{a['title']}\n\n{a['text']}", posted, a["course"][:40]):
                t.source = "announcement"
                found.append(t)
        merged = dedupe(state.tasks + found)
        res["ann_tasks"] = len([t for t in merged if t not in state.tasks])
        state.tasks = merged
        state.announcements = [{"id": a["id"], "course": a["course"],
                                "title": a["title"], "posted": a["posted"]}
                               for a in anns][:30]
    except RuntimeError as e:
        res["errors"].append(f"announcements: {e}")
    return res


# --------------------------------------------------------------------------
# Autonomous run cycle — the "agentic" loop in one call
# --------------------------------------------------------------------------

def run_cycle(state: State, agent: "Agent", opts, today: date) -> None:
    """
    sense -> think -> monitor -> notify -> (propose) act, with every external
    step isolated: one source failing (e.g. an expired ERP token) is reported
    and turned into an alert, but never stops the rest of the cycle.
    Calendar writes still require approval (opts.approve_gcal).
    """
    now = datetime.now()
    print(f"\n=== CollegeOS run {now:%Y-%m-%d %H:%M} ===")
    problems: List[Alert] = []

    # 1. Canvas -----------------------------------------------------------
    if not opts.no_canvas and os.path.exists(opts.canvas_config):
        try:
            r = canvas_pull(state, opts.canvas_config, today)
            print(f"[canvas] {r['added']} new / {r['updated']} updated assignment(s), "
                  f"{len(r['grades'])} course grade(s), "
                  f"{r['ann_tasks']} task(s) from announcements")
            for ch in r["grade_changes"]:
                print(f"         grade changed: {ch}")
            for err in r["errors"]:
                print(f"[canvas] partial: {err}")
        except RuntimeError as e:
            print(f"[canvas] failed: {e}")
            problems.append(Alert("warn", "sync", f"Canvas sync failed: {e}", key="sync:canvas"))
    else:
        print("[canvas] skipped (no canvas_token.json or --no-canvas)")

    # 2. ERP (attendance + timetable) --------------------------------------
    if not opts.no_erp and (opts.erp_user_id or os.path.exists("erp_token.json")
                            or os.environ.get("ERP_TOKEN")):
        try:
            erp = ERPConnector(opts.erp_user_id or None)
            state.attendance = erp.fetch_attendance()
            low = [a for a in state.attendance if a.held and a.ratio < a.threshold]
            print(f"[erp] attendance for {len(state.attendance)} course(s), "
                  f"{len(low)} below threshold")
            try:
                sessions = erp.fetch_schedule()
                state.timetable = sessions
                state.capacity.apply_classes(sessions, opts.day_budget)
                print(f"[erp] timetable: {len(sessions)} class(es)")
            except RuntimeError as e:
                print(f"[erp] timetable skipped: {e}")
        except RuntimeError as e:
            print(f"[erp] failed: {e}")
            problems.append(Alert("warn", "sync", f"ERP sync failed: {e}", key="sync:erp"))
    else:
        print("[erp] skipped (no erp_token.json, or --no-erp)")

    # 3. Gmail --------------------------------------------------------------
    if opts.senders:
        if not os.path.exists(opts.gmail_token):
            print("[gmail] skipped: not logged in yet. Run `gmail-sync --from ...` "
                  "once so the login can be saved.")
        else:
            try:
                known = sorted({a.course for a in state.attendance}
                               | {c.course_code for c in state.timetable}
                               | {t.course for t in state.tasks if t.source == "canvas"})
                extractor = LLMExtractor() if opts.llm_extract else RuleExtractor()
                gm = GmailConnector(opts.senders, "GEN", extractor, known,
                                    days=opts.gmail_days, token_path=opts.gmail_token)
                found = gm.fetch(today)
                merged = dedupe(state.tasks + found)
                new = [t for t in merged if t not in state.tasks]
                state.tasks = merged
                print(f"[gmail] {gm.stats['read']} email(s) read, {len(new)} new task(s)")
            except RuntimeError as e:
                print(f"[gmail] failed: {e}")
                problems.append(Alert("warn", "sync", f"Gmail sync failed: {e}", key="sync:gmail"))
    else:
        print("[gmail] skipped (no senders: add gmail_senders.json or --from)")

    # 4. Replan --------------------------------------------------------------
    plan, churn = agent.think(today, opts.days, "repair", use_llm=opts.use_llm)
    print(f"[plan] replanned (repair): {len(plan.blocks)} block(s), "
          f"churn {churn:.0%}, {len(plan.at_risk)} task(s) at risk")
    if agent.last_advice.get("rationale"):
        print(f"[advisor] {agent.last_advice['rationale']}")

    # 5. Monitor + notify ------------------------------------------------------
    alerts = agent.alerts(today) + problems
    for a in alerts:
        print(f"  [{a.level}] {a.message}")
    if not alerts:
        print("  no alerts")
    channels = {"none": [], "desktop": ["desktop"], "email": ["email"],
                "both": ["desktop", "email"]}[opts.notify]
    Notifier(channels, opts.cooldown).notify_alerts(alerts, state.notified, now)
    cutoff = now - timedelta(days=60)
    for k in list(state.notified):
        try:
            if datetime.fromisoformat(state.notified[k]) < cutoff:
                del state.notified[k]
        except ValueError:
            del state.notified[k]
    state.save()

    # 6. Google Calendar (approval-gated) ------------------------------------------
    if opts.gcal:
        if not os.path.exists(opts.gcal_token):
            print("[calendar] skipped: not logged in yet. Run `gcal-sync` once first.")
        else:
            try:
                cal = GoogleCalendarConnector({t.id: t for t in state.tasks},
                                              tz=opts.tz, token_path=opts.gcal_token)
                actions = cal.propose(plan)
                adds = sum(1 for a in actions if a["action"] == "create_event")
                dels = len(actions) - adds
                if not actions:
                    print("[calendar] already up to date")
                elif opts.approve_gcal:
                    cal.commit(actions)
                    print(f"[calendar] applied: {adds} added, {dels} replaced")
                else:
                    print(f"[calendar] {adds} to add, {dels} to replace — waiting for "
                          f"your approval (run `gcal-sync --approve`)")
            except RuntimeError as e:
                print(f"[calendar] failed: {e}")
    print("=== run finished ===")


# --------------------------------------------------------------------------
# Evaluation harness
# --------------------------------------------------------------------------

@dataclass
class SimResult:
    on_time: int
    late: int
    total_lateness_days: int
    mean_churn: float
    crunch_days: int           # days where >90% of capacity was scheduled


def simulate(seed: int, policy: str, semester_days: int = 45,
             n_tasks: int = 22, weights: Optional[dict] = None) -> SimResult:
    """
    One synthetic semester.

    Hidden truth: every task has a true effort the student discovers only by
    working on it. The planner sees a noisy estimate. Each day the student
    works the scheduled blocks (with imperfect compliance). Overruns are fed
    back as re-estimates, which is what forces replanning.

    Policies:
      edf   — earliest deadline first, no effort pressure, replan by rebuild
      agent — full prioritizer + repair-mode replanning + re-estimation

    `weights`, when given, overrides the agent policy's Prioritizer weights
    (keys: w_urgency, w_weight, w_pressure, w_boost) — used for ablations.
    Ignored for the edf policy, which always uses its fixed urgency-only
    weighting.
    """
    rng = random.Random(seed)
    today = date(2026, 8, 3)
    cap = Capacity()

    tasks: List[Task] = []
    truth: Dict[str, int] = {}
    arrival: Dict[str, date] = {}
    courses = ["CS401", "CS402", "CS403", "MA401", "HS401"]

    for _ in range(n_tasks):
        ttype = rng.choices(list(TASK_TYPES),
                            weights=[5, 3, 1, 3, 2, 2])[0]
        arrive_off = rng.randint(0, semester_days - 8)
        lead = rng.randint(4, 18)
        est = int(DEFAULT_EFFORT[ttype] * rng.uniform(0.6, 1.4))
        t = Task(
            id=new_id(),
            title=f"{ttype}-{rng.randint(100, 999)}",
            due=today + timedelta(days=arrive_off + lead),
            course=rng.choice(courses),
            type=ttype,
            est_effort=est,
            importance=DEFAULT_IMPORTANCE[ttype],
            created=today + timedelta(days=arrive_off),
        )
        tasks.append(t)
        truth[t.id] = max(20, int(est * rng.uniform(0.7, 2.0)))
        arrival[t.id] = today + timedelta(days=arrive_off)

    if policy == "edf":
        pri = Prioritizer(w_urgency=1.0, w_weight=0.0, w_pressure=0.0, w_boost=0.0)
        mode = "rebuild"
    else:
        pri = Prioritizer(**weights) if weights else Prioritizer()
        mode = "repair"

    sched = Scheduler(cap, pri)
    replanner = Replanner(sched)

    plan: Optional[Plan] = None
    churns: List[float] = []
    crunch = 0
    worked: Dict[str, int] = {t.id: 0 for t in tasks}

    for day_i in range(semester_days):
        day = today + timedelta(days=day_i)
        visible = [t for t in tasks if arrival[t.id] <= day]

        new_plan = replanner.replan(visible, plan, day, horizon=21, mode=mode)
        churns.append(Replanner.churn(plan, new_plan))
        plan = new_plan

        todays = plan.for_day(day)
        if sum(b.minutes for b in todays) > 0.9 * cap.available(day):
            crunch += 1

        budget = cap.available(day)
        for b in todays:
            if budget <= 0:
                break
            t = next((x for x in tasks if x.id == b.task_id), None)
            if not t or t.done:
                continue
            # Imperfect compliance: sometimes the student works less.
            actual = int(b.minutes * rng.uniform(0.6, 1.05))
            actual = min(actual, budget)
            budget -= actual
            worked[t.id] += actual
            t.spent = min(worked[t.id], t.est_effort)

            if worked[t.id] >= truth[t.id]:
                t.done = True
                t.done_on = day
            elif worked[t.id] >= t.est_effort:
                # Discovered an overrun — re-estimate (agent) or nudge (edf).
                bump = 1.35 if policy == "agent" else 1.10
                t.est_effort = int(max(t.est_effort * bump, worked[t.id] + 30))

    on_time = sum(1 for t in tasks if t.due and t.done and t.done_on and t.done_on <= t.due)
    late = len(tasks) - on_time
    lateness = 0
    end = today + timedelta(days=semester_days)
    for t in tasks:
        if t.due and t.done and t.done_on and t.done_on > t.due:
            lateness += (t.done_on - t.due).days
        elif t.due and not t.done:
            lateness += max(0, (end - t.due).days)

    return SimResult(on_time, late, lateness,
                     statistics.mean(churns) if churns else 0.0, crunch)


def _mean_ci95(xs: List[float]) -> Tuple[float, float]:
    """Mean and a normal-approximation 95% CI half-width (1.96 * SEM).
    Fine for n>=25ish; this is a teaching/eval harness, not a stats paper."""
    if not xs:
        return 0.0, 0.0
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, 0.0
    sd = statistics.stdev(xs)
    return m, 1.96 * sd / (len(xs) ** 0.5)


def evaluate(runs: int = 40, weights: Optional[dict] = None) -> dict:
    """
    Runs both policies, prints a comparison table with 95% CIs, and returns
    the raw per-metric (mean, ci95) values so callers (e.g. the ablation)
    can reuse them without re-simulating.
    """
    rows = {}
    raw = {}
    for policy in ("edf", "agent"):
        res = [simulate(seed, policy, weights=weights) for seed in range(runs)]
        on_time_pcts = [100 * r.on_time / (r.on_time + r.late) for r in res]
        lateness = [r.total_lateness_days for r in res]
        churns = [r.mean_churn for r in res]
        crunch = [r.crunch_days for r in res]
        raw[policy] = {
            "on_time_pct": on_time_pcts, "lateness_days": lateness,
            "churn": churns, "crunch_days": crunch,
        }
        rows[policy] = {k: _mean_ci95(v) for k, v in raw[policy].items()}

    print(f"\n{runs} simulated semesters, 22 tasks each (95% CI shown as \u00b1)\n")
    print(f"{'metric':<22}{'EDF baseline':>20}{'agent':>20}")
    print("-" * 62)
    for k, label, fmt in [
        ("on_time_pct", "on-time completion %", "{:.1f}"),
        ("lateness_days", "total lateness (days)", "{:.1f}"),
        ("churn", "mean plan churn", "{:.3f}"),
        ("crunch_days", "crunch days", "{:.1f}"),
    ]:
        (bm, bc), (am, ac) = rows["edf"][k], rows["agent"][k]
        bstr = f"{fmt.format(bm)} \u00b1{fmt.format(bc)}"
        astr = f"{fmt.format(am)} \u00b1{fmt.format(ac)}"
        print(f"{label:<22}{bstr:>20}{astr:>20}")
    print("\nHigher on-time is better; lower lateness, churn and crunch are better.")
    print("Overlapping CIs on a metric mean the two policies are not")
    print("distinguishable on it at this sample size.\n")
    return rows


def evaluate_ablation(runs: int = 40) -> None:
    """
    Zeroes out each Prioritizer weight one at a time (vs the agent's default
    weights) and reports the change in on-time completion, so you can see
    which signal is actually pulling weight in the simulation rather than
    assuming all four matter equally.
    """
    base = Prioritizer()
    configs = {
        "full (default)": None,
        "no urgency (w_urgency=0)": {"w_urgency": 0.0, "w_weight": base.w_weight,
                                     "w_pressure": base.w_pressure, "w_boost": base.w_boost},
        "no importance (w_weight=0)": {"w_urgency": base.w_urgency, "w_weight": 0.0,
                                       "w_pressure": base.w_pressure, "w_boost": base.w_boost},
        "no workload pressure (w_pressure=0)": {"w_urgency": base.w_urgency, "w_weight": base.w_weight,
                                                "w_pressure": 0.0, "w_boost": base.w_boost},
        "no attendance boost (w_boost=0)": {"w_urgency": base.w_urgency, "w_weight": base.w_weight,
                                            "w_pressure": base.w_pressure, "w_boost": 0.0},
    }

    print(f"\nWeight ablation, {runs} simulated semesters per configuration")
    print("(agent policy only; each row zeroes one signal vs the full default)\n")
    print(f"{'configuration':<38}{'on-time %':>14}{'lateness (d)':>16}{'churn':>10}")
    print("-" * 78)
    for name, w in configs.items():
        res = [simulate(seed, "agent", weights=w) for seed in range(runs)]
        on_time_pcts = [100 * r.on_time / (r.on_time + r.late) for r in res]
        lateness = [r.total_lateness_days for r in res]
        churns = [r.mean_churn for r in res]
        om, oc = _mean_ci95(on_time_pcts)
        lm, lc = _mean_ci95(lateness)
        cm, cc = _mean_ci95(churns)
        print(f"{name:<38}{f'{om:.1f}\u00b1{oc:.1f}':>14}"
              f"{f'{lm:.1f}\u00b1{lc:.1f}':>16}{f'{cm:.3f}\u00b1{cc:.3f}':>10}")
    print("\nA row close to the full-default row means that signal contributed")
    print("little to the outcome in this simulation; a row far from it means")
    print("the signal mattered.\n")
    print("Report these with confidence intervals and an ablation over the")
    print("Prioritizer weights — that table is the evaluation section.\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def default_gmail_senders(path: str = "gmail_senders.json") -> List[str]:
    """Saved sender list, so --from isn't needed every time.
    Order: GMAIL_SENDERS env var, then gmail_senders.json, e.g.
        {"senders": ["cse.xyz@ncuindia.edu", "ritachikara@ncuindia.edu"]}
    (a plain JSON list of addresses also works)."""
    env = [x.strip() for x in os.environ.get("GMAIL_SENDERS", "").split(",") if x.strip()]
    if env:
        return env
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("senders", []) if isinstance(data, dict) else data
            return [str(x).strip() for x in items if str(x).strip()]
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"[warn] {path} unreadable ({e}); expected "
                  '{"senders": ["a@x.com", "b@y.com"]}', file=sys.stderr)
    return []


def fmt_minutes(m: int) -> str:
    h, mm = divmod(m, 60)
    return f"{h}h{mm:02d}" if h else f"{mm}m"


def show_plan(state: State, today: date, days: int = 7) -> None:
    plan = state.plan
    if not plan or not plan.blocks:
        print("No plan. Run: python collegeos.py plan")
        return
    by_id = {t.id: t for t in state.tasks}
    print(f"\nPlan made {plan.made_on.isoformat()}\n")
    for i in range(days):
        day = today + timedelta(days=i)
        blocks = plan.for_day(day)
        if not blocks:
            continue
        total = sum(b.minutes for b in blocks)
        cap = state.capacity.available(day)
        print(f"{day.strftime('%a %d %b')}  [{fmt_minutes(total)} / {fmt_minutes(cap)}]")
        for b in sorted(blocks, key=lambda x: -x.minutes):
            t = by_id.get(b.task_id)
            if not t:
                continue
            left = (f"due in {(t.due - day).days}d" if t.due else "no deadline")
            print(f"   {fmt_minutes(b.minutes):>6}  {t.title[:46]:<46} "
                  f"{t.course:<7} {left}")
        print()


def show_status(state: State, today: date) -> None:
    agent = Agent(state)
    open_tasks = [t for t in state.tasks if not t.done]
    open_tasks.sort(key=lambda t: (t.due is None, t.due or date.max))
    print(f"\n{len(open_tasks)} open task(s), "
          f"{sum(t.remaining() for t in open_tasks)//60}h of work remaining\n")
    for t in open_tasks[:20]:
        left = t.days_left(today)
        flag = "!!" if t.due and left < 0 else ("!" if t.due and left <= 2 else "  ")
        due_label = t.due.isoformat() if t.due else "No due date"
        print(f"{flag} {t.id}  {t.title[:44]:<44} {t.course:<7} "
              f"{due_label:<12}  {fmt_minutes(t.remaining()):>6} left")
    alerts = agent.alerts(today)
    if alerts:
        print("\nAlerts:")
        for a in alerts:
            print(f"  [{a.level}] {a.message}")
    print()


def seed_demo(state: State, today: date) -> None:
    samples = [
        ("Data Structures Assignment 3 — AVL rotations", 9, "CS401", "assignment", 4),
        ("OS quiz on scheduling algorithms", 3, "CS402", "quiz", 3),
        ("DBMS mid-sem exam", 16, "CS403", "exam", 5),
        ("Read Chapter 7 — normalization", 5, "CS403", "reading", 2),
        ("Capstone synopsis submission", 12, "CS499", "project", 5),
        ("Maths tutorial sheet 4", 2, "MA401", "assignment", 2),
        ("Fill semester feedback form", 6, "HS401", "admin", 1),
    ]
    for title, off, course, ttype, imp in samples:
        state.tasks.append(Task(
            id=new_id(), title=title, due=today + timedelta(days=off),
            course=course, type=ttype, est_effort=DEFAULT_EFFORT[ttype],
            importance=imp, created=today))
    state.attendance = [
        Attendance("CS401", held=40, attended=34),
        Attendance("CS402", held=38, attended=25),
        Attendance("CS403", held=36, attended=30),
    ]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="collegeos", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="seed sample data and plan")

    a = sub.add_parser("add", help="add a task")
    a.add_argument("title")
    a.add_argument("--due", required=True, help="YYYY-MM-DD")
    a.add_argument("--course", default="GEN")
    a.add_argument("--type", default="assignment", choices=TASK_TYPES)
    a.add_argument("--effort", type=int, default=None, help="minutes")
    a.add_argument("--importance", type=int, default=None, choices=range(1, 6))

    e = sub.add_parser("extract", help="pull tasks out of a text file")
    e.add_argument("--file", required=True)
    e.add_argument("--course", default="GEN")
    e.add_argument("--llm", action="store_true", help="use the LLM extractor")

    pl = sub.add_parser("plan", help="build a schedule")
    pl.add_argument("--days", type=int, default=21)
    pl.add_argument("--show", type=int, default=7)
    pl.add_argument("--llm", action="store_true",
                    help="use the LLM advisor for priority hints + rationale")

    rp = sub.add_parser("replan", help="replan against current state")
    rp.add_argument("--mode", default="repair", choices=["repair", "rebuild"])
    rp.add_argument("--days", type=int, default=21)
    rp.add_argument("--llm", action="store_true",
                    help="use the LLM advisor for priority hints + rationale")

    ex = sub.add_parser("explain", help="ask the LLM advisor why, without changing the plan")

    sub.add_parser("status", help="tasks and alerts")

    clean = sub.add_parser("cleanup-demo", help="remove only the built-in sample tasks")
    clean.add_argument("--apply", action="store_true",
                       help="back up state and remove the identified sample tasks")

    d = sub.add_parser("done", help="mark a task complete")
    d.add_argument("task_id")

    lg = sub.add_parser("log", help="record time worked")
    lg.add_argument("task_id")
    lg.add_argument("--minutes", type=int, required=True)

    at = sub.add_parser("attendance", help="update attendance for a course")
    at.add_argument("course")
    at.add_argument("--held", type=int, required=True)
    at.add_argument("--attended", type=int, required=True)
    at.add_argument("--threshold", type=float, default=0.75)

    lm = sub.add_parser("lms-sync", help="pull courses and assignments from CanPlus LMS")
    lm.add_argument("--config", default="canvas_token.json",
                    help="path to the CanPlus token JSON file")

    cp = sub.add_parser("erp-sync", help="pull attendance from the college ERP")
    cp.add_argument("--user-id", default=None,
                    help="the userId query-param value from your ERP request URL")
    cp.add_argument("--threshold", type=float, default=0.75)

    cs = sub.add_parser("erp-schedule", help="pull the timetable from the college ERP")
    cs.add_argument("--user-id", default=None,
                    help="optional; defaults to erp_token.json / ERP_USER_ID")
    cs.add_argument("--day-budget", type=int, default=480,
                    help="total academic minutes per day (classes + study); default 480")

    ev = sub.add_parser("evaluate", help="agent vs EDF baseline")
    ev.add_argument("--runs", type=int, default=40)
    ev.add_argument("--ablation", action="store_true",
                    help="also report a per-signal weight ablation")

    gc = sub.add_parser("gcal-sync", help="put the study plan on Google Calendar")
    gc.add_argument("--approve", action="store_true",
                    help="actually write to your calendar (default: just show changes)")
    gc.add_argument("--days", type=int, default=14, help="how many days ahead to sync")
    gc.add_argument("--start", default="18:00", help="weekday study start time")
    gc.add_argument("--end", default="23:00", help="latest study end time")
    gc.add_argument("--tz", default="Asia/Kolkata")

    gm = sub.add_parser("gmail-sync", help="pull deadlines from emails by chosen senders")
    gm.add_argument("--from", dest="senders", action="append", default=[],
                    help="allowed sender address (repeat for each). "
                         "Or set GMAIL_SENDERS=a@x.com,b@y.com")
    gm.add_argument("--days", type=int, default=14, help="how many days back to look")
    gm.add_argument("--max", type=int, default=25, help="max emails to read")
    gm.add_argument("--course", default="GEN", help="course to use if the email names none")
    gm.add_argument("--llm", action="store_true", help="use the LLM extractor "
                    "(sends email text to the LLM provider)")
    gm.add_argument("--show", action="store_true",
                    help="print each email checked (preview + tasks found)")
    gm.add_argument("--dry-run", action="store_true",
                    help="look only: don't save any tasks")

    gr = sub.add_parser("grades", help="show saved course grades and assignment scores")

    nt = sub.add_parser("notify-test", help="send a test notification")
    nt.add_argument("--notify", default="desktop", choices=["desktop", "email", "both"])

    sc = sub.add_parser("schedule", help="show how to run CollegeOS automatically")
    sc.add_argument("--minutes", type=int, default=60, help="how often to run")

    rn = sub.add_parser("run", help="sync everything, check for problems, replan, notify")
    rn.add_argument("--every", type=int, default=0, metavar="MIN",
                    help="repeat every MIN minutes (0 = run once and exit)")
    rn.add_argument("--days", type=int, default=21, help="planning horizon in days")
    rn.add_argument("--canvas-config", default="canvas_token.json")
    rn.add_argument("--no-canvas", action="store_true")
    rn.add_argument("--erp-user-id", default="",
                    help="optional; the ERP is used automatically if erp_token.json exists")
    rn.add_argument("--no-erp", action="store_true", help="skip the ERP this run")
    rn.add_argument("--day-budget", type=int, default=480)
    rn.add_argument("--from", dest="senders", action="append", default=[],
                    help="allowed Gmail sender (repeat; or GMAIL_SENDERS=a@x.com,b@y.com)")
    rn.add_argument("--gmail-days", type=int, default=14)
    rn.add_argument("--gmail-token", default="gmail_token.json")
    rn.add_argument("--llm", action="store_true", help="LLM advisor for priority hints")
    rn.add_argument("--llm-extract", action="store_true",
                    help="use the LLM to read emails (sends email text to the provider)")
    rn.add_argument("--notify", default="desktop", choices=["none", "desktop", "email", "both"])
    rn.add_argument("--cooldown", type=float, default=24.0,
                    help="hours before the same alert is sent again")
    rn.add_argument("--gcal", action="store_true", help="also propose Google Calendar changes")
    rn.add_argument("--approve-gcal", action="store_true",
                    help="apply the calendar changes without asking (off by default)")
    rn.add_argument("--gcal-token", default="google_token.json")
    rn.add_argument("--tz", default="Asia/Kolkata")

    cal = sub.add_parser("sync", help="propose calendar changes")
    cal.add_argument("--approve", action="store_true")

    args = p.parse_args(argv)
    today = date.today()

    if args.cmd == "evaluate":
        evaluate(args.runs)
        if args.ablation:
            evaluate_ablation(args.runs)
        return 0

    state = State.load()
    agent = Agent(state)

    if args.cmd == "demo":
        if not state.tasks:
            seed_demo(state, today)
        plan, churn = agent.think(today)
        state.save()
        show_status(state, today)
        show_plan(state, today)
        return 0

    if args.cmd == "add":
        ttype = args.type
        t = Task(id=new_id(), title=args.title, due=date.fromisoformat(args.due),
                 course=args.course, type=ttype,
                 est_effort=args.effort or DEFAULT_EFFORT[ttype],
                 importance=args.importance or DEFAULT_IMPORTANCE[ttype],
                 created=today)
        state.tasks.append(t)
        state.save()
        print(f"added {t.id}  {t.title}  due {t.due.isoformat()}")
        return 0

    if args.cmd == "extract":
        extractor = LLMExtractor() if args.llm else RuleExtractor()
        conn = FileInboxConnector(args.file, args.course, extractor)
        added = agent.sense([conn], today)
        state.save()
        print(f"{len(added)} task(s) extracted via {extractor.name}")
        for t in added:
            print(f"  {t.id}  {t.due.isoformat()}  [{t.type}] {t.title}")
        return 0

    if args.cmd in ("plan", "replan"):
        mode = getattr(args, "mode", "rebuild" if args.cmd == "plan" else "repair")
        plan, churn = agent.think(today, args.days, mode, use_llm=getattr(args, "llm", False))
        state.save()
        if args.cmd == "replan":
            print(f"replanned ({mode}); churn = {churn:.1%}")
        if plan.at_risk:
            print(f"{len(plan.at_risk)} task(s) at risk — see status")
        if agent.last_advice.get("rationale"):
            print(f"\nAdvisor: {agent.last_advice['rationale']}")
        for c in agent.last_advice.get("conflicts", []):
            print(f"  [conflict] {c}")
        show_plan(state, today, getattr(args, "show", 7))
        return 0

    if args.cmd == "explain":
        alerts = agent.alerts(today)
        advice = agent.advisor.advise(state.tasks, alerts, state.plan, today)
        if not advice.get("rationale") and not advice.get("boosts"):
            if llm_available():
                print("Advisor call didn't return anything usable — "
                     "see the [warn] line above for why.")
            else:
                print("No ANTHROPIC_API_KEY or GEMINI_API_KEY set, or nothing to "
                     "explain — set one and make sure you have open tasks.")
            return 0
        print(f"\n{advice['rationale']}\n")
        if advice["boosts"]:
            by_id = {t.id: t for t in state.tasks}
            print("Extra attention suggested on:")
            for tid, b in sorted(advice["boosts"].items(), key=lambda x: -x[1]):
                t = by_id.get(tid)
                if t:
                    print(f"  +{b:.2f}  {t.title[:50]}")
        if advice["conflicts"]:
            print("\nPossible conflicts:")
            for c in advice["conflicts"]:
                print(f"  - {c}")
        return 0

    if args.cmd == "status":
        show_status(state, today)
        return 0

    if args.cmd == "cleanup-demo":
        # Exact signatures from seed_demo(). Never remove Canvas/ERP tasks or
        # user-created tasks that merely have a similar course code.
        demo_signatures = {
            ("Data Structures Assignment 3 — AVL rotations", "CS401"),
            ("OS quiz on scheduling algorithms", "CS402"),
            ("DBMS mid-sem exam", "CS403"),
            ("Read Chapter 7 — normalization", "CS403"),
            ("Capstone synopsis submission", "CS499"),
            ("Maths tutorial sheet 4", "MA401"),
            ("Fill semester feedback form", "HS401"),
        }
        candidates = [t for t in state.tasks
                      if t.source not in ("canvas", "erp")
                      and not t.id.startswith("canvas_")
                      and (t.title, t.course) in demo_signatures]
        if not candidates:
            print("No built-in demo tasks found. Nothing changed.")
            return 0
        print(f"Identified {len(candidates)} built-in demo task(s):")
        for t in candidates:
            print(f"  - {t.title} ({t.course}, id={t.id})")
        if not args.apply:
            print("\nPreview only; no data changed. Run with --apply to remove these tasks.")
            return 0

        backup_path = STATE_PATH + ".pre-demo-cleanup.bak"
        suffix = 1
        while os.path.exists(backup_path):
            backup_path = STATE_PATH + f".pre-demo-cleanup-{suffix}.bak"
            suffix += 1
        if os.path.exists(STATE_PATH):
            shutil.copy2(STATE_PATH, backup_path)
            print(f"Backup created: {backup_path}")

        removed_ids = {t.id for t in candidates}
        state.tasks = [t for t in state.tasks if t.id not in removed_ids]
        if state.plan:
            state.plan.blocks = [b for b in state.plan.blocks if b.task_id not in removed_ids]
            state.plan.at_risk = {tid: minutes for tid, minutes in state.plan.at_risk.items()
                                  if tid not in removed_ids}
        state.save()
        print(f"\nRemoved {len(removed_ids)} demo task(s). Canvas tasks, ERP data, attendance, and timetable were preserved.")
        return 0

    if args.cmd == "done":
        t = state.task(args.task_id)
        if not t:
            print("no unique task with that id prefix")
            return 1
        t.done, t.done_on = True, today
        state.save()
        print(f"done: {t.title}")
        return 0

    if args.cmd == "log":
        t = state.task(args.task_id)
        if not t:
            print("no unique task with that id prefix")
            return 1
        t.spent += args.minutes
        if t.spent > t.est_effort:
            old = t.est_effort
            t.est_effort = int(max(t.est_effort * 1.3, t.spent + 30))
            print(f"estimate revised {old}m -> {t.est_effort}m")
        state.save()
        print(f"logged {args.minutes}m on {t.title} ({fmt_minutes(t.remaining())} left)")
        return 0

    if args.cmd == "attendance":
        rec = next((a for a in state.attendance if a.course == args.course), None)
        if rec:
            rec.held, rec.attended, rec.threshold = args.held, args.attended, args.threshold
        else:
            state.attendance.append(Attendance(args.course, args.held,
                                               args.attended, args.threshold))
            rec = state.attendance[-1]
        state.save()
        print(f"{rec.course}: {rec.ratio*100:.1f}%"
              + (f" — {rec.classes_needed()} classes to recover"
                 if rec.ratio < rec.threshold else " — above threshold"))
        return 0

    if args.cmd == "lms-sync":
        try:
            r = canvas_pull(state, args.config, today)
        except RuntimeError as e:
            print(f"[error] {e}")
            return 1
        conn = r["conn"]
        state.save()
        print("\nCanPlus LMS sync complete (read-only).")
        print(f"Courses found: {len(conn.courses)}")
        for course in conn.courses:
            print(f"  - {course.get('name', 'Unnamed course')} (ID: {course.get('id')})")
        print(f"\nAssignments returned: {conn.assignment_count}")
        print(f"Tasks imported: {r['added']} new, {r['updated']} updated")
        print(f"Submitted: {conn.submitted_count} | Missing: {conn.missing_count}")
        if conn.assignment_rows:
            print("\nAssignment details:")
            for row in conn.assignment_rows:
                print(f"  [{row['status']:<9}] {row['due']}  {row['course']}: {row['title']}")
        if conn.skipped_without_due:
            print(f"Skipped (invalid due date): {conn.skipped_without_due}")
        if conn.skipped_without_id:
            print(f"Skipped (missing assignment ID): {conn.skipped_without_id}")
        print(f"\nGrades: {len(r['grades'])} course(s) saved \u2014 run `grades` to view")
        for ch in r["grade_changes"]:
            print(f"  grade changed: {ch}")
        print(f"Announcements read: {len(r['announcements'])}, "
              f"{r['ann_tasks']} new task(s) found in them")
        for err in r["errors"]:
            print(f"[warn] {err}")
        print("\nRun 'python3 collegeos.py status' to see your tasks, or "
              "'python3 collegeos.py plan' to schedule them.")
        return 0

    if args.cmd == "grades":
        if not state.grades:
            print("No grades saved yet. Run: python3 collegeos.py lms-sync")
            return 0
        print("\nCourse grades (from Canvas)\n")
        for g in state.grades:
            cs, fs = g.get("current_score"), g.get("final_score")
            cur = f"{cs:.1f}%" if cs is not None else "hidden/none"
            fin = f"{fs:.1f}%" if fs is not None else "-"
            letter = f" ({g['current_grade']})" if g.get("current_grade") else ""
            print(f"  {g['course']:<14} current {cur}{letter:<6}  final {fin}")
        scored = [t for t in state.tasks if t.score is not None and t.points_possible]
        if scored:
            print("\nGraded assignments\n")
            for t in sorted(scored, key=lambda x: (x.course, x.due or date.min)):
                pct = 100 * t.score / t.points_possible
                print(f"  {t.course:<14} {t.title[:40]:<40} "
                      f"{t.score:g}/{t.points_possible:g}  ({pct:.0f}%)")
        print()
        return 0

    if args.cmd == "notify-test":
        chans = ["desktop", "email"] if args.notify == "both" else [args.notify]
        ok = Notifier(chans).send("CollegeOS test", ["If you can see this, alerts work."])
        print(f"delivered via: {', '.join(ok) if ok else 'nothing (see warnings above)'}")
        return 0 if ok else 1

    if args.cmd == "schedule":
        here = os.path.dirname(os.path.abspath(__file__))
        py, me = sys.executable, os.path.abspath(__file__)
        print(f"""
Run CollegeOS automatically every {args.minutes} minutes.

Option 1 \u2014 built-in loop (leave a terminal open):
  {py} {me} run --every {args.minutes} --notify desktop

Option 2 \u2014 cron (macOS/Linux). Run `crontab -e` and add:
  */{args.minutes} * * * * cd {here} && {py} {me} run >> collegeos.log 2>&1

Option 3 \u2014 macOS launchd. Save as ~/Library/LaunchAgents/com.collegeos.run.plist,
then run: launchctl load ~/Library/LaunchAgents/com.collegeos.run.plist

<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.collegeos.run</string>
  <key>ProgramArguments</key><array>
    <string>{py}</string><string>{me}</string><string>run</string>
  </array>
  <key>WorkingDirectory</key><string>{here}</string>
  <key>StartInterval</key><integer>{args.minutes * 60}</integer>
  <key>StandardOutPath</key><string>{here}/collegeos.log</string>
  <key>StandardErrorPath</key><string>{here}/collegeos.log</string>
</dict></plist>

Notes:
  * Scheduled jobs can't open a browser, so run `gmail-sync` and `gcal-sync`
    by hand once first; `run` then reuses the saved logins.
  * The ERP token expires. When it does, `run` sends you a "ERP sync failed"
    alert \u2014 update "token" in erp_token.json. Don't share or commit that file.
  * Calendar changes are only proposed unless you add --approve-gcal.
""")
        return 0

    if args.cmd == "run":
        args.use_llm = args.llm
        if not args.senders:
            args.senders = default_gmail_senders()
        while True:
            try:
                state = State.load()
                agent = Agent(state)
                run_cycle(state, agent, args, date.today())
            except KeyboardInterrupt:
                print("\nstopped.")
                return 0
            except Exception as e:                    # noqa: BLE001
                print(f"[error] run failed: {e}", file=sys.stderr)
                if not args.every:
                    return 1
            if not args.every:
                return 0
            print(f"next run in {args.every} min (Ctrl+C to stop)")
            try:
                time.sleep(args.every * 60)
            except KeyboardInterrupt:
                print("\nstopped.")
                return 0

    if args.cmd == "erp-sync":
        try:
            conn = ERPConnector(args.user_id)
            records = conn.fetch_attendance(args.threshold)
        except RuntimeError as e:
            print(f"[error] {e}")
            return 1
        state.attendance = records
        state.save()
        print(f"synced {len(records)} course(s) from ERP")
        for a in records:
            flag = "  [below threshold]" if a.ratio < a.threshold else ""
            print(f"  {a.course}: {a.ratio*100:.1f}%{flag}")
        return 0

    if args.cmd == "erp-schedule":
        try:
            conn = ERPConnector(args.user_id)
            sessions = conn.fetch_schedule()
        except RuntimeError as e:
            print(f"[error] {e}")
            return 1
        state.timetable = sessions
        state.capacity.apply_classes(sessions, args.day_budget)
        state.save()
        print(f"synced {len(sessions)} class(es) from the ERP")
        last = None
        for c in sessions:
            if c.day != last:
                print(f"\n{c.day.strftime('%a %d %b')}")
                last = c.day
            print(f"  {c.start_min//60:02d}:{c.start_min%60:02d}-"
                  f"{c.end_min//60:02d}:{c.end_min%60:02d}  "
                  f"{c.course_code:<7} {c.course_name[:38]:<38} {c.room}")
        print("\nStudy capacity adjusted for these days. Run: python3 collegeos.py replan")
        return 0

    if args.cmd == "gcal-sync":
        if not state.plan or not state.plan.blocks:
            print("No plan yet. Run: python3 collegeos.py plan")
            return 1
        conn = GoogleCalendarConnector({t.id: t for t in state.tasks}, tz=args.tz,
                                       study_start=args.start, study_end=args.end,
                                       days=args.days)
        try:
            actions = conn.propose(state.plan)
        except RuntimeError as e:
            print(f"[error] {e}")
            return 1
        dels = [a for a in actions if a["action"] == "delete_event"]
        news = [a for a in actions if a["action"] == "create_event"]
        print(f"\nChanges to your Google Calendar: {len(news)} to add, "
              f"{len(dels)} old CollegeOS events to replace")
        for a in dels:
            print(f"  - remove  {a['when'][:16]}  {a['label']}")
        for a in news:
            print(f"  + add     {a['when'][:16]}  {a['minutes']:>3}m  {a['label']}")
        for b, why in conn.last_skipped:
            t = state.task(b.task_id)
            print(f"  ! skipped {b.day}  {t.title if t else b.task_id}: {why}")
        if not actions:
            print("Nothing to do.")
            return 0
        if not args.approve:
            print("\nNothing was changed. Run again with --approve to apply.")
            return 0
        try:
            conn.commit(actions)
        except RuntimeError as e:
            print(f"[error] {e}")
            return 1
        print("\nDone. Check your Google Calendar.")
        return 0

    if args.cmd == "gmail-sync":
        senders = args.senders or default_gmail_senders()
        known = sorted({a.course for a in state.attendance}
                       | {c.course_code for c in state.timetable}
                       | {t.course for t in state.tasks if t.source == "canvas"})
        extractor = LLMExtractor() if args.llm else RuleExtractor()
        try:
            conn = GmailConnector(senders, args.course, extractor, known,
                                  days=args.days, max_messages=args.max)
            found = conn.fetch(today)
        except RuntimeError as e:
            print(f"[error] {e}")
            return 1
        merged = dedupe(state.tasks + found)
        added = [t for t in merged if t not in state.tasks]
        if not args.dry_run:
            state.tasks = merged
            state.save()
        st = conn.stats
        print(f"Senders allowed: {', '.join(conn.senders)}")
        print(f"Emails found: {st['found']} | read: {st['read']} | "
              f"wrong sender: {st['wrong_sender']} | failed spoof check: {st['failed_auth']}")
        tag = " (dry run: nothing saved)" if args.dry_run else ""
        print(f"{len(added)} new task(s) via {extractor.name}{tag}")
        for t in added:
            due = t.due.isoformat() if t.due else "no date"
            print(f"  {t.id}  {due}  {t.course:<7} [{t.type}] {t.title}")
        if args.show:
            print("\nEmails checked (shown on screen only, never saved):")
            for i, m in enumerate(conn.log, 1):
                print(f"\n[{i}] {m['date']}  {m['status']}")
                print(f"    From:    {m['from']}")
                print(f"    Subject: {m['subject']}")
                if m["preview"]:
                    print(f"    Text:    {m['preview']}")
                for due, title in m["tasks"]:
                    print(f"    -> task found: {due}  {title}")
        if st["found"] == 0:
            print("No emails matched. Check the addresses are spelled exactly, "
                  "or try --days 30.")
        return 0

    if args.cmd == "sync":
        agent.act([DryRunCalendarConnector()], approve=args.approve)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
