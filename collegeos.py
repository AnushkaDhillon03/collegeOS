#!/usr/bin/env python3
"""
CollegeOS — agentic academic planning core.

Single-file, stdlib-only prototype of the defensible core of the project:
task model -> extraction -> prioritization -> scheduling -> monitoring ->
incremental replanning, plus an evaluation harness that compares the agent
against an earliest-deadline-first baseline on simulated semesters.

Deliberately NOT included here: Gmail/Canvas/Calendar OAuth clients. Those are
plumbing behind the Connector interface at the bottom of this file; the core
runs and can be evaluated without any credentials.

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
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
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
    due: date
    course: str = "GEN"
    type: str = "assignment"
    est_effort: int = 180          # minutes, the planner's current belief
    spent: int = 0                 # minutes actually logged
    importance: int = 3            # 1-5
    done: bool = False
    done_on: Optional[date] = None
    source: str = "manual"         # manual | email | canvas | erp
    created: date = field(default_factory=date.today)

    def remaining(self) -> int:
        return 0 if self.done else max(0, self.est_effort - self.spent)

    def days_left(self, today: date) -> int:
        return (self.due - today).days

    def to_json(self) -> dict:
        d = asdict(self)
        d["due"] = self.due.isoformat()
        d["created"] = self.created.isoformat()
        d["done_on"] = self.done_on.isoformat() if self.done_on else None
        return d

    @staticmethod
    def from_json(d: dict) -> "Task":
        d = dict(d)
        d["due"] = date.fromisoformat(d["due"])
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
class Capacity:
    weekday_minutes: int = 180
    weekend_minutes: int = 300
    committed: Dict[str, int] = field(default_factory=dict)  # iso date -> busy minutes

    def available(self, day: date) -> int:
        base = self.weekend_minutes if day.weekday() >= 5 else self.weekday_minutes
        return max(0, base - self.committed.get(day.isoformat(), 0))

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
        days_left = max(0, (task.due - day).days)
        urgency = 1.0 / (1.0 + days_left)

        weight = (task.importance - 1) / 4.0

        avail = cap.cumulative(day, task.due)
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
                            if not t.done and remaining.get(t.id, 0) > 0 and t.due >= day]
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
    kind: str       # overdue | at_risk | attendance | overrun
    message: str


class Monitor:
    """Turns state + plan into alerts. Alerts are what trigger replanning."""

    def check(self, tasks: List[Task], plan: Optional[Plan], today: date,
              attendance: Iterable[Attendance]) -> List[Alert]:
        alerts: List[Alert] = []
        by_id = {t.id: t for t in tasks}

        for t in tasks:
            if t.done:
                continue
            if t.due < today:
                alerts.append(Alert("critical", "overdue",
                                    f"OVERDUE {t.days_left(today)*-1}d: {t.title} ({t.course})"))
            elif t.spent > t.est_effort * 1.25:
                alerts.append(Alert("warn", "overrun",
                                    f"Effort overrun on {t.title}: "
                                    f"{t.spent}m spent vs {t.est_effort}m estimated"))

        if plan:
            for tid, short in plan.at_risk.items():
                t = by_id.get(tid)
                if t:
                    alerts.append(Alert("critical", "at_risk",
                                        f"Cannot fit {short}m of {t.title} before "
                                        f"{t.due.isoformat()} — deadline at risk"))

        for a in attendance:
            if a.held and a.ratio < a.threshold:
                need = a.classes_needed()
                alerts.append(Alert("warn", "attendance",
                                    f"{a.course} attendance {a.ratio*100:.1f}% "
                                    f"(below {a.threshold*100:.0f}%) — "
                                    f"{need} consecutive classes to recover"))
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
            if b.day < today or b.day > t.due:
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
    """Same course + same due date + overlapping title words == same task."""
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
            f"due={t.due.isoformat()} remaining_min={t.remaining()} "
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


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------

@dataclass
class State:
    tasks: List[Task] = field(default_factory=list)
    attendance: List[Attendance] = field(default_factory=list)
    plan: Optional[Plan] = None
    capacity: Capacity = field(default_factory=Capacity)

    def save(self, path: str = STATE_PATH) -> None:
        payload = {
            "tasks": [t.to_json() for t in self.tasks],
            "attendance": [asdict(a) for a in self.attendance],
            "plan": self.plan.to_json() if self.plan else None,
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
             n_tasks: int = 22) -> SimResult:
    """
    One synthetic semester.

    Hidden truth: every task has a true effort the student discovers only by
    working on it. The planner sees a noisy estimate. Each day the student
    works the scheduled blocks (with imperfect compliance). Overruns are fed
    back as re-estimates, which is what forces replanning.

    Policies:
      edf   — earliest deadline first, no effort pressure, replan by rebuild
      agent — full prioritizer + repair-mode replanning + re-estimation
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
        pri = Prioritizer()
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

    on_time = sum(1 for t in tasks if t.done and t.done_on and t.done_on <= t.due)
    late = len(tasks) - on_time
    lateness = 0
    end = today + timedelta(days=semester_days)
    for t in tasks:
        if t.done and t.done_on and t.done_on > t.due:
            lateness += (t.done_on - t.due).days
        elif not t.done:
            lateness += max(0, (end - t.due).days)

    return SimResult(on_time, late, lateness,
                     statistics.mean(churns) if churns else 0.0, crunch)


def evaluate(runs: int = 40) -> None:
    rows = {}
    for policy in ("edf", "agent"):
        res = [simulate(seed, policy) for seed in range(runs)]
        rows[policy] = {
            "on_time_pct": 100 * statistics.mean(
                r.on_time / (r.on_time + r.late) for r in res),
            "lateness_days": statistics.mean(r.total_lateness_days for r in res),
            "churn": statistics.mean(r.mean_churn for r in res),
            "crunch_days": statistics.mean(r.crunch_days for r in res),
        }

    print(f"\n{runs} simulated semesters, 22 tasks each\n")
    print(f"{'metric':<22}{'EDF baseline':>16}{'agent':>16}{'delta':>12}")
    print("-" * 66)
    for k, label, fmt in [
        ("on_time_pct", "on-time completion %", "{:.1f}"),
        ("lateness_days", "total lateness (days)", "{:.1f}"),
        ("churn", "mean plan churn", "{:.3f}"),
        ("crunch_days", "crunch days", "{:.1f}"),
    ]:
        b, a = rows["edf"][k], rows["agent"][k]
        print(f"{label:<22}{fmt.format(b):>16}{fmt.format(a):>16}"
              f"{fmt.format(a - b):>12}")
    print("\nHigher on-time is better; lower lateness, churn and crunch are better.")
    print("Report these with confidence intervals and an ablation over the")
    print("Prioritizer weights — that table is the evaluation section.\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

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
            left = (t.due - day).days
            print(f"   {fmt_minutes(b.minutes):>6}  {t.title[:46]:<46} "
                  f"{t.course:<7} due in {left}d")
        print()


def show_status(state: State, today: date) -> None:
    agent = Agent(state)
    open_tasks = [t for t in state.tasks if not t.done]
    open_tasks.sort(key=lambda t: t.due)
    print(f"\n{len(open_tasks)} open task(s), "
          f"{sum(t.remaining() for t in open_tasks)//60}h of work remaining\n")
    for t in open_tasks[:20]:
        left = t.days_left(today)
        flag = "!!" if left < 0 else ("!" if left <= 2 else "  ")
        print(f"{flag} {t.id}  {t.title[:44]:<44} {t.course:<7} "
              f"{t.due.isoformat()}  {fmt_minutes(t.remaining()):>6} left")
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

    ev = sub.add_parser("evaluate", help="agent vs EDF baseline")
    ev.add_argument("--runs", type=int, default=40)

    cal = sub.add_parser("sync", help="propose calendar changes")
    cal.add_argument("--approve", action="store_true")

    args = p.parse_args(argv)
    today = date.today()

    if args.cmd == "evaluate":
        evaluate(args.runs)
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

    if args.cmd == "sync":
        agent.act([DryRunCalendarConnector()], approve=args.approve)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
