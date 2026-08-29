from __future__ import annotations

import hashlib
import re

from adr.core.types import (
    ActionType,
    Budget,
    Evidence,
    OrchestratorAction,
    Query,
    Report,
    StepRecord,
    Subtask,
    TokenUsage,
    Trajectory,
)


_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]{3,}")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def evidence_id(url: str, title: str = "") -> str:
    digest = hashlib.sha1(f"{url}|{title}".encode("utf-8")).hexdigest()[:12]
    return f"ev_{digest}"


def subtask_id(goal: str, index: int) -> str:
    digest = hashlib.sha1(f"{index}|{goal}".encode("utf-8")).hexdigest()[:8]
    return f"st_{index}_{digest}"


class ResearchState:
    """Mutable evidence pool + frontier + budget for one query."""

    def __init__(self, query: Query, budget: Budget | None = None) -> None:
        self.query = query
        self.budget = budget or Budget()
        self.subtasks: dict[str, Subtask] = {}
        self.evidence: dict[str, Evidence] = {}
        self.steps: list[StepRecord] = []
        self.report: Report | None = None
        self.terminated: bool = False
        self.termination_reason: str = ""

    # ── frontier ──────────────────────────────────────────────────
    def add_subtasks(self, goals: list[str], *, parent_id: str | None = None) -> list[Subtask]:
        created: list[Subtask] = []
        existing = {_norm(st.goal) for st in self.subtasks.values()}
        for goal in goals:
            cleaned = goal.strip()
            if not cleaned or _norm(cleaned) in existing:
                continue
            idx = len(self.subtasks) + 1
            item = Subtask(id=subtask_id(cleaned, idx), goal=cleaned, parent_id=parent_id)
            self.subtasks[item.id] = item
            existing.add(_norm(cleaned))
            created.append(item)
        if not self.subtasks:
            fallback = Subtask(id=subtask_id(self.query.text, 1), goal=self.query.text)
            self.subtasks[fallback.id] = fallback
            created.append(fallback)
        return created

    def open_subtasks(self) -> list[Subtask]:
        return [st for st in self.subtasks.values() if st.status in {"open", "active"}]

    def mark_active(self, subtask_id: str) -> None:
        if subtask_id in self.subtasks:
            self.subtasks[subtask_id].status = "active"

    def mark_done(self, subtask_id: str) -> None:
        if subtask_id in self.subtasks:
            self.subtasks[subtask_id].status = "done"

    def allocate(self, subtask_id: str, boost: float = 1.0) -> None:
        if subtask_id in self.subtasks:
            st = self.subtasks[subtask_id]
            st.priority += boost
            st.status = "active"

    def pick_branch(self) -> Subtask | None:
        open_items = self.open_subtasks()
        if not open_items:
            return None
        return min(open_items, key=lambda st: (len(st.evidence_ids), -st.priority, st.searches))

    # ── evidence pool ─────────────────────────────────────────────
    def add_evidence(self, items: list[Evidence]) -> list[Evidence]:
        added: list[Evidence] = []
        for item in items:
            if item.id in self.evidence:
                continue
            if len(self.retained()) >= self.budget.max_evidence:
                item.retained = False
            item.added_step = self.budget.used_steps
            self.evidence[item.id] = item
            if item.subtask_id and item.subtask_id in self.subtasks:
                self.subtasks[item.subtask_id].evidence_ids.append(item.id)
            added.append(item)
        return added

    def retained(self) -> list[Evidence]:
        return [ev for ev in self.evidence.values() if ev.retained and ev.superseded_by is None]

    def pruned(self) -> list[Evidence]:
        return [ev for ev in self.evidence.values() if not ev.retained or ev.superseded_by]

    def prune(
        self,
        *,
        evidence_ids: list[str] | None = None,
        drop_duplicates: bool = True,
        min_score: float | None = None,
        max_keep: int | None = None,
    ) -> list[str]:
        dropped: list[str] = []
        if evidence_ids:
            for eid in evidence_ids:
                if eid in self.evidence and self.evidence[eid].retained:
                    self.evidence[eid].retained = False
                    dropped.append(eid)

        if drop_duplicates:
            seen_urls: dict[str, str] = {}
            for ev in sorted(self.retained(), key=lambda e: (-e.score, e.added_step)):
                key = ev.url.rstrip("/").lower()
                if key in seen_urls:
                    ev.retained = False
                    ev.superseded_by = seen_urls[key]
                    dropped.append(ev.id)
                else:
                    seen_urls[key] = ev.id

        if min_score is not None:
            for ev in self.retained():
                if ev.score < min_score:
                    ev.retained = False
                    dropped.append(ev.id)

        if max_keep is not None and len(self.retained()) > max_keep:
            ranked = sorted(self.retained(), key=lambda e: (-e.score, e.added_step))
            for ev in ranked[max_keep:]:
                ev.retained = False
                dropped.append(ev.id)
        return dropped

    def attach_text(self, evidence_id: str, text: str) -> None:
        if evidence_id in self.evidence:
            self.evidence[evidence_id].text = text

    # ── accounting ────────────────────────────────────────────────
    def record_step(
        self,
        action: OrchestratorAction,
        *,
        observation: str = "",
        tokens: TokenUsage | None = None,
        latency_s: float = 0.0,
        extra: dict | None = None,
    ) -> StepRecord:
        stats = self.compact_stats()
        self.budget.charge(
            steps=1,
            searches=1 if action.type is ActionType.SEARCH else 0,
            reads=1 if action.type is ActionType.READ else 0,
            tokens=(tokens or TokenUsage()).total_tokens,
            latency_s=latency_s,
        )
        record = StepRecord(
            step=len(self.steps) + 1,
            action=action,
            stats_before=stats,
            observation=observation,
            tokens=tokens or TokenUsage(),
            latency_s=latency_s,
            extra=extra or {},
        )
        self.steps.append(record)
        if action.type is ActionType.TERMINATE:
            self.terminated = True
            self.termination_reason = action.rationale or "terminate"
        if action.type is ActionType.WRITE and action.report_draft:
            self.report = Report(article=action.report_draft, citations=self.citation_urls())
        return record

    def should_stop(self) -> bool:
        return self.terminated or self.budget.exhausted()

    def citation_urls(self) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()
        for ev in self.retained():
            if ev.url and ev.url not in seen:
                urls.append(ev.url)
                seen.add(ev.url)
        return urls

    def compact_stats(self) -> dict:
        """Compact observation used by PILOT instead of raw passages."""
        retained = self.retained()
        open_items = self.open_subtasks()
        scores = [ev.score for ev in retained]
        lengths = [len(ev.body()) for ev in retained]
        query_terms = set(_WORD.findall(self.query.text.lower()))
        overlap = 0.0
        for ev in retained:
            terms = set(_WORD.findall((ev.title + " " + ev.body(400)).lower()))
            if query_terms:
                overlap += len(terms & query_terms) / max(1, len(query_terms))
        branch_rows = []
        for st in self.subtasks.values():
            branch_rows.append(
                {
                    "id": st.id,
                    "goal": st.goal,
                    "status": st.status,
                    "priority": round(st.priority, 3),
                    "searches": st.searches,
                    "n_evidence": len(st.evidence_ids),
                    "n_retained": sum(
                        1 for eid in st.evidence_ids if eid in self.evidence and self.evidence[eid].retained
                    ),
                }
            )
        return {
            "query_id": self.query.id,
            "n_evidence": len(self.evidence),
            "n_retained": len(retained),
            "n_pruned": len(self.pruned()),
            "n_open_branches": len(open_items),
            "n_branches": len(self.subtasks),
            "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "mean_chars": int(sum(lengths) / len(lengths)) if lengths else 0,
            "query_term_coverage": round(overlap / len(retained), 4) if retained else 0.0,
            "unique_urls": len({ev.url for ev in retained}),
            "budget": {
                "step_frac": round(self.budget.step_frac(), 4),
                "token_frac": round(self.budget.token_frac(), 4),
                "remaining_steps": self.budget.remaining_steps(),
                "remaining_searches": self.budget.remaining_searches(),
                "remaining_reads": self.budget.remaining_reads(),
                "remaining_tokens": self.budget.remaining_tokens(),
                "used_latency_s": round(self.budget.used_latency_s, 3),
            },
            "branches": branch_rows,
        }

    def trajectory(self) -> Trajectory:
        return Trajectory(
            query=self.query,
            steps=list(self.steps),
            report=self.report,
            final_stats=self.compact_stats(),
            error=None if self.report else (self.termination_reason or "no_report"),
        )
