"""
llm_failures_collect
====================

Data collection only, for the study
  How LLM users explain factual errors, memory lapses and instruction-following
  failures: Expressed mental models and expectation violations

Collects public GitHub Issues and Discussions from 30 repositories: every
unique Issue returned by the search lexicon (opening post and all comments)
and every matching Discussion (opening post, comments and nested replies).
Raw records are stored losslessly as JSONL with a completion manifest so that
runs resume without duplication; assembled full-text thread and message
datasets are saved as CSV and JSONL with a collection summary.

There is no screening, coding, marker, sampling or LLM step in this package,
and no retrieved thread is discarded by local keyword checks during collection.
A separate, optional Step 8 (clean_for_coding) writes cleaned coding-ready
copies to a `coding_ready` subfolder without modifying the collected datasets.

Pipeline version: 3.1.0
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import requests

PIPELINE_VERSION = "3.1.0"


# =============================================================================
# 0. Repository scope and search lexicon
# =============================================================================

# Exactly 30 repositories, five in each of six categories. Paths are resolved to
# canonical locations at run time (resolve_repositories).
REPO_INFO = {

    # 1. General chat interfaces and personal assistants (5)
    "open-webui/open-webui": {"category": "General chat interfaces and personal assistants", "subgroup": "Chat interface", "system_type": "Conversational assistants"},
    "LibreChat-AI/LibreChat": {"category": "General chat interfaces and personal assistants", "subgroup": "Chat interface", "system_type": "Conversational assistants"},
    "lobehub/lobehub": {"category": "General chat interfaces and personal assistants", "subgroup": "Chat interface", "system_type": "Conversational assistants"},
    "janhq/jan": {"category": "General chat interfaces and personal assistants", "subgroup": "Chat interface", "system_type": "Conversational assistants"},
    "CherryHQ/cherry-studio": {"category": "General chat interfaces and personal assistants", "subgroup": "Chat interface", "system_type": "Conversational assistants"},

    # 2. Document, knowledge and search assistants (5)
    "Mintplex-Labs/anything-llm": {"category": "Document, knowledge and search assistants", "subgroup": "Knowledge assistant", "system_type": "Knowledge and memory systems"},
    "nomic-ai/gpt4all": {"category": "Document, knowledge and search assistants", "subgroup": "Knowledge assistant", "system_type": "Knowledge and memory systems"},
    "khoj-ai/khoj": {"category": "Document, knowledge and search assistants", "subgroup": "Knowledge assistant", "system_type": "Knowledge and memory systems"},
    "infiniflow/ragflow": {"category": "Document, knowledge and search assistants", "subgroup": "Knowledge assistant", "system_type": "Knowledge and memory systems"},
    "onyx-dot-app/onyx": {"category": "Document, knowledge and search assistants", "subgroup": "Knowledge assistant", "system_type": "Knowledge and memory systems"},

    # 3. Coding assistants and action-taking agents (5)
    "openai/codex": {"category": "Coding assistants and action-taking agents", "subgroup": "Coding agent", "system_type": "AI agents and orchestration"},
    "cline/cline": {"category": "Coding assistants and action-taking agents", "subgroup": "Coding agent", "system_type": "AI agents and orchestration"},
    "Aider-AI/aider": {"category": "Coding assistants and action-taking agents", "subgroup": "Coding agent", "system_type": "AI agents and orchestration"},
    "continuedev/continue": {"category": "Coding assistants and action-taking agents", "subgroup": "Coding agent", "system_type": "AI agents and orchestration"},
    "OpenHands/OpenHands": {"category": "Coding assistants and action-taking agents", "subgroup": "Coding agent", "system_type": "AI agents and orchestration"},

    # 4. Role-play and local-model environments (5)
    "SillyTavern/SillyTavern": {"category": "Role-play and local-model environments", "subgroup": "Role-play and creative writing", "system_type": "Conversational assistants"},
    "kwaroran/Risuai": {"category": "Role-play and local-model environments", "subgroup": "Role-play and creative writing", "system_type": "Conversational assistants"},
    "LostRuins/koboldcpp": {"category": "Role-play and local-model environments", "subgroup": "Local model interface and serving", "system_type": "Model infrastructure"},
    "oobabooga/textgen": {"category": "Role-play and local-model environments", "subgroup": "Local model interface and serving", "system_type": "Model infrastructure"},
    "ollama/ollama": {"category": "Role-play and local-model environments", "subgroup": "Local model interface and serving", "system_type": "Model infrastructure"},

    # 5. Agent orchestration and LLM application frameworks (5)
    "microsoft/autogen": {"category": "Agent orchestration and LLM application frameworks", "subgroup": "Orchestration framework", "system_type": "AI agents and orchestration"},
    "crewAIInc/crewAI": {"category": "Agent orchestration and LLM application frameworks", "subgroup": "Orchestration framework", "system_type": "AI agents and orchestration"},
    "langchain-ai/langgraph": {"category": "Agent orchestration and LLM application frameworks", "subgroup": "Orchestration framework", "system_type": "AI agents and orchestration"},
    "FlowiseAI/Flowise": {"category": "Agent orchestration and LLM application frameworks", "subgroup": "Orchestration framework", "system_type": "AI agents and orchestration"},
    "langflow-ai/langflow": {"category": "Agent orchestration and LLM application frameworks", "subgroup": "Orchestration framework", "system_type": "AI agents and orchestration"},

    # 6. Agent memory and model-access infrastructure (5)
    "letta-ai/letta": {"category": "Agent memory and model-access infrastructure", "subgroup": "Persistent memory and knowledge management", "system_type": "Knowledge and memory systems"},
    "mem0ai/mem0": {"category": "Agent memory and model-access infrastructure", "subgroup": "Persistent memory and knowledge management", "system_type": "Knowledge and memory systems"},
    "getzep/graphiti": {"category": "Agent memory and model-access infrastructure", "subgroup": "Persistent memory and knowledge management", "system_type": "Knowledge and memory systems"},
    "BerriAI/litellm": {"category": "Agent memory and model-access infrastructure", "subgroup": "Model access, routing and switching", "system_type": "Model infrastructure"},
    "mostlygeek/llama-swap": {"category": "Agent memory and model-access infrastructure", "subgroup": "Model access, routing and switching", "system_type": "Model infrastructure"},
}

# Search lexicon: three failure dimensions plus folk
# explanatory idioms, each in a core and an exploratory layer. Issues are
# searched as OR-bundles; Discussions one term per query (core layer).
FAILURE_TERMS = {
    "Factual errors and fabrication": {
        "core": [
            "hallucination", "hallucinations", "hallucinate", "hallucinates", "hallucinating",
            "hallucinated", "confabulation", "confabulates", "fabricated", "fabricates",
            "fabricating", "confidently wrong", "factually incorrect", "factually wrong",
            "makes up", "makes things up", "making things up", "made up"
        ],
        "exploratory": [
            "invents", "invented", "non-existent", "nonexistent", "wrong answer",
            "incorrect answer", "wrong information", "incorrect information", "false information",
            "misinformation", "lying", "lied", "is lying", "pretends", "pretending",
            "claims to have", "says it did", "didn't actually", "did not actually",
            "fake citation", "fake citations", "fake references", "fake source"
        ],
    },
    "Memory and context lapses": {
        "core": [
            "forgets", "forgetting", "it forgot", "loses context", "lost context",
            "losing context", "doesn't remember", "does not remember", "no memory", "memory loss",
            "loses track", "lost track", "context loss", "forgets everything", "forgets what"
        ],
        "exploratory": [
            "forgot", "amnesia", "goldfish", "short-term memory", "short term memory",
            "doesn't recall", "does not recall", "can't remember", "cannot remember",
            "remember previous", "remember the previous", "previous conversation",
            "earlier in the conversation", "earlier messages", "ignores previous",
            "repeats itself", "repeating itself", "stuck in a loop", "keeps repeating",
            "lost the plot", "memory doesn't work", "memory does not work",
            "memory is not working"
        ],
    },
    "Instruction-following failures": {
        "core": [
            "ignores instructions", "ignore instructions", "ignoring instructions",
            "ignored instructions", "ignores my instructions", "doesn't follow instructions",
            "does not follow instructions", "not following instructions", "follow my instructions",
            "ignores the system prompt", "ignores system prompt", "ignoring the system prompt",
            "disregards", "doesn't listen", "does not listen", "refuses to follow", "won't follow",
            "does not obey", "doesn't obey"
        ],
        "exploratory": [
            "didn't follow", "doesn't follow", "does not follow", "ignores my", "ignored my",
            "ignoring my", "not respecting", "doesn't respect", "does not respect",
            "doesn't adhere", "does not adhere", "does whatever it wants", "off the rails",
            "does the opposite", "despite being told", "even though I told", "I told it",
            "I asked it to", "keeps doing", "stopped following", "breaks character",
            "out of character", "ignores the rules", "ignoring the rules", "overrides my",
            "overrode my", "doesn't comply", "does not comply"
        ],
    },
    "Folk explanations and attributions": {
        "core": [
            "stochastic parrot", "next token prediction", "next-token prediction",
            "predicting the next token", "predict the next token", "doesn't actually understand",
            "does not actually understand", "doesn't really understand", "no real understanding",
            "just a language model", "just an LLM", "it's just predicting",
            "autocomplete on steroids", "the model is lazy", "being lazy", "is lazy", "too lazy",
            "laziness", "gaslighting", "gaslights", "lobotomized", "lobotomised", "dumbed down",
            "nerfed", "got dumber", "getting dumber", "became dumber"
        ],
        "exploratory": [
            "the model thinks", "the model believes", "the model is confused", "gets confused",
            "model is guessing", "just guessing", "it's guessing", "it is guessing",
            "the model assumes", "it assumes", "mind of its own", "the model wants", "it wants to",
            "the model decided", "it decided to", "the model doesn't know", "it doesn't know",
            "model is stupid", "so stupid", "too dumb", "not smart enough", "stupid model",
            "dumb model", "the model refuses", "it refuses", "stubborn", "inherent limitation",
            "fundamental limitation", "all LLMs", "LLMs are", "any LLM", "lost in the middle",
            "overthinking", "overthinks"
        ],
    },
}

# =============================================================================
# 1. Configuration
# =============================================================================

MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
CONTRIBUTOR_ASSOCIATIONS = {"CONTRIBUTOR"}

REST_API = "https://api.github.com"
GRAPHQL_API = "https://api.github.com/graphql"
EXCEL_CELL_LIMIT = 32767


@dataclass
class Config:
    """All run parameters. Search-relevant fields contribute to config_hash()."""
    data_dir: str = "./llm_failures_data_v3"
    run_label: str = "collect"
    # thread-creation window (inclusive, dates)
    start_date: date = date(2023, 9, 1)
    end_date: date = date(2026, 9, 30)
    # analytic message cutoff (UTC). Later messages are retained and flagged
    # outside_window = True; nothing is discarded.
    message_cutoff_utc: str = "2026-09-30T23:59:59Z"
    # issue search (REST)
    bundle_levels: Tuple[str, ...] = ("core", "exploratory")
    dimensions: Tuple[str, ...] = tuple(FAILURE_TERMS.keys())
    max_query_chars: int = 256
    max_or_operators: int = 5          # documented REST limit: five AND/OR/NOT operators
    per_page: int = 100
    max_pages: int = 10
    result_cap: int = 1000
    near_cap_threshold: int = 900
    search_sleep_seconds: float = 2.1  # 30 search requests/minute when authenticated
    # discussions (GraphQL)
    collect_discussions: bool = True
    discussion_levels: Tuple[str, ...] = ("core",)
    graphql_sleep_seconds: float = 1.0
    graphql_page_size: int = 100
    # transport
    request_timeout: float = 30.0
    max_retries: int = 5
    comment_sleep_seconds: float = 0.8
    checkpoint_every: int = 25
    # exports
    preview_chars: int = 2000
    write_thread_preview_xlsx: bool = True
    write_message_preview_xlsx: bool = False   # large; enable if wanted

    SEARCH_FIELDS = ("start_date", "end_date", "bundle_levels", "dimensions", "max_query_chars",
                     "max_or_operators", "per_page", "result_cap", "collect_discussions",
                     "discussion_levels")

    def cutoff_dt(self) -> datetime:
        return parse_iso(self.message_cutoff_utc)

    def config_hash(self) -> str:
        payload = {k: (str(getattr(self, k))) for k in self.SEARCH_FIELDS}
        payload["lexicon_hash"] = lexicon_hash()
        payload["repos_hash"] = hashlib.sha1(json.dumps(sorted(REPO_INFO.keys())).encode()).hexdigest()[:12]
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def describe(self) -> Dict[str, Any]:
        d = asdict(self)
        d["start_date"] = self.start_date.isoformat()
        d["end_date"] = self.end_date.isoformat()
        d["config_hash"] = self.config_hash()
        return d


def lexicon_hash() -> str:
    return hashlib.sha1(json.dumps(FAILURE_TERMS, sort_keys=True).encode()).hexdigest()[:12]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return now_utc().isoformat(timespec="seconds")


def parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts or not isinstance(ts, str):
        return None
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def author_role(association: Optional[str], account_type: Optional[str], login: Optional[str]) -> str:
    """Descriptive label from GitHub metadata (association with the repository, account type)."""
    if account_type == "Bot" or (isinstance(login, str) and login.endswith("[bot]")):
        return "bot"
    if association in MAINTAINER_ASSOCIATIONS:
        return "maintainer"
    if association in CONTRIBUTOR_ASSOCIATIONS:
        return "contributor"
    return "user"


# =============================================================================
# 2. Storage: lossless JSONL + completion manifest
# =============================================================================

class Store:
    """
    Append-only JSONL files are the authoritative raw record. A JSON manifest
    tracks per-unit completion so interrupted runs resume without duplicating
    or silently mixing incompatible configurations.
    """
    FILES = {
        "requests": "log_requests.jsonl",
        "search_log": "log_search_queries.jsonl",
        "repo_metadata": "raw_repo_metadata.jsonl",
        "issue_hits": "raw_issue_search_hits.jsonl",
        "issue_comments": "raw_issue_comments.jsonl",
        "discussion_hits": "raw_discussion_search_hits.jsonl",
        "discussions": "raw_discussions_full.jsonl",
        "bundle_validation": "log_bundle_validation.jsonl",
    }

    def __init__(self, config: Config, allow_config_change: bool = False):
        self.config = config
        self.dir = config.data_dir
        os.makedirs(self.dir, exist_ok=True)
        self.manifest_path = os.path.join(self.dir, "manifest.json")
        self.manifest = self._load_manifest(allow_config_change)

    # -- manifest -------------------------------------------------------------
    def _load_manifest(self, allow_config_change: bool) -> Dict[str, Any]:
        current_hash = self.config.config_hash()
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, encoding="utf-8") as f:
                m = json.load(f)
            if m.get("config_hash") != current_hash:
                if not allow_config_change:
                    raise RuntimeError(
                        f"Existing data in {self.dir} was collected with config_hash "
                        f"{m.get('config_hash')} but the current configuration hashes to "
                        f"{current_hash}. Use a new data_dir, or pass allow_config_change=True "
                        f"to record the change and continue (results will mix configurations).")
                m.setdefault("events", []).append({
                    "ts": now_iso(), "event": "config_changed",
                    "from": m.get("config_hash"), "to": current_hash})
                m["config_hash"] = current_hash
                m["config"] = self.config.describe()
            m.setdefault("events", []).append({"ts": now_iso(), "event": "session_resumed"})
            return m
        m = {
            "run_id": f"{self.config.run_label}-{now_utc().strftime('%Y%m%dT%H%M%SZ')}",
            "created_at": now_iso(),
            "config_hash": current_hash,
            "config": self.config.describe(),
            "pipeline_version": PIPELINE_VERSION,
            "events": [{"ts": now_iso(), "event": "created"}],
            "repos": {},            # canonical repo -> status
            "issue_search": {},     # "repo|bundle_id" -> {status, ...}
            "discussion_search": {},
            "issue_threads": {},    # thread_id -> {status, expected, retrieved, retrieved_at}
            "discussion_threads": {},
            "bundle_validation": {},
        }
        self._save_manifest(m)
        return m

    def _save_manifest(self, m: Optional[Dict[str, Any]] = None) -> None:
        m = m if m is not None else self.manifest
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=1, ensure_ascii=False, default=str)
        os.replace(tmp, self.manifest_path)

    def save(self) -> None:
        self._save_manifest()

    def mark(self, section: str, key: str, **fields) -> None:
        entry = self.manifest.setdefault(section, {}).get(key, {})
        entry.update(fields)
        entry["updated_at"] = now_iso()
        self.manifest[section][key] = entry

    def status(self, section: str, key: str) -> Optional[str]:
        return self.manifest.get(section, {}).get(key, {}).get("status")

    # -- jsonl ----------------------------------------------------------------
    def path(self, name: str) -> str:
        return os.path.join(self.dir, self.FILES[name])

    def append(self, name: str, record: Dict[str, Any]) -> None:
        with open(self.path(name), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def append_many(self, name: str, records: Iterable[Dict[str, Any]]) -> None:
        with open(self.path(name), "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    def read(self, name: str) -> List[Dict[str, Any]]:
        """
        Read a JSONL file tolerantly. Appending to a file on a mounted drive for
        hours can leave blank, NUL-padded or truncated lines; such lines are
        skipped, copied verbatim to <file>.badlines for inspection, counted in
        the manifest (events) and reported, instead of aborting the run.
        """
        p = self.path(name)
        if not os.path.exists(p):
            return []
        out = []
        bad = []
        with open(p, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.replace("\x00", "").strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    bad.append((lineno, line))
        if bad:
            with open(p + ".badlines", "a", encoding="utf-8") as g:
                for lineno, line in bad:
                    g.write(f"{lineno}\t{line}\n")
            self.manifest.setdefault("events", []).append(
                {"ts": now_iso(), "event": "unparseable_jsonl_lines", "file": self.FILES[name], "n": len(bad)})
            self._save_manifest()
            print(f"WARNING: {len(bad)} unparseable line(s) skipped in {self.FILES[name]} "
                  f"(saved to {self.FILES[name]}.badlines); {len(out)} records read.")
        return out

    def summary(self) -> Dict[str, Any]:
        def count_status(section):
            c = Counter(v.get("status") for v in self.manifest.get(section, {}).values())
            return dict(c)
        return {
            "run_id": self.manifest.get("run_id"),
            "config_hash": self.manifest.get("config_hash"),
            "repos": count_status("repos"),
            "issue_search": count_status("issue_search"),
            "discussion_search": count_status("discussion_search"),
            "issue_threads": count_status("issue_threads"),
            "discussion_threads": count_status("discussion_threads"),
            "raw_record_counts": {k: len(self.read(k)) for k in self.FILES},
        }


# =============================================================================
# 3. GitHub client: timeouts, retries, rate limits, full request logging
# =============================================================================

@dataclass
class Outcome:
    ok: bool
    status: Optional[int]
    data: Any = None
    error: Optional[str] = None
    attempts: int = 0
    headers: Dict[str, str] = field(default_factory=dict)
    retrieved_at: str = ""


class GitHubClient:
    def __init__(self, token: Optional[str], config: Config, store: Optional[Store] = None,
                 session: Optional[requests.Session] = None, sleep_fn: Callable[[float], None] = time.sleep):
        self.config = config
        self.store = store
        self.sleep = sleep_fn
        self.session = session or requests.Session()
        self.headers = {"Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.n_requests = 0

    def _log(self, kind: str, url: str, params: Any, outcome: Outcome, extra: Optional[Dict] = None) -> None:
        if self.store is None:
            return
        rec = {"ts": outcome.retrieved_at, "kind": kind, "url": url,
               "params": params if kind != "graphql" else None,
               "status": outcome.status, "ok": outcome.ok, "attempts": outcome.attempts,
               "error": (outcome.error or "")[:500],
               "ratelimit_remaining": outcome.headers.get("x-ratelimit-remaining"),
               "ratelimit_reset": outcome.headers.get("x-ratelimit-reset")}
        if extra:
            rec.update(extra)
        self.store.append("requests", rec)

    def _wait_for_rate_limit(self, resp: requests.Response, attempt: int) -> None:
        retry_after = resp.headers.get("retry-after")
        remaining = resp.headers.get("x-ratelimit-remaining")
        reset = resp.headers.get("x-ratelimit-reset")
        if retry_after:
            wait = max(int(float(retry_after)), 1) + 1
        elif remaining == "0" and reset:
            wait = max(int(reset) - int(time.time()) + 5, 10)
        else:
            wait = 60 * (attempt + 1)
        self.sleep(wait)

    def get(self, url: str, params: Optional[Dict[str, Any]] = None, kind: str = "rest") -> Outcome:
        """GET with timeout, bounded retries and rate-limit handling. Never raises."""
        last_err = None
        status = None
        headers: Dict[str, str] = {}
        for attempt in range(1, self.config.max_retries + 1):
            self.n_requests += 1
            try:
                resp = self.session.get(url, headers=self.headers, params=params,
                                        timeout=self.config.request_timeout)
            except requests.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
                self.sleep(min(2 ** attempt, 60))
                continue
            status = resp.status_code
            headers = {k.lower(): v for k, v in resp.headers.items()}
            if status == 200:
                try:
                    data = resp.json()
                except ValueError as e:
                    last_err = f"JSONDecodeError: {e}"
                    continue
                out = Outcome(True, status, data, None, attempt, headers, now_iso())
                self._log(kind, url, params, out)
                return out
            if status in (403, 429):
                last_err = f"HTTP {status}: {resp.text[:200]}"
                self._wait_for_rate_limit(resp, attempt)
                continue
            if status >= 500:
                last_err = f"HTTP {status}: {resp.text[:200]}"
                self.sleep(min(2 ** attempt, 60))
                continue
            # 4xx other than rate limits: not retryable (404, 410, 422 validation failed)
            out = Outcome(False, status, None, f"HTTP {status}: {resp.text[:300]}", attempt, headers, now_iso())
            self._log(kind, url, params, out)
            return out
        out = Outcome(False, status, None, last_err or "exhausted retries",
                      self.config.max_retries, headers, now_iso())
        self._log(kind, url, params, out)
        return out

    def graphql(self, query: str, variables: Dict[str, Any]) -> Outcome:
        """POST a GraphQL query. GraphQL errors arrive with HTTP 200 and are surfaced as failures."""
        last_err = None
        status = None
        headers: Dict[str, str] = {}
        for attempt in range(1, self.config.max_retries + 1):
            self.n_requests += 1
            try:
                resp = self.session.post(GRAPHQL_API, headers=self.headers,
                                         json={"query": query, "variables": variables},
                                         timeout=self.config.request_timeout)
            except requests.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
                self.sleep(min(2 ** attempt, 60))
                continue
            status = resp.status_code
            headers = {k.lower(): v for k, v in resp.headers.items()}
            if status in (403, 429):
                last_err = f"HTTP {status}: {resp.text[:200]}"
                self._wait_for_rate_limit(resp, attempt)
                continue
            if status >= 500:
                last_err = f"HTTP {status}: {resp.text[:200]}"
                self.sleep(min(2 ** attempt, 60))
                continue
            if status != 200:
                out = Outcome(False, status, None, f"HTTP {status}: {resp.text[:300]}", attempt, headers, now_iso())
                self._log("graphql", GRAPHQL_API, None, out, {"variables": variables})
                return out
            try:
                payload = resp.json()
            except ValueError as e:
                last_err = f"JSONDecodeError: {e}"
                continue
            if payload.get("errors"):
                types = {e.get("type") for e in payload["errors"]}
                msg = "; ".join(e.get("message", "") for e in payload["errors"])[:300]
                if "RATE_LIMITED" in types:
                    last_err = msg
                    self._wait_for_rate_limit(resp, attempt)
                    continue
                # Partial data with NOT_FOUND etc. is returned as a failure with the data attached
                out = Outcome(False, status, payload.get("data"), f"GraphQL errors: {msg}", attempt, headers, now_iso())
                self._log("graphql", GRAPHQL_API, None, out, {"variables": variables})
                return out
            rl = (payload.get("data") or {}).get("rateLimit") or {}
            out = Outcome(True, status, payload.get("data"), None, attempt, headers, now_iso())
            self._log("graphql", GRAPHQL_API, None, out, {"variables": variables, "cost": rl.get("cost"),
                                                           "gql_remaining": rl.get("remaining")})
            if rl.get("remaining") is not None and rl["remaining"] < 50:
                reset = parse_iso(rl.get("resetAt"))
                if reset:
                    self.sleep(max((reset - now_utc()).total_seconds() + 5, 10))
            return out
        out = Outcome(False, status, None, last_err or "exhausted retries", self.config.max_retries, headers, now_iso())
        self._log("graphql", GRAPHQL_API, None, out, {"variables": variables})
        return out


# =============================================================================
# 4. Repository resolution
# =============================================================================

def resolve_repositories(client: GitHubClient, store: Store, repo_info: Dict[str, Dict[str, str]] = None,
                         sleep_seconds: float = 0.5) -> pd.DataFrame:
    """
    Resolve each supplied path to its canonical location and record stable IDs,
    redirects, availability and whether Issues and Discussions are enabled.
    Unavailable repositories are kept in the metadata with their HTTP status.
    """
    repo_info = repo_info or REPO_INFO
    rows = []
    for supplied, info in repo_info.items():
        out = client.get(f"{REST_API}/repos/{supplied}")
        rec = {"supplied_path": supplied, **info, "retrieved_at": out.retrieved_at,
               "http_status": out.status, "resolved": out.ok}
        if out.ok:
            m = out.data
            canonical = m.get("full_name", supplied)
            rec.update({
                "canonical_path": canonical,
                "redirected": canonical.lower() != supplied.lower(),
                "repo_id": m.get("id"), "node_id": m.get("node_id"),
                "owner": (m.get("owner") or {}).get("login"), "name": m.get("name"),
                "description": m.get("description"), "repo_created_at": m.get("created_at"),
                "pushed_at": m.get("pushed_at"), "stargazers_count": m.get("stargazers_count"),
                "forks_count": m.get("forks_count"), "open_issues_count": m.get("open_issues_count"),
                "has_issues": m.get("has_issues"), "has_discussions": m.get("has_discussions"),
                "archived": m.get("archived"), "disabled": m.get("disabled"),
                "default_branch": m.get("default_branch"), "homepage": m.get("homepage"),
                "error": None})
            store.mark("repos", canonical, status="resolved", supplied_path=supplied,
                       has_issues=m.get("has_issues"), has_discussions=m.get("has_discussions"))
        else:
            rec.update({"canonical_path": None, "redirected": None, "repo_id": None, "node_id": None,
                        "has_issues": None, "has_discussions": None, "error": out.error})
            store.mark("repos", supplied, status="unavailable", supplied_path=supplied, error=out.error)
        rows.append(rec)
        store.append("repo_metadata", rec)
        client.sleep(sleep_seconds)
    store.save()
    df = pd.DataFrame(rows)
    return df


def resolved_repos(repo_df: pd.DataFrame, require_issues: bool = True) -> Dict[str, Dict[str, Any]]:
    """Canonical path -> info for repositories that resolved (and have issues enabled if required)."""
    out = {}
    for _, r in repo_df.iterrows():
        if not r.get("resolved"):
            continue
        if require_issues and r.get("has_issues") is False:
            continue
        out[r["canonical_path"]] = {
            "category": r["category"], "subgroup": r["subgroup"], "system_type": r["system_type"],
            "supplied_path": r["supplied_path"], "repo_id": r.get("repo_id"),
            "has_discussions": bool(r.get("has_discussions")),
        }
    return out


# =============================================================================
# 5. Query construction (documented limits: 256 chars, five boolean operators)
# =============================================================================

def or_clause(terms: Sequence[str]) -> str:
    return "(" + " OR ".join(f'"{t}"' for t in terms) + ")" if len(terms) > 1 else f'"{terms[0]}"'


def query_base(repo: str, window_start: date, window_end: date) -> str:
    return (f"repo:{repo} is:issue in:title,body,comments "
            f"created:{window_start.isoformat()}..{window_end.isoformat()} ")


def pack_terms(repo: str, terms: Sequence[str], window_start: date, window_end: date,
               max_chars: int, max_or_operators: int) -> List[List[str]]:
    """
    Greedily pack terms into OR-bundles that respect BOTH documented limits:
    the whole query string stays within max_chars (counted conservatively,
    including qualifiers and operators), and no bundle uses more than
    max_or_operators OR operators (i.e. max_or_operators + 1 terms).
    """
    base = query_base(repo, window_start, window_end)
    chunks: List[List[str]] = []
    current: List[str] = []
    for term in terms:
        candidate = current + [term]
        too_long = len(base + or_clause(candidate)) > max_chars
        too_many = len(candidate) - 1 > max_or_operators
        if (too_long or too_many) and current:
            chunks.append(current)
            current = [term]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def iter_bundles(config: Config, repo: str) -> Iterator[Tuple[str, str, str, List[str]]]:
    for dimension in config.dimensions:
        for level in config.bundle_levels:
            terms = FAILURE_TERMS[dimension].get(level, [])
            if not terms:
                continue
            chunks = pack_terms(repo, terms, config.start_date, config.end_date,
                                config.max_query_chars, config.max_or_operators)
            for i, chunk in enumerate(chunks):
                yield dimension, level, f"{dimension} | {level} | {i + 1}", chunk


def count_base_queries(config: Config, repos: Iterable[str]) -> int:
    return sum(1 for r in repos for _ in iter_bundles(config, r))


# =============================================================================
# 6. Issue search (REST) with window splitting, caps and checkpointing
# =============================================================================

def issue_hit_record(item: Dict[str, Any], repo: str, info: Dict[str, Any], dimension: str, level: str,
                     bundle_id: str, bundle_terms: Sequence[str], window: Tuple[date, date],
                     retrieved_at: str) -> Dict[str, Any]:
    user = item.get("user") or {}
    return {
        "source_type": "issue",
        "thread_id": f"I:{repo}#{item.get('number')}",
        "repo": repo, "repo_id": info.get("repo_id"), "repo_category": info.get("category"),
        "repo_subgroup": info.get("subgroup"), "system_type": info.get("system_type"),
        "number": item.get("number"), "github_id": item.get("id"), "node_id": item.get("node_id"),
        "title": item.get("title"), "body": item.get("body"),
        "author_login": user.get("login"), "author_id": user.get("id"), "author_type": user.get("type"),
        "author_association": item.get("author_association"),
        "state": item.get("state"), "state_reason": item.get("state_reason"),
        "created_at": item.get("created_at"), "updated_at": item.get("updated_at"),
        "closed_at": item.get("closed_at"), "comments_count": item.get("comments"),
        "reactions_total": (item.get("reactions") or {}).get("total_count"),
        "labels": [l.get("name") for l in item.get("labels", [])],
        "html_url": item.get("html_url"), "comments_url": item.get("comments_url"),
        "locked": item.get("locked"),
        "bundle_dimension": dimension, "bundle_level": level, "bundle_id": bundle_id,
        "bundle_terms": list(bundle_terms),
        "window_start": window[0].isoformat(), "window_end": window[1].isoformat(),
        "retrieved_at": retrieved_at,
    }


def _search_page(client: GitHubClient, query: str, page: int, order: str, config: Config) -> Outcome:
    params = {"q": query, "advanced_search": "true", "sort": "created", "order": order,
              "per_page": config.per_page, "page": page}
    out = client.get(f"{REST_API}/search/issues", params=params, kind="search")
    client.sleep(config.search_sleep_seconds)
    return out


def _collect_query(client: GitHubClient, store: Store, config: Config, query: str, order: str,
                   log_base: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Retrieve every page of one query (up to the cap). Returns (items, status dict)."""
    items: List[Dict[str, Any]] = []
    first = _search_page(client, query, 1, order, config)
    log = {**log_base, "query": query, "order": order, "page": 1, "status": first.status,
           "ok": first.ok, "retrieved_at": first.retrieved_at, "error": first.error}
    if not first.ok or not isinstance(first.data, dict) or "items" not in first.data:
        log.update({"total_count": None, "incomplete_results": None, "retrieved": 0,
                    "action": "request_failed", "complete": False})
        store.append("search_log", log)
        return items, {"complete": False, "total_count": None, "incomplete": True, "failed_pages": [1]}
    total = int(first.data.get("total_count", 0))
    incomplete = bool(first.data.get("incomplete_results", False))
    items.extend(first.data["items"])
    pages_needed = min(math.ceil(total / config.per_page), config.max_pages)
    failed_pages: List[int] = []
    for page in range(2, pages_needed + 1):
        out = _search_page(client, query, page, order, config)
        page_log = {**log_base, "query": query, "order": order, "page": page, "status": out.status,
                    "ok": out.ok, "retrieved_at": out.retrieved_at, "error": out.error,
                    "total_count": total, "action": "page"}
        if not out.ok or not isinstance(out.data, dict) or "items" not in out.data:
            failed_pages.append(page)
            page_log.update({"retrieved": 0, "complete": False})
            store.append("search_log", page_log)
            continue
        incomplete = incomplete or bool(out.data.get("incomplete_results", False))
        items.extend(out.data["items"])
        page_log.update({"retrieved": len(out.data["items"]), "complete": True})
        store.append("search_log", page_log)
    expected = min(total, config.result_cap)
    discrepancy = len(items) != expected
    log.update({"total_count": total, "incomplete_results": incomplete, "retrieved": len(items),
                "expected_retrievable": expected, "count_discrepancy": discrepancy,
                "failed_pages": failed_pages, "action": "collected",
                "complete": not failed_pages and not incomplete and not discrepancy})
    store.append("search_log", log)
    return items, {"complete": log["complete"], "total_count": total, "incomplete": incomplete,
                   "failed_pages": failed_pages}


def search_bundle_window(client: GitHubClient, store: Store, config: Config, repo: str, info: Dict[str, Any],
                         dimension: str, level: str, bundle_id: str, bundle_terms: Sequence[str],
                         window_start: date, window_end: date) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Exhaustive retrieval for one bundle within one creation-date window.
      * total_count > cap and window > 1 day  -> split the window in half (recursively)
      * total_count > cap and window == 1 day -> retrieve ascending AND descending
        (up to 2 x cap distinct results); if total_count > 2 x cap the remainder is
        unreachable through search and the window is logged as cap_exceeded_single_day
    Returns (hit records, complete flag). 'complete' is False if any page failed,
    incomplete_results was set, a count discrepancy occurred, or a cap was exceeded.
    """
    query = query_base(repo, window_start, window_end) + or_clause(bundle_terms)
    log_base = {"repo": repo, "dimension": dimension, "bundle_level": level, "bundle_id": bundle_id,
                "n_terms": len(bundle_terms), "n_or_operators": len(bundle_terms) - 1,
                "query_chars": len(query), "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat()}
    probe = _search_page(client, query, 1, "asc", config)
    store.append("search_log", {**log_base, "query": query, "order": "asc", "page": 1, "status": probe.status,
                                "ok": probe.ok, "retrieved_at": probe.retrieved_at, "error": probe.error,
                                "total_count": (probe.data or {}).get("total_count") if probe.ok else None,
                                "incomplete_results": (probe.data or {}).get("incomplete_results") if probe.ok else None,
                                "action": "probe"})
    if not probe.ok or not isinstance(probe.data, dict) or "items" not in probe.data:
        store.append("search_log", {**log_base, "query": query, "status": probe.status, "ok": False,
                                    "retrieved_at": probe.retrieved_at, "error": probe.error,
                                    "action": "request_failed", "complete": False})
        return [], False
    total = int(probe.data.get("total_count", 0))

    if total > config.result_cap and window_start < window_end:
        mid = window_start + (window_end - window_start) / 2
        store.append("search_log", {**log_base, "query": query, "total_count": total, "action": "split",
                                    "retrieved_at": probe.retrieved_at})
        left, ok_l = search_bundle_window(client, store, config, repo, info, dimension, level, bundle_id,
                                          bundle_terms, window_start, mid)
        right, ok_r = search_bundle_window(client, store, config, repo, info, dimension, level, bundle_id,
                                           bundle_terms, mid + timedelta(days=1), window_end)
        return left + right, ok_l and ok_r

    window = (window_start, window_end)
    if total > config.result_cap:
        # single day over the cap: ascending + descending passes
        items_asc, st_asc = _collect_query(client, store, config, query, "asc", log_base)
        items_desc, st_desc = _collect_query(client, store, config, query, "desc", log_base)
        seen = set()
        merged = []
        for it in items_asc + items_desc:
            if it.get("number") not in seen:
                seen.add(it.get("number"))
                merged.append(it)
        exceeded = total > 2 * config.result_cap
        store.append("search_log", {**log_base, "query": query, "total_count": total,
                                    "retrieved": len(merged), "action": "cap_exceeded_single_day",
                                    "unreachable_estimate": max(total - len(merged), 0),
                                    "truncated": exceeded, "retrieved_at": now_iso()})
        recs = [issue_hit_record(it, repo, info, dimension, level, bundle_id, bundle_terms, window, now_iso())
                for it in merged]
        return recs, (st_asc["complete"] and st_desc["complete"] and not exceeded)

    items, st = _collect_query(client, store, config, query, "asc", log_base)
    near_cap = total >= config.near_cap_threshold
    if near_cap:
        store.append("search_log", {**log_base, "query": query, "total_count": total, "action": "near_cap",
                                    "retrieved_at": now_iso()})
    recs = [issue_hit_record(it, repo, info, dimension, level, bundle_id, bundle_terms, window, now_iso())
            for it in items]
    return recs, st["complete"]


def search_issues(client: GitHubClient, store: Store, config: Config, repos: Dict[str, Dict[str, Any]],
                  progress: Callable[[Iterable], Iterable] = lambda x: x) -> Dict[str, int]:
    """
    Run every bundle for every repository with per-bundle checkpointing. Bundles
    already marked complete in the manifest are skipped; bundles marked partial
    are re-run in full (their earlier hits are superseded at deduplication by
    retrieval timestamp, so nothing is duplicated in the assembled corpus).
    """
    counts = Counter()
    for repo in progress(list(repos.keys())):
        info = repos[repo]
        for dimension, level, bundle_id, terms in iter_bundles(config, repo):
            key = f"{repo}|{bundle_id}"
            if store.status("issue_search", key) == "complete":
                counts["skipped_complete"] += 1
                continue
            store.mark("issue_search", key, status="running", started_at=now_iso(), n_terms=len(terms))
            recs, complete = search_bundle_window(client, store, config, repo, info, dimension, level,
                                                  bundle_id, terms, config.start_date, config.end_date)
            run_tag = now_iso()
            for r in recs:
                r["search_run_at"] = run_tag
            store.append_many("issue_hits", recs)
            store.mark("issue_search", key, status="complete" if complete else "partial",
                       n_hits=len(recs), finished_at=run_tag)
            counts["complete" if complete else "partial"] += 1
            counts["hits"] += len(recs)
            store.save()
    store.save()
    return dict(counts)


def dry_run_counts(client: GitHubClient, store: Store, config: Config, repos: Dict[str, Dict[str, Any]],
                   progress: Callable[[Iterable], Iterable] = lambda x: x) -> pd.DataFrame:
    """Count-only pass (per_page=1) over every repo-bundle pair: exact pre-deduplication
    hit totals, HTTP outcomes and incomplete_results flags, before committing to retrieval."""
    rows = []
    for repo in progress(list(repos.keys())):
        for dimension, level, bundle_id, terms in iter_bundles(config, repo):
            q = query_base(repo, config.start_date, config.end_date) + or_clause(terms)
            out = client.get(f"{REST_API}/search/issues", params={"q": q, "advanced_search": "true", "per_page": 1},
                             kind="search")
            client.sleep(config.search_sleep_seconds)
            rec = {"repo": repo, "system_type": repos[repo].get("system_type"), "dimension": dimension,
                   "bundle_level": level, "bundle_id": bundle_id, "bundle_terms": list(terms), "query": q,
                   "query_chars": len(q), "n_or_operators": len(terms) - 1, "status": out.status, "ok": out.ok,
                   "error": out.error, "retrieved_at": out.retrieved_at, "action": "dry_run",
                   "total_count": out.data.get("total_count") if out.ok else None,
                   "incomplete_results": out.data.get("incomplete_results") if out.ok else None}
            rows.append(rec)
            store.append("search_log", rec)
    store.save()
    return pd.DataFrame(rows)


def validate_bundles(client: GitHubClient, store: Store, config: Config, repo: str,
                     max_bundles: Optional[int] = None) -> pd.DataFrame:
    """
    Empirical check that OR-bundles behave as unions: for each bundle in one
    repository, compare total_count(bundle) with the per-term counts. A valid
    bundle satisfies max(term counts) <= bundle count <= sum(term counts). HTTP
    422 ('Validation failed') indicates the query violates a documented limit.
    Results are logged; failing bundles should be split before the full harvest.
    """
    rows = []
    bundles = list(iter_bundles(config, repo))
    if max_bundles:
        bundles = bundles[:max_bundles]
    for dimension, level, bundle_id, terms in bundles:
        q = query_base(repo, config.start_date, config.end_date) + or_clause(terms)
        out = client.get(f"{REST_API}/search/issues", params={"q": q, "advanced_search": "true", "per_page": 1},
                         kind="search")
        client.sleep(config.search_sleep_seconds)
        bundle_total = out.data.get("total_count") if out.ok else None
        term_counts = {}
        for t in terms:
            tq = query_base(repo, config.start_date, config.end_date) + f'"{t}"'
            o = client.get(f"{REST_API}/search/issues", params={"q": tq, "advanced_search": "true", "per_page": 1},
                           kind="search")
            client.sleep(config.search_sleep_seconds)
            term_counts[t] = o.data.get("total_count") if o.ok else None
        known = [v for v in term_counts.values() if v is not None]
        if bundle_total is None or len(known) != len(terms):
            verdict = "request_failed"
        elif max(known) <= bundle_total <= sum(known):
            verdict = "consistent"
        else:
            verdict = "inconsistent"
        rec = {"repo": repo, "bundle_id": bundle_id, "n_terms": len(terms), "query_chars": len(q),
               "http_status": out.status, "error": out.error, "bundle_total": bundle_total,
               "term_counts": term_counts, "verdict": verdict, "checked_at": out.retrieved_at}
        rows.append(rec)
        store.append("bundle_validation", rec)
        store.mark("bundle_validation", f"{repo}|{bundle_id}", status=verdict)
    store.save()
    return pd.DataFrame(rows)


# =============================================================================
# 7. Discussions (GraphQL): per-term search, full retrieval with nested replies
# =============================================================================

GQL_SEARCH_DISCUSSIONS = """
query($q: String!, $first: Int!, $after: String) {
  rateLimit { cost remaining resetAt }
  search(query: $q, type: DISCUSSION, first: $first, after: $after) {
    discussionCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on Discussion {
        id databaseId number title createdAt updatedAt url isAnswered locked
        author { login __typename ... on User { databaseId } }
        authorAssociation
        category { name slug isAnswerable }
        comments { totalCount }
        repository { nameWithOwner databaseId }
      }
    }
  }
}
"""

GQL_DISCUSSION_FULL = """
query($owner: String!, $name: String!, $number: Int!, $first: Int!, $after: String, $replyFirst: Int!) {
  rateLimit { cost remaining resetAt }
  repository(owner: $owner, name: $name) {
    databaseId nameWithOwner
    discussion(number: $number) {
      id databaseId number title body bodyText createdAt updatedAt lastEditedAt url isAnswered locked
      upvoteCount
      author { login __typename ... on User { databaseId } }
      authorAssociation
      category { name slug isAnswerable }
      labels(first: 50) { nodes { name } }
      answer { id }
      answerChosenAt
      comments(first: $first, after: $after) {
        totalCount
        pageInfo { hasNextPage endCursor }
        nodes {
          id databaseId body createdAt updatedAt lastEditedAt url isAnswer upvoteCount
          author { login __typename ... on User { databaseId } }
          authorAssociation
          replyTo { id }
          replies(first: $replyFirst) {
            totalCount
            pageInfo { hasNextPage endCursor }
            nodes {
              id databaseId body createdAt updatedAt lastEditedAt url isAnswer upvoteCount
              author { login __typename ... on User { databaseId } }
              authorAssociation
              replyTo { id }
            }
          }
        }
      }
    }
  }
}
"""

GQL_REPLIES_PAGE = """
query($id: ID!, $first: Int!, $after: String) {
  rateLimit { cost remaining resetAt }
  node(id: $id) {
    ... on DiscussionComment {
      id
      replies(first: $first, after: $after) {
        totalCount
        pageInfo { hasNextPage endCursor }
        nodes {
          id databaseId body createdAt updatedAt lastEditedAt url isAnswer upvoteCount
          author { login __typename ... on User { databaseId } }
          authorAssociation
          replyTo { id }
        }
      }
    }
  }
}
"""


def discussion_query(repo: str, term: str, window_start: date, window_end: date) -> str:
    # Discussion search matches title, body and comments by default; boolean OR is
    # not documented for discussions, so one term per query.
    return f'repo:{repo} created:{window_start.isoformat()}..{window_end.isoformat()} "{term}"'


def _gql_author(a: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    a = a or {}
    return {"author_login": a.get("login"), "author_id": a.get("databaseId"),
            "author_type": a.get("__typename")}


def search_discussions_window(client: GitHubClient, store: Store, config: Config, repo: str, info: Dict[str, Any],
                              dimension: str, level: str, term: str, window_start: date, window_end: date
                              ) -> Tuple[List[Dict[str, Any]], bool]:
    q = discussion_query(repo, term, window_start, window_end)
    log_base = {"repo": repo, "dimension": dimension, "bundle_level": level, "bundle_id": f"{dimension} | {level} | {term}",
                "n_terms": 1, "query_chars": len(q), "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(), "source_type": "discussion"}
    recs: List[Dict[str, Any]] = []
    after = None
    total = None
    complete = True
    page = 0
    while True:
        page += 1
        out = client.graphql(GQL_SEARCH_DISCUSSIONS, {"q": q, "first": config.graphql_page_size, "after": after})
        client.sleep(config.graphql_sleep_seconds)
        log = {**log_base, "query": q, "page": page, "status": out.status, "ok": out.ok,
               "retrieved_at": out.retrieved_at, "error": out.error}
        if not out.ok or not (out.data or {}).get("search"):
            log.update({"action": "request_failed", "complete": False})
            store.append("search_log", log)
            return recs, False
        s = out.data["search"]
        total = int(s.get("discussionCount", 0))
        if page == 1 and total > config.result_cap and window_start < window_end:
            mid = window_start + (window_end - window_start) / 2
            log.update({"total_count": total, "action": "split"})
            store.append("search_log", log)
            left, ok_l = search_discussions_window(client, store, config, repo, info, dimension, level, term,
                                                   window_start, mid)
            right, ok_r = search_discussions_window(client, store, config, repo, info, dimension, level, term,
                                                    mid + timedelta(days=1), window_end)
            return left + right, ok_l and ok_r
        for n in s.get("nodes", []):
            if not n:
                continue
            recs.append({
                "source_type": "discussion", "thread_id": f"D:{repo}#{n.get('number')}",
                "repo": repo, "repo_id": info.get("repo_id"), "repo_category": info.get("category"),
                "repo_subgroup": info.get("subgroup"), "system_type": info.get("system_type"),
                "number": n.get("number"), "github_id": n.get("databaseId"), "node_id": n.get("id"),
                "title": n.get("title"), "created_at": n.get("createdAt"), "updated_at": n.get("updatedAt"),
                "html_url": n.get("url"), "is_answered": n.get("isAnswered"), "locked": n.get("locked"),
                **_gql_author(n.get("author")), "author_association": n.get("authorAssociation"),
                "discussion_category": (n.get("category") or {}).get("name"),
                "comments_count": (n.get("comments") or {}).get("totalCount"),
                "bundle_dimension": dimension, "bundle_level": level, "bundle_id": log_base["bundle_id"],
                "bundle_terms": [term], "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(), "retrieved_at": out.retrieved_at,
            })
        log.update({"total_count": total, "retrieved": len(s.get("nodes", [])), "action": "collected" if not s["pageInfo"]["hasNextPage"] else "page"})
        store.append("search_log", log)
        if not s["pageInfo"]["hasNextPage"] or page >= config.max_pages:
            break
        after = s["pageInfo"]["endCursor"]
    if total is not None and total > config.result_cap:
        store.append("search_log", {**log_base, "query": q, "total_count": total, "retrieved": len(recs),
                                    "action": "cap_exceeded_single_day", "truncated": True, "retrieved_at": now_iso()})
        complete = False
    return recs, complete


def search_discussions(client: GitHubClient, store: Store, config: Config, repos: Dict[str, Dict[str, Any]],
                       progress: Callable[[Iterable], Iterable] = lambda x: x) -> Dict[str, int]:
    counts = Counter()
    for repo in progress([r for r, i in repos.items() if i.get("has_discussions")]):
        info = repos[repo]
        for dimension in config.dimensions:
            for level in config.discussion_levels:
                for term in FAILURE_TERMS[dimension].get(level, []):
                    key = f"{repo}|{dimension} | {level} | {term}"
                    if store.status("discussion_search", key) == "complete":
                        counts["skipped_complete"] += 1
                        continue
                    store.mark("discussion_search", key, status="running", started_at=now_iso())
                    recs, complete = search_discussions_window(client, store, config, repo, info, dimension, level,
                                                               term, config.start_date, config.end_date)
                    tag = now_iso()
                    for r in recs:
                        r["search_run_at"] = tag
                    store.append_many("discussion_hits", recs)
                    store.mark("discussion_search", key, status="complete" if complete else "partial",
                               n_hits=len(recs), finished_at=tag)
                    counts["complete" if complete else "partial"] += 1
                    counts["hits"] += len(recs)
        store.save()
    store.save()
    return dict(counts)


def _flatten_reply_nodes(nodes: List[Dict[str, Any]], parent_id: str) -> List[Dict[str, Any]]:
    out = []
    for r in nodes or []:
        if not r:
            continue
        out.append({"message_id": r.get("id"), "github_id": r.get("databaseId"), "parent_message_id": parent_id,
                    "depth": 2, "body": r.get("body"), "created_at": r.get("createdAt"),
                    "updated_at": r.get("updatedAt"), "last_edited_at": r.get("lastEditedAt"), "url": r.get("url"),
                    "is_answer": r.get("isAnswer"), "upvotes": r.get("upvoteCount"),
                    **_gql_author(r.get("author")), "author_association": r.get("authorAssociation")})
    return out


def fetch_discussion_full(client: GitHubClient, store: Store, config: Config, repo: str, number: int
                          ) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Fetch one discussion with all comments and all nested replies. Returns (record, complete)."""
    owner, name = repo.split("/", 1)
    after = None
    rec: Optional[Dict[str, Any]] = None
    comments: List[Dict[str, Any]] = []
    complete = True
    expected_comments = None
    while True:
        out = client.graphql(GQL_DISCUSSION_FULL, {"owner": owner, "name": name, "number": int(number),
                                                   "first": config.graphql_page_size, "after": after,
                                                   "replyFirst": config.graphql_page_size})
        client.sleep(config.graphql_sleep_seconds)
        d = ((out.data or {}).get("repository") or {}).get("discussion") if out.data else None
        if not out.ok or not d:
            return rec, False
        if rec is None:
            rec = {"source_type": "discussion", "thread_id": f"D:{repo}#{number}", "repo": repo,
                   "repo_id": ((out.data or {}).get("repository") or {}).get("databaseId"),
                   "number": d.get("number"), "github_id": d.get("databaseId"), "node_id": d.get("id"),
                   "title": d.get("title"), "body": d.get("body"), "created_at": d.get("createdAt"),
                   "updated_at": d.get("updatedAt"), "last_edited_at": d.get("lastEditedAt"), "html_url": d.get("url"),
                   "is_answered": d.get("isAnswered"), "answer_message_id": (d.get("answer") or {}).get("id"),
                   "answer_chosen_at": d.get("answerChosenAt"), "locked": d.get("locked"),
                   "upvotes": d.get("upvoteCount"), **_gql_author(d.get("author")),
                   "author_association": d.get("authorAssociation"),
                   "discussion_category": (d.get("category") or {}).get("name"),
                   "labels": [l.get("name") for l in ((d.get("labels") or {}).get("nodes") or []) if l],
                   "retrieved_at": out.retrieved_at}
        conn = d.get("comments") or {}
        expected_comments = conn.get("totalCount")
        for c in conn.get("nodes") or []:
            if not c:
                continue
            comments.append({"message_id": c.get("id"), "github_id": c.get("databaseId"), "parent_message_id": None,
                             "depth": 1, "body": c.get("body"), "created_at": c.get("createdAt"),
                             "updated_at": c.get("updatedAt"), "last_edited_at": c.get("lastEditedAt"),
                             "url": c.get("url"), "is_answer": c.get("isAnswer"), "upvotes": c.get("upvoteCount"),
                             **_gql_author(c.get("author")), "author_association": c.get("authorAssociation"),
                             "replies_total": (c.get("replies") or {}).get("totalCount")})
            replies = c.get("replies") or {}
            comments.extend(_flatten_reply_nodes(replies.get("nodes"), c.get("id")))
            r_after = (replies.get("pageInfo") or {}).get("endCursor")
            while (replies.get("pageInfo") or {}).get("hasNextPage"):
                ro = client.graphql(GQL_REPLIES_PAGE, {"id": c.get("id"), "first": config.graphql_page_size,
                                                       "after": r_after})
                client.sleep(config.graphql_sleep_seconds)
                node = (ro.data or {}).get("node") if ro.ok else None
                if not node:
                    complete = False
                    break
                replies = node.get("replies") or {}
                comments.extend(_flatten_reply_nodes(replies.get("nodes"), c.get("id")))
                r_after = (replies.get("pageInfo") or {}).get("endCursor")
        if not (conn.get("pageInfo") or {}).get("hasNextPage"):
            break
        after = conn["pageInfo"]["endCursor"]
    top_level = sum(1 for c in comments if c["depth"] == 1)
    if expected_comments is not None and top_level != expected_comments:
        complete = False
    rec["comments"] = comments
    rec["expected_top_level_comments"] = expected_comments
    rec["retrieved_top_level_comments"] = top_level
    rec["retrieved_total_messages"] = len(comments)
    rec["complete"] = complete
    rec["retrieved_at"] = now_iso()
    return rec, complete


def fetch_discussions(client: GitHubClient, store: Store, config: Config,
                      progress: Callable[[Iterable], Iterable] = lambda x: x) -> Dict[str, int]:
    hits = store.read("discussion_hits")
    unique = {}
    for h in hits:
        unique.setdefault(h["thread_id"], h)
    counts = Counter()
    for tid in progress(sorted(unique)):
        if store.status("discussion_threads", tid) == "complete":
            counts["skipped_complete"] += 1
            continue
        h = unique[tid]
        rec, complete = fetch_discussion_full(client, store, config, h["repo"], h["number"])
        if rec is None:
            store.mark("discussion_threads", tid, status="failed")
            counts["failed"] += 1
            continue
        rec.update({"repo_category": h.get("repo_category"), "repo_subgroup": h.get("repo_subgroup"),
                    "system_type": h.get("system_type")})
        store.append("discussions", rec)
        store.mark("discussion_threads", tid, status="complete" if complete else "partial",
                   expected=rec["expected_top_level_comments"], retrieved=rec["retrieved_top_level_comments"],
                   retrieved_at=rec["retrieved_at"])
        counts["complete" if complete else "partial"] += 1
        if sum(counts.values()) % config.checkpoint_every == 0:
            store.save()
    store.save()
    return dict(counts)


# =============================================================================
# 8. Issue comments with completion manifest
# =============================================================================

def unique_issue_hits(store: Store) -> List[Dict[str, Any]]:
    """Deduplicate raw issue hits on thread_id, keeping the most recently retrieved record
    and aggregating bundle membership across all hits."""
    hits = store.read("issue_hits")
    by_id: Dict[str, Dict[str, Any]] = {}
    membership: Dict[str, Dict[str, set]] = defaultdict(lambda: {"dims": set(), "levels": set(), "bundles": set(), "terms": set()})
    for h in hits:
        tid = h["thread_id"]
        m = membership[tid]
        m["dims"].add(h["bundle_dimension"]); m["levels"].add(h["bundle_level"])
        m["bundles"].add(h["bundle_id"]); m["terms"].update(h.get("bundle_terms") or [])
        if tid not in by_id or (h.get("retrieved_at") or "") >= (by_id[tid].get("retrieved_at") or ""):
            by_id[tid] = h
    out = []
    for tid, h in by_id.items():
        h = dict(h)
        m = membership[tid]
        h["all_bundle_dimensions"] = sorted(m["dims"]); h["all_bundle_levels"] = sorted(m["levels"])
        h["n_bundles"] = len(m["bundles"]); h["all_bundle_terms"] = sorted(m["terms"])
        out.append(h)
    return out


def fetch_issue_comments(client: GitHubClient, store: Store, config: Config,
                         progress: Callable[[Iterable], Iterable] = lambda x: x) -> Dict[str, int]:
    """
    Retrieve all comments for EVERY unique issue identified by the searches.
    A thread is marked complete only when the final page was shorter than
    per_page (or empty) and no page failed; otherwise it is 'partial' and is
    re-fetched in full on resume. Comments are deduplicated at assembly on
    message_id keeping only the latest fetch of each thread, so re-fetching
    never duplicates messages. Zero-comment issues are recorded as complete.
    """
    issues = unique_issue_hits(store)
    counts = Counter()
    since_ckpt = 0
    for h in progress(issues):
        tid = h["thread_id"]
        if store.status("issue_threads", tid) == "complete":
            counts["skipped_complete"] += 1
            continue
        if not h.get("comments_count"):
            store.mark("issue_threads", tid, status="complete", expected=0, retrieved=0, retrieved_at=now_iso())
            counts["zero_comments"] += 1
            continue
        page = 1
        retrieved = []
        failed = False
        error = None
        fetch_tag = now_iso()
        while True:
            out = client.get(h["comments_url"], params={"per_page": config.per_page, "page": page}, kind="comments")
            client.sleep(config.comment_sleep_seconds)
            if not out.ok or not isinstance(out.data, list):
                failed = True
                error = out.error or f"HTTP {out.status}"
                break
            for c in out.data:
                u = c.get("user") or {}
                retrieved.append({
                    "source_type": "issue", "thread_id": tid, "repo": h["repo"], "number": h["number"],
                    "message_id": c.get("node_id"), "github_id": c.get("id"), "parent_message_id": None, "depth": 1,
                    "author_login": u.get("login"), "author_id": u.get("id"), "author_type": u.get("type"),
                    "author_association": c.get("author_association"),
                    "created_at": c.get("created_at"), "updated_at": c.get("updated_at"),
                    "body": c.get("body"), "url": c.get("html_url"),
                    "reactions_total": (c.get("reactions") or {}).get("total_count"),
                    "fetch_tag": fetch_tag, "retrieved_at": out.retrieved_at})
            if len(out.data) < config.per_page:
                break
            page += 1
        store.append_many("issue_comments", retrieved)
        complete = not failed
        store.mark("issue_threads", tid, status="complete" if complete else "partial",
                   expected=h.get("comments_count"), retrieved=len(retrieved), retrieved_at=fetch_tag,
                   fetch_tag=fetch_tag, error=error, failed_page=page if failed else None)
        counts["complete" if complete else "partial"] += 1
        since_ckpt += 1
        if since_ckpt >= config.checkpoint_every:
            store.save()
            since_ckpt = 0
    store.save()
    return dict(counts)


# =============================================================================
# 9. Assembly: full-text thread and message tables (no filtering, no coding)
# =============================================================================

def _dedupe_latest(records: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for r in records:
        k = r.get(key)
        if k is None:
            continue
        if k not in best or (r.get("retrieved_at") or "") >= (best[k].get("retrieved_at") or ""):
            best[k] = r
    return list(best.values())


def _message_row(rec: Dict[str, Any], thread: Dict[str, Any], cutoff: datetime, message_type: str,
                 index: int, depth: int, parent: Optional[str], title: Optional[str]) -> Dict[str, Any]:
    created = parse_iso(rec.get("created_at"))
    in_window = bool(created and created <= cutoff)
    body = rec.get("body")
    return {
        "thread_id": thread["thread_id"], "source_type": thread["source_type"], "repo": thread["repo"],
        "repo_id": thread.get("repo_id"), "repo_category": thread.get("repo_category"),
        "repo_subgroup": thread.get("repo_subgroup"), "system_type": thread.get("system_type"),
        "message_id": rec.get("message_id") if message_type != "opening_post" else rec.get("node_id"),
        "github_id": rec.get("github_id"),
        "message_type": message_type, "message_index": index, "depth": depth,
        "parent_message_id": parent, "title": title, "body": body,
        "body_n_chars": len(body) if isinstance(body, str) else 0,
        "author_login": rec.get("author_login"), "author_id": rec.get("author_id"),
        "author_type": rec.get("author_type"), "author_association": rec.get("author_association"),
        "author_role": author_role(rec.get("author_association"), rec.get("author_type"), rec.get("author_login")),
        "created_at": rec.get("created_at"), "updated_at": rec.get("updated_at"),
        "last_edited_at": rec.get("last_edited_at"),
        "url": rec.get("url") if message_type != "opening_post" else rec.get("html_url"),
        "is_answer": rec.get("is_answer"), "reactions_total": rec.get("reactions_total"),
        "upvotes": rec.get("upvotes"),
        "in_window": in_window, "outside_window": not in_window,
        "retrieved_at": rec.get("retrieved_at"),
    }


def assemble(store: Store, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the full-text message table and the thread table from JSONL only.
    Every retrieved thread and every retrieved message is kept: there is no
    local keyword eligibility filter. Messages created after the analytic
    cutoff are kept and flagged outside_window = True. Issues and Discussions
    share one schema; Discussion replies keep parent_message_id (depth 2).
    """
    cutoff = config.cutoff_dt()
    messages: List[Dict[str, Any]] = []
    threads: List[Dict[str, Any]] = []

    # ---- issues
    issues = unique_issue_hits(store)
    manifest_threads = store.manifest.get("issue_threads", {})
    latest_fetch = {tid: v.get("fetch_tag") for tid, v in manifest_threads.items()}
    comments = _dedupe_latest(store.read("issue_comments"), "message_id")
    by_thread: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in comments:
        if latest_fetch.get(c["thread_id"]) in (None, c.get("fetch_tag")):
            by_thread[c["thread_id"]].append(c)

    for h in issues:
        tid = h["thread_id"]
        msgs = [_message_row({**h, "author_login": h.get("author_login")}, h, cutoff, "opening_post", 0, 0, None,
                             h.get("title"))]
        for i, c in enumerate(sorted(by_thread.get(tid, []), key=lambda x: (x.get("created_at") or "", x.get("github_id") or 0)), 1):
            msgs.append(_message_row(c, h, cutoff, "comment", i, 1, None, None))
        st = manifest_threads.get(tid, {})
        threads.append(_thread_row(h, msgs, st, cutoff))
        messages.extend(msgs)

    # ---- discussions
    if config.collect_discussions:
        dhits: Dict[str, Dict[str, Any]] = {}
        for h in store.read("discussion_hits"):
            m = dhits.setdefault(h["thread_id"], {"hit": h, "dims": set(), "levels": set(), "bundles": set(), "terms": set()})
            m["dims"].add(h["bundle_dimension"]); m["levels"].add(h["bundle_level"]); m["bundles"].add(h["bundle_id"])
            m["terms"].update(h.get("bundle_terms") or [])
        full = {r["thread_id"]: r for r in _dedupe_latest(store.read("discussions"), "thread_id")}
        dmanifest = store.manifest.get("discussion_threads", {})
        for tid, m in dhits.items():
            h = dict(m["hit"])
            h["all_bundle_dimensions"] = sorted(m["dims"]); h["all_bundle_levels"] = sorted(m["levels"])
            h["n_bundles"] = len(m["bundles"]); h["all_bundle_terms"] = sorted(m["terms"])
            f = full.get(tid)
            if f:
                for k in ["body", "last_edited_at", "is_answered", "answer_message_id", "answer_chosen_at", "labels",
                          "upvotes", "github_id", "node_id", "updated_at"]:
                    if k in f:
                        h[k] = f.get(k)
                h["retrieved_at"] = f.get("retrieved_at")
            msgs = [_message_row(h, h, cutoff, "opening_post", 0, 0, None, h.get("title"))]
            if f:
                ordered = sorted(f.get("comments", []), key=lambda x: (x.get("created_at") or ""))
                for i, c in enumerate(ordered, 1):
                    c = {**c, "retrieved_at": f.get("retrieved_at")}
                    is_reply = c.get("depth") == 2
                    msgs.append(_message_row(c, h, cutoff, "reply" if is_reply else "comment", i,
                                             2 if is_reply else 1, c.get("parent_message_id"), None))
            st = dmanifest.get(tid, {"status": "not_fetched"} if not f else {})
            row = _thread_row(h, msgs, st, cutoff)
            if f:
                row["expected_comments"] = f.get("expected_top_level_comments")
                row["retrieved_comments"] = f.get("retrieved_top_level_comments")
            threads.append(row)
            messages.extend(msgs)

    messages_df = pd.DataFrame(messages)
    threads_df = pd.DataFrame(threads)
    if len(messages_df):
        messages_df = messages_df.sort_values(["thread_id", "message_index"], kind="stable").reset_index(drop=True)
    if len(threads_df):
        threads_df = threads_df.sort_values(["source_type", "repo", "number"], kind="stable").reset_index(drop=True)
    return threads_df, messages_df


def _thread_row(h: Dict[str, Any], msgs: List[Dict[str, Any]], status: Dict[str, Any], cutoff: datetime) -> Dict[str, Any]:
    created = parse_iso(h.get("created_at"))
    kinds = Counter(m["message_type"] for m in msgs)
    retrieval_status = status.get("status", "not_fetched")
    return {
        "thread_id": h["thread_id"], "source_type": h["source_type"], "repo": h["repo"], "repo_id": h.get("repo_id"),
        "repo_category": h.get("repo_category"), "repo_subgroup": h.get("repo_subgroup"),
        "system_type": h.get("system_type"), "number": h.get("number"), "github_id": h.get("github_id"),
        "node_id": h.get("node_id"), "title": h.get("title"), "html_url": h.get("html_url"),
        "author_login": h.get("author_login"), "author_id": h.get("author_id"), "author_type": h.get("author_type"),
        "author_association": h.get("author_association"),
        "created_at": h.get("created_at"), "updated_at": h.get("updated_at"), "closed_at": h.get("closed_at"),
        "state": h.get("state"), "state_reason": h.get("state_reason"), "locked": h.get("locked"),
        "labels": h.get("labels"), "discussion_category": h.get("discussion_category"),
        "is_answered": h.get("is_answered"), "answer_message_id": h.get("answer_message_id"),
        "comments_count_at_search": h.get("comments_count"),
        "all_bundle_dimensions": h.get("all_bundle_dimensions"), "all_bundle_levels": h.get("all_bundle_levels"),
        "n_bundles": h.get("n_bundles"), "all_bundle_terms": h.get("all_bundle_terms"),
        "thread_created_in_window": bool(created and created <= cutoff),
        "retrieval_status": retrieval_status,
        "retrieval_complete": retrieval_status == "complete",
        "retrieval_error": status.get("error"),
        "expected_comments": status.get("expected"), "retrieved_comments": status.get("retrieved"),
        "n_messages": len(msgs), "n_opening_posts": kinds.get("opening_post", 0),
        "n_comments": kinds.get("comment", 0), "n_replies": kinds.get("reply", 0),
        "n_messages_outside_window": sum(1 for m in msgs if m["outside_window"]),
        "n_bot_messages": sum(1 for m in msgs if m["author_role"] == "bot"),
        "retrieved_at": h.get("retrieved_at"),
    }


# =============================================================================
# 10. Exports: full-text CSV/JSONL (authoritative) and optional Excel previews
# =============================================================================

# Characters that XML 1.0 (and therefore openpyxl with lxml) rejects: C0 controls
# other than tab/LF/CR, the non-characters U+FFFE/U+FFFF, and unpaired surrogates.
# Removed from Excel previews only; JSONL and CSV keep the original text.
ILLEGAL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿\ud800-\udfff]")


def _xl(value):
    if isinstance(value, (list, dict, set, tuple)):
        value = json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False, default=str)
    if isinstance(value, str):
        return ILLEGAL_RE.sub("", value)
    return value


def to_excel_preview(df: pd.DataFrame, path: str, text_cols: Sequence[str], preview_chars: int,
                     id_col: str = "thread_id", reference_file: str = "") -> pd.DataFrame:
    """
    Excel preview for browsing. Each long text column is replaced by a preview of
    preview_chars characters, its full character count (*_n_chars) and an explicit
    *_preview_truncated flag; full_text_reference names the full-text file and id.
    Any other cell longer than Excel's 32,767-character limit is cut and marked
    [TRUNCATED_FOR_EXCEL], with the column listed in excel_truncated_columns.
    """
    out = df.copy()
    for col in text_cols:
        if col not in out.columns:
            continue
        full = out[col].fillna("").astype(str)
        out[f"{col}_n_chars"] = full.str.len()
        out[f"{col}_preview_truncated"] = full.str.len() > preview_chars
        out[f"{col}_preview"] = full.str.slice(0, preview_chars)
        out = out.drop(columns=[col])
    if id_col in out.columns:
        out["full_text_reference"] = reference_file + " :: " + out[id_col].astype(str)
    out = out.apply(lambda c: c.map(_xl))
    cut_cols = []
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]) or pd.api.types.is_bool_dtype(out[col]):
            continue
        s = out[col]
        lens = s.map(lambda v: len(v) if isinstance(v, str) else 0)
        if len(lens) and lens.max() > EXCEL_CELL_LIMIT:
            over = lens > EXCEL_CELL_LIMIT
            out.loc[over, col] = s[over].map(lambda v: v[:EXCEL_CELL_LIMIT - 25] + " [TRUNCATED_FOR_EXCEL]")
            cut_cols.append(col)
    out["excel_truncated_columns"] = ";".join(cut_cols)
    out.to_excel(path, index=False)
    return out


def _write_jsonl(path: str, df: pd.DataFrame) -> None:
    with open(path, "w", encoding="utf-8", errors="backslashreplace") as f:
        for rec in df.to_dict("records"):
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def write_outputs(store: Store, config: Config, threads_df: pd.DataFrame, messages_df: pd.DataFrame,
                  repo_df: Optional[pd.DataFrame] = None, prefix: str = "corpus") -> Dict[str, str]:
    """
    Save the full-text datasets (CSV and JSONL), repository metadata, the search
    log as CSV, and optional Excel previews. Full text is never shortened in CSV
    or JSONL.
    """
    d = store.dir
    paths = {
        "threads_jsonl": os.path.join(d, f"{prefix}_threads.jsonl"),
        "messages_jsonl": os.path.join(d, f"{prefix}_messages.jsonl"),
        "threads_csv": os.path.join(d, f"{prefix}_threads.csv"),
        "messages_csv": os.path.join(d, f"{prefix}_messages.csv"),
    }
    _write_jsonl(paths["threads_jsonl"], threads_df)
    _write_jsonl(paths["messages_jsonl"], messages_df)
    threads_df.to_csv(paths["threads_csv"], index=False, encoding="utf-8-sig", errors="backslashreplace")
    messages_df.to_csv(paths["messages_csv"], index=False, encoding="utf-8-sig", errors="backslashreplace")
    meta = repo_df if repo_df is not None else pd.DataFrame(store.read("repo_metadata"))
    if len(meta):
        meta = meta.drop_duplicates(subset=["supplied_path"], keep="last")
    paths["repo_metadata_csv"] = os.path.join(d, "repository_metadata.csv")
    meta.to_csv(paths["repo_metadata_csv"], index=False, encoding="utf-8-sig")
    paths["search_log_csv"] = os.path.join(d, "search_log.csv")
    pd.DataFrame(store.read("search_log")).to_csv(paths["search_log_csv"], index=False, encoding="utf-8-sig")
    if config.write_thread_preview_xlsx and len(threads_df):
        tx = threads_df.copy()
        if len(messages_df):
            op = messages_df[messages_df["message_type"] == "opening_post"].set_index("thread_id")["body"]
            tx["opening_post_body"] = tx["thread_id"].map(op)
        paths["threads_xlsx"] = os.path.join(d, f"{prefix}_threads_preview.xlsx")
        to_excel_preview(tx, paths["threads_xlsx"], ["opening_post_body"], config.preview_chars,
                         reference_file=os.path.basename(paths["messages_jsonl"]))
    if config.write_message_preview_xlsx and len(messages_df):
        paths["messages_xlsx"] = os.path.join(d, f"{prefix}_messages_preview.xlsx")
        to_excel_preview(messages_df, paths["messages_xlsx"], ["body"], config.preview_chars, id_col="message_id",
                         reference_file=os.path.basename(paths["messages_jsonl"]))
    return paths


def build_thread_text(messages_df: pd.DataFrame, thread_id: str, in_window_only: bool = False) -> str:
    """Render one thread as structured text from the message table (for reading; not saved by default)."""
    m = messages_df[messages_df["thread_id"] == thread_id]
    if in_window_only:
        m = m[m["in_window"]]
    parts = []
    for _, r in m.sort_values("message_index").iterrows():
        tag = "OPENING POST" if r["message_type"] == "opening_post" else f"{str(r['message_type']).upper()} {int(r['message_index'])}"
        if r.get("depth") == 2:
            tag += f" (reply to {r.get('parent_message_id')})"
        parts.append(f"[{tag} | {r['author_login']} ({r['author_association']}) | {r['created_at']} | id={r['message_id']}]")
        if r["message_type"] == "opening_post":
            parts.append(f"TITLE: {r.get('title') or ''}")
        parts.append(r.get("body") or "")
        parts.append("")
    return "\n".join(parts).strip()


# =============================================================================
# 11. Collection summary
# =============================================================================

def collection_summary(store: Store, threads_df: pd.DataFrame, messages_df: pd.DataFrame,
                       repo_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Counts describing what was collected and what is incomplete. Written to collection_summary.json."""
    log = pd.DataFrame(store.read("search_log"))
    act = log["action"] if "action" in log.columns else pd.Series(dtype=str)

    def count(df, col, val):
        return int((df[col] == val).sum()) if len(df) and col in df.columns else 0

    issue_bundles = Counter(v.get("status") for v in store.manifest.get("issue_search", {}).values())
    disc_searches = Counter(v.get("status") for v in store.manifest.get("discussion_search", {}).values())
    repos = Counter(v.get("status") for v in store.manifest.get("repos", {}).values())
    t = threads_df
    m = messages_df
    status_by_source = {}
    if len(t):
        for (src, st), n in t.groupby(["source_type", "retrieval_status"]).size().items():
            status_by_source.setdefault(src, {})[st] = int(n)
    comment_gap = {}
    if len(t):
        it = t[(t["source_type"] == "issue") & (t["retrieval_status"] == "complete")]
        exp = pd.to_numeric(it["expected_comments"], errors="coerce")
        got = pd.to_numeric(it["retrieved_comments"], errors="coerce")
        comment_gap = {"fewer_than_count_at_search": int((got < exp).sum()),
                       "more_than_count_at_search": int((got > exp).sum())}
    bad_lines = sum(e.get("n", 0) for e in store.manifest.get("events", []) if e.get("event") == "unparseable_jsonl_lines")
    summary = {
        "generated_at": now_iso(),
        "pipeline_version": PIPELINE_VERSION,
        "run_id": store.manifest.get("run_id"),
        "config_hash": store.manifest.get("config_hash"),
        "thread_creation_window": [store.config.start_date.isoformat(), store.config.end_date.isoformat()],
        "message_cutoff_utc": store.config.message_cutoff_utc,
        "repositories": {
            "supplied": len(REPO_INFO), "resolved": repos.get("resolved", 0),
            "unavailable": repos.get("unavailable", 0),
            "issues_disabled": int((repo_df["has_issues"] == False).sum()) if repo_df is not None and len(repo_df) else None,  # noqa: E712
            "discussions_enabled": int((repo_df["has_discussions"] == True).sum()) if repo_df is not None and len(repo_df) else None,  # noqa: E712
        },
        "search": {
            "queries_logged": int(len(log)),
            "requests_failed": int((act == "request_failed").sum()),
            "window_splits": int((act == "split").sum()),
            "near_cap_flags": int((act == "near_cap").sum()),
            "single_day_cap_events": int((act == "cap_exceeded_single_day").sum()),
            "incomplete_results_flags": int(log["incomplete_results"].map(lambda v: v is True).sum()) if "incomplete_results" in log.columns else 0,
            "issue_bundles": dict(issue_bundles),
            "discussion_term_searches": dict(disc_searches),
            "raw_issue_hits": len(store.read("issue_hits")),
            "raw_discussion_hits": len(store.read("discussion_hits")),
            "unparseable_jsonl_lines_skipped": bad_lines,
        },
        "threads": {
            "total": int(len(t)),
            "issues": count(t, "source_type", "issue"),
            "discussions": count(t, "source_type", "discussion"),
            "retrieval_complete": int(t["retrieval_complete"].sum()) if len(t) else 0,
            "retrieval_incomplete": int((~t["retrieval_complete"]).sum()) if len(t) else 0,
            "retrieval_status_by_source": status_by_source,
            "without_replies": int((t["n_messages"] <= 1).sum()) if len(t) else 0,
            "issue_comment_count_differences": comment_gap,
        },
        "messages": {
            "total": int(len(m)),
            "opening_posts": count(m, "message_type", "opening_post"),
            "comments": count(m, "message_type", "comment"),
            "nested_replies": count(m, "message_type", "reply"),
            "outside_window": int(m["outside_window"].sum()) if len(m) else 0,
            "bot_authored": count(m, "author_role", "bot"),
        },
        "coverage_note": ("Coverage is exhaustive only if requests_failed, single_day_cap_events, incomplete_results_flags "
                          "and retrieval_incomplete are all zero and every issue bundle and discussion search is complete."),
    }
    with open(os.path.join(store.dir, "collection_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    if len(t):
        per_repo = (t.groupby(["repo_category", "repo", "source_type", "retrieval_status"]).size()
                    .unstack(fill_value=0).reset_index())
        per_repo.to_csv(os.path.join(store.dir, "collection_by_repository.csv"), index=False, encoding="utf-8-sig")
    incomplete = t[~t["retrieval_complete"]] if len(t) else t
    if len(incomplete):
        incomplete[["thread_id", "source_type", "repo", "html_url", "retrieval_status", "retrieval_error",
                     "expected_comments", "retrieved_comments"]].to_csv(
            os.path.join(store.dir, "incomplete_retrievals.csv"), index=False, encoding="utf-8-sig")
    return summary


# =============================================================================
# 12. Step 8: cleaned, coding-ready files (separate from the collection)
# =============================================================================
# Reads the saved full-text JSONL datasets and writes cleaned copies to a
# `coding_ready` subfolder. The collected datasets are never modified.

_FENCE_RE = re.compile(r"```.*?(?:```|\Z)", re.S)
_LOW_CONTENT_RE = r"(?is)\W*(\+1|same( here| issue| problem)?|me too|any (update|news)s?\??|bump)\W*"


def clean_text_for_coding(text: Any, max_code_lines: int = 15) -> str:
    """Remove material that carries no user explanation but costs LLM tokens.

    Hidden issue-template comments are deleted; collapsed <details> sections,
    fenced code or log blocks longer than `max_code_lines` lines and images are
    replaced by placeholders; surplus blank lines are collapsed.
    """
    if not isinstance(text, str):
        return ""

    def _code(m):
        block = m.group(0)
        lines = block.count("\n") - 1
        return block if lines <= max_code_lines else f"[code/log block removed: {lines} lines]"

    t = text.replace("\r\n", "\n")
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<details>.*?</details>", "[collapsed section removed]", t, flags=re.S | re.I)
    t = _FENCE_RE.sub(_code, t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "[image]", t)
    t = re.sub(r"<img[^>]*>", "[image]", t, flags=re.I)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def clean_for_coding(data_dir: str, max_code_lines: int = 15, max_thread_chars: int = 30000,
                     out_subdir: str = "coding_ready") -> Dict[str, Any]:
    """Build coding-ready files from corpus_threads.jsonl and corpus_messages.jsonl.

    Steps (each counted in cleaning_log.json):
      1. keep fully retrieved threads;
      2. drop threads opened by bot accounts (with all their messages);
      3. drop remaining bot messages and messages after the analytic cutoff;
      4. clean text into `body_clean` (original `body` kept);
      5. drop comments and replies left empty (opening posts always kept);
         flag, without removing, very short "+1"-style messages (`low_content`).
    Outputs (CSV and JSONL): coding_opening_posts, coding_threads (ordered,
    labelled conversation; shortened above `max_thread_chars` and flagged) and
    coding_messages_clean (never shortened).
    """
    out_dir = os.path.join(data_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    def _read(name):
        return pd.read_json(os.path.join(data_dir, name), lines=True, dtype=False, convert_dates=False)

    threads = _read("corpus_threads.jsonl")
    msgs = _read("corpus_messages.jsonl")
    log: Dict[str, Any] = {"generated_at": now_iso(), "pipeline_version": PIPELINE_VERSION,
                           "start_threads": len(threads), "start_messages": len(msgs)}

    threads = threads[threads["retrieval_complete"] == True]  # noqa: E712
    log["threads_removed_incomplete"] = log["start_threads"] - len(threads)

    bot_threads = set(msgs.loc[(msgs["message_type"] == "opening_post") & (msgs["author_role"] == "bot"),
                               "thread_id"])
    threads = threads[~threads["thread_id"].isin(bot_threads)]
    log["threads_removed_bot_opened"] = len(bot_threads)
    msgs = msgs[msgs["thread_id"].isin(set(threads["thread_id"]))]

    n = len(msgs)
    msgs = msgs[msgs["author_role"] != "bot"]
    log["messages_removed_bot"] = n - len(msgs)
    n = len(msgs)
    msgs = msgs[msgs["outside_window"] != True]  # noqa: E712
    log["messages_removed_after_cutoff"] = n - len(msgs)

    msgs = msgs.copy()
    msgs["body_clean"] = msgs["body"].map(lambda t: clean_text_for_coding(t, max_code_lines))
    msgs["body_clean_n_chars"] = msgs["body_clean"].str.len()

    n = len(msgs)
    msgs = msgs[(msgs["message_type"] == "opening_post") | (msgs["body_clean"] != "")].copy()
    log["messages_removed_empty"] = n - len(msgs)
    msgs["low_content"] = (msgs["message_type"] != "opening_post") & (
        msgs["body_clean"].str.fullmatch(_LOW_CONTENT_RE) | (msgs["body_clean_n_chars"] < 15))
    msgs = msgs.sort_values(["thread_id", "message_index"], kind="stable")

    keep_t = ["thread_id", "source_type", "repo", "repo_category", "repo_subgroup", "system_type", "number",
              "title", "html_url", "created_at", "state", "labels", "discussion_category"]
    tmeta = threads[[c for c in keep_t if c in threads.columns]].set_index("thread_id")

    op = msgs[msgs["message_type"] == "opening_post"].set_index("thread_id")
    openings = tmeta.join(op[["author_role", "body_clean", "body_clean_n_chars"]], how="inner").reset_index()
    openings = openings.rename(columns={"body_clean": "opening_post_text",
                                        "body_clean_n_chars": "opening_post_n_chars"})

    kinds = {"opening_post": "OPENING POST", "comment": "COMMENT", "reply": "REPLY"}
    blocks = [f"--- [{i}] {kinds.get(k, k)} | {r} | {str(c)[:10]} ---\n{b}"
              for i, k, r, c, b in zip(msgs["message_index"], msgs["message_type"], msgs["author_role"],
                                       msgs["created_at"], msgs["body_clean"])]
    conv = (msgs.assign(_block=blocks)
                .groupby("thread_id")
                .agg(thread_text=("_block", "\n\n".join), n_messages_kept=("message_id", "size"),
                     n_low_content=("low_content", "sum")))
    whole = tmeta.join(conv, how="inner").reset_index()
    whole["thread_text"] = "TITLE: " + whole["title"].fillna("").astype(str) + "\n\n" + whole["thread_text"]
    whole["thread_n_chars"] = whole["thread_text"].str.len()
    whole["thread_text_truncated"] = whole["thread_n_chars"] > max_thread_chars
    cut = whole["thread_text_truncated"]
    whole.loc[cut, "thread_text"] = whole.loc[cut, "thread_text"].str[:max_thread_chars] + "\n[... thread shortened ...]"

    paths: Dict[str, str] = {}
    for name, df in [("coding_opening_posts", openings), ("coding_threads", whole),
                     ("coding_messages_clean", msgs)]:
        paths[name + "_csv"] = os.path.join(out_dir, name + ".csv")
        paths[name + "_jsonl"] = os.path.join(out_dir, name + ".jsonl")
        df.to_csv(paths[name + "_csv"], index=False, encoding="utf-8-sig", errors="backslashreplace")
        _write_jsonl(paths[name + "_jsonl"], df)

    log.update({"final_threads": len(whole), "final_messages": len(msgs), "final_opening_posts": len(openings),
                "low_content_messages_flagged": int(msgs["low_content"].sum()),
                "threads_shortened": int(cut.sum()),
                "settings": {"max_code_lines": max_code_lines, "max_thread_chars": max_thread_chars}})
    paths["cleaning_log"] = os.path.join(out_dir, "cleaning_log.json")
    with open(paths["cleaning_log"], "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    log["paths"] = paths
    return log
