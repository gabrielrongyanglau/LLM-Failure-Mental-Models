"""
Offline tests for llm_failures_collect against the scripted FakeGitHub server.
Run:  python -m pytest tests -q
"""
import json
import os
import sys
from collections import Counter
from datetime import date, datetime, timezone

import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import llm_failures_collect as p  # noqa: E402
from fake_github import FakeGitHub  # noqa: E402

NOSLEEP = lambda s: None  # noqa: E731

REQUIRED_REPOS = {
    "General chat interfaces and personal assistants": [
        "open-webui/open-webui", "LibreChat-AI/LibreChat", "lobehub/lobehub", "janhq/jan", "CherryHQ/cherry-studio"],
    "Document, knowledge and search assistants": [
        "Mintplex-Labs/anything-llm", "nomic-ai/gpt4all", "khoj-ai/khoj", "infiniflow/ragflow", "onyx-dot-app/onyx"],
    "Coding assistants and action-taking agents": [
        "openai/codex", "cline/cline", "Aider-AI/aider", "continuedev/continue", "OpenHands/OpenHands"],
    "Role-play and local-model environments": [
        "SillyTavern/SillyTavern", "kwaroran/Risuai", "LostRuins/koboldcpp", "oobabooga/textgen", "ollama/ollama"],
    "Agent orchestration and LLM application frameworks": [
        "microsoft/autogen", "crewAIInc/crewAI", "langchain-ai/langgraph", "FlowiseAI/Flowise", "langflow-ai/langflow"],
    "Agent memory and model-access infrastructure": [
        "letta-ai/letta", "mem0ai/mem0", "getzep/graphiti", "BerriAI/litellm", "mostlygeek/llama-swap"],
}


def dt(y, m, d, h=12):
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def make_config(tmp_path, **kw):
    base = dict(data_dir=str(tmp_path / "data"), search_sleep_seconds=0, graphql_sleep_seconds=0,
                comment_sleep_seconds=0, per_page=2, max_pages=10, result_cap=4, near_cap_threshold=3,
                graphql_page_size=2, bundle_levels=("core",), dimensions=("Memory and context lapses",),
                discussion_levels=("core",), checkpoint_every=1)
    base.update(kw)
    return p.Config(**base)


FIXTURE_REPOS = {
    "acme/chat": {"category": "C1", "subgroup": "S", "system_type": "T1"},
    "acme/agent": {"category": "C2", "subgroup": "S", "system_type": "T2"},
    "acme/oldname": {"category": "C1", "subgroup": "S", "system_type": "T1"},
    "acme/noissues": {"category": "C2", "subgroup": "S", "system_type": "T2"},
    "acme/gone": {"category": "C2", "subgroup": "S", "system_type": "T2"},
}


@pytest.fixture
def fake():
    g = FakeGitHub(result_cap=4)
    g.add_repo("acme/chat", has_discussions=True)
    g.add_repo("acme/agent")
    g.add_repo("acme/newname", redirect_from="acme/oldname")
    g.add_repo("acme/noissues", has_issues=False)
    g.missing.add("acme/gone")
    for i in range(1, 7):
        g.add_issue("acme/chat", i, f"Model forgets context #{i}", "It forgets everything after a few turns.",
                    dt(2024, 1 + i, 10),
                    comments=[{"body": "Same here.", "author": "u2"},
                              {"body": "Context length is the limit.", "author": "maint", "assoc": "MEMBER"},
                              {"body": "stale", "author": "stale[bot]", "type": "Bot"}])
    g.add_issue("acme/chat", 50, "Docker build fails", "npm error", dt(2024, 3, 3))          # does not match
    g.add_issue("acme/chat", 55, "It forgets", "old issue", dt(2023, 8, 31))                 # before window
    g.add_issue("acme/chat", 60, "Odd behaviour", "Something is off.", dt(2026, 9, 1),        # matches via late comment
                comments=[{"body": "It forgets everything after an update.", "author": "late", "created": dt(2026, 10, 5)}])
    g.add_issue("acme/chat", 70, "Forgets long docs", "It forgets. " + ("x" * 40000) + "\x0b￾ end", dt(2024, 5, 5))
    g.add_issue("acme/chat", 90, "Assistant forgets the persona", "Persona disappears.", dt(2024, 7, 7))  # title-only
    for i in range(1, 11):
        g.add_issue("acme/agent", i, f"Loses context {i}", "It loses context mid task", dt(2024, 8, 8, h=i))
    g.add_discussion("acme/chat", 5, "Why does it forget?", "The assistant forgets everything.", dt(2024, 2, 2),
                     comments=[{"body": "Context window.", "author": "c1",
                                "replies": [{"body": f"reply {k}"} for k in range(5)]},
                               {"body": "Try a bigger model.", "author": "c2", "assoc": "MEMBER"},
                               {"body": "Stale", "author": "bot[bot]", "type": "Bot"}])
    g.add_discussion("acme/chat", 6, "Theme question", "How to theme the UI?", dt(2024, 2, 3))
    return g


def run_resolve(cfg, fake, allow=False):
    store = p.Store(cfg, allow_config_change=allow)
    client = p.GitHubClient("tok", cfg, store, session=fake, sleep_fn=NOSLEEP)
    repo_df = p.resolve_repositories(client, store, FIXTURE_REPOS, sleep_seconds=0)
    return store, client, repo_df


def full_run(tmp_path, fake, repos=("acme/chat",), **kw):
    cfg = make_config(tmp_path, **kw)
    store, client, df = run_resolve(cfg, fake)
    rr = {k: v for k, v in p.resolved_repos(df).items() if k in repos}
    p.search_issues(client, store, cfg, rr)
    p.fetch_issue_comments(client, store, cfg)
    p.search_discussions(client, store, cfg, rr)
    p.fetch_discussions(client, store, cfg)
    threads, messages = p.assemble(store, cfg)
    return cfg, store, client, df, threads, messages


# --------------------------------------------------------------------------- scope and settings
def test_repository_scope_is_exactly_the_30_requested():
    assert len(p.REPO_INFO) == 30
    got = {}
    for repo, info in p.REPO_INFO.items():
        got.setdefault(info["category"], []).append(repo)
    assert got == REQUIRED_REPOS


def test_default_dates_and_cutoff():
    c = p.Config()
    assert c.start_date == date(2023, 9, 1) and c.end_date == date(2026, 9, 30)
    assert c.message_cutoff_utc == "2026-09-30T23:59:59Z"
    assert c.cutoff_dt() == datetime(2026, 9, 30, 23, 59, 59, tzinfo=timezone.utc)
    assert c.data_dir.endswith("llm_failures_data_v3")


def test_no_sampling_screening_or_marker_code_remains():
    names = set(dir(p))
    for gone in ["build_retrieval_sample", "load_retrieval_sample_ids", "make_screening_workbook",
                 "make_coding_workbook", "build_sampling_plan", "pilot_sample", "audit_sample",
                 "provisional_markers", "segment_text", "EXPLANATION_FRAMES", "add_final_eligibility"]:
        assert gone not in names, gone
    src = open(p.__file__, encoding="utf-8").read()
    assert "openai.com/v1" not in src and "anthropic" not in src.lower()


def test_pack_terms_respects_documented_limits_for_all_30_repositories():
    cfg = p.Config()
    for repo in p.REPO_INFO:
        searched = []
        for dim, lvl, bid, terms in p.iter_bundles(cfg, repo):
            q = p.query_base(repo, cfg.start_date, cfg.end_date) + p.or_clause(terms)
            assert len(terms) - 1 <= cfg.max_or_operators and len(q) <= cfg.max_query_chars, (repo, bid)
            searched += terms
        expected = [t for d in cfg.dimensions for l in cfg.bundle_levels for t in p.FAILURE_TERMS[d][l]]
        assert sorted(searched) == sorted(expected)
        assert "created:2023-09-01..2026-09-30" in p.query_base(repo, cfg.start_date, cfg.end_date)


# --------------------------------------------------------------------------- repositories
def test_resolve_records_ids_redirects_unavailable_and_features(tmp_path, fake):
    cfg = make_config(tmp_path)
    store, client, df = run_resolve(cfg, fake)
    r = df.set_index("supplied_path")
    assert r.loc["acme/oldname", "canonical_path"] == "acme/newname" and bool(r.loc["acme/oldname", "redirected"])
    assert r.loc["acme/gone", "http_status"] == 404 and not r.loc["acme/gone", "resolved"]
    assert r.loc["acme/chat", "repo_id"] == fake.repos["acme/chat"]["id"]
    assert bool(r.loc["acme/chat", "has_discussions"]) and not bool(r.loc["acme/noissues", "has_issues"])
    assert not any(c.startswith("screen_") for c in df.columns)
    assert "acme/noissues" not in p.resolved_repos(df)


# --------------------------------------------------------------------------- search
def test_search_window_split_dedupe_and_logging(tmp_path, fake):
    cfg = make_config(tmp_path)
    store, client, df = run_resolve(cfg, fake)
    c = p.search_issues(client, store, cfg, {"acme/chat": p.resolved_repos(df)["acme/chat"]})
    log = pd.DataFrame(store.read("search_log"))
    assert (log["action"] == "split").sum() >= 1 and log["query"].str.len().max() <= 256
    nums = sorted(h["number"] for h in p.unique_issue_hits(store))
    assert nums == [1, 2, 3, 4, 5, 6, 60, 70, 90], nums          # 50 no match; 55 before window
    assert c.get("partial", 0) == 0


def test_single_day_over_cap_is_reported_partial(tmp_path, fake):
    cfg = make_config(tmp_path)
    store, client, df = run_resolve(cfg, fake)
    c = p.search_issues(client, store, cfg, {"acme/agent": p.resolved_repos(df)["acme/agent"]})
    log = pd.DataFrame(store.read("search_log"))
    assert (log["action"] == "cap_exceeded_single_day").sum() >= 1
    assert c["partial"] >= 1 and len(p.unique_issue_hits(store)) == 8


def test_validation_failure_is_failed_not_empty(tmp_path, fake):
    cfg = make_config(tmp_path, max_or_operators=7)
    store, client, df = run_resolve(cfg, fake)
    c = p.search_issues(client, store, cfg, {"acme/chat": p.resolved_repos(df)["acme/chat"]})
    log = pd.DataFrame(store.read("search_log"))
    assert (log["status"] == 422).any() and (log["action"] == "request_failed").any()
    assert c.get("complete", 0) == 0


def test_client_retries_and_rate_limits(tmp_path, fake):
    cfg = make_config(tmp_path)
    store = p.Store(cfg)
    slept = []
    client = p.GitHubClient("tok", cfg, store, session=fake, sleep_fn=slept.append)
    fake.fail_once["comments:acme/chat#1:1"] = 2
    out = client.get("https://api.github.com/repos/acme/chat/issues/1/comments", params={"per_page": 2, "page": 1})
    assert out.ok and out.attempts == 3
    fake.rate_limit_once = True
    assert client.get("https://api.github.com/repos/acme/chat").ok and len(slept) >= 3


# --------------------------------------------------------------------------- comments, resume, completeness
def test_all_issues_fetched_failure_reported_then_resumed_without_duplicates(tmp_path, fake):
    cfg = make_config(tmp_path)
    store, client, df = run_resolve(cfg, fake)
    repos = {"acme/chat": p.resolved_repos(df)["acme/chat"]}
    p.search_issues(client, store, cfg, repos)
    fake.fail_always.add("comments:acme/chat#3:2")
    c1 = p.fetch_issue_comments(client, store, cfg)
    n_issues = len(p.unique_issue_hits(store))
    assert sum(v for k, v in c1.items() if k in ("complete", "partial", "zero_comments")) == n_issues
    assert c1["partial"] == 1
    threads, messages = p.assemble(store, cfg)
    t3 = threads.set_index("thread_id").loc["I:acme/chat#3"]
    assert t3["retrieval_status"] == "partial" and not t3["retrieval_complete"] and t3["retrieval_error"]
    summ = p.collection_summary(store, threads, messages, df)
    assert summ["threads"]["retrieval_incomplete"] == 1
    assert os.path.exists(os.path.join(cfg.data_dir, "incomplete_retrievals.csv"))
    # resume
    fake.fail_always.clear()
    store2 = p.Store(cfg)
    client2 = p.GitHubClient("tok", cfg, store2, session=fake, sleep_fn=NOSLEEP)
    assert p.search_issues(client2, store2, cfg, repos).get("hits", 0) == 0
    c2 = p.fetch_issue_comments(client2, store2, cfg)
    assert c2["complete"] == 1 and c2["skipped_complete"] == n_issues - 1
    threads, messages = p.assemble(store2, cfg)
    assert messages["message_id"].is_unique
    t3 = threads.set_index("thread_id").loc["I:acme/chat#3"]
    assert t3["retrieval_complete"] and t3["n_comments"] == 3 and t3["n_opening_posts"] == 1


def test_config_change_refused_unless_allowed(tmp_path):
    p.Store(make_config(tmp_path))
    with pytest.raises(RuntimeError):
        p.Store(make_config(tmp_path, start_date=date(2022, 1, 1)))
    s = p.Store(make_config(tmp_path, start_date=date(2022, 1, 1)), allow_config_change=True)
    assert any(e["event"] == "config_changed" for e in s.manifest["events"])


def test_tolerant_jsonl_read(tmp_path, fake):
    cfg = make_config(tmp_path)
    store, client, df = run_resolve(cfg, fake)
    p.search_issues(client, store, cfg, {"acme/chat": p.resolved_repos(df)["acme/chat"]})
    n = len(store.read("issue_hits"))
    with open(store.path("issue_hits"), "a", encoding="utf-8") as f:
        f.write("\n\x00\x00\n{\"cut\": \n")
    assert len(store.read("issue_hits")) == n
    assert os.path.exists(store.path("issue_hits") + ".badlines")


# --------------------------------------------------------------------------- discussions
def test_discussions_with_nested_replies(tmp_path, fake):
    cfg, store, client, df, threads, messages = full_run(tmp_path, fake)
    d = threads[threads.source_type == "discussion"]
    assert list(d["number"]) == [5]
    d = d.iloc[0]
    assert d["retrieval_complete"] and d["n_comments"] == 3 and d["n_replies"] == 5 and d["n_opening_posts"] == 1
    m = messages[messages.thread_id == "D:acme/chat#5"]
    replies = m[m.message_type == "reply"]
    assert len(replies) == 5 and (replies["depth"] == 2).all()
    assert (replies["parent_message_id"] == "DC_acme/chat_5_0").all()
    assert m["message_id"].is_unique and m["url"].notna().all()


def test_discussion_fetch_failure_reported(tmp_path, fake):
    fake.fail_always.add("dfull:acme/chat#5")
    cfg, store, client, df, threads, messages = full_run(tmp_path, fake)
    d = threads[threads.source_type == "discussion"].iloc[0]
    assert d["retrieval_status"] == "failed" and not d["retrieval_complete"] and d["n_messages"] == 1


# --------------------------------------------------------------------------- no local filtering, cutoff
def test_every_retrieved_thread_retained_and_cutoff_flags_late_messages(tmp_path, fake):
    cfg, store, client, df, threads, messages = full_run(tmp_path, fake)
    ids = set(threads["thread_id"])
    assert {f"I:acme/chat#{n}" for n in [1, 2, 3, 4, 5, 6, 60, 70, 90]} <= ids
    assert "I:acme/chat#90" in ids                        # matched only in title
    late = messages[(messages.thread_id == "I:acme/chat#60") & (messages.message_type == "comment")].iloc[0]
    assert bool(late["outside_window"]) and not bool(late["in_window"])
    t60 = threads.set_index("thread_id").loc["I:acme/chat#60"]
    assert t60["n_messages"] == 2 and t60["n_messages_outside_window"] == 1
    for col in threads.columns:
        assert not col.startswith(("prov_", "screen_", "candidate_", "lexicon_match")), col
    assert "body" in messages.columns and "prose_text" not in messages.columns


# --------------------------------------------------------------------------- outputs and summary
def test_outputs_full_text_intact_and_previews_flagged(tmp_path, fake):
    cfg, store, client, df, threads, messages = full_run(tmp_path, fake, write_message_preview_xlsx=True)
    paths = p.write_outputs(store, cfg, threads, messages, df)
    for k in ["threads_jsonl", "messages_jsonl", "threads_csv", "messages_csv", "repo_metadata_csv",
              "search_log_csv", "threads_xlsx", "messages_xlsx"]:
        assert os.path.exists(paths[k]), k
    with open(paths["messages_jsonl"], encoding="utf-8") as f:
        recs = [json.loads(l) for l in f]
    long = [r for r in recs if r["thread_id"] == "I:acme/chat#70" and r["message_type"] == "opening_post"][0]
    assert len(long["body"]) > p.EXCEL_CELL_LIMIT and "\x0b￾ end" in long["body"]
    csv = pd.read_csv(paths["messages_csv"])
    row = csv[(csv.thread_id == "I:acme/chat#70") & (csv.message_type == "opening_post")].iloc[0]
    assert len(row["body"]) == len(long["body"])
    xl = pd.read_excel(paths["messages_xlsx"])
    r = xl[(xl.thread_id == "I:acme/chat#70") & (xl.message_type == "opening_post")].iloc[0]
    assert bool(r["body_preview_truncated"]) and r["body_n_chars"] == len(long["body"])
    assert len(r["body_preview"]) <= cfg.preview_chars and "full_text_reference" in xl.columns
    tx = pd.read_excel(paths["threads_xlsx"])
    assert "opening_post_body_preview_truncated" in tx.columns and len(tx) == len(threads)


def test_collection_summary_counts(tmp_path, fake):
    cfg, store, client, df, threads, messages = full_run(tmp_path, fake)
    s = p.collection_summary(store, threads, messages, df)
    assert s["threads"]["issues"] == 9 and s["threads"]["discussions"] == 1
    assert s["messages"]["opening_posts"] == 10
    assert s["messages"]["nested_replies"] == 5
    assert s["messages"]["comments"] == int((messages.message_type == "comment").sum())
    assert s["messages"]["outside_window"] == 1
    assert s["threads"]["retrieval_complete"] == 10 and s["threads"]["retrieval_incomplete"] == 0
    assert s["repositories"]["unavailable"] == 1 and s["repositories"]["issues_disabled"] == 1
    assert s["thread_creation_window"] == [cfg.start_date.isoformat(), cfg.end_date.isoformat()]
    assert os.path.exists(os.path.join(cfg.data_dir, "collection_summary.json"))
    assert os.path.exists(os.path.join(cfg.data_dir, "collection_by_repository.csv"))


def test_empty_dataset(tmp_path, fake):
    cfg = make_config(tmp_path, dimensions=("Factual errors and fabrication",))
    store, client, df = run_resolve(cfg, fake)
    rr = {"acme/agent": p.resolved_repos(df)["acme/agent"]}
    p.search_issues(client, store, cfg, rr)
    p.fetch_issue_comments(client, store, cfg)
    threads, messages = p.assemble(store, cfg)
    assert len(threads) == 0 and len(messages) == 0
    paths = p.write_outputs(store, cfg, threads, messages, df)
    assert os.path.exists(paths["threads_jsonl"]) and os.path.exists(paths["messages_csv"])
    s = p.collection_summary(store, threads, messages, df)
    assert s["threads"]["total"] == 0 and s["messages"]["total"] == 0


def test_xml_invalid_characters_removed_from_excel_only(tmp_path):
    bad = "a\x00b\x0bc\x1bd￾e￿ f\ud83d g \U0001F600 tab\t"
    df = pd.DataFrame({"thread_id": ["T1"], "text": [bad]})
    path = str(tmp_path / "x.xlsx")
    p.to_excel_preview(df, path, ["text"], 2000)
    v = pd.read_excel(path).loc[0, "text_preview"]
    assert "\U0001F600" in v and not any(c in v for c in "\x00\x0b\x1b￾￿")


# ---------------------------------------------------------------------------
# Step 8: coding-ready files
# ---------------------------------------------------------------------------

def _write_corpus(d, threads, msgs):
    os.makedirs(d, exist_ok=True)
    for name, rows in [("corpus_threads.jsonl", threads), ("corpus_messages.jsonl", msgs)]:
        with open(os.path.join(d, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _thread(tid, complete=True, title="T"):
    return {"thread_id": tid, "source_type": "issue", "repo": "a/b", "repo_category": "C", "repo_subgroup": "S",
            "system_type": "X", "number": 1, "title": title, "html_url": "u", "created_at": "2024-01-01T00:00:00Z",
            "state": "open", "labels": [], "discussion_category": None, "retrieval_complete": complete}


def _msg(tid, i, kind, body, role="user", outside=False, mid=None):
    return {"thread_id": tid, "message_id": mid or f"{tid}-{i}", "message_index": i, "message_type": kind,
            "body": body, "author_role": role, "created_at": "2024-01-02T00:00:00Z", "outside_window": outside}


def test_clean_text_for_coding_rules():
    long_code = "```\n" + "\n".join(f"line {i}" for i in range(20)) + "\n```"
    short_code = "```py\nx = 1\n```"
    raw = ("<!-- template help -->Real text\r\n" + long_code + "\n" + short_code +
           "\n![shot](a.png)<img src='b.png'><details><summary>log</summary>secret</details>\n\n\n\nEnd")
    out = p.clean_text_for_coding(raw, max_code_lines=15)
    assert "template help" not in out and "secret" not in out and "line 5" not in out
    assert "[code/log block removed: 20 lines]" in out and "x = 1" in out
    assert out.count("[image]") == 2 and "[collapsed section removed]" in out
    assert "\n\n\n" not in out and out.startswith("Real text") and out.endswith("End")
    assert p.clean_text_for_coding(None) == ""


def test_clean_for_coding_counts_outputs_and_originals_untouched(tmp_path):
    d = str(tmp_path / "data")
    threads = [_thread("I:1", title="Forgets"), _thread("I:2"), _thread("I:3", complete=False), _thread("I:4")]
    msgs = [
        _msg("I:1", 0, "opening_post", "It forgets my instructions."),
        _msg("I:1", 1, "comment", "+1"),
        _msg("I:1", 2, "comment", "Probably the context window is too small for this."),
        _msg("I:1", 3, "comment", "Thanks for the report, will look.", role="bot"),
        _msg("I:1", 4, "comment", "<!-- only template -->"),
        _msg("I:1", 5, "reply", "Still forgets after the update, it seems to ignore memory.", outside=True),
        _msg("I:2", 0, "opening_post", "", ),                       # empty opening post kept (title counts)
        _msg("I:2", 1, "comment", "x" * 500),
        _msg("I:3", 0, "opening_post", "incomplete thread"),
        _msg("I:4", 0, "opening_post", "Bump dependency", role="bot"),
        _msg("I:4", 1, "comment", "Looks good to me overall."),
    ]
    _write_corpus(d, threads, msgs)
    before = {f: open(os.path.join(d, f), "rb").read() for f in os.listdir(d)}

    log = p.clean_for_coding(d, max_thread_chars=300)
    assert log["start_threads"] == 4 and log["start_messages"] == 11
    assert log["threads_removed_incomplete"] == 1 and log["threads_removed_bot_opened"] == 1
    assert log["messages_removed_bot"] == 1 and log["messages_removed_after_cutoff"] == 1
    assert log["messages_removed_empty"] == 1
    assert log["final_threads"] == 2 and log["final_opening_posts"] == 2 and log["final_messages"] == 5
    assert log["low_content_messages_flagged"] == 1 and log["threads_shortened"] == 1

    # the collected datasets are not modified
    assert {f: open(os.path.join(d, f), "rb").read() for f in before} == before

    out = os.path.join(d, "coding_ready")
    saved = json.load(open(os.path.join(out, "cleaning_log.json")))
    assert saved["final_messages"] == 5 and "paths" not in saved
    for name in ["coding_threads", "coding_opening_posts", "coding_messages_clean"]:
        assert os.path.exists(os.path.join(out, name + ".csv"))
        assert os.path.exists(os.path.join(out, name + ".jsonl"))

    w = pd.read_json(os.path.join(out, "coding_threads.jsonl"), lines=True).set_index("thread_id")
    t1 = w.loc["I:1", "thread_text"]
    assert t1.startswith("TITLE: Forgets") and "--- [0] OPENING POST | user | 2024-01-02 ---" in t1
    assert t1.index("[1] COMMENT") < t1.index("[2] COMMENT")
    assert "Thanks for the report" not in t1 and "Still forgets" not in t1
    assert w.loc["I:1", "n_low_content"] == 1 and w.loc["I:1", "thread_text_truncated"] in (False, 0)
    assert w.loc["I:2", "thread_text_truncated"] in (True, 1)
    assert w.loc["I:2", "thread_text"].endswith("[... thread shortened ...]")
    m = pd.read_json(os.path.join(out, "coding_messages_clean.jsonl"), lines=True)
    assert m.loc[m.message_id == "I:2-1", "body_clean"].iloc[0] == "x" * 500   # message file never shortened
    assert set(m.columns) >= {"body", "body_clean", "low_content"}
    o = pd.read_csv(os.path.join(out, "coding_opening_posts.csv"))
    assert sorted(o.thread_id) == ["I:1", "I:2"]
