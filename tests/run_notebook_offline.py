"""
Execute every code cell of the notebook offline against FakeGitHub.

    python tests/run_notebook_offline.py LLM_Failure_GitHub_Data_Collection_v3.ipynb /tmp/nb3

Includes Step 8 (coding-ready files). The %%writefile cell is written to the run directory; requests.Session is
replaced with the fake server; the 30-repository scope is swapped for a fixture
corpus; sleeps are disabled; the optional Step 2 is switched on. The run is
executed twice to check that the second pass resumes without duplication.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import nbformat as nbf
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fake_github import FakeGitHub  # noqa: E402

NB_PATH, RUN_DIR = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.makedirs(RUN_DIR, exist_ok=True)
os.chdir(RUN_DIR)
os.environ["LLM_FAILURES_DATA_DIR"] = os.path.join(RUN_DIR, "data")
os.environ["GITHUB_TOKEN"] = "fake-token"
time.sleep = lambda s: None


def dt(y, m, d, h=12):
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def build_fake():
    g = FakeGitHub(result_cap=5)
    g.add_repo("acme/chat", has_discussions=True)
    g.add_repo("acme/agent")
    g.add_repo("acme/newname", redirect_from="acme/oldname")
    g.add_repo("acme/noissues", has_issues=False)
    g.missing.add("acme/gone")
    for i in range(1, 9):
        g.add_issue("acme/chat", i, f"Forgets context #{i}", "It forgets what I told it.", dt(2024, 1 + i, 10),
                    comments=[{"body": f"comment {k}", "author": f"u{k}"} for k in range(5)])
    g.add_issue("acme/chat", 60, "Late", "Nothing.", dt(2026, 9, 2),
                comments=[{"body": "Now it forgets everything.", "created": dt(2026, 10, 3)}])
    g.add_issue("acme/chat", 70, "Long body forgets", "It forgets. " + "y" * 40000 + "\x0b￾", dt(2024, 4, 4))
    for i in range(1, 13):
        g.add_issue("acme/agent", i, f"Ignores instructions {i}", "It ignores my instructions and loses context",
                    dt(2024, 8, 8, h=i))
    g.add_discussion("acme/chat", 5, "Why does it forget?", "The assistant forgets everything.", dt(2024, 2, 2),
                     comments=[{"body": "Context.", "replies": [{"body": f"reply {k}"} for k in range(4)]},
                               {"body": "Bigger model.", "assoc": "MEMBER"}])
    return g


FAKE = build_fake()
FIXTURE_REPOS = {
    "acme/chat": {"category": "General chat interfaces and personal assistants", "subgroup": "Chat interface",
                  "system_type": "Conversational assistants"},
    "acme/agent": {"category": "Coding assistants and action-taking agents", "subgroup": "Coding agent",
                   "system_type": "AI agents and orchestration"},
    "acme/oldname": {"category": "Document, knowledge and search assistants", "subgroup": "Knowledge assistant",
                     "system_type": "Knowledge and memory systems"},
    "acme/noissues": {"category": "Role-play and local-model environments", "subgroup": "Local model interface",
                      "system_type": "Model infrastructure"},
    "acme/gone": {"category": "Agent memory and model-access infrastructure", "subgroup": "Model access",
                  "system_type": "Model infrastructure"},
}


def run_once(label):
    nb = nbf.read(NB_PATH, as_version=4)
    ns = {"display": print, "__name__": "__main__", "FAKE": FAKE, "FIXTURE_REPOS": FIXTURE_REPOS}
    n = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code" or cell.source.startswith("!pip"):
            continue
        src = cell.source
        if src.startswith("%%writefile"):
            first, body = src.split("\n", 1)
            with open(os.path.join(RUN_DIR, first.split()[1]), "w", encoding="utf-8") as f:
                f.write(body)
            continue
        src = (src.replace("RUN_BUNDLE_CHECK = False", "RUN_BUNDLE_CHECK = True")
                  .replace('"ollama/ollama"', '"acme/chat"')
                  .replace("RUN_DRY_RUN = False", "RUN_DRY_RUN = True")
                  .replace("write_message_preview_xlsx=False,", "write_message_preview_xlsx=True, search_sleep_seconds=0, "
                           "graphql_sleep_seconds=0, comment_sleep_seconds=0, result_cap=5, near_cap_threshold=4, "
                           "per_page=3, graphql_page_size=2,"))
        if "import llm_failures_collect as p" in src:
            sys.path.insert(0, RUN_DIR)
            src = src.replace("importlib.reload(p)",
                              "importlib.reload(p); p.requests.Session = lambda: FAKE; p.REPO_INFO = FIXTURE_REPOS")
        exec(compile(src, f"{label}-cell{i}", "exec"), ns)
        n += 1
    print(f"[{label}] {n} code cells executed")
    return ns


ns1 = run_once("run1")
s1 = ns1["SUMMARY"]
ns2 = run_once("run2")
s2 = ns2["SUMMARY"]
data_dir = ns2["DATA_DIR"]
for f in ["corpus_threads.csv", "corpus_threads.jsonl", "corpus_messages.csv", "corpus_messages.jsonl",
          "repository_metadata.csv", "search_log.csv", "manifest.json", "collection_summary.json",
          "collection_by_repository.csv", "corpus_threads_preview.xlsx", "corpus_messages_preview.xlsx"]:
    assert os.path.exists(os.path.join(data_dir, f)), f
msgs = pd.read_csv(os.path.join(data_dir, "corpus_messages.csv"))
assert msgs["message_id"].is_unique
assert s1["messages"] == s2["messages"] and s1["threads"]["total"] == s2["threads"]["total"]
assert s2["messages"]["nested_replies"] == 4 and s2["messages"]["outside_window"] == 1
long = msgs[(msgs.thread_id == "I:acme/chat#70") & (msgs.message_type == "opening_post")].iloc[0]
assert len(long["body"]) > 32767
# Step 8: coding-ready files written, originals unchanged, counts consistent
c1, c2 = ns1["CLEAN_LOG"], ns2["CLEAN_LOG"]
assert {k: v for k, v in c1.items() if k not in ("generated_at", "paths")} == \
       {k: v for k, v in c2.items() if k not in ("generated_at", "paths")}
assert c2["start_messages"] == s2["messages"]["total"] and c2["final_threads"] == s2["threads"]["total"]
assert c2["messages_removed_after_cutoff"] == 1
for f in ["coding_threads.jsonl", "coding_opening_posts.jsonl", "coding_messages_clean.jsonl",
          "coding_threads.csv", "cleaning_log.json"]:
    assert os.path.exists(os.path.join(data_dir, "coding_ready", f)), f
assert len(pd.read_csv(os.path.join(data_dir, "corpus_messages.csv"))) == s2["messages"]["total"]
print(json.dumps({"threads": s2["threads"], "messages": s2["messages"], "search": s2["search"],
                  "cleaning": {k: v for k, v in c2.items() if k not in ("paths", "generated_at")}}, indent=1))
print("NOTEBOOK OFFLINE RUN PASSED (two passes, resumed without duplication)")
