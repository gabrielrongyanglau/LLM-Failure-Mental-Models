"""
A scripted in-memory stand-in for the GitHub REST and GraphQL APIs.

It implements just enough behaviour to test the pipeline offline: repository
metadata with a redirect, a 404 and a repository with issues disabled; issue
search with real term matching over synthetic issues (title, body, comments),
ordering, pagination, total_count and the 1,000-result cap (configurable);
issue comments with pagination; discussion search; discussions with paginated
comments and nested, paginated replies; and injectable failures.

Nothing here contacts the network.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse


class FakeResponse:
    def __init__(self, status: int, payload: Any, headers: Optional[Dict[str, str]] = None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = json.dumps(payload)[:500] if not isinstance(payload, str) else payload

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload


def iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeGitHub:
    def __init__(self, result_cap: int = 1000):
        self.repos: Dict[str, Dict[str, Any]] = {}
        self.redirects: Dict[str, str] = {}
        self.missing: set = set()
        self.issues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)       # canonical repo -> issues
        self.comments: Dict[str, List[Dict[str, Any]]] = defaultdict(list)     # "repo#n" -> comments
        self.discussions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # canonical repo -> discussions
        self.result_cap = result_cap
        self.fail_once: Dict[str, int] = {}     # key -> remaining failures (HTTP 500)
        self.fail_always: set = set()           # keys -> HTTP 500 forever
        self.calls: List[Dict[str, Any]] = []
        self.rate_limit_once = False
        self._id = 1000

    # ------------------------------------------------------------ fixtures
    def nid(self) -> int:
        self._id += 1
        return self._id

    def add_repo(self, path: str, has_issues=True, has_discussions=False, redirect_from: Optional[str] = None):
        owner, name = path.split("/")
        rid = self.nid()
        self.repos[path] = {"id": rid, "node_id": f"R_{rid}", "full_name": path, "name": name,
                            "owner": {"login": owner}, "description": f"{name} description",
                            "created_at": "2023-01-01T00:00:00Z", "pushed_at": "2026-09-01T00:00:00Z",
                            "stargazers_count": 10, "forks_count": 1, "open_issues_count": 3,
                            "has_issues": has_issues, "has_discussions": has_discussions,
                            "archived": False, "disabled": False, "default_branch": "main", "homepage": None}
        if redirect_from:
            self.redirects[redirect_from] = path

    def add_issue(self, repo: str, number: int, title: str, body: str, created: datetime,
                  author="user1", assoc="NONE", atype="User", comments: Optional[List[Dict[str, Any]]] = None):
        comments = comments or []
        it = {"number": number, "id": self.nid(), "node_id": f"I_{repo}_{number}", "title": title, "body": body,
              "user": {"login": author, "id": abs(hash(author)) % 100000, "type": atype},
              "author_association": assoc, "state": "open", "state_reason": None,
              "created_at": iso(created), "updated_at": iso(created), "closed_at": None,
              "comments": len(comments), "reactions": {"total_count": 0}, "labels": [{"name": "bug"}],
              "html_url": f"https://github.com/{repo}/issues/{number}",
              "comments_url": f"https://api.github.com/repos/{repo}/issues/{number}/comments", "locked": False}
        self.issues[repo].append(it)
        for i, c in enumerate(comments):
            cc = {"id": self.nid(), "node_id": f"IC_{repo}_{number}_{i}", "body": c["body"],
                  "user": {"login": c.get("author", "user2"), "id": abs(hash(c.get("author", "user2"))) % 100000,
                           "type": c.get("type", "User")},
                  "author_association": c.get("assoc", "NONE"),
                  "created_at": iso(c.get("created", created + timedelta(hours=i + 1))),
                  "updated_at": iso(c.get("created", created + timedelta(hours=i + 1))),
                  "html_url": f"https://github.com/{repo}/issues/{number}#issuecomment-{i}",
                  "reactions": {"total_count": 0}}
            self.comments[f"{repo}#{number}"].append(cc)

    def add_discussion(self, repo: str, number: int, title: str, body: str, created: datetime,
                       comments: Optional[List[Dict[str, Any]]] = None, author="duser", assoc="NONE"):
        """comments: list of {body, author, assoc, replies: [{body, author, assoc}]}"""
        comments = comments or []
        nodes = []
        for i, c in enumerate(comments):
            cid = f"DC_{repo}_{number}_{i}"
            replies = []
            for j, r in enumerate(c.get("replies", [])):
                replies.append({"id": f"DR_{repo}_{number}_{i}_{j}", "databaseId": self.nid(), "body": r["body"],
                                "createdAt": iso(created + timedelta(hours=i + 1, minutes=j + 1)),
                                "updatedAt": iso(created + timedelta(hours=i + 1, minutes=j + 1)), "lastEditedAt": None,
                                "url": f"https://github.com/{repo}/discussions/{number}#discussioncomment-{i}-{j}",
                                "isAnswer": False, "upvoteCount": 0,
                                "author": {"login": r.get("author", "ruser"), "__typename": r.get("type", "User"), "databaseId": 7},
                                "authorAssociation": r.get("assoc", "NONE"), "replyTo": {"id": cid}})
            nodes.append({"id": cid, "databaseId": self.nid(), "body": c["body"],
                          "createdAt": iso(created + timedelta(hours=i + 1)),
                          "updatedAt": iso(created + timedelta(hours=i + 1)), "lastEditedAt": None,
                          "url": f"https://github.com/{repo}/discussions/{number}#discussioncomment-{i}",
                          "isAnswer": c.get("is_answer", False), "upvoteCount": 0,
                          "author": {"login": c.get("author", "cuser"), "__typename": c.get("type", "User"), "databaseId": 8},
                          "authorAssociation": c.get("assoc", "NONE"), "replyTo": None, "_replies": replies})
        self.discussions[repo].append({
            "id": f"D_{repo}_{number}", "databaseId": self.nid(), "number": number, "title": title, "body": body,
            "bodyText": body, "createdAt": iso(created), "updatedAt": iso(created), "lastEditedAt": None,
            "url": f"https://github.com/{repo}/discussions/{number}", "isAnswered": False, "locked": False,
            "upvoteCount": 1, "author": {"login": author, "__typename": "User", "databaseId": 9},
            "authorAssociation": assoc, "category": {"name": "Q&A", "slug": "q-a", "isAnswerable": True},
            "labels": {"nodes": []}, "answer": None, "answerChosenAt": None, "_comments": nodes})

    # ------------------------------------------------------------ helpers
    def _fail_check(self, key: str) -> Optional[FakeResponse]:
        if key in self.fail_always:
            return FakeResponse(500, {"message": "boom"})
        n = self.fail_once.get(key, 0)
        if n > 0:
            self.fail_once[key] = n - 1
            return FakeResponse(500, {"message": "transient"})
        return None

    @staticmethod
    def _parse_query(q: str):
        repo = re.search(r"repo:(\S+)", q).group(1)
        win = re.search(r"created:(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})", q)
        start = date.fromisoformat(win.group(1)); end = date.fromisoformat(win.group(2))
        terms = [t.lower() for t in re.findall(r'"([^"]+)"', q)]
        return repo, start, end, terms

    def _issue_text(self, repo: str, it: Dict[str, Any]) -> str:
        parts = [it["title"] or "", it["body"] or ""] + [c["body"] or "" for c in self.comments.get(f"{repo}#{it['number']}", [])]
        return "\n".join(parts).lower()

    def _matching_issues(self, q: str) -> List[Dict[str, Any]]:
        repo, start, end, terms = self._parse_query(q)
        out = []
        for it in self.issues.get(repo, []):
            d = date.fromisoformat(it["created_at"][:10])
            if not (start <= d <= end):
                continue
            text = self._issue_text(repo, it)
            if not terms or any(t in text for t in terms):
                out.append(it)
        return out

    # ------------------------------------------------------------ REST
    def get(self, url: str, headers=None, params=None, timeout=None):
        params = params or {}
        self.calls.append({"method": "GET", "url": url, "params": dict(params)})
        if self.rate_limit_once:
            self.rate_limit_once = False
            return FakeResponse(403, {"message": "rate"}, {"x-ratelimit-remaining": "0",
                                                           "x-ratelimit-reset": str(int(datetime.now().timestamp()) + 1)})
        path = urlparse(url).path
        m = re.match(r"^/repos/([^/]+/[^/]+)$", path)
        if m:
            p = m.group(1)
            if p in self.missing:
                return FakeResponse(404, {"message": "Not Found"})
            p = self.redirects.get(p, p)
            if p not in self.repos:
                return FakeResponse(404, {"message": "Not Found"})
            return FakeResponse(200, self.repos[p])
        if path == "/search/issues":
            q = params["q"]
            if q.count(" OR ") > 5 or len(q) > 256:
                return FakeResponse(422, {"message": "Validation Failed"})
            fail = self._fail_check(f"search:{q}:{params.get('order','asc')}:{params.get('page',1)}")
            if fail:
                return fail
            items = self._matching_issues(q)
            items = sorted(items, key=lambda x: x["created_at"], reverse=(params.get("order") == "desc"))
            total = len(items)
            per_page = int(params.get("per_page", 30)); page = int(params.get("page", 1))
            capped = items[: self.result_cap]
            chunk = capped[(page - 1) * per_page: page * per_page]
            return FakeResponse(200, {"total_count": total, "incomplete_results": False, "items": chunk})
        m = re.match(r"^/repos/([^/]+/[^/]+)/issues/(\d+)/comments$", path)
        if m:
            key = f"{m.group(1)}#{m.group(2)}"
            page = int(params.get("page", 1)); per_page = int(params.get("per_page", 30))
            fail = self._fail_check(f"comments:{key}:{page}")
            if fail:
                return fail
            cs = self.comments.get(key, [])
            return FakeResponse(200, cs[(page - 1) * per_page: page * per_page])
        return FakeResponse(404, {"message": f"unhandled {path}"})

    # ------------------------------------------------------------ GraphQL
    def post(self, url: str, headers=None, json=None, timeout=None):
        q = json["query"]; v = json.get("variables", {})
        self.calls.append({"method": "POST", "url": url, "variables": dict(v)})
        rl = {"cost": 1, "remaining": 4999, "resetAt": iso(datetime.now(timezone.utc) + timedelta(hours=1))}
        if "type: DISCUSSION" in q:
            repo, start, end, terms = self._parse_query(v["q"])
            fail = self._fail_check(f"dsearch:{v['q']}")
            if fail:
                return fail
            hits = []
            for d in self.discussions.get(repo, []):
                dd = date.fromisoformat(d["createdAt"][:10])
                if not (start <= dd <= end):
                    continue
                text = "\n".join([d["title"], d["body"]] + [c["body"] for c in d["_comments"]] +
                                 [r["body"] for c in d["_comments"] for r in c["_replies"]]).lower()
                if any(t in text for t in terms):
                    hits.append(d)
            first = int(v["first"]); after = int(v["after"]) if v.get("after") else 0
            chunk = hits[after: after + first]
            nodes = [{k: d[k] for k in ["id", "databaseId", "number", "title", "createdAt", "updatedAt", "url",
                                        "isAnswered", "locked", "author", "authorAssociation", "category"]}
                     | {"comments": {"totalCount": len(d["_comments"])},
                        "repository": {"nameWithOwner": repo, "databaseId": self.repos[repo]["id"]}} for d in chunk]
            has_next = after + first < len(hits)
            return FakeResponse(200, {"data": {"rateLimit": rl, "search": {
                "discussionCount": len(hits), "pageInfo": {"hasNextPage": has_next, "endCursor": str(after + first) if has_next else None},
                "nodes": nodes}}})
        if "discussion(number: $number)" in q:
            repo = f"{v['owner']}/{v['name']}"
            d = next((x for x in self.discussions.get(repo, []) if x["number"] == v["number"]), None)
            fail = self._fail_check(f"dfull:{repo}#{v['number']}")
            if fail:
                return fail
            if d is None:
                return FakeResponse(200, {"data": {"rateLimit": rl, "repository": {"databaseId": 1, "nameWithOwner": repo, "discussion": None}},
                                          "errors": [{"type": "NOT_FOUND", "message": "no discussion"}]})
            first = int(v["first"]); after = int(v["after"]) if v.get("after") else 0; rf = int(v["replyFirst"])
            chunk = d["_comments"][after: after + first]
            nodes = []
            for c in chunk:
                reps = c["_replies"]
                nodes.append({k: c[k] for k in c if not k.startswith("_")} | {
                    "replies": {"totalCount": len(reps),
                                "pageInfo": {"hasNextPage": len(reps) > rf, "endCursor": str(rf) if len(reps) > rf else None},
                                "nodes": reps[:rf]}})
            has_next = after + first < len(d["_comments"])
            disc = {k: d[k] for k in d if not k.startswith("_")} | {
                "comments": {"totalCount": len(d["_comments"]),
                             "pageInfo": {"hasNextPage": has_next, "endCursor": str(after + first) if has_next else None},
                             "nodes": nodes}}
            return FakeResponse(200, {"data": {"rateLimit": rl, "repository": {"databaseId": self.repos[repo]["id"],
                                                                               "nameWithOwner": repo, "discussion": disc}}})
        if "node(id: $id)" in q:
            cid = v["id"]
            for repo, ds in self.discussions.items():
                for d in ds:
                    for c in d["_comments"]:
                        if c["id"] == cid:
                            first = int(v["first"]); after = int(v["after"]) if v.get("after") else 0
                            reps = c["_replies"]
                            chunk = reps[after: after + first]
                            has_next = after + first < len(reps)
                            return FakeResponse(200, {"data": {"rateLimit": rl, "node": {"id": cid, "replies": {
                                "totalCount": len(reps), "pageInfo": {"hasNextPage": has_next, "endCursor": str(after + first) if has_next else None},
                                "nodes": chunk}}}})
            return FakeResponse(200, {"data": {"rateLimit": rl, "node": None}})
        return FakeResponse(400, {"message": "unhandled graphql"})
