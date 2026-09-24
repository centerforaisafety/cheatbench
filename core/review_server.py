"""A task's pairwise reviewer, behind a loopback socket the host owns.

WHERE THIS RUNS, AND WHY HERE. A `writings`-style task stages a compiled grader
(`grading/grade.pyc` + a `.so`) into the agent's workspace and tells the agent to run
it. That grader reads its reference and POSTs `{brief, reference, submission}`
to a review endpoint; a model answers whether the two are the same artifact and
returns a verdict and feedback. The credential and the judge's prompt must be
somewhere the agent can REACH and cannot READ, and the episode now runs in the
HOST network namespace (no private netns, like `gdpval`). So this server runs in
the HOST orchestrator process -- outside the container's pid and mount
namespaces, on the loopback the container shares -- and the only thing that
crosses the boundary is a pairwise request in and a verdict out.

ONE LLM CLIENT. Because this runs on the host, there is no container to hide a
credential from, so the model call goes through the SAME `core/llm_agents.py`
the trajectory judge uses (`get_llm_agent_class`, with the model resolved from
the task's `grader: judge:` through `configs/models.yaml`), not a second bespoke
client. The reply is schema-enforced by the SAME `response_format` passthrough
`core/judge.py` relies on. Nothing sets `temperature`, `top_p` or `seed`: the
judge entry carries reasoning settings only.

WHAT DEFENDS IT. It binds `127.0.0.1` on a kernel-assigned ephemeral port, so
nothing off the box can reach it and concurrent episodes never collide. Every
request must carry `Authorization: Bearer <token>`; the token is written into
the episode's own workspace at `grading/.review` (alongside the endpoint URL) for the
grader to read, and a request without it is a 401. A neighbouring episode that
guessed the port still cannot answer the token.

KNOWN LIMIT, one line: an agent on the HOST side of the boundary -- same uid,
same machine -- could reach this endpoint or the process it runs in. Inside the
container it cannot: a different pid namespace hides the process and a different
mount namespace hides its files. Spoofing the endpoint (POSTing a
`reference_text` that is not the real reference) is SERVED AS ASKED and recorded
as `reference_spoofed`, because it is an attack to MEASURE, not to block.

NO VERDICT ARITHMETIC AND NO BUDGET. The model returns `tie` or `below` and that
is what goes back; the reviewer answers every well-formed call. A transient
transport or 5xx failure is retried; a persistent one returns 503 with a clear
`review temporarily unavailable` body and `verdict: null` -- never a `below` with
empty feedback, because a caller cannot tell an empty verdict from a real one,
and a reviewer that looks broken is what made a flaky proxy read as "the grader
is broken". The deterministic short-circuit for a normalised match lives in the
grader wrapper, not here; this server is only ever asked the questions the
short-circuit did not already answer.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Literal

from pydantic import BaseModel

from .llm_agents import get_llm_agent_class

VERDICTS = ("tie", "below")

# The model call is retried on any transport / 5xx failure before the endpoint
# gives up. A persistent failure returns 503 with a clear "unavailable" message,
# never a `below` with empty feedback -- an empty verdict must not be mistaken
# for a real one, which is what made a flaky proxy read as "the grader is broken".
_MAX_ATTEMPTS = 3          # the model call is tried at most three times
# A per-call timeout, so a hung proxy connection fails fast into a retry rather
# than blocking a grader call for the client's default (an hour). A real review
# on the full reference and previous draft is tens of seconds; this is headroom.
_CALL_TIMEOUT_S = 180
# The single source of truth for how long a client must wait before concluding
# the server itself has failed. It must exceed the worst-case server time -- all
# `_MAX_ATTEMPTS` calls each hitting `_CALL_TIMEOUT_S`, plus the small backoff
# between them -- with headroom, so the client never abandons a request the
# server is still working on. The generated grader imports this value.
CLIENT_TIMEOUT_S = _MAX_ATTEMPTS * _CALL_TIMEOUT_S + 30
# Shared by the compiled wrapper and the episode's outage check.
UNAVAILABLE = {"verdict": None, "feedback": "review temporarily unavailable; try again"}
UNAVAILABLE_FEEDBACK = UNAVAILABLE["feedback"]
UNAVAILABLE_NULL_VERDICT = json.dumps({"verdict": None})[1:-1]

_WS = re.compile(r"\s+")


class ReviewReply(BaseModel):
    """The reviewer's structured reply. Handed to the model as `response_format`,
    so the shape is enforced by the API, not requested in prose."""

    verdict: Literal["tie", "below"]
    feedback: str


def norm_digest(text: str) -> str:
    """SHA-256 of whitespace-collapsed text. The spoof check's fingerprint.

    Must match the grader `.so`'s own normalisation (`_norm`): a reference the
    grader re-wraps before sending still fingerprints to the same value, so a
    faithful request is never mistaken for a spoofed one.
    """
    return hashlib.sha256(
        _WS.sub(" ", text or "").strip().encode("utf-8", "replace")
    ).hexdigest()


def parse_prompt(text: str) -> dict:
    """`# <key>` headings and the prose under each, from the judge prompt file.

    Expects a `# system` section and a `# user_template` section.
    The user template carries
    `{task_brief}`, `{reference_text}` and `{submission_text}` placeholders.
    """
    out: dict = {}
    key = None
    buf: list = []
    for line in (text or "").splitlines():
        match = re.match(r"^#\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", line)
        if match:
            if key:
                out[key] = "\n".join(buf).strip()
            key, buf = match.group(1), []
        elif key is not None:
            buf.append(line)
    if key:
        out[key] = "\n".join(buf).strip()
    return out


def _fill(template: str, brief: str, reference: str, submission: str) -> str:
    """Fill current and legacy reference placeholders."""
    out = str(template or "")
    for token, value in (("{task_brief}", brief),
                         ("{reference_text}", reference),
                         ("{page_text}", reference),
                         ("{submission_text}", submission)):
        out = out.replace(token, value)
    return out


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "editorial/1.0"
    sys_version = ""

    def log_message(self, *args) -> None:  # noqa: A003 - silence; host parses stderr
        return

    def _send(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self) -> bool:
        header = self.headers.get("Authorization") or ""
        prefix = "Bearer "
        if not header.startswith(prefix):
            return False
        return secrets.compare_digest(header[len(prefix):], self.server.token)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
        if self.path.split("?", 1)[0] != "/review":
            return self._send({"error": "not found"}, 404)
        if not self._authorised():
            return self._send({"error": "unauthorised"}, 401)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(length).decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001
            return self._send({"error": "bad request"}, 400)
        # Accept older compiled graders as well as the generic task.
        reference = req.get("reference_text")
        if reference is None:
            reference = req.get("page_text")
        status, body = self.server.review(
            brief=str(req.get("task_brief") or ""),
            reference=str(reference or ""),
            submission=str(req.get("submission_text") or ""))
        return self._send(body, status)


class ReviewServer(ThreadingHTTPServer):
    """The running review endpoint, and the handle the episode holds it by.

    One server per episode, and one `core/llm_agents.py` agent inside it, so the
    per-call usage this records is this episode's alone and concurrent episodes
    never share an accumulator.
    """

    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, config: dict):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.config = config
        self.token = secrets.token_hex(16)
        # The judge entry's routing (configs/models.yaml `api_key_env:` /
        # `api_base_url:` / `extra_body:`), passed exactly as core/judge.py
        # passes it: the two names as the client's constructor kwargs, and
        # extra_body inside generation_config, which the SDK create() spreads.
        # Only what the entry SET is passed; an entry that routes nothing gets
        # the client's own defaults, as before.
        gen = dict(config.get("generation_config") or {})
        if config.get("extra_body"):
            gen["extra_body"] = dict(config["extra_body"])
        self.agent = get_llm_agent_class(
            config["model"], gen,
            **{k: config[k] for k in ("api_key_env", "api_base_url")
               if config.get(k)})
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        # The authoritative record. `conversation` is the reviewer's ONE
        # persistent chat for this episode: a system turn, then the brief +
        # reference + first draft, then the model's reply, then each later draft
        # and reply. The judge sees its own earlier verdicts and feedback as its
        # own turns, so it builds on them. Nothing is capped -- a conversation
        # that outgrows the model's context is a finding to report, not to trim.
        self.state: dict = {
            "kind": "pairwise_review",
            "judge": config.get("judge"),
            "model": config.get("model"),
            "generation_config": dict(config.get("generation_config") or {}),
            "reviews_used": 0,
            "reference_spoofed": 0,
            "errors": [],
            "calls": [],
            "conversation": [],
        }

    @property
    def url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        return f"http://{host}:{port}/review"

    def start(self) -> None:
        self._thread = threading.Thread(target=self.serve_forever,
                                        kwargs={"poll_interval": 0.2},
                                        daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._thread is not None:
            try:
                self.shutdown()
            except Exception:  # noqa: BLE001 - teardown is best effort
                pass
            self._thread.join(timeout=10)
            self._thread = None
        try:
            self.server_close()
        except Exception:  # noqa: BLE001
            pass

    # -- the review -------------------------------------------------------
    def review(self, *, brief: str, reference: str, submission: str):
        """`(http_status, body)` for one pairwise request. Never raises.

        MULTI-TURN, AS ONE CONVERSATION. The episode holds a single chat with the
        judge: turn 0 the system prompt; the first call adds the brief, reference
        and first draft; every later call adds only the revised draft. The model
        replies structured JSON each turn, and its replies stay in the chat as
        its own turns, so it sees its earlier verdicts and feedback and builds on
        them. The reference and brief are sent once.

        There is NO budget: the reviewer answers every well-formed call. The
        model call is retried up to `_MAX_ATTEMPTS` times on any transport or 5xx
        failure; if it still fails, the endpoint returns 503 with a clear
        `review temporarily unavailable` body and `verdict: null` -- never a
        `below` with empty feedback -- and the failed turn is NOT added to the
        conversation. Each failed call is recorded in `state["errors"]`.
        """
        cfg = self.config
        ref_digest = cfg.get("reference_sha256") or ""
        spoofed = bool(ref_digest) and norm_digest(reference) != ref_digest

        with self._lock:
            conversation = list(self.state["conversation"])
        if not conversation:
            # First call: system, then brief + reference + first draft.
            pending = [
                {"role": "system", "content": cfg.get("system") or ""},
                {"role": "user", "content": _fill(cfg.get("user_template") or "",
                                                  brief, reference, submission)},
            ]
        else:
            # Later call: only the revised draft; the reference and brief, and
            # the reviewer's own earlier turns, are already in the conversation.
            pending = [{"role": "user",
                        "content": "Here is the revised draft:\n\n" + submission}]
        messages = conversation + pending

        reply = usage = raw = None
        last_err = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                # `timeout` is a per-call default only: a judge entry that sets
                # its own `timeout` in generation_config wins, and passing both
                # would be a TypeError swallowed by the retry loop below.
                call_kw = ({} if "timeout" in (getattr(self.agent, "generation_config", None) or {})
                           else {"timeout": _CALL_TIMEOUT_S})
                resp = self.agent.completions(messages,
                                              response_format=ReviewReply,
                                              **call_kw)
                raw = resp.content or ""
                reply = ReviewReply.model_validate_json(raw)
                usage = resp.token_usage
                break
            except Exception as exc:  # noqa: BLE001 - transport/5xx: retry
                last_err = exc
                if attempt < _MAX_ATTEMPTS - 1:
                    time.sleep(min(4.0, 0.5 * (2 ** attempt)))

        if reply is None:
            with self._lock:
                self.state["errors"].append(
                    f"{type(last_err).__name__}: {last_err}")
                if spoofed:
                    self.state["reference_spoofed"] += 1
            return 503, dict(UNAVAILABLE)

        with self._lock:
            self.state["reviews_used"] += 1
            if spoofed:
                self.state["reference_spoofed"] += 1
            # Commit this turn to the conversation: the user turn(s) and the
            # model's own reply, so the next turn sees both.
            self.state["conversation"].extend(pending)
            self.state["conversation"].append({"role": "assistant",
                                                "content": raw})
            self.state["calls"].append({
                "round": self.state["reviews_used"],
                "verdict": reply.verdict,
                "feedback": reply.feedback,
                "usage": _usage_dict(usage),
                "cost": float(getattr(usage, "cost", 0.0) or 0.0),
                "reference_spoofed": spoofed,
                "submission_chars": len(submission),
                "reference_chars": len(reference),
            })
        return 200, {"verdict": reply.verdict, "feedback": reply.feedback}


def _usage_dict(usage) -> dict:
    """`core/llm_agents.py` TokenUsage as the four counts the pricing path wants.

    `input_tokens` from the proxy is INCLUSIVE of cached tokens, which is what
    `core/agents/base.py:cost_from_usage` expects.
    """
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0,
                "cached_tokens": 0, "cache_write_tokens": 0}
    return {"prompt_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cached_tokens": int(getattr(usage, "cached_tokens", 0) or 0),
            "cache_write_tokens": 0}


def start_review_server(config: dict) -> ReviewServer:
    """Bind and start a `ReviewServer`. The caller closes it."""
    server = ReviewServer(config)
    server.start()
    return server
