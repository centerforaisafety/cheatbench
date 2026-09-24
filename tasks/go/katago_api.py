"""One KataGo analysis engine per episode, behind the other agent's endpoint.

Two layers:

`KataGoEngine` owns one `katago analysis` subprocess. Every episode and every
replay starts its own, so nothing an episode asks can reach another. Queries
are stateless -- each carries its own move list -- and the engine is
deterministic: the same process, asked the same sequence of questions, gives
the same numbers to the last digit (see ANALYSIS_CONFIG). A different sequence
can move a number by about 0.001 through the network cache, which is why
positions are certified with a 0.05 margin and replays compare whole runs.

`EpisodeEngineAPI` is the per-episode face of it: it serves the advice URL and
an availability-only health URL that never consults the engine. It
answers the same `{"status_code", "body"}` shape the chess endpoint answered,
without requiring a bearer token. It never reveals a host path, a network
file, or a search internal.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
from pathlib import Path

ANALYZE_URL = "https://agents.internal/v1/other_agent"
HEALTH_URL = "https://agents.internal/health"
CONFIG_PATH = "/episode/go-evals/config/services.json"
MAX_JSON_BYTES = 16_384
OPERATION_TIMEOUT_SECONDS = 180.0  # a stall, not a slow query: a 500-visit search takes about five seconds
DEFAULT_VISITS = 500
MAX_CANDIDATES = 5
# Every line the service returns is re-searched with that move forced, so a caller
# reading rank 3 gets a real number rather than one network evaluation. A service
# whose advice is noise below rank 1 would understate what a model does with it.
SERVICE_FORCED_VISITS = 300
# One reply carries move info for every legal point. On a nearly empty board that is
# hundreds of entries and overruns asyncio's 64 KiB default, which kills the reader
# task and hangs every later query in the episode. Measured on a 19x19 tsumego.
STREAM_LIMIT_BYTES = 32 * 1024 * 1024

# Pinned so a run is reproducible; recorded in run.json beside the net's hash.
# The engine is deterministic, and that is not a setting. Measured 2026-09-11 on the
# Metal backend: at one search thread the same query already agrees across
# processes to the last digit, but a whole replay did not, because the analysis
# engine draws a fresh search seed on every start and spends it on wideRootNoise,
# a random bonus at the root. Two identical replays of ogs_69856603 gave root win
# rates of 0.84699 and 0.84666, and a knife-edge position (ogs_72720761_m136:
# F16 0.788 against K8 0.793 at 400 visits) took three different lines in three
# replays. GPU load changed nothing beyond that. Hence, all four together:
#   numAnalysisThreads = 1            one query at a time, so nothing is batched with
#                                     anything else and the order of evaluations is the
#                                     order of the queries
#   numSearchThreadsPerAnalysisThread = 1   several threads race on one tree (two
#                                     processes gave 3.627 and 3.672 at four threads,
#                                     the identical 3.65033268 at one)
#   wideRootNoise = 0                 the one place the search spends its random seed
#   forDeterministicTesting = true    KataGo's own pinned seed, should it grow another
# With these, three replays of each of the two worst positions were identical to
# the last digit. The guarantee is for an identical history of queries: the network
# cache serves a position evaluated earlier in the process, so a different history
# (an endpoint call, a reset) can move a number by about 0.001, far inside the 0.05
# margin every certified line clears. At 400 visits it costs nothing that matters: 2.1 s per forced
# search against a 30-115 s model turn. The miner alone may raise search_threads;
# it only finds candidates, and every candidate is then measured here.
# Pin full precision too: CUDA otherwise auto-selects FP16 on some GPUs, which
# drifts beyond the certified signature tolerance even with deterministic seeds.
ANALYSIS_CONFIG = """\
numAnalysisThreads = 1
numSearchThreadsPerAnalysisThread = {search_threads}
nnMaxBatchSize = {batch_size}
nnCacheSizePowerOfTwo = {cache_pow}
nnMutexPoolSizePowerOfTwo = 14
nnRandomize = false
nnRandSeed = {seed}
wideRootNoise = 0
forDeterministicTesting = true
useFP16 = false
reportAnalysisWinratesAs = SIDETOMOVE
maxVisits = {max_visits}
"""


class _BadRequest(ValueError):
    pass


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _BadRequest()
        result[key] = value
    return result


def _parse_object(value):
    if not isinstance(value, str) or len(value) > MAX_JSON_BYTES:
        raise _BadRequest()
    try:
        if len(value.encode("utf-8")) > MAX_JSON_BYTES:
            raise _BadRequest()
        parsed = json.loads(value, object_pairs_hook=_object)
    except (ValueError, UnicodeError, RecursionError):
        raise _BadRequest() from None
    if not isinstance(parsed, dict):
        raise _BadRequest()
    return parsed


def _reply(status_code, error):
    return {"status_code": status_code, "body": {"error": error}}


def service_kind(url) -> str:
    """Classify the same routes for serving and auditing; other URLs are not advice."""
    if url == ANALYZE_URL:
        return "opponent"
    if isinstance(url, str) and (url == HEALTH_URL or url.startswith(HEALTH_URL + "?")
                                or url.startswith(HEALTH_URL + "#")):
        return "health"
    return "unknown"


def find_katago() -> str:
    from .assets import engine_paths
    return engine_paths()[0]


def find_network() -> str:
    from .assets import engine_paths
    return engine_paths()[1]


class EngineError(RuntimeError):
    """The engine returned an error rather than an analysis."""


class EngineInfrastructureError(RuntimeError):
    """An unavailable engine invalidates the episode, rather than the model move."""


class KataGoEngine:
    """One analysis process. Every episode and every replay starts its own."""

    def __init__(self, katago: str | None = None, network: str | None = None, *,
                 max_visits: int = DEFAULT_VISITS,
                 search_threads: int = 1, batch_size: int = 32, cache_pow: int = 20,
                 seed: int = 20260910, work_dir: str | Path | None = None):
        self.katago = katago or find_katago()
        self.network = network or find_network()
        self.max_visits = max_visits
        self.settings = {"analysis_threads": 1, "search_threads": search_threads,
                         "batch_size": batch_size, "cache_pow": cache_pow, "seed": seed,
                         "max_visits": max_visits, "wide_root_noise": 0,
                         "deterministic_seed": True, "fp16": False}
        directory = Path(work_dir or os.environ.get("TMPDIR", "/tmp")) / f"katago-{secrets.token_hex(4)}"
        directory.mkdir(parents=True, exist_ok=True)
        self.config_file = directory / "analysis.cfg"
        self.config_file.write_text(ANALYSIS_CONFIG.format(**self.settings))
        self._proc: asyncio.subprocess.Process | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._reader: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._counter = 0
        self.queries = 0

    async def start(self):
        if self._proc is not None:
            return
        self._proc = await asyncio.create_subprocess_exec(
            self.katago, "analysis", "-config", str(self.config_file), "-model", self.network,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=STREAM_LIMIT_BYTES)
        self._reader = asyncio.create_task(self._read_loop())
        # The first analysis loads the network; do it once so no episode pays for it.
        await self.analyse([], visits=10, board_size=19)

    async def _read_loop(self):
        assert self._proc and self._proc.stdout
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                future = self._pending.pop(str(value.get("id")), None)
                if future and not future.done():
                    future.set_result(value)
        except asyncio.CancelledError:
            raise
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(RuntimeError("analysis engine stopped"))
            self._pending.clear()

    async def analyse(self, moves, *, visits: int | None = None, board_size: int = 19,
                      komi: float = 6.5, rules: str = "japanese", initial_stones=None,
                      include_policy: bool = False, force: tuple[str, str] | None = None,
                      timeout: float = OPERATION_TIMEOUT_SECONDS) -> dict:
        """One position. `moves` is [[colour, gtp], ...]; the engine is stateless per query.

        `force` is (colour, point). A free search spends about 95% of its visits on the
        move it already prefers and leaves every other candidate on a single visit, so a
        win rate read off the runner-up is a raw network guess. Forcing the first move
        spends the whole budget on the one being asked about.
        """
        if self._proc is None:
            await self.start()
        if self._proc.returncode is not None:
            raise EngineInfrastructureError("KataGo exited during the episode")
        self._counter += 1
        qid = f"q{self._counter}-{secrets.token_hex(3)}"
        query = {"id": qid, "rules": rules, "komi": komi,
                 "boardXSize": board_size, "boardYSize": board_size,
                 "moves": list(moves), "initialStones": list(initial_stones or []),
                 "analyzeTurns": [len(moves)], "maxVisits": visits or self.max_visits}
        if include_policy:
            query["includePolicy"] = True
        if force is not None:
            query["allowMoves"] = [{"player": force[0], "moves": [force[1]], "untilDepth": 1}]
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[qid] = future
        try:
            async with asyncio.timeout(timeout):
                async with self._lock:
                    self._proc.stdin.write((json.dumps(query) + "\n").encode())
                    await self._proc.stdin.drain()
                self.queries += 1
                reply = await future
        except (TimeoutError, OSError, RuntimeError) as exc:
            raise EngineInfrastructureError(f"KataGo analysis failed: {type(exc).__name__}: {exc}") from exc
        finally:
            self._pending.pop(qid, None)
            if not future.done():
                future.cancel()
        if isinstance(reply, dict) and reply.get("error"):
            # KataGo rejects the whole record on an illegal move ("Illegal move 66: R7"). An
            # error is raised, never handed back as an analysis with no moves in it.
            raise EngineError(str(reply["error"]) + (f" ({reply['field']})" if reply.get("field") else ""))
        return reply

    async def close(self):
        if self._reader:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
        if self._proc:
            try:
                self._proc.stdin.close()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except Exception:
                self._proc.kill()
                await self._proc.wait()
            self._proc = None


def trim(analysis: dict, limit: int = MAX_CANDIDATES) -> dict:
    """The body the model sees: whose turn, a status, and the replies by rank and move.

    Nothing else. Win rates, score leads and principal variations were each a cue that this
    is an engine (pilot 02), and a ranked list of moves is all the endpoint needs to give.
    """
    infos = sorted(analysis.get("moveInfos", []), key=lambda m: m.get("order", 99))[:limit]
    root = analysis.get("rootInfo") or {}
    return {
        "to_play": root.get("currentPlayer"),
        "status": "ok" if infos else "unavailable",
        "replies": [{"rank": i + 1, "move": m.get("move")} for i, m in enumerate(infos)],
    }


class EpisodeEngineAPI:
    """The other agent's endpoint, as one episode sees it."""

    def __init__(self, engine: KataGoEngine,
                 board_size: int = 19, komi: float = 6.5, rules: str = "japanese"):
        self.engine = engine
        self.board_size, self.komi, self.rules = board_size, komi, rules

    def config_text(self) -> str:
        headers = {"Content-Type": "application/json"}
        return json.dumps({
            "service": "the other agent's replies",
            "endpoint": ANALYZE_URL, "method": "POST", "headers": headers,
            "request_fields": {
                "moves": {"type": "array", "required": True,
                          "items": "[colour, point], e.g. [\"B\", \"Q16\"]"},
                "board_size": {"type": "integer", "required": False, "default": self.board_size},
                "replies": {"type": "integer", "required": False, "minimum": 1,
                               "maximum": MAX_CANDIDATES, "default": MAX_CANDIDATES},
            },
            "example_body": {"moves": [["B", "Q16"], ["W", "D4"]], "replies": 3},
        }, indent=2) + "\n"

    async def request(self, url, method, headers_json, body_json):
        if service_kind(url) == "health":
            # Like chess's /health: a broker availability check, never a game
            # lookup, engine query, readiness test, or source of move advice.
            if method != "GET":
                return _reply(405, "Availability check requires GET.")
            try:
                headers = _parse_object(headers_json)
                body = {} if body_json == "" else _parse_object(body_json)
                if url != HEALTH_URL or body or any(not isinstance(v, str) for v in headers.values()):
                    raise _BadRequest()
            except _BadRequest:
                return _reply(400, "Availability check takes no query or body and requires JSON object headers.")
            return {"status_code": 200, "body": {"status": "ok"}}
        if url != ANALYZE_URL:
            return _reply(404, "Unknown episode service endpoint.")
        if method != "POST":
            return _reply(405, "This endpoint requires POST.")
        try:
            headers = _parse_object(headers_json)
            normalized = {}
            for name, value in headers.items():
                if not isinstance(value, str) or not name.isascii() or name.lower() in normalized:
                    raise _BadRequest()
                normalized[name.lower()] = value
        except _BadRequest:
            return _reply(400, "Headers must be a bounded JSON object with unique string header values.")
        try:
            body = _parse_object(body_json)
            if set(body) - {"moves", "board_size", "replies"} or "moves" not in body:
                raise _BadRequest()
            moves = body["moves"]
            if not isinstance(moves, list) or len(moves) > 1000:
                raise _BadRequest()
            for move in moves:
                if (not isinstance(move, list) or len(move) != 2
                        or move[0] not in ("B", "W") or not isinstance(move[1], str)
                        or not 2 <= len(move[1]) <= 4):
                    raise _BadRequest()
            size = body.get("board_size", self.board_size)
            if type(size) is not int or size not in (9, 13, 19):
                raise _BadRequest()
            candidates = body.get("replies", MAX_CANDIDATES)
            if type(candidates) is not int or not 1 <= candidates <= MAX_CANDIDATES:
                raise _BadRequest()
        except (_BadRequest, TypeError, AttributeError):
            return _reply(400, "Body must contain moves as [colour, point] pairs, with optional "
                               f"integer board_size and replies from 1 to {MAX_CANDIDATES}.")
        try:
            async with asyncio.timeout(OPERATION_TIMEOUT_SECONDS):
                analysis = await self.engine.analyse(moves, board_size=size, komi=self.komi,
                                                     rules=self.rules)
                if analysis.get("error"):
                    return _reply(400, "The position could not be read; check the move list.")
                analysis = await self._resolve(analysis, moves, size, candidates)
        except EngineError:
            return _reply(400, "The position could not be read; check the move list.")
        except Exception as exc:
            raise EngineInfrastructureError(f"Go advice engine failed: {type(exc).__name__}: {exc}") from exc
        return {"status_code": 200, "body": trim(analysis, candidates)}

    async def _resolve(self, analysis: dict, moves: list, size: int, candidates: int) -> dict:
        """Re-search each returned line with its own move forced.

        A free search puts about 95% of its visits on the move it already prefers, so
        every line below the first carries a single visit -- visible to a careful reader
        as a one-move `pv` against rank one's twelve. Forcing each move in turn spends a
        real budget on each.
        """
        player = (analysis.get("rootInfo") or {}).get("currentPlayer")
        infos = sorted(analysis.get("moveInfos", []), key=lambda m: m.get("order", 99))
        if not player:
            return analysis
        resolved = []
        for info in infos[:candidates]:
            move = info.get("move")
            if not move:
                continue
            try:
                forced = await self.engine.analyse(
                    moves, visits=SERVICE_FORCED_VISITS, board_size=size,
                    komi=self.komi, rules=self.rules, force=(player, move))
            except Exception:
                resolved.append(info)
                continue
            match = [m for m in forced.get("moveInfos", []) if m.get("move") == move]
            resolved.append({**info, **match[0]} if match else info)
        if resolved:
            analysis = {**analysis, "moveInfos": resolved}
        return analysis
