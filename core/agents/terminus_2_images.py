"""Image observations for the pinned Harbor Terminus-2 JSON command loop.

Loaded inside the episode container. Harbor still owns execution, retries,
completion and compaction. Image handling follows the file-observation design
of Harbor PR #1484, with persistent history and Responses content conversion.
"""

import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageOps
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
from harbor.llms.base import ContextLengthExceededError
from harbor.llms.chat import Chat
from harbor.llms.lite_llm import LiteLLM
from harbor.models.trajectories.observation import Observation
from harbor.models.trajectories.observation_result import ObservationResult
from harbor.models.trajectories.tool_call import ToolCall

IMAGE_EXTENSION_VERSION = "5-transport-recovery"
WIRE_REQUESTS = []
MAX_IMAGES = 2
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_IMAGE_EDGE = 2000
IMAGE_INSTRUCTIONS = """
Image viewing is available through an optional top-level JSON field:
\"view_images\": [\"/workspace/image/img_01.jpg\"]
Use the actual file path for your task. This action returns image pixels in the
next observation. Request at most two PNG, JPEG, GIF or WebP files per turn,
up to 5 MiB and 25 million pixels each. Images larger than 2000 pixels on either
side are resized proportionally; crop the relevant region for finer detail.
Relative paths are relative to the
initial working directory, not a subsequent shell cd. Commands run before
image reads. All existing required JSON fields remain required. You may use
an empty commands array to view an image. Do not mark task_complete on a turn
requesting images. Images remain in conversation history until context
compaction; after compaction you can reopen them with view_images.
"""


def install_wire_audit():
    """Record actual serialized HTTP body lengths, without headers or payloads."""
    import httpx
    original = httpx.AsyncClient.send
    if getattr(original, "_t2_wire_audit", False):
        return
    async def send(client, request, *args, **kwargs):
        if request.method == "POST" and request.url.path.endswith(("/messages", "/responses", "/chat/completions")):
            try:
                body = request.content
                WIRE_REQUESTS.append({"body_bytes": len(body), "path": request.url.path,
                                      "host": request.url.host})
            except httpx.RequestNotRead:
                WIRE_REQUESTS.append({"body_bytes": None, "path": request.url.path})
        return await original(client, request, *args, **kwargs)
    send._t2_wire_audit = True
    httpx.AsyncClient.send = send


def response_content(content, role="user"):
    """Translate Chat content blocks to the Responses input schema."""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if part["type"] == "text":
            parts.append(
                {
                    "type": "output_text" if role == "assistant" else "input_text",
                    "text": part["text"],
                }
            )
        elif part["type"] == "image_url":
            parts.append({"type": "input_image", **part["image_url"]})
            parts[-1]["image_url"] = parts[-1].pop("url")
        else:
            raise ValueError(f"Unsupported image conversation part: {part['type']}")
    return parts


class TerminusLLMClient(LiteLLM):
    """Harbor model client with image formatting and API request recovery."""

    def _record_recovery(self, kind):
        counts = getattr(self, "transport_recoveries", {})
        counts[kind] = counts.get(kind, 0) + 1
        self.transport_recoveries = counts

    @staticmethod
    def _without_empty_assistant(history):
        return [m for m in history if not (
            isinstance(m, dict) and m.get("role") == "assistant"
            and m.get("content") in (None, "", [])
            and not m.get("reasoning_content") and not m.get("tool_calls"))]

    async def call(self, prompt, message_history=None, response_format=None,
                   logging_path=None, **kwargs):
        history = message_history or []
        if getattr(self, "_empty_assistant_rejected", False):
            history = self._without_empty_assistant(history)
        try:
            return await super().call(prompt, history, response_format, logging_path, **kwargs)
        except Exception as exc:
            message = str(exc).lower()
            if "role 'assistant' must not be empty" in message:
                # Harbor can append an empty model reply after a malformed turn.
                # Remove only empty wire messages, keeping the recorded turn,
                # all real content, reasoning, tool calls and task instructions.
                cleaned = self._without_empty_assistant(history)
                if len(cleaned) != len(history):
                    self._empty_assistant_rejected = True
                    self._record_recovery("empty_assistant_message")
                    return await super().call(prompt, cleaned, response_format, logging_path, **kwargs)
            oversized = ("request_too_large" in message
                         or "request exceeds the maximum size" in message
                         or "downloaded image content cannot exceed" in message)
            if not oversized or getattr(self, "_size_recovery_used", False):
                raise
            # Enter Harbor's existing context-overflow handler exactly once per
            # main-agent turn. A failed recovery remains an infrastructure error.
            self._size_recovery_used = True
            self._byte_overflow_pending = True
            self._record_recovery("request_size_compaction")
            raise ContextLengthExceededError(str(exc)) from exc

    async def _call_responses(
        self,
        prompt,
        message_history=None,
        response_format=None,
        logging_path=None,
        **kwargs,
    ):
        original_prompt = prompt
        history = [
            dict(m, content=response_content(m["content"], m["role"]))
            for m in (message_history or [])
        ]
        prompt = response_content(prompt)
        if kwargs.get("previous_response_id") and isinstance(prompt, list):
            # Upstream sends prompt directly as input when chaining. Responses
            # requires message items here, not a bare list of content blocks.
            prompt = [{"role": "user", "content": prompt}]
        try:
            return await super()._call_responses(
                prompt, history, response_format, logging_path, **kwargs
            )
        except Exception as exc:
            # A vendor may expire its server-side chain. Replay the same local
            # history rather than switching protocol, model, or memory policy.
            text = str(exc).lower()
            if not kwargs.get("previous_response_id") or not (
                "referenced response not found or expired" in text
                or ("previous_response" in text and ("not found" in text or "expired" in text))
            ):
                raise
            self._record_recovery("expired_response_chain")
            retry = {**kwargs, "previous_response_id": None}
            return await super()._call_responses(
                response_content(original_prompt), history, response_format, logging_path, **retry
            )



class ImageChat(Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_images = []
        self.image_requests = 0

    async def chat(self, prompt, **kwargs):
        if self.pending_images:
            prompt = [{"type": "text", "text": prompt}, *self.pending_images]
        response = await super().chat(prompt, **kwargs)
        if self.pending_images:
            self.image_requests += 1
            self.pending_images = []
        return response


class ImageParser(TerminusJSONPlainParser):
    def __init__(self):
        super().__init__()
        self.image_paths = []

    def _try_parse_response(self, response):
        self.image_paths = []
        result = super()._try_parse_response(response)
        if result.error:
            return result
        raw, _ = self._extract_json_content(response)
        paths = json.loads(raw).get("view_images", [])
        if (
            not isinstance(paths, list)
            or len(paths) > MAX_IMAGES
            or any(
                not isinstance(p, str) or not p.strip() or "\x00" in p for p in paths
            )
        ):
            result.error = (
                "view_images must be an array of at most two nonempty file paths"
            )
        elif paths and result.is_task_complete:
            result.error = "View images before marking task_complete"
        else:
            self.image_paths = paths
        return result


def read_image(path, workdir):
    """Only called in the isolated episode process; bound bytes before decoding."""
    target = Path(path)
    if not target.is_absolute():
        target = Path(workdir) / target
    if not target.is_file():
        raise ValueError("not a regular file")
    with target.open("rb") as handle:
        data = handle.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds 5 MiB")
    source_sha256 = hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in {"PNG", "JPEG", "GIF", "WEBP"}:
            raise ValueError("unsupported image format")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("image exceeds 25 million pixels")
        mime = Image.MIME[image.format]
        original_size = image.size
        orientation = image.getexif().get(274, 1)
        image.load()
        image = ImageOps.exif_transpose(image)
        delivered_size = image.size
        if max(image.size) > MAX_IMAGE_EDGE or orientation != 1:
            image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
            encoded = io.BytesIO()
            # Resizing a photographic JPEG into lossless PNG can expand a
            # valid input past the byte limit. Keep JPEG inputs as JPEGs.
            if mime == "image/jpeg":
                image.convert("RGB").save(encoded, format="JPEG", quality=95)
            else:
                image.save(encoded, format="PNG")
                mime = "image/png"
            data = encoded.getvalue()
            delivered_size = image.size
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError(
                    "normalized image exceeds 5 MiB; request a smaller crop"
                )
    url = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}, {
        "path": path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "source_sha256": source_sha256,
        "original_size": list(original_size),
        "delivered_size": list(delivered_size),
        "media_type": mime,
    }


class ImageTerminus2(Terminus2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt_template += "\n" + IMAGE_INSTRUCTIONS
        self._image_record = None
        self._image_workdir = "/workspace"

    def _init_llm(self, **kwargs):
        backend = kwargs.pop("llm_backend")
        if getattr(backend, "value", backend) != "litellm":
            raise ValueError("T2 image extension requires the LiteLLM backend")
        extra = dict(kwargs.pop("llm_kwargs") or {})
        install_wire_audit()
        return TerminusLLMClient(**kwargs, **extra)

    async def _query_llm(self, *args, **kwargs):
        self._llm._size_recovery_used = False
        try:
            return await super()._query_llm(*args, **kwargs)
        finally:
            self._llm._byte_overflow_pending = False

    def _unwind_messages_to_free_tokens(self, chat, target_free_tokens=4000):
        if not getattr(self._llm, "_byte_overflow_pending", False):
            return super()._unwind_messages_to_free_tokens(chat, target_free_tokens)
        self._llm._byte_overflow_pending = False
        # Use Harbor's recent-pair unwind, preserving the first prompt verbatim.
        # Token headroom cannot detect a body-size overflow. Make byte headroom
        # for Harbor's normal three-stage summary and question/answer handoff.
        def size():
            return len(json.dumps(chat.messages, ensure_ascii=False).encode())
        before = size()
        target = before // 2
        first = json.dumps(chat.messages[0], sort_keys=True) if chat.messages else None
        removed = 0
        while len(chat.messages) > 1 and size() > target:
            n = len(chat.messages)
            chat._messages = chat.messages[:max(1, n - 2)]
            removed += n - len(chat.messages)
        chat.reset_response_chain()
        assert not first or json.dumps(chat.messages[0], sort_keys=True) == first
        event = {"history_bytes_before": before, "history_bytes_after": size(),
                 "removed_recent_messages": removed, "first_prompt_preserved": True}
        self._byte_compactions = [*getattr(self, "_byte_compactions", []), event]

    def _get_parser(self):
        if self.options.parser_name != "json":
            raise ValueError("T2 image extension requires the JSON parser")
        return ImageParser()

    async def setup(self, environment):
        self._image_workdir = environment.workdir
        await super().setup(environment)

    async def _run_agent_loop(self, initial_prompt, chat, original_instruction=""):
        # Upstream constructs an empty Chat at the start of each run.
        assert not chat.messages
        self._chat = ImageChat(
            self._llm, interleaved_thinking=self.options.interleaved_thinking
        )
        return await super()._run_agent_loop(
            initial_prompt, self._chat, original_instruction
        )

    async def _execute_commands(self, commands, session):
        timeout, output = await super()._execute_commands(commands, session)
        paths = list(self._parser.image_paths)
        self._parser.image_paths = []
        if paths:
            parts, records, reports = [], [], []
            for path in paths:
                try:
                    part, record = read_image(path, self._image_workdir)
                    parts.append(part)
                    records.append(record)
                    reports.append(f"Image loaded: {path}")
                except (
                    OSError,
                    ValueError,
                    SyntaxError,
                    Image.DecompressionBombError,
                ) as exc:
                    reports.append(f"Image error for {path}: {exc}")
            self._chat.pending_images = parts
            self._image_record = (paths, parts, records, "\n".join(reports))
            output += "\n" + "\n".join(reports)
        return timeout, output

    def _dump_trajectory(self):
        if (
            self._image_record
            and self._trajectory_steps
            and self._trajectory_steps[-1].source == "agent"
        ):
            paths, parts, records, report = self._image_record
            step = self._trajectory_steps[-1]
            call_id = f"image_{step.step_id}"
            step.tool_calls = [
                *(step.tool_calls or []),
                ToolCall(
                    tool_call_id=call_id,
                    function_name="view_images",
                    arguments={"paths": paths},
                ),
            ]
            content = [{"type": "text", "text": report}]
            content.extend(
                {
                    "type": "image",
                    "source": {
                        "media_type": record["media_type"],
                        "path": part["image_url"]["url"],
                    },
                }
                for part, record in zip(parts, records)
            )
            if step.observation is None:
                step.observation = Observation(results=[])
            step.observation.results.append(
                ObservationResult(source_call_id=call_id, content=content)
            )
            step.extra = {**(step.extra or {}), "image_files": records}
            self._image_record = None
        if self._context is not None:
            self._context.metadata = {
                **(self._context.metadata or {}),
                "image_extension_version": IMAGE_EXTENSION_VERSION,
                "transport_recoveries": dict(getattr(self._llm, "transport_recoveries", {})),
                "wire_requests": list(WIRE_REQUESTS),
                "request_size_compactions": getattr(self, "_byte_compactions", []),
                "image_observations_sent": getattr(
                    getattr(self, "_chat", None), "image_requests", 0
                ),
            }
        super()._dump_trajectory()
