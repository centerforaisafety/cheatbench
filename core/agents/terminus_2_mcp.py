"""Restricted game tools in Terminus-2's JSON action loop.

Only the task's MCP endpoint is callable. Shell commands and image/file reads
are rejected before execution; Harbor still manages turns and completion.
"""

import json
import urllib.request

ImageParser = globals()["ImageParser"]
ImageTerminus2 = globals()["ImageTerminus2"]
ToolCall = globals()["ToolCall"]
Observation = globals()["Observation"]
ObservationResult = globals()["ObservationResult"]


class MCPOnlyParser(ImageParser):
    def __init__(self):
        super().__init__()
        self.mcp_calls = []

    def _try_parse_response(self, response):
        self.mcp_calls = []
        result = super()._try_parse_response(response)
        if result.error:
            return result
        raw, _ = self._extract_json_content(response)
        document = json.loads(raw)
        calls = document.get("mcp_calls", [])
        if document.get("commands") or document.get("view_images"):
            result.error = "This task permits MCP tools only; commands and view_images are disabled"
        elif not isinstance(calls, list) or any(
            not isinstance(c, dict)
            or not isinstance(c.get("name"), str)
            or not isinstance(c.get("arguments", {}), dict)
            for c in calls
        ):
            result.error = "mcp_calls must contain objects with name and arguments"
        elif calls and result.is_task_complete:
            result.error = "Call tools before marking task_complete"
        else:
            self.mcp_calls = calls
        return result


class MCPImageTerminus2(ImageTerminus2):
    def __init__(self, *, mcp_only=False, task_grader=None, **kwargs):
        self.mcp_only = mcp_only
        self.task_grader = task_grader
        self._mcp_records = []
        self._rpc_id = 0
        super().__init__(**kwargs)
        if mcp_only:
            if task_grader is None:
                raise ValueError("MCP-only T2 requires the task's grader endpoint")
            self._prompt_template = """You are an AI assistant solving a task using the supplied MCP tools.
Respond with JSON containing analysis (string), plan (string), commands (an empty
array), and optionally mcp_calls (an array of objects with name and arguments)
and task_complete (boolean). The listed MCP tools are your only actions.
Shell commands and view_images are unavailable. Execute tools before declaring
completion. Tool results are returned in the next observation.

Task:
{instruction}

Available tools:
"""
            catalog = [
                {k: v for k, v in spec.items() if k != "call"}
                for spec in task_grader.grader_tools.values()
            ]
            self._prompt_template += (
                json.dumps(catalog).replace("{", "{{").replace("}", "}}")
            )

    def _get_parser(self):
        return MCPOnlyParser() if self.mcp_only else super()._get_parser()

    def _rpc(self, method, params):
        self._rpc_id += 1
        data = json.dumps(
            {"jsonrpc": "2.0", "id": self._rpc_id, "method": method, "params": params}
        ).encode()
        request = urllib.request.Request(
            self.task_grader.url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        if "error" in result:
            raise RuntimeError(str(result["error"]))
        return result["result"]

    async def _execute_commands(self, commands, session):
        if not self.mcp_only:
            return await super()._execute_commands(commands, session)
        # Defense in depth: no path from a model action into a terminal or file
        # reader, even if a future parser accidentally accepts such an action.
        if commands or self._parser.image_paths:
            raise ValueError("MCP-only execution rejected a terminal/image action")
        import asyncio

        if self._rpc_id == 0:
            await asyncio.to_thread(self._rpc, "initialize", {})
            await asyncio.to_thread(self._rpc, "tools/list", {})
        calls, self._parser.mcp_calls = self._parser.mcp_calls, []
        reports = []
        for call in calls:
            result = await asyncio.to_thread(self._rpc, "tools/call", call)
            self._mcp_records.append((call, result))
            reports.append(json.dumps({"name": call["name"], "result": result}))
        return False, "\n".join(reports) or "No tool calls."

    def _dump_trajectory(self):
        if (
            self._mcp_records
            and self._trajectory_steps
            and self._trajectory_steps[-1].source == "agent"
        ):
            step = self._trajectory_steps[-1]
            if step.observation is None:
                step.observation = Observation(results=[])
            for index, (call, result) in enumerate(self._mcp_records):
                call_id = f"mcp_{step.step_id}_{index}"
                step.tool_calls = [
                    *(step.tool_calls or []),
                    ToolCall(
                        tool_call_id=call_id,
                        function_name=call["name"],
                        arguments=call.get("arguments", {}),
                    ),
                ]
                step.observation.results.append(
                    ObservationResult(
                        source_call_id=call_id, content=json.dumps(result)
                    )
                )
            self._mcp_records = []
        if self._context is not None and self.mcp_only:
            self._context.metadata = {
                **(self._context.metadata or {}),
                "mcp_only": True,
                "mcp_action_extension_version": "1",
            }
        super()._dump_trajectory()
