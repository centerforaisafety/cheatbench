"""Deterministic MCP fixture for the real upstream T2 runtime test."""

TOOL_BASENAME = "verify"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
}


def make_tool(row, staged, workdir):
    state = {"observed_before_call": False}

    def call(args):
        state["answer"] = args["answer"]
        return {"content": [{"type": "text", "text": f"verified {args['answer']}"}]}

    return state, call


def make_observer(row, staged, workdir, state):
    class Observer:
        def observe(self, record):
            for block in record.get("content") or []:
                if block.get("_type") == "ToolUseBlock" and "answer" not in state:
                    state["observed_before_call"] = True

        def finish(self):
            state["observer_finished"] = True

    return Observer()
