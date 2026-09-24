"""Run with the frozen Harbor interpreter; no provider or container required."""

import asyncio
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from harbor.models.trial.paths import EnvironmentPaths


async def check(root):
    with tempfile.TemporaryDirectory() as folder:
        logs = Path(folder)
        support = {
            "WORKDIR": folder,
            "task_install_env": lambda: {"PATH": os.environ["PATH"]},
            "_emit_msg": lambda record: None,
        }
        bridge = {"__name__": "bridge", "SUPPORT": support}
        exec(
            compile(
                (root / "core/agents/terminus_2_bridge.py").read_text(),
                "<bridge>",
                "exec",
            ),
            bridge,
        )
        observed = bridge["ObservedTerminus2"]
        task = {
            "id": "timeout-probe",
            "model": "openai/probe",
            "api_key_env": "T2_TEST_KEY",
            "api_base": "http://127.0.0.1:1/v1",
            "generation_config": {},
            "max_turns": 4,
            "timeout_s": 0.05,
            "content": "A test task",
        }
        os.environ["T2_TEST_KEY"] = "timeout-probe-secret"

        class Session:
            async def get_incremental_output(self):
                return "$ "

            async def stop(self):
                pass

        async def setup(self, environment):
            self._session = Session()
            # Credential metadata must be sanitized before any run/setup.
            assert self.options.llm_kwargs == {}
            assert self._llm._llm_kwargs["api_key"] == "timeout-probe-secret"

        async def loop(self, **kwargs):
            # Upstream run() has already appended its actual initial prompt.
            await asyncio.sleep(5)

        with patch.object(EnvironmentPaths, "agent_dir", logs), patch.object(
            observed, "setup", setup
        ), patch.object(observed, "_run_agent_loop", loop):
            context, options, error = await bridge["episode"](task, logs, None, None)
        assert error == "timeout after 0.05s", error
        native = json.loads((logs / "trajectory.json").read_text())
        assert native["steps"][0]["source"] == "user"
        assert "A test task" in native["steps"][0]["message"]
        assert "timeout-probe-secret" not in json.dumps(native)
        assert options["enable_summarize"] is True
        assert options["proactive_summarization_threshold"] == 8000
        assert options["temperature"] is None
        environment = bridge["EpisodeEnvironment"](
            folder, logs, "probe", support["task_install_env"]()
        )
        try:
            await environment.exec("sleep 5", timeout_sec=0.05)
        except TimeoutError:
            pass
        else:
            raise AssertionError("exec deadline was ignored")
        print(
            json.dumps(
                {
                    "timeout_preserved_trajectory": True,
                    "credentials_excluded": True,
                    "exec_timeout": True,
                    "upstream_defaults": True,
                }
            )
        )


if __name__ == "__main__":
    import sys

    asyncio.run(check(Path(sys.argv[1])))
