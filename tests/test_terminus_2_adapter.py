import json
import os
from pathlib import Path
import subprocess

import pytest

from core.agents import make_agent, trajectory_from_transcript
from core.agents.terminus_2 import HARBOR_COMMIT

ROOT = Path(__file__).resolve().parents[1]


def agent(**kwargs):
    a = make_agent("terminus-2", model="openai/probe", **kwargs)
    a.apply_tool_policy({"web_search": False, "web_fetch": False})
    return a


def payload(a, prompt="Compute the answer and call the verifier."):
    return json.loads(
        a.blob(
            row={"id": "probe"},
            prompt=prompt,
            files=[],
            skip_dirs=(),
            stage_src=(ROOT / "core/stage.py").read_text(),
            timeout_s=60,
            env={},
            install=a.install(),
        )
    )


def test_pinned_install_and_routing():
    a = agent(
        api_base_url="https://gateway.example/v1",
        extra_body={"provider": {"only": ["example"]}},
        generation_config={"reasoning_effort": "high"},
    )
    assert a.setup() == []
    p = payload(a)
    assert p["task"]["harbor_commit"] == HARBOR_COMMIT
    assert p["task"]["api_base"] == "https://gateway.example/v1"
    assert p["task"]["extra_body"] == {"provider": {"only": ["example"]}}
    assert "--frozen" in a.install()["install"]
    assert "sha256sum -c" in a.install()["install"]
    assert p["modules"]["runner"] == p["code"]
    assert agent(generation_config={"made_up": 1}).setup()
    with pytest.raises(ValueError, match="pinned"):
        agent(version="latest")


def test_images_fail_instead_of_disappearing():
    with pytest.raises(ValueError, match="image input"):
        payload(
            agent(), [{"type": "image", "source": {}}, {"type": "text", "text": "look"}]
        )
    assert (
        payload(agent(), [{"type": "text", "text": "hello"}])["task"]["content"]
        == "hello"
    )


def test_native_atif_and_estimated_cost_are_preserved():
    a = agent()
    native = {
        "schema_version": "ATIF-v1.8",
        "agent": {"name": "terminus-2", "version": "2.0.0"},
        "steps": [
            {
                "step_id": 1,
                "source": "agent",
                "message": "plan",
                "reasoning_content": "reason",
                "tool_calls": [
                    {
                        "tool_call_id": "c1",
                        "function_name": "bash_command",
                        "arguments": {"keystrokes": "cat data\n", "duration": 1},
                    }
                ],
                "observation": {
                    "results": [{"source_call_id": "c1", "content": "answer"}]
                },
            }
        ],
    }
    raw = {
        "agent": "terminus-2",
        "model": "openai/probe",
        "harbor_trajectory": native,
        "harbor_cost_usd": 0.25,
    }
    assert trajectory_from_transcript(raw).to_json_dict() == native
    historical = {
        "agent": "terminus-2",
        "model": "openai/probe",
        "messages": [
            {"type": "harbor_trajectory", "trajectory": native, "artifacts": {}}
        ],
    }
    assert trajectory_from_transcript(historical).to_json_dict() == native
    assert a.episode_cost(raw) == (0.25, "estimated")
    assert "cat data" in a.readable(native["steps"][0])


@pytest.mark.skipif(
    not os.environ.get("HARBOR_TEST_PYTHON"),
    reason="requires the pinned Harbor interpreter",
)
def test_bridge_timeout_retains_partial_trajectory_and_credentials_stay_private():
    proc = subprocess.run(
        [
            os.environ["HARBOR_TEST_PYTHON"],
            str(ROOT / "tests/terminus_2_bridge_probe.py"),
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert all(json.loads(proc.stdout).values())


@pytest.mark.skipif(
    os.environ.get("RH_TERMINUS2_CONTAINER_TEST") != "1",
    reason="opt-in enroot + pinned Harbor installation",
)
def test_real_upstream_loop_terminal_mcp_observer_and_secrets(tmp_path):
    from core import sandbox

    a = agent(
        max_turns=5,
        generation_config={
            "model_info": {
                "max_input_tokens": 32000,
                "max_output_tokens": 4096,
                "input_cost_per_token": 0.000001,
                "output_cost_per_token": 0.000002,
            }
        },
    )
    p = payload(a)
    p["task"]["deliverable_files"] = ["submission.txt"]
    p["modules"]["grader"] = (ROOT / "tests/terminus_2_probe_grader.py").read_text()
    p["code"] = (ROOT / "tests/terminus_2_probe.py").read_text()
    sandbox.ensure_path()
    proc = subprocess.run(
        sandbox.container_argv(
            sandbox.DEFAULT_IMAGE,
            a.bootstrap,
            private_net=False,
            key_env=a.API_KEY_ENV,
            pass_key=False,
        ),
        env=sandbox.spawn_env(""),
        input=json.dumps(p),
        text=True,
        capture_output=True,
        timeout=1000,
    )
    (tmp_path / "stderr.log").write_text(proc.stderr)
    assert proc.returncode == 0, proc.stderr[-8000:]
    assert proc.stdout.strip(), proc.stderr[-8000:]
    raw = json.loads(proc.stdout)
    assert raw.get("error") is None, (raw, proc.stderr[-5000:])
    assert raw["grader_state"]["answer"] == 42
    assert raw["grader_state"]["observed_before_call"]
    assert raw["n_turns"] == 3
    assert "verified 42" in json.dumps(raw["harbor_trajectory"])
    assert "probe-secret-key" not in proc.stdout
    assert "probe-secret-key" not in proc.stderr
    assert raw["harbor_commit"] == HARBOR_COMMIT
    assert a.to_trajectory(raw).steps
    assert (
        a.to_trajectory({"messages": raw["messages"]}).to_json_dict()
        == raw["harbor_trajectory"]
    )
    assert raw["usage"]["input"] > 0
    assert {d["name"]: d["b64"] for d in raw["deliverables"]}[
        "submission.txt"
    ] == "NDI="


def test_policy_outputs_and_independent_runtime_helpers():
    for key in ("web_search", "web_fetch"):
        with pytest.raises(ValueError, match="no provider web tools"):
            agent().apply_tool_policy(
                {"web_search": key == "web_search", "web_fetch": key == "web_fetch"}
            )
    assert agent(permission_mode="default").setup()
    a = agent()
    p = json.loads(
        a.blob(
            row={"id": "test", "deliverable_files": ["submission.txt"]},
            prompt="p",
            files=[],
            skip_dirs=(),
            stage_src="",
            timeout_s=60,
            env={},
        )
    )
    assert p["task"]["deliverable_files"] == ["submission.txt"]
    assert "def main(" not in p["modules"]["support"]
    assert "gemini" not in p["code"].lower()
    from core.sandbox import egress

    restricted = egress.hosts_for(
        egress.NetworkPolicy("deny"), "https://gateway.example", a.INSTALL_HOSTS
    )
    assert set(a.INSTALL_HOSTS).issubset(restricted)
    assert not set(a.INSTALL_HOSTS).intersection(
        egress.hosts_for(egress.NetworkPolicy("deny"), "https://gateway.example")
    )
