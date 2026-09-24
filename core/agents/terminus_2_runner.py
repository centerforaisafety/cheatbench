"""Run task services in the image's Python and Harbor in its frozen runtime."""

import json
import os
import subprocess
import threading
from types import SimpleNamespace


class RemoteObserver:
    """Synchronous callbacks to the task process, without exposing a new tool."""

    def __init__(self, fds):
        self.requests = os.fdopen(fds[0], "w")
        self.replies = os.fdopen(fds[1])

    def observe(self, event):
        self.requests.write(json.dumps(event) + "\n")
        self.requests.flush()
        reply = json.loads(self.replies.readline())
        if reply.get("error"):
            raise RuntimeError(reply["error"])


def observe_events(observer, request_fd, reply_fd):
    with os.fdopen(request_fd) as requests, os.fdopen(reply_fd, "w") as replies:
        for line in requests:
            try:
                observer.observe(json.loads(line))
                reply = {}
            except Exception as exc:
                reply = {"error": f"Observer failed: {type(exc).__name__}: {exc}"}
            replies.write(json.dumps(reply) + "\n")
            replies.flush()


def main(task, modules):
    output = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    support = {"__name__": "runner_support", "WORKDIR": "/workspace"}
    exec(compile(modules["support"], "<support>", "exec"), support)
    install = task.get("install_result")
    record = {"id": task["id"], "model": task["model"], "messages": []}
    grader = observer = task_install = None
    child_fds = ()
    observer_thread = None
    try:
        os.environ.update({str(k): str(v) for k, v in task.get("env", {}).items()})
        os.chdir(support["WORKDIR"])
        if not task.get("runtime_ready"):
            install = support["install_agent"](task.get("install"))
            if not install or install.get("status") == "failed":
                raise RuntimeError("Terminus-2 runtime installation failed")
            # Task Python owns staging, dependencies, grading and observation.
            # Native extensions installed in the task image cannot be imported
            # by Harbor's separate Python version and frozen dependency set.
            namespaces = {}
            for name in ("stage", "grader"):
                if modules.get(name):
                    ns = {"__name__": f"runner_{name}"}
                    exec(compile(modules[name], f"<{name}>", "exec"), ns)
                    namespaces[name] = ns
            staged = namespaces["stage"]["write"](
                task.get("files") or [], support["WORKDIR"]
            )
            task_install = support["install_agent"](
                task.get("task_install"), env=support["task_install_env"]()
            )
            if task_install and task_install.get("status") == "failed":
                raise RuntimeError("Task dependency installation failed")
            module = namespaces.get("grader") or {}
            descriptor = None
            if module:
                grader = support["serve_grader"](
                    module, task.get("row") or {}, staged, support["WORKDIR"]
                )
                descriptor = dict(
                    url=grader.url,
                    grader_server_name=grader.grader_server_name,
                    grader_tools={
                        name: {k: v for k, v in spec.items() if k != "call"}
                        for name, spec in grader.grader_tools.items()
                    },
                )
            if module.get("make_observer"):
                observer = module["make_observer"](
                    task.get("row") or {},
                    staged,
                    support["WORKDIR"],
                    grader.grader_state if grader else {},
                )
                request_read, request_write = os.pipe()
                reply_read, reply_write = os.pipe()
                child_fds = (request_write, reply_read)
                observer_thread = threading.Thread(
                    target=observe_events,
                    args=(observer, request_read, reply_write),
                    daemon=True,
                )
                observer_thread.start()
            payload = {
                "task": dict(
                    task,
                    runtime_ready=True,
                    install_result=install,
                    grader_descriptor=descriptor,
                    observer_fds=child_fds,
                ),
                "modules": modules,
                "code": modules["runner"],
            }
            child = subprocess.run(
                [
                    task["runtime"] + "/.venv/bin/python",
                    "-c",
                    'import sys,json;d=json.load(sys.stdin);exec(compile(d["code"],"<r>","exec"),'
                    '{"TASK":d["task"],"MODULES":d["modules"]})',
                ],
                input=json.dumps(payload),
                text=True,
                stdout=subprocess.PIPE,
                pass_fds=child_fds,
            )
            if child.returncode:
                raise RuntimeError(f"Harbor runtime exited {child.returncode}")
            record = json.loads(child.stdout)
            record["task_install"] = task_install
        else:
            descriptor = task.get("grader_descriptor")
            remote_grader = None
            if descriptor:
                remote_grader = SimpleNamespace(
                    **descriptor,
                    grader_state={},
                    report=lambda: {
                        "name": descriptor["grader_server_name"],
                        "transport": "streamable-http",
                        "status": "configured",
                    },
                )
            remote_observer = (
                RemoteObserver(task["observer_fds"])
                if task.get("observer_fds")
                else None
            )
            bridge = {"__name__": "terminal_bridge", "SUPPORT": support}
            exec(compile(modules["images"], "<images>", "exec"), bridge)
            exec(compile(modules["mcp"], "<mcp>", "exec"), bridge)
            exec(compile(modules["bridge"], "<bridge>", "exec"), bridge)
            record = bridge["run"](task, install, remote_grader, remote_observer)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for fd in child_fds:
            os.close(fd)
        if observer_thread:
            observer_thread.join(timeout=10)
        if observer:
            try:
                observer.finish()
                record["deliverables"] = support["collect_deliverables"](
                    support["WORKDIR"],
                    tuple(task["skip_dirs"]),
                    tuple(task.get("deliverable_files") or []),
                )
            except Exception as exc:
                record["error"] = (
                    f"{record.get('error') or ''} observer finish failed: {exc}".strip()
                )
        if grader:
            record["grader_state"] = grader.grader_state
            record["init_mcp_servers"] = [grader.report()]
            grader.close()
    record["install"] = install
    output.write(json.dumps(record) + "\n")
    output.flush()


if "TASK" in globals():
    main(globals()["TASK"], globals()["MODULES"])
