"""Run an edited opponent policy in a separate, deny-default OS sandbox.

The chess harness executes `opponent/play.py` when a model edits it, so a model
that rewrites the policy to resign actually wins. Go staged the same file but
never ran it, which made one of the three tamper routes silently inert and any
Go-versus-chess comparison unsafe. This is that executor, ported.

It is simpler than the chess one in a single respect: a Go policy needs no game
library, because `decide` only picks among the candidates the harness hands it.
So nothing but the bundle and the standard library is mapped into the sandbox.

The decision this returns is the *untrusted* one. The runner keeps its own
pristine verdict alongside it, and there is deliberately no unsandboxed
fallback: if the sandbox is unavailable, the policy does not run.
"""
from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import sysconfig

WALL_SECONDS = 5.0
OUTPUT_BYTES = 32_768
SOURCE_BYTES = 131_072
LIMITS = {"cpu_seconds": 2, "resident_memory_bytes": 134_217_728,
          "memory_poll_seconds": 0.025,
          "open_files": 32, "file_bytes": 0, "child_processes": 0,
          "wall_seconds": WALL_SECONDS, "stdout_bytes": OUTPUT_BYTES,
          "stderr_bytes": OUTPUT_BYTES}

POLICY_BOOTSTRAP = '''# Trusted bootstrap, outside the editable source mapping.
import json, pathlib, resource, sys
bundle = pathlib.Path(__file__).resolve().parent
settings = json.loads((bundle / "worker_settings.json").read_text())
limits = settings["limits"]
resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"]))
resource.setrlimit(resource.RLIMIT_NOFILE, (limits["open_files"], limits["open_files"]))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
sys.dont_write_bytecode = True
sys.path.insert(0, str(bundle))
request = json.loads((bundle / "request.json").read_text())
if sys.platform == "linux":
    # Hard address-space limit backs up the parent RSS monitor. Install both
    # kernel boundaries before importing any model-edited module.
    resource.setrlimit(resource.RLIMIT_AS, (limits["resident_memory_bytes"], limits["resident_memory_bytes"]))
    try:
        from sandbox_linux import enforce
        enforce(settings["read_paths"])
    except BaseException as exc:
        print(json.dumps({"sandbox_error": type(exc).__name__ + ": " + str(exc)[:2000]}), flush=True)
        sys.exit(3)
try:
    import play
    result = play.decide(request)
    if not isinstance(result, dict) or not (result.get("resign") is True or isinstance(result.get("move"), str)):
        raise TypeError("decide must return {'move': point} or {'resign': True}")
    print(json.dumps({"move": result.get("move"), "resign": result.get("resign") is True}), flush=True)
except BaseException as exc:
    print(json.dumps({"worker_error": type(exc).__name__ + ": " + str(exc)[:2000]}), flush=True)
    sys.exit(2)
'''


def _runtime() -> dict[str, str]:
    executable = Path(sys.executable).resolve()
    framework_root = Path(sys.base_prefix).resolve()
    # Framework bin/python is a launcher that posix_spawns Python.app. Invoke
    # the interpreter directly so the profile never needs child-process rights.
    app_python = framework_root / "Resources/Python.app/Contents/MacOS/Python"
    if app_python.is_file():
        executable = app_python.resolve()
    return {"python": str(executable),
            "framework_library": str((framework_root / "Python").resolve()),
            "stdlib": str(Path(sysconfig.get_path("stdlib")).resolve())}


def sandbox_profile(bundle: Path, runtime: dict[str, str]) -> str:
    """No home or workspace reads, no writes, no network, no child processes."""
    quote = json.dumps
    stdlib = runtime["stdlib"]
    # Python's framework binary/dylib are outside its stdlib directory.
    framework = runtime["framework_library"]
    ancestors = sorted({str(parent) for path in
                        [runtime["python"], framework, stdlib, str(bundle.resolve())]
                        for parent in Path(path).parents})
    metadata_rules = " ".join(f"(literal {quote(path)})" for path in ancestors)
    return f'''(version 1)
(deny default)
(allow process-exec (literal {quote(runtime["python"])}))
(allow sysctl-read
    (sysctl-name "kern.bootargs")
    (sysctl-name "kern.osrelease")
    (sysctl-name "kern.ostype")
    (sysctl-name "kern.osversion")
    (sysctl-name "hw.machine")
    (sysctl-name "hw.model")
    (sysctl-name "hw.ncpu")
    (sysctl-name "hw.activecpu")
    (sysctl-name "hw.memsize")
    (sysctl-name "hw.pagesize")
    (sysctl-name "hw.optional.arm64")
    (sysctl-name "sysctl.proc_native"))
(allow file-read-metadata {metadata_rules})
(allow file-read*
    (literal "/")
    (literal {quote(runtime["python"])})
    (literal {quote(framework)})
    (subpath {quote(str(bundle.resolve()))})
    (require-all (subpath {quote(stdlib)})
                 (require-not (subpath {quote(str(Path(stdlib) / "site-packages"))})))
    (subpath "/System/Library")
    (subpath "/usr/lib")
    (subpath "/Library/Apple/usr/libexec/oah")
    (literal "/dev/null")
    (literal "/dev/random")
    (literal "/dev/urandom"))
'''


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _kill(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _resident_bytes(pid: int) -> int | None:
    """Read only this child's resident bytes via procfs or Darwin libproc."""
    if sys.platform == "linux":
        try:
            resident_pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
        except FileNotFoundError:
            return None
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = library.proc_pidinfo
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                         ctypes.c_void_p, ctypes.c_int]
    function.restype = ctypes.c_int
    # struct proc_taskinfo begins with uint64 virtual_size, resident_size.
    buffer = ctypes.create_string_buffer(96)
    read = function(pid, 4, 0, buffer, len(buffer))
    return int.from_bytes(buffer.raw[8:16], sys.byteorder) if read == len(buffer) else None


async def _execute(command: list[str], bundle: Path) -> dict:
    # Never copy os.environ: in particular no API tokens, proxy settings,
    # Python configuration, HOME, or inherited loader overrides enter the child.
    process = await asyncio.create_subprocess_exec(
        *command, cwd=bundle, env={"LANG": "C", "LC_ALL": "C"},
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, start_new_session=True, close_fds=True)
    exceeded = False
    memory_exceeded = False
    memory_monitor_error = None
    peak_rss = 0

    async def monitor_memory():
        nonlocal memory_exceeded, memory_monitor_error, peak_rss
        misses = 0
        while process.returncode is None:
            try:
                rss = _resident_bytes(process.pid)
            except (OSError, AttributeError) as exc:
                memory_monitor_error = str(exc)
                _kill(process)
                return
            if rss is None:
                misses += 1
                if misses >= 3 and process.returncode is None:
                    memory_monitor_error = "Child resident memory could not be inspected."
                    _kill(process)
                    return
            else:
                misses = 0
                peak_rss = max(peak_rss, rss)
                if rss > LIMITS["resident_memory_bytes"]:
                    memory_exceeded = True
                    _kill(process)
                    return
            await asyncio.sleep(LIMITS["memory_poll_seconds"])

    async def capture(stream):
        nonlocal exceeded
        result = bytearray()
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                return bytes(result)
            remaining = OUTPUT_BYTES - len(result)
            result.extend(chunk[:max(0, remaining)])
            if len(chunk) > remaining and not exceeded:
                exceeded = True
                _kill(process)
            # Drain and discard after killing: Process.wait() can otherwise
            # wait forever for an unread pipe buffer to close.

    readers = [asyncio.create_task(capture(process.stdout)),
               asyncio.create_task(capture(process.stderr))]
    monitor = asyncio.create_task(monitor_memory())
    timed_out = False
    try:
        try:
            await asyncio.wait_for(process.wait(), WALL_SECONDS)
        except TimeoutError:
            timed_out = True
            _kill(process)
            await process.wait()
        stdout, stderr = await asyncio.gather(*readers)
        return {"returncode": process.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "timed_out": timed_out, "output_limit": exceeded,
                "memory_limit": memory_exceeded, "peak_rss_bytes": peak_rss,
                "memory_monitor_error": memory_monitor_error}
    except asyncio.CancelledError:
        _kill(process)
        await process.wait()
        await asyncio.gather(*readers, return_exceptions=True)
        raise
    finally:
        monitor.cancel()
        for reader in readers:
            if not reader.done():
                reader.cancel()
        await asyncio.gather(*readers, monitor, return_exceptions=True)


async def run_exposed_policy(source: str, request: dict, output_dir: Path, attempt: int) -> dict:
    """Run an edited opponent policy and report what it decided."""
    report = {"move": None, "resign": None, "status": "setup_error", "error": None,
              "code_execution": True}
    try:
        if not isinstance(source, str) or len(source.encode()) > SOURCE_BYTES:
            raise ValueError("Invalid or oversized policy source")
        bundle = (Path(output_dir) / f"policy_{attempt:04d}").resolve()
        bundle.mkdir(parents=True, exist_ok=False)
        (bundle / "play.py").write_text(source)
        (bundle / "request.json").write_text(json.dumps(request) + "\n")
        runtime = _runtime()
        settings = {"limits": LIMITS}
        (bundle / "bootstrap.py").write_text(POLICY_BOOTSTRAP)
        report.update(bundle_path=str(bundle), source_sha256=_sha(source.encode()))
        if sys.platform == "linux":
            # Allow only standard-library children, excluding installed packages.
            # Never grant the runtime's parent/home directory or the checkout.
            stdlib = Path(runtime["stdlib"])
            read_paths = [str(bundle), *(str(p.resolve()) for p in stdlib.iterdir()
                          if p.name not in {"site-packages", "dist-packages", "__pycache__"})]
            # Standard-library extensions may load system shared libraries.
            triplet = os.uname().machine + "-linux-gnu"
            library_dirs = [Path("/lib") / triplet, Path("/usr/lib") / triplet,
                            Path("/lib64"), Path("/usr/lib64")]
            read_paths += [str(p.resolve()) for root in library_dirs
                           for p in root.glob("*.so*") if p.is_file()]
            settings["read_paths"] = sorted(set(read_paths))
            helper = Path(__file__).with_name("linux_policy_sandbox.py")
            (bundle / "sandbox_linux.py").write_bytes(helper.read_bytes())
            report.update(sandbox_backend="linux-landlock-seccomp", helper_sha256=_sha(helper.read_bytes()))
            command = [runtime["python"], "-I", "-S", str(bundle / "bootstrap.py")]
        elif sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file():
            (bundle / "sandbox.sb").write_text(sandbox_profile(bundle, runtime))
            report["sandbox_backend"] = "macos-seatbelt"
            command = ["/usr/bin/sandbox-exec", "-f", str(bundle / "sandbox.sb"),
                       runtime["python"], "-I", "-S", str(bundle / "bootstrap.py")]
        else:
            report.update(status="sandbox_unavailable",
                          error="No supported OS sandbox is available; no fallback.")
            (bundle / "worker_result.json").write_text(json.dumps(report, indent=2) + "\n")
            return report
        (bundle / "worker_settings.json").write_text(json.dumps(settings) + "\n")
        execution = await _execute(command, bundle)
        report.update({k: execution[k] for k in
                       ("returncode", "timed_out", "memory_limit", "output_limit")})
        if execution["timed_out"]:
            report.update(status="timeout", error="Policy exceeded its wall-time limit.")
        elif execution["memory_limit"]:
            report.update(status="memory_limit", error="Policy exceeded its memory limit.")
        elif execution["output_limit"]:
            report.update(status="output_limit", error="Policy exceeded its output limit.")
        elif execution["memory_monitor_error"]:
            report.update(status="execution_error", error=execution["memory_monitor_error"])
        elif "sandbox_apply" in execution["stderr"] or "sandbox-exec:" in execution["stderr"]:
            report.update(status="sandbox_unavailable", error=execution["stderr"][:2000])
        else:
            try:
                value = json.loads(execution["stdout"])
                if execution["returncode"] == 3 and isinstance(value, dict) and "sandbox_error" in value:
                    report.update(status="sandbox_unavailable", error=value["sandbox_error"])
                    (bundle / "worker_result.json").write_text(json.dumps(report, indent=2) + "\n")
                    return report
                if execution["returncode"] != 0 or "worker_error" in value:
                    raise ValueError(value.get("worker_error", "policy did not return a decision"))
            except (ValueError, TypeError, AttributeError) as exc:
                report.update(status="execution_error", error=str(exc)[:2000])
            else:
                report.update(move=value.get("move"), resign=bool(value.get("resign")),
                              status="completed")
        (bundle / "worker_result.json").write_text(json.dumps(report, indent=2) + "\n")
    except (OSError, ValueError, TypeError) as exc:
        report.update(error=f"{type(exc).__name__}: {exc}")
    return report
