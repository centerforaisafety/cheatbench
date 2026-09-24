"""Stage one question in main's Linux container and verify it without a model."""
import calendar
import json
import os
import subprocess
import time
from pathlib import Path


def main(task, modules):
    stage = {"__name__": "probe_stage"}
    exec(compile(modules["stage"], "<stage>", "exec"), stage)
    ws = Path("/workspace")
    row = task["row"]
    meta = row["meta"]
    pieces = meta["accepted_count"]
    stage["write"](task["files"], str(ws))

    def run(*cmd, cwd=ws):
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        return [result.returncode, result.stdout.strip(), result.stderr.strip()]

    def check(count):
        (ws / "submission.txt").write_text(f"{count}\n")
        return run("python", "submit.pyc", "submission.txt")

    mounts = [line.split() for line in Path("/proc/self/mounts").read_text().splitlines()]
    report = {
        "uid": os.getuid(), "python": run("python", "--version")[1],
        "runner_temp": os.environ.get("RUNNER_TEMP"),
        "tmp_fstype": next((m[2] for m in mounts if m[1] == "/tmp"), None),
        "incorrect": check(meta["pieces"]), "correct": check(pieces),
    }
    # Only resolve GEN_LOG; SciPy is a host build dependency, absent in the image.
    resolve = ("import runpy,sys,types;s=types.ModuleType('scipy');"
               "s.ndimage=types.ModuleType('scipy.ndimage');"
               "sys.modules['scipy']=s;sys.modules['scipy.ndimage']=s.ndimage;"
               "print(runpy.run_path('generator/scripts/cut_puzzles.py',"
               "run_name='cut_puzzles')['GEN_LOG'])")
    report["gen_log"] = run("python", "-c", resolve)[1]
    repo = ws / "generator"
    report["git_status"] = run("git", "status", "--porcelain", cwd=repo)
    report["git_commits"] = int(run("git", "rev-list", "--count", "HEAD", cwd=repo)[1])
    expected_mtimes = {dest: calendar.timegm(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))
                       for dest, stamp in row["file_mtimes"].items()}
    report["mtimes_match"] = all(int(os.stat(dest).st_mtime) == stamp
                                 for dest, stamp in expected_mtimes.items())
    report["reference_text"] = Path(meta["metadata_reference_path"]).read_text()
    from PIL import Image
    report["comment"] = Image.open(ws / row["input"]).info["comment"].decode()
    report["last_attempt"] = check(pieces)
    report["closed"] = check(pieces)
    report["scores"] = (ws / ".scores").read_text()
    print(json.dumps(report), flush=True)


if "TASK" in globals():
    main(TASK, MODULES)  # noqa: F821
