import base64
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "promotion_saved_grader", ROOT / "tasks/protein_design/grader.py")
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)


def test_saved_work_keeps_changes_and_excludes_inputs_git_and_symlinks(tmp_path):
    repo = tmp_path / "promotion_worktest"
    repo.mkdir()
    (repo / "input.txt").write_text("original input")
    state = {}
    observer = grader.make_observer({"vars": {"work_dir": str(repo)}}, {}, "", state)
    (repo / "report.md").write_text("Final explanation")
    (repo / ".git").mkdir()
    (repo / ".git/config").write_text("private repo metadata")
    outside = tmp_path / "outside.txt"
    outside.write_text("must not export")
    (repo / "link.txt").symlink_to(outside)
    observer.finish()
    assert [f["name"] for f in state["saved_work"]] == ["report.md"]
    assert base64.b64decode(state["saved_work"][0]["b64"]) == b"Final explanation"


def test_checker_initializes_the_named_repository(tmp_path):
    repo = tmp_path / "promotion_worktest"
    repo.mkdir()
    grader.make_tool({"vars": {"work_dir": str(repo)}}, {}, str(tmp_path))
    assert (repo / ".git").is_dir()
