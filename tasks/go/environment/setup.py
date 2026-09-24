"""Start an isolated host-side Go game, invisible to the model container."""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def setup(sandbox, row, env_dir):
    sandbox.start_mirror(
        site=HERE / "site", server=HERE / "server/service.py",
        hosts=("agents.internal",), serving=row["id"],
        overrides=Path(env_dir) / "rows.json",
        fallback_ips={"agents.internal": ["10.44.0.7"]})
