"""OURS: no Harbor counterpart -- Harbor's `src/harbor/models/trial/paths.py` fixes
the paths INSIDE the container (/logs, /tests, /solution); these are the HOST
paths enroot needs, environment-variable driven so no default names a person.

Where the sandbox's files live, and which image a run picks up.

Every path here is an environment variable with a default, and no default names
a person. A committed file that said `/data/<someone>/images/...` would be both
a machine-specific literal and, if it ever reached the container, a way for the
agent to name the operator.

    RH_IMAGES_DIR      where built images are kept.        default ~/images
    RH_ENROOT_IMAGE    an explicit image path, wins over
                       everything including a task's
                       `image:` key.                        default unset
    RH_ENROOT_SCRATCH  node-local scratch for enroot's
                       data/runtime/temp/cache dirs.        default
                       /mnt/localdisk/enroot where that disk exists, else
                       $TMPDIR/enroot
    ENROOT_DATA_PATH   } enroot's own variables, honoured as-is when set;
    ENROOT_RUNTIME_ROOT} otherwise derived from RH_ENROOT_SCRATCH.
    ENROOT_TEMP_PATH   }
    ENROOT_CACHE_PATH  }
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# The image FILE NAME is visible to the agent: enroot leaves squashfuse running
# alongside the payload, and while the private pid namespace hides it from `ps`,
# the name must stay something an ordinary Python container would be called. It
# says "python311" and nothing else.
DEFAULT_IMAGE_NAME = "python311"

IMAGES_DIR = os.environ.get("RH_IMAGES_DIR") or os.path.expanduser("~/images")

# enroot writes several large trees; on a cluster they belong on node-local
# disk. /mnt/localdisk is that disk here, and $TMPDIR is the portable fallback,
# so a fresh clone on a laptop needs no configuration.
SCRATCH = os.environ.get("RH_ENROOT_SCRATCH") or (
    "/mnt/localdisk/enroot" if os.path.isdir("/mnt/localdisk")
    else os.path.join(tempfile.gettempdir(), "enroot"))


def image_path(name: str = "") -> str:
    """A task's `image:` value, resolved to a file.

    A bare name (`python311`) is a file in RH_IMAGES_DIR; anything with a
    separator or an extension is taken as the path it is. That is what lets a
    task name a different image without any code changing.
    """
    name = name or DEFAULT_IMAGE_NAME
    if os.sep in name or name.endswith(".sqsh"):
        return os.path.abspath(os.path.expanduser(name))
    return os.path.join(IMAGES_DIR, f"{name}.sqsh")


DEFAULT_IMAGE = os.environ.get("RH_ENROOT_IMAGE") or image_path()


def resolve_image(name: str = "") -> str:
    """The image a run should use: RH_ENROOT_IMAGE wins, then the task's name."""
    return os.environ.get("RH_ENROOT_IMAGE") or image_path(name)


# enroot's data dir is derived from `id -u`, which is 0 inside our user
# namespace; without this it would look under .../user-0 and fail.
DEFAULT_DATA_PATH = os.environ.get("ENROOT_DATA_PATH") or \
    f"{SCRATCH}/enroot_data/user-{os.getuid()}"
DEFAULT_RUNTIME_ROOT = os.environ.get("ENROOT_RUNTIME_ROOT") or \
    f"{SCRATCH}/enroot_runtime"

# Our replacement for /etc/enroot: every stock hook and both stock fstabs are
# dropped. It lives next to the code that sets ENROOT_SYSCONF_PATH to it, not in
# an `image/` directory -- it configures the RUNTIME, not the image. See the
# files themselves; every line in them is load-bearing.
SYSCONF = Path(__file__).resolve().parent / "enroot" / "sysconf"

# enroot merges ${ENROOT_CONFIG_PATH}/{hooks,mounts,environ}.d on top of the
# sysconf ones. Point it at an empty directory of ours so a stray
# ~/.config/enroot on some node cannot inject a mount into the sandbox.
USERCONF = Path(__file__).resolve().parent / "enroot" / "userconf"
