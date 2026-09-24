#!/usr/bin/env python3
"""Build the interviewer environment from twenty fixed brief/reference pairs."""
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK.parent.parent))
from tasks.creative_writings import _builder as builder
from tasks.creative_writings.rows import ROWS

builder.configure_task(TASK, ROWS, "writing.md")

if __name__ == "__main__":
    raise SystemExit(builder.main())
