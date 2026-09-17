"""Splice the bundled ARC-AGI-3 agent into notebooks/submission.ipynb.

The notebook follows the exact pattern used by Kaggle's official ARC-AGI-3 starter:
  Cell 1: install the arc-agi wheel and deps from offline competition dataset
  Cell 2: write my_agent.py to /tmp/my_agent.py
  Cell 3: in competition rerun, wait for gateway sidecar, wire framework, and run
  Cell 4: in commit / save-and-run-all, write dummy submission.parquet
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import nbformat

ACCELERATOR = "t4"

_ACCELERATORS = {
    "cpu": {"name": "none", "gpu": False},
    "t4": {"name": "nvidiaTeslaT4", "gpu": True},
    "p100": {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000", "gpu": True},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "bundled_my_agent.py"
if not AGENT_SRC.exists():
    AGENT_SRC = ROOT / "agent" / "my_agent.py"

NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"


def code_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source,
    }


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find agent source at {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text(encoding="utf-8")

    header_cell = markdown_cell(
        "# ARC Prize 2026 — ARC-AGI-3 Submission\n\n"
        "Built from `agent/bundled_my_agent.py` via `scripts/build_notebook.py`.\n"
        "Uncertainty-aware dual-loop agent with cryptographic provenance.",
        cell_id="cell-header",
    )

    install_cell = code_cell(
        "!pip install --no-index --find-links \\\n"
        "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv",
        cell_id="cell-install",
    )

    write_agent_cell = code_cell(
        "%%writefile /tmp/my_agent.py\n" + agent_body,
        cell_id="cell-agent",
    )

    run_cell_source = dedent(
        """\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # Copy the framework into a writable location.
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents

            # Drop our agent in as a framework template.
            !cp /tmp/my_agent.py \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

            # Register MyAgent in the framework's agent registry. We rewrite
            # __init__.py because the upstream version eagerly imports
            # templates with deps we don't ship (langgraph, smolagents, etc.).
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': MyAgent,
        }
        \"\"\")

            # Point the framework at the gateway sidecar.
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write(\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        \"\"\")

            # Run it. The gateway records every action and emits submission.parquet.
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
                python main.py --agent myagent
        """
    )
    run_cell = code_cell(run_cell_source, cell_id="cell-run")

    dummy_submission_cell = code_cell(
        dedent(
            """\
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                # Save-and-run-all (commit) mode: emit a dummy submission so the
                # commit succeeds. The real submission.parquet is produced by the
                # gateway during competition rerun.
                import pandas as pd
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'])
                submission.to_parquet('/kaggle/working/submission.parquet', index=False)
                submission.head()
            """
        ),
        cell_id="cell-dummy",
    )

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: {sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 5,
        "nbformat": 4,
        "cells": [
            header_cell,
            install_cell,
            write_agent_cell,
            run_cell,
            dummy_submission_cell,
        ],
    }

    # Validate notebook schema compliance
    nbformat.validate(notebook)
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb_dict = build()
    NOTEBOOK_PATH.write_text(json.dumps(nb_dict, indent=1), encoding="utf-8")
    print(
        f"[build_notebook] Successfully generated and validated {NOTEBOOK_PATH.relative_to(ROOT)}"
    )

    # Sync kernel-metadata.json
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        wanted = _ACCELERATORS[ACCELERATOR]["gpu"]
        changed = False
        if meta.get("enable_gpu") != wanted:
            meta["enable_gpu"] = wanted
            changed = True
        if "REPLACE_WITH_YOUR_USERNAME" in meta.get("id", ""):
            meta["id"] = meta["id"].replace("REPLACE_WITH_YOUR_USERNAME", "simonjosiah")
            changed = True
        if changed:
            METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(f"[build_notebook] Synced {METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
