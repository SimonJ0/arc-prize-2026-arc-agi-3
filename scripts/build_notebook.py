"""Splice bundled agent into notebooks/submission.ipynb."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "bundled_my_agent.py"
if not AGENT_SRC.exists():
    AGENT_SRC = ROOT / "agent" / "my_agent.py"

NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

agent_body = AGENT_SRC.read_text(encoding="utf-8")

notebook = {
    "cells": [
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": "!pip install --no-index --find-links /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels arc-agi python-dotenv\n"
        },
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": "with open('/tmp/my_agent.py', 'w', encoding='utf-8') as f:\n    f.write(" + json.dumps(agent_body) + ")\n"
        },
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": "import os\n# Kaggle competition execution stub\n"
        }
    ],
    "metadata": {
        "accelerator": "t4",
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Built submission notebook at: {NOTEBOOK_PATH}")
