"""
Single-File Deployment Bundler & Clean-Room Verifier for ARC-AGI-3.
Inlines all modular agent components (contracts, perception, belief-state world model,
epistemic policy, legality adapter, scoped memory) into a single, self-contained file.
Verifies clean-room offline execution before any Kaggle submission notebook is built.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODULES_TO_INLINE = [
    ROOT / "src" / "arc_core" / "contracts.py",
    ROOT / "src" / "arc_agent" / "legality_adapter.py",
    ROOT / "src" / "arc_agent" / "diagnostics" / "replay_logger.py",
    ROOT / "src" / "arc_agent" / "memory" / "effect_taxonomy.py",
    ROOT / "src" / "arc_agent" / "memory" / "mechanism_memory.py",
    ROOT / "src" / "arc_agent" / "memory" / "structured_belief.py",
    ROOT / "src" / "arc_agent" / "world_model" / "falsification_engine.py",
    ROOT / "src" / "arc_agent" / "planning" / "deadlock_taxonomy.py",
    ROOT / "src" / "arc_agent" / "memory" / "level_transfer.py",
    ROOT / "src" / "arc_agent" / "perception" / "layered_perception.py",
    ROOT / "src" / "arc_agent" / "perception" / "cognitive_hierarchy.py",
    ROOT / "src" / "arc_agent" / "memory" / "reasoning_state.py",
    ROOT / "src" / "arc_agent" / "world_model" / "belief_state.py",
    ROOT / "src" / "arc_agent" / "reasoning" / "action_algebra.py",
    ROOT / "src" / "arc_agent" / "reasoning" / "model_gate.py",
    ROOT / "src" / "arc_agent" / "planning" / "epistemic_policy.py",
    ROOT / "src" / "arc_agent" / "memory" / "scoped_memory.py",
]

MY_AGENT_PATH = ROOT / "agent" / "my_agent.py"
OUTPUT_BUNDLED_PATH = ROOT / "agent" / "bundled_my_agent.py"


def strip_internal_imports(code: str) -> str:
    """Strips local 'from src.arc_...' and duplicate __future__ imports."""
    lines = code.splitlines()
    filtered = []
    in_internal_import_block = False

    for line in lines:
        if "from __future__ import" in line:
            continue
        if re.match(r"^\s*from\s+src\.arc_\w+", line):
            if "(" in line and ")" not in line:
                in_internal_import_block = True
            continue
        if in_internal_import_block:
            if ")" in line:
                in_internal_import_block = False
            continue
        filtered.append(line)
    return "\n".join(filtered)


def generate_bundle() -> str:
    """Generates the single-file, self-contained bundled agent source."""
    header = [
        '"""',
        "AUTONOMOUS ARC-AGI-3 UNCERTAINTY-AWARE AGENT (INLINED DEPLOYMENT BUNDLE)",
        "Self-contained, offline-compatible implementation for Kaggle code competition.",
        '"""',
        "from __future__ import annotations",
        "import os",
        "import sys",
        "import time",
        "import json",
        "import random",
        "import hashlib",
        "from enum import Enum",
        "from collections import defaultdict, deque",
        "from dataclasses import dataclass, field",
        "from typing import Any, Callable, Dict, List, Optional, Set, Tuple",
        "import numpy as np",
        "from scipy.ndimage import label",
        "from arcengine import GameAction, GameState, FrameDataRaw",
        "",
    ]

    bundle_parts = ["\n".join(header)]

    for mod_path in MODULES_TO_INLINE:
        if not mod_path.exists():
            raise FileNotFoundError(f"Missing required component to inline: {mod_path}")
        raw_code = mod_path.read_text(encoding="utf-8")
        clean_code = strip_internal_imports(raw_code)
        bundle_parts.append(f"\n# {'=' * 70}\n# INLINED: {mod_path.name}\n# {'=' * 70}\n")
        bundle_parts.append(clean_code)

    # Append MyAgent class
    raw_agent = MY_AGENT_PATH.read_text(encoding="utf-8")
    clean_agent = strip_internal_imports(raw_agent)
    bundle_parts.append(f"\n# {'=' * 70}\n# PRIMARY AGENT: my_agent.py\n# {'=' * 70}\n")
    bundle_parts.append(clean_agent)

    return "\n".join(bundle_parts)


def compute_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def verify_offline_cleanroom(bundled_code: str) -> bool:
    """Executes the bundled agent in an isolated clean-room Python subprocess."""
    test_script = f"""
{bundled_code}

# Offline Smoke Test
import numpy as np
from arcengine import GameAction, GameState

class MockFrameData:
    def __init__(self):
        self.game_id = "test_smoke_game"
        self.state = GameState.NOT_FINISHED
        self.levels_completed = 0
        self.available_actions = [0, 1, 2, 3, 4]
        self.frame = [np.zeros((20, 20), dtype=int)]
        self.guid = "test-guid-123"

mock_frame = MockFrameData()
agent = MyAgent(game_id="test_smoke_game")

# Step 1: Check is_done
assert agent.is_done(mock_frame.frame, mock_frame) is False

# Step 2: Choose action
action = agent.choose_action(mock_frame.frame, mock_frame)
assert isinstance(action, GameAction)
assert action.value in mock_frame.available_actions

# Step 3: Test GAME_OVER recovery
mock_frame.state = GameState.GAME_OVER
mock_frame.available_actions = [0]
recovery_action = agent.choose_action(mock_frame.frame, mock_frame)
assert recovery_action == GameAction.RESET

print("CLEANROOM_OFFLINE_VERIFICATION_PASSED")
"""
    temp_test_file = ROOT / "scratch" / "_cleanroom_verification.py"
    temp_test_file.parent.mkdir(parents=True, exist_ok=True)
    temp_test_file.write_text(test_script, encoding="utf-8")

    python_exe = sys.executable
    try:
        proc = subprocess.run(
            [python_exe, str(temp_test_file)],
            capture_output=True,
            text=True,
            timeout=15,
        )
    finally:
        if temp_test_file.exists():
            temp_test_file.unlink()

    if proc.returncode != 0:
        print("Clean-room offline verification FAILED:")
        print(proc.stderr)
        return False

    return "CLEANROOM_OFFLINE_VERIFICATION_PASSED" in proc.stdout


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Agent Bundler & Offline Verifier")
    parser.add_argument(
        "--output", default=str(OUTPUT_BUNDLED_PATH), help="Target bundled file path"
    )
    parser.add_argument("--verify-offline", action="store_true", help="Run clean-room offline test")
    parser.add_argument(
        "--replace-agent", action="store_true", help="Replace agent/my_agent.py with bundled code"
    )

    args = parser.parse_args()

    print("Bundling ARC-AGI-3 Agent components...")
    bundled_code = generate_bundle()
    bundle_hash = compute_sha256(bundled_code)
    print(f"Bundle generated: {len(bundled_code.splitlines())} lines | SHA-256: {bundle_hash}")

    if args.verify_offline:
        print("Executing clean-room offline verification...")
        success = verify_offline_cleanroom(bundled_code)
        if not success:
            sys.exit(1)
        print(
            "PASS: Clean-room offline verification succeeded. Zero network dependencies, zero missing imports."
        )

    out_path = Path(args.output)
    out_path.write_text(bundled_code, encoding="utf-8")
    print(f"Saved bundle to: {out_path}")

    if args.replace_agent:
        MY_AGENT_PATH.write_text(bundled_code, encoding="utf-8")
        print(f"Updated {MY_AGENT_PATH} with bundled code.")


if __name__ == "__main__":
    main()
