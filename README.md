# ARC-AGI-3 Cognitive Agent Architecture (ARC Prize 2026)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](http://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 33/33 Passing](https://img.shields.io/badge/tests-33%2F33%20passing-brightgreen.svg)]()

An autonomous, uncertainty-aware epistemic agent built for the **ARC Prize 2026 (ARC-AGI-3)** competition. The architecture replaces brittle heuristic templates with **empirical 1-step dynamics learning**, a **4-level cognitive hierarchy (DRE-Bench)**, **Bayesian belief-state tracking**, **surprise-driven plan invalidation**, **deadlock loop breaking**, and strict **runtime legality adaptation**.

---

## Repository Architecture

```
arc-prize-2026-arc-agi-3/
├── .agents/
│   └── skills/self-improving-ai-ml/ # Autonomous agent research skill
├── agent/
│   ├── my_agent.py                  # Modular ARC-AGI-3 agent entrypoint
│   └── bundled_my_agent.py          # Standalone, zero-dependency Kaggle bundle (1,924 lines)
├── configs/
│   └── arc_hypotheses.yaml          # ARC-AGI-3 agent exploration hypotheses
├── docs/
│   └── adr/                         # Architecture Decision Records (ADR-001)
├── notebooks/
│   ├── kernel-metadata.json         # Kaggle notebook competition metadata
│   └── submission.ipynb             # Standalone competition submission notebook (< 1MB)
├── scripts/
│   ├── bundle_agent.py              # Inlines modular agent into zero-dependency bundle
│   └── build_notebook.py            # Generates validated Kaggle submission notebook
├── src/
│   ├── arc_agent/                   # ARC-AGI-3 Cognitive Agent Subsystem
│   │   ├── legality_adapter.py      # Hard guardrail (0% illegal actions, GAME_OVER handling)
│   │   ├── memory/
│   │   │   ├── reasoning_state.py   # OpenAI Reasoning Persistence & Context Compaction
│   │   │   └── scoped_memory.py     # Episode scoped memory & transition event logging
│   │   ├── perception/
│   │   │   ├── cognitive_hierarchy.py # DRE-Bench 4-level perception hierarchy
│   │   │   └── layered_perception.py  # Spatial component & motion extraction
│   │   ├── planning/
│   │   │   └── epistemic_policy.py  # Epistemic exploration & heuristic A* pathfinding
│   │   └── world_model/
│   │       └── belief_state.py      # Bayesian belief updates & empirical dynamics learning
│   ├── arc_core/                    # ARC Platform Evaluation & Gating
│   │   ├── arc_loop.py              # Autonomous hypothesis testing loop for ARC
│   │   ├── contracts.py             # Immutable domain dataclasses & game contracts
│   │   ├── gating.py                # Verification gates (legality, efficiency, RHAE)
│   │   ├── metrics.py               # Robust Highest Action Efficiency (RHAE) metrics
│   │   └── platform_bench.py        # Official Arcade platform benchmark & multi-budget suite
│   └── submit/
│       └── submission_gate.py       # Cryptographic Iron Rule authorization gate
├── tests/
│   ├── microworlds/                 # Deterministic synthetic verification environments
│   │   ├── test_coordinate_selection.py
│   │   ├── test_delayed_effects.py
│   │   ├── test_movement_induction.py
│   │   └── test_reversibility.py
│   ├── test_arc_contracts_and_legality.py
│   ├── test_arc_metrics.py
│   ├── test_cognitive_hierarchy_and_persistence.py
│   ├── test_platform_bench.py
│   └── test_submission_gate.py
├── .env.example                     # Environment template (ARC API credentials)
├── .gitignore                       # Industry-standard comprehensive gitignore
├── CONTRIBUTING.md                  # Contribution guidelines & Conventional Commits
├── LICENSE                          # MIT License (2026)
├── pyproject.toml                   # PEP 517/621 packaging (ruff, mypy, pytest)
└── requirements.txt                 # Pinned, reproducible dependency manifest
```

---

## Core Cognitive System Pillars

### 1. DRE-Bench 4-Level Cognitive Hierarchy
- **Level 1 (Attribute)**: Extracts foreground entities, background palette, and dynamic bounding boxes.
- **Level 2 (Spatial Relations)**: Computes centroid distances, cardinal alignments, and contact manifolds.
- **Level 3 (Sequential Macro-Planning)**: Maintains a FIFO queue of verified subgoals, preventing single-step oscillation.
- **Level 4 (Intuitive Physics)**: Learns causal motion vectors $(dy, dx)$ and static obstacles through empirical observation.

### 2. Empirical Transition Dynamics & 1-Step Verification
- No hardcoded movement assumptions. The agent records empirical $(dy, dx)$ displacements and marks actions verified only upon reaching $\ge 85\%$ prediction consistency across at least 2 non-blocked executions.

### 3. Reasoning Persistence & Context Compaction
- Persists causal learning (`hazard_colors`, `death_coords`, `action_effects`) across game levels.
- Re-initializes ephemeral state (`visitation_counts`, `falsified_goals`, `active_macro_plan`) upon level progression.
- Compacts historical action transitions into semantic summaries to eliminate context window degradation.

### 4. Surprise Detection & Loop Breaking
- Predicts expected avatar coordinates for the next turn. If the environment diverges, `check_and_handle_surprise` instantly aborts the macro-plan and clears invalid trajectories.
- Tracks coordinate visitation frequency; if oscillation ($\ge 3$ visits) occurs, an epistemic deadlock-breaker injects orthogonal exploratory actions.

### 5. Runtime Legality Adapter
- **Zero 400 Bad Requests**: Intercepts `GAME_OVER` to force `RESET`, strips coordinates from non-coordinate actions, and clamps `ACTION6` coordinates to $[0, 63]$.

---

## Quickstart Guide

### 1. Installation

```bash
git clone https://github.com/SimonJ0/arc-prize-2026-arc-agi-3.git
cd arc-prize-2026-arc-agi-3

# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install in editable mode with development tools
uv pip install -e ".[dev]"
```

### 2. Configuration

```bash
cp .env.example .env
```

Set your credentials in `.env`:
```ini
ARC_API_KEY=your_arc_api_key_here
ARC_BASE_URL=https://arcprize.org/api
```

---

## CLI & Evaluation Reference

```bash
# 1. Run the autonomous self-improving research loop over ARC hypotheses
python cli.py run-arc --hypotheses configs/arc_hypotheses.yaml

# 2. Play a game locally (supports official Arcade environments or synthetic navigation)
python cli.py play --game synthetic_nav --max-steps 30

# 3. Evaluate against official ARC-AGI-3 platform games
python cli.py eval-platform --split train --max-actions 100

# 4. Run multi-budget holdout benchmark (e.g. 50, 100, 200 actions)
python cli.py eval-platform --split holdout --budgets 50,100,200

# 5. Build submission notebook and generate SHA-256 provenance hashes (Offline, Zero network)
python cli.py build-submission --accelerator t4

# 6. Request formal authorization brief with HMAC signed token
python cli.py request-approval --exp-id EXP_ARC_001

# 7. Execute authorized submission upload to Kaggle
python cli.py submit --approval "<signed-hmac-token>"
```

---

## Offline Clean-Room Bundler

The entire agent can be bundled into a standalone, zero-dependency Python script for Kaggle execution:

```bash
python scripts/bundle_agent.py --output agent/bundled_my_agent.py --verify-offline
```

---

## Quality Assurance & Verification

```bash
# 1. Execute full unit and integration test suite (33/33 passing)
pytest -v

# 2. Strict type checking
mypy src tests

# 3. Code formatting and linting
ruff check .
ruff format --check .
```

---

## License

Distributed under the [MIT License](LICENSE).
