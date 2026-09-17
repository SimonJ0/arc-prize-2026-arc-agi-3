# ARC-AGI-3 Cognitive Agent & LMSYS Autonomous Research Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](http://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 63/63 Passing](https://img.shields.io/badge/tests-63%2F63%20passing-brightgreen.svg)]()

A unified, production-grade autonomous intelligence repository housing two state-of-the-art competitive ML and cognitive agent architectures:

1. **ARC-AGI-3 Cognitive Agent Platform (`src/arc_agent`, `src/arc_core`, `agent/`)**  
   An uncertainty-aware epistemic agent built for the **ARC Prize 2026 (ARC-AGI-3)**. Incorporates a 4-level cognitive hierarchy, persistent reasoning states, automatic context compaction, Bayesian world-model belief states, heuristic A* pathfinding, and strict runtime legality adaptation.

2. **LMSYS Chatbot Arena Autonomous Research Loop (`src/core`, `src/features`, `src/models`, `src/submit`)**  
   An autonomous self-improving machine learning research loop for pairwise LLM preference prediction. Features anti-symmetric feature engineering, dense Latent Semantic Analysis (LSA) projections, LightGBM GBDT models with exact Test-Time Augmentation (TTA) symmetry, nested CV ensemble blending, temperature calibration, and the **Cryptographic Iron Rule** submission gate.

---

## Repository Architecture

```
charming-hertz/
├── .agents/
│   └── skills/self-improving-ai-ml/ # Autonomous research loop skill & templates
├── agent/
│   ├── my_agent.py                  # Modular ARC-AGI-3 agent entrypoint
│   └── bundled_my_agent.py          # Standalone, zero-dependency Kaggle bundle
├── configs/
│   ├── default_config.yaml          # LMSYS experiment loop & feature configs
│   ├── hypotheses.yaml              # LMSYS prioritized hypothesis queue
│   └── arc_hypotheses.yaml          # ARC-AGI-3 agent exploration hypotheses
├── docs/
│   └── adr/                         # Architecture Decision Records
├── notebooks/
│   ├── kernel-metadata.json         # Kaggle notebook competition metadata
│   └── submission.ipynb             # Standalone competition submission notebook
├── scripts/
│   ├── bundle_agent.py              # Inlines agent into zero-dependency bundle
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
│   │   │   └── epistemic_policy.py  # Epistemic exploration & A* pathfinding
│   │   └── world_model/
│   │       └── belief_state.py      # Bayesian belief updates & hypothesis falsification
│   ├── arc_core/                    # ARC Platform Evaluation & Gating
│   │   ├── arc_loop.py              # Autonomous hypothesis testing loop for ARC
│   │   ├── contracts.py             # Immutable domain dataclasses & types
│   │   ├── gating.py                # Verification gates (legality, efficiency, rhae)
│   │   ├── metrics.py               # Robust Highest Action Efficiency (RHAE) metrics
│   │   └── platform_bench.py        # Official Arcade platform benchmark & multi-budget suite
│   ├── core/                        # LMSYS Research Engine Core
│   │   ├── adaptive.py              # Dynamic hypothesis generation & mutation
│   │   ├── calibration.py           # Post-hoc temperature calibration
│   │   ├── gates.py                 # Leakage, symmetry, and CV improvement gates
│   │   ├── guardrails.py            # Entropy bounds & anomaly detection
│   │   ├── loop.py                  # SelfDrivingResearchLoop execution engine
│   │   └── metrics.py               # Multi-class Log Loss, ECE, and symmetry divergence
│   ├── data/
│   │   ├── loader.py                # Safe dataset loading with validation
│   │   └── splitter.py              # Stratified K-Fold cross-validation splitter
│   ├── evaluation/
│   │   ├── diagnostics.py           # Model failure analysis & calibration diagnostics
│   │   └── html_reporter.py         # Visual HTML report generator
│   ├── features/
│   │   ├── extractor.py             # Anti-symmetric tabular feature engineering
│   │   ├── lsa_vectorizer.py        # Dense TruncatedSVD / LSA text projections
│   │   └── text_vectorizer.py       # Differential TF-IDF token vectorizer
│   ├── models/
│   │   ├── base.py                  # Abstract base model with strict TTA symmetry
│   │   ├── ensemble.py              # Nested cross-validation convex ensemble blender
│   │   ├── gbdt_classifier.py       # LightGBM multi-class GBDT predictor
│   │   ├── length_prior.py          # Empirical response length prior baseline
│   │   ├── tfidf_linear.py          # Differential sparse linear classifier
│   │   └── transformer_head.py      # Symmetric Cross-Encoder / DeBERTa head interface
│   └── submit/
│       ├── generator.py             # Submissions & LZMA-compressed Kaggle kernel (< 1MB)
│       └── submission_gate.py       # Cryptographic Iron Rule authorization gate
├── tests/
│   ├── microworlds/                 # Synthetic deterministic test environments
│   │   ├── test_coordinate_selection.py
│   │   ├── test_delayed_effects.py
│   │   ├── test_movement_induction.py
│   │   └── test_reversibility.py
│   └── test_*.py                    # 16 unit & integration test modules (63 tests)
├── .env.example                     # Environment template (ARC & Kaggle credentials)
├── .gitignore                       # Industry-standard comprehensive gitignore
├── CONTRIBUTING.md                  # Contribution guidelines & Conventional Commits
├── LICENSE                          # MIT License (2026)
├── pyproject.toml                   # PEP 517/621 configuration (ruff, mypy, pytest)
└── requirements.txt                 # Pinned, reproducible dependency manifest
```

---

## Core System Highlights

### 1. ARC-AGI-3 Cognitive Agent Architecture
- **DRE-Bench 4-Level Cognitive Hierarchy**:
  - *Level 1 (Attribute)*: Detects color palettes, spatial bounds, and foreground clusters.
  - *Level 2 (Spatial Relations)*: Computes distances, alignment, and contact manifolds.
  - *Level 3 (Sequential Planning)*: Deque-based macro-action queue preventing single-step myopia.
  - *Level 4 (Intuitive Physics)*: Caches movement step sizes, blocked corridors, and lethal hazards.
- **Reasoning Persistence & Context Compaction**:
  - Eliminates context rot by compressing past steps into concise semantic transition summaries.
  - Retains active goals, death coordinates, and verified causal action effects across turns.
- **Hard Legality Adapter**:
  - Enforces zero `400 Bad Request` errors: ensures `GAME_OVER` strictly triggers `RESET`, strips coordinates from non-coordinate actions, and clamps `ACTION6` coordinates to $[0, 63]$.
- **Cryptographic Iron Rule Submission Gate**:
  - Zero network access during compilation; issues HMAC-signed authorization tokens expiring in 2 hours. Submissions require explicit operator confirmation.

### 2. LMSYS Preference Prediction Research Loop
- **Anti-Symmetric Feature Engineering**:
  - Exact anti-symmetry guarantee: $\phi(B, A) = -\phi(A, B)$ on differential features.
  - Token counts, markdown header counts, code block metrics, readability scores, and lexical overlap.
- **Dense LSA Vectorization**:
  - Dimensionality reduction via `TruncatedSVD` on differential TF-IDF matrices, captured in `float16` for compact kernel serialization.
- **Exact Test-Time Augmentation (TTA) Symmetry**:
  - Enforces invariant probability assignments:
    $$P_{\text{sym}}(\text{Win A}) = \frac{1}{2} \left[ P(A > B) + P'(B > A) \right]$$
- **Nested Cross-Validation Ensemble Blending**:
  - Non-negative convex optimization with out-of-fold cross-validation, avoiding in-sample weight overfitting.
- **Kernel Compression**:
  - Serializes booster models and LSA projections via extreme LZMA compression (`preset=9 | PRESET_EXTREME`) down to $< 1\text{ MB}$, satisfying Kaggle API notebook constraints.

---

## Quickstart Guide

### 1. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/SimonJ0/arc-prize-2026-arc-agi-3.git
cd arc-prize-2026-arc-agi-3

# Create and activate virtual environment using uv or standard venv
uv venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies in editable mode with development tools
uv pip install -e ".[dev]"
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and configure your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
ARC_API_KEY=your_arc_api_key_here
ARC_BASE_URL=https://arcprize.org/api
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

---

## CLI & Entrypoint Reference

### ARC-AGI-3 Agent CLI (`cli.py` / `arc-cli`)

The CLI enforces strict command boundaries in compliance with the **Cryptographic Iron Rule**:

```bash
# 1. Run the autonomous self-improving research loop over ARC hypotheses
python cli.py run-arc --hypotheses configs/arc_hypotheses.yaml

# 2. Play a game locally (supports official Arcade environments or synthetic navigation)
python cli.py play --game synthetic_nav --max-steps 30

# 3. Evaluate the agent against official ARC-AGI-3 platform games
python cli.py eval-platform --split train --max-actions 100

# 4. Run multi-budget efficiency benchmark (e.g. 25, 50, 100 actions)
python cli.py eval-platform --split train --budgets 25,50,100

# 5. Build submission notebook and generate SHA-256 provenance hashes (Offline, Zero network)
python cli.py build-submission --accelerator t4

# 6. Request formal authorization brief with HMAC signed token
python cli.py request-approval --exp-id EXP_ARC_001

# 7. Execute authorized submission upload to Kaggle
python cli.py submit --approval "<signed-hmac-token>"
```

### LMSYS Research Loop Entrypoint (`run_loop.py`)

```bash
# Run the autonomous hypothesis research loop with fast cross-validation
python run_loop.py
```

### Bundling & Notebook Generation Scripts

```bash
# Bundle modular agent into standalone single-file agent
python scripts/bundle_agent.py

# Build and validate Kaggle submission notebook
python scripts/build_notebook.py
```

---

## Quality Assurance & Verification

We adhere to tier-one software engineering rigor with automated linting, formatting, and test suites:

### 1. Test Suite (100% Passing)
```bash
pytest
```
Executes all 63 unit, integration, and microworld tests covering:
- ARC contracts, legality adapters, and RHAE metrics
- DRE-Bench cognitive hierarchy and OpenAI reasoning persistence
- Synthetic microworlds (movement induction, delayed effects, reversibility, coordinate selection)
- LMSYS feature extraction, TTA symmetry, GBDT classifiers, calibration, and nested ensembles
- Standalone kernel generation and cryptographic submission gates

### 2. Linting & Static Analysis
```bash
# Check code style with Ruff
ruff check .

# Automated style formatting
ruff format --check .

# Strict type checking with Mypy
mypy src tests
```

---

## Contributing & Governance

- Please consult [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines on branching, code style, and [Conventional Commits](https://www.conventionalcommits.org/).
- All pull requests must pass `ruff check`, `mypy src tests`, and `pytest` with zero warnings before approval.

---

## License

This project is licensed under the [MIT License](LICENSE) — see the LICENSE file for details.
