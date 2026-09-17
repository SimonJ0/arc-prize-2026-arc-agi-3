# Contributing Guidelines

Thank you for contributing to the **ARC-AGI-3 Agent & LMSYS Research Loop** platform. We maintain rigorous, tier-one engineering standards across all contributions.

---

## 1. Development Environment Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- `uv` (recommended) or standard `venv` + `pip`
- Git

### Initializing the Workspace
```bash
# Clone the repository
git clone <repo-url>
cd charming-hertz

# Create and activate virtual environment
uv venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies (including development tools)
uv pip install -e ".[dev]"
```

### Environment Variables
Copy `.env.example` to `.env` and provide your credentials:
```bash
cp .env.example .env
```
> **Security Notice**: Never commit `.env` or credentials to git. The `.gitignore` strictly protects sensitive patterns.

---

## 2. Code Quality & Standards

All code submitted to this repository must meet strict automated quality gates before merge.

### Formatting & Linting
We enforce automated code formatting and linting via [Ruff](https://docs.astral.sh/ruff/):
```bash
# Check code for linting issues
ruff check .

# Automatically fix supported linting violations
ruff check --fix .

# Enforce consistent code formatting
ruff format .
```

### Static Type Analysis
Type annotations are expected on all public functions, classes, and interfaces:
```bash
mypy src tests
```

### Testing Suite
All tests must pass 100% cleanly without regressions before committing:
```bash
pytest
```

---

## 3. Git Workflow & Commit Conventions

We strictly adhere to the [Conventional Commits specification](https://www.conventionalcommits.org/):

### Commit Types
- `feat:` Introduces a new capability, model, or perception module.
- `fix:` Patches a bug, numerical instability, or test failure.
- `docs:` Documentation updates, docstrings, or README revisions.
- `style:` Formatting, whitespace, or import ordering adjustments (no logic change).
- `refactor:` Code restructuring without changing observable behavior.
- `perf:` Performance optimizations or memory reductions.
- `test:` Adding or refining unit tests, integration tests, or benchmarks.
- `chore:` Dependency bumps, CI updates, configuration changes.

### Example Commit Messages
```
feat(perception): add color-clustering and object persistence tracking
fix(planning): resolve 1D A* search collapse on narrow corridors
docs(readme): add dual-platform architecture diagrams and quickstart
chore(deps): pin requirements and update pyproject.toml
```

### Branching Strategy
- `main`: Protected production branch. All code must pass CI and reviews before merging.
- Feature branches: `feat/<feature-name>`, `fix/<issue-name>`, `chore/<task-name>`.

---

## 4. Cryptographic Iron Rule & Submission Gate

For Kaggle and ARC-AGI-3 platform submissions, this codebase enforces the **Cryptographic Iron Rule**:
1. **Compilation & Hashing**: Notebooks and agent bundles must be compiled locally with SHA-256 hashes generated offline (`arc-cli build-submission`).
2. **Signed Token Authorization**: No code may upload or transmit model artifacts without an explicit, time-limited HMAC approval token signed by an authorized human operator (`arc-cli request-approval`).
3. **Audit Trail**: Every submission event logs SHA-256 hashes, timestamps, and evaluator metrics for full reproducibility.

---

## 5. Submitting Pull Requests
1. Ensure your branch is rebased on latest `main`.
2. Run `ruff format .`, `ruff check .`, and `pytest` locally.
3. Open a Pull Request with a clear description of the problem solved, architectural implications, and test verification output.
