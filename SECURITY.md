# Security Policy

## Supported Versions

The following table outlines the security update lifecycle for releases of this repository:

| Version | Supported          | Security Maintenance Status |
| :---    | :---:              | :---                        |
| 0.1.x   | :white_check_mark: | Active Support              |
| < 0.1.0 | :x:                | Deprecated / EOL            |

---

## Reporting a Vulnerability

We prioritize the security and integrity of our codebase, agent runtime, and platform evaluation infrastructure. If you discover a security vulnerability or sensitive information disclosure, please report it responsibly.

### Responsible Disclosure Protocol
- **Do NOT** open a public issue, discussion, or pull request on GitHub for suspected security vulnerabilities.
- Send an encrypted email or report detailing the vulnerability to **`security@arcprize.org`** or directly to the project maintainers.
- Please include:
  1. Description of the vulnerability or security flaw.
  2. Steps to reproduce or proof-of-concept (PoC) code.
  3. Potential impact on execution environments, credential isolation, or platform integrity.
  4. Any proposed patches or mitigations.

### Response Timelines
- **Initial Acknowledgment:** Within 24 hours of report receipt.
- **Triage & Assessment:** Within 48 hours.
- **Patch Resolution & Advisory:** Within 7 calendar days depending on severity.

---

## Platform Security Architecture

This repository enforces stringent security controls designed to prevent data leakage, untrusted code execution, and unauthorized remote transmissions:

### 1. The Cryptographic Iron Rule & Submission Gate
- All competition submission artifacts (`submission.ipynb`, `submission.csv`, `agent/bundled_my_agent.py`) undergo local deterministic compilation and SHA-256 integrity hashing.
- Autonomous loops are **cryptographically barred** from transmitting or uploading submissions to external platform APIs (such as Kaggle or ARC Prize endpoints) without an explicit, time-bounded HMAC authorization token signed by an authenticated human operator (`cli.py request-approval`).

### 2. Zero Network Exfiltration & Clean-Room Bundling
- The ARC-AGI-3 deployment bundle (`agent/bundled_my_agent.py`) is verified using a clean-room offline test harness (`--verify-offline`) to guarantee that the agent operates 100% locally with zero external network requests or dynamic remote imports.
- Synthetic microworlds and validation environments execute in isolated local sub-processes.

### 3. Credential & Secret Management
- API tokens (`ARC_API_KEY`, `KAGGLE_KEY`, etc.) are never hardcoded or committed to source control.
- Credentials must be supplied via runtime environment variables or a local `.env` file that is strictly excluded via `.gitignore`.
- Automated pre-commit hooks and CI security scans reject any staged files containing credential tokens or private key patterns.
