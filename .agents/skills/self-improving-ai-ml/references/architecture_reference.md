# Architecture Reference: Self-Improving AI/ML Framework

This document provides in-depth mathematical formulations, structural design patterns, and concrete code contracts for implementing the self-improving AI/ML research framework.

---

## 1. System Architecture & Lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant Loop as Research Loop Controller
    participant Task as TaskSpec & Metrics
    participant Splitter as Grouped K-Fold Splitter
    participant Model as Model Predictor (with TTA)
    participant Gate as Validation Gatekeeper
    participant Blender as Outer Meta-CV Blender
    participant Reporter as Diagnostics & HTML Engine
    participant Gatekeeper as Cryptographic Iron Rule Gate

    Loop->>Task: Load task specification & probability semantics
    Loop->>Splitter: Generate leak-free grouped/temporal splits
    loop For each Fold k in 1..K
        Loop->>Model: Fit fold model on training fold
        Loop->>Model: Predict validation fold with TTA
    end
    Loop->>Gate: Evaluate validation gates & anomaly guardrails
    alt Any Gate Fails
        Gate-->>Loop: Status: REJECTED (reason)
    else All Gates Pass
        Gate-->>Loop: Status: PASSED
        Loop->>Blender: Compute true outer meta-OOF & unbiased score
        Loop->>Blender: Refit final weights on 100% OOF for deployment only
        Loop->>Reporter: Generate deep diagnostics HTML report
        alt Candidate beats best score
            Loop->>Gatekeeper: Freeze artifacts, compute SHA-256 hashes & create SUBMISSION_AUTHORIZATION_REQUEST.md
        end
    end
    Loop->>Gatekeeper: Verify artifact hashes before executing external submission/deployment
```

---

## 2. Mathematical Formulations

### 2.1 Multi-Class Log Loss
Given true target $y_i \in \{0, 1\}^C$ and predicted probabilities $p_i \in (0, 1)^C$ on the probability simplex ($\sum_{c=1}^C p_{ic} = 1$):
$$\mathcal{L}_{\text{log}} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \ln(\max(\epsilon, \min(1-\epsilon, p_{ic})))$$
where $\epsilon = 10^{-12}$ guarantees numerical safety.

### 2.2 Binary Classification Log Loss
For binary targets $y_i \in \{0, 1\}$ with scalar prediction $p_i = P(y_i = 1)$:
$$\mathcal{L}_{\text{binary}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\max(\epsilon, p_i)) + (1 - y_i) \ln(\max(\epsilon, 1 - p_i)) \right]$$

### 2.3 Continuous Regression Objectives
For continuous target problems:
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}, \quad \text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
Continuous ensemble blending minimizes residual sum of squares over the unit simplex:
$$\min_{\mathbf{w} \in \Delta^M} \sum_{i=1}^N \left( y_i - \sum_{m=1}^M w_m \hat{y}_{im} \right)^2 \quad \text{s.t.} \quad \sum_{m=1}^M w_m = 1, \quad w_m \ge 0$$

### 2.4 Expected Calibration Error (ECE) & Brier Score
Reliability is evaluated by partitioning maximum confidence estimates into $B$ equal-width bins $I_b = (\frac{b-1}{B}, \frac{b}{B}]$:
$$\text{ECE} = \sum_{b=1}^B \frac{|I_b|}{N} \left| \text{acc}(I_b) - \text{conf}(I_b) \right|$$
The multi-class Brier score provides a strictly proper scoring rule:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C (p_{ic} - y_{ic})^2$$

### 2.5 Domain Symmetry & Total Variation Divergence
When an input $x$ possesses an invariant transformation $\tilde{x} = \mathcal{T}(x)$, the predicted distribution satisfies a known mapping:
- **Invariance**: $f(\mathcal{T}(x)) = f(x)$ (e.g. image horizontal flip for non-directional object detection).
- **Equivariance**: $f(\mathcal{T}(x)) = \mathcal{T}'(f(x))$ (e.g. spatial segmentation under rotation).
- **Label Permutation**: $f(\mathcal{T}(x)) = \pi(f(x))$ (e.g. prompt order swap in LLM preference modeling).

To enforce consistent evaluation across the framework, symmetry divergence is formally defined as the **Total Variation (TV) distance**:
$$D_{\text{TV}}(p, \tilde{p}) = \frac{1}{2N} \sum_{i=1}^N \sum_{c=1}^C \left| p_{ic} - \pi(\tilde{p}_{ic}) \right|$$
Total Variation is bounded in $[0, 1]$, making the gate threshold `max_symmetry_divergence: 0.10` mathematically consistent.

Test-Time Augmentation (TTA) enforces invariance at inference:
$$p_i^{\text{sym}} = \text{Normalize}\left( \frac{p_i^{\text{norm}} + \pi(p_i^{\text{aug}})}{2} \right)$$

### 2.6 Post-Hoc Temperature Calibration
Given normalized probabilities $\mathbf{p}_i \in \Delta^C$, we compute log-odds $\mathbf{z}_i = \ln(\mathbf{p}_i)$ and optimize a single scalar temperature $T > 0$ on out-of-fold predictions:
$$\min_{T > 0} -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \ln \sigma\left(\frac{z_{ic}}{T}\right)$$
Optimization is performed via Nelder-Mead with a lower bound $T \ge 0.05$.

### 2.7 True Outer Meta-CV Ensemble Blending
Given $M$ candidate models with out-of-fold prediction matrices $P_1, P_2, \dots, P_M \in \mathbb{R}^{N \times C}$, we separate **unbiased evaluation** from **production model fitting**:

1. **Outer Meta-CV Evaluation (Unbiased)**:
   Split the $N$ out-of-fold samples into $K_{\text{outer}}$ splits. For each split $k$:
   - Solve SLSQP constrained to $\Delta^M$ on training split $N \setminus k$:
     $$\mathbf{w}^{(k)} = \arg\min_{\mathbf{w} \in \Delta^M} \mathcal{L}\left(Y_{\text{train}}, \sum_{m=1}^M w_m P_{m, \text{train}}\right)$$
   - Generate held-out predictions on test split $k$:
     $$\text{meta\_oof}_k = \sum_{m=1}^M w_m^{(k)} P_{m, k}$$
   - Concatenate all $\text{meta\_oof}_k$ and compute the reported validation loss:
     $$\mathcal{L}_{\text{meta}} = \mathcal{L}(Y, \text{meta\_oof})$$
2. **Final Deployment Refitting**:
   Solve SLSQP over 100% of the OOF data to produce $\mathbf{w}^*$. $\mathbf{w}^*$ is **strictly used for test set inference**, never for computing the reported CV score.

---

## 3. Core Interface Contracts

### 3.1 Task Specification (`TaskSpec`)
```python
@dataclass(frozen=True)
class TaskSpec:
    task_type: Literal[
        "binary_classification",
        "multiclass_classification",
        "multilabel_classification",
        "regression",
        "forecasting",
        "ranking",
    ]
    primary_metric: str
    minimize: bool
    probability_semantics: Literal[
        "simplex",               # sum(p) == 1 (multiclass softmax)
        "independent_bernoulli", # p in [0, 1] per class (multilabel sigmoid)
        "continuous",            # unbounded continuous (regression)
        "ranking_score",         # real-valued sorting scores
    ]
    group_key: Optional[str] = None
    time_key: Optional[str] = None
    supports_calibration: bool = True
    supports_tta: bool = False
```

> [!IMPORTANT]
> Probability simplex normalization (`normalize_probabilities`) MUST NOT be applied to multi-label, continuous regression, forecasting, or ranking tasks.

### 3.2 Model Predictor Contract (`BasePredictor`)
```python
class BasePredictor(ABC):
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}
        self.feature_importances: Dict[str, float] = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BasePredictor":
        """Fits model parameters on training fold."""
        pass

    @abstractmethod
    def predict_proba_with_tta(
        self,
        X_norm: pd.DataFrame,
        X_aug: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Returns: (p_symmetric, p_normal, p_augmented)"""
        pass
```

### 3.3 Cryptographically Bound Iron Rule Contract
```python
@dataclass(frozen=True)
class AuthorizationRequest:
    experiment_id: str
    artifact_sha256: str
    prediction_sha256: str
    config_sha256: str
    code_revision: str
    metric_name: str
    metric_value: float
    created_at: str
    expires_at: str

class ISubmissionAuthorizationGate(ABC):
    @abstractmethod
    def create_authorization_request(...) -> Tuple[AuthorizationRequest, Path]: ...

    @abstractmethod
    def verify_authorization(self, request: AuthorizationRequest, ...) -> None:
        """Fails closed if any artifact or configuration hash changed."""
        pass
```

---

## 4. Domain Adaptation Matrix

| Domain | Invariant Transformation ($\mathcal{T}$) | Grouping / Temporal Key | Primary Objective | Probability Semantics |
| :--- | :--- | :--- | :--- | :--- |
| **NLP Preference / Arena** | Swap (Response A $\leftrightarrow$ Response B) | `prompt_id` | Multi-class Log Loss | `simplex` |
| **Tabular Fraud Detection** | Feature permutation / jitter | `customer_id` / `day` | PR-AUC / Brier Score | `independent_bernoulli` |
| **Continuous Regression** | Unit scaling / translation | `group_id` / `time_block`| RMSE / MAE | `continuous` |
| **Computer Vision / Medical** | Horizontal flip / 90° rotation | `patient_id` / `device` | Multi-label BCE / Dice | `independent_bernoulli` |
| **Audio Classification** | Time-shift / Frequency masking | `recording_site` / `hour` | Macro F1 / Cross-Entropy| `simplex` |
| **Ranking & Recommendations**| Candidate permutation | `session_id` / `user_id` | NDCG@K / MAP | `ranking_score` |
