"""
Visual HTML Diagnostic Report Generator.
Produces self-contained, interactive HTML dashboards modeled after Tom's BirdCLEF-2026 reports.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment Report: {{ exp_id }} | Self-Driving Kaggle Agent</title>
<style>
  :root {
    --bg: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #eab308;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 30px;
  }
  .container { max-width: 1200px; margin: 0 auto; }
  .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 25px; }
  h1 { margin: 0; font-size: 26px; color: var(--accent); }
  .badge { background: #0369a1; padding: 4px 12px; border-radius: 999px; font-size: 13px; font-weight: 600; }
  .badge.passed { background: #15803d; }
  .badge.failed { background: #b91c1c; }
  .grid-kpi { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 30px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }
  .card-title { font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 8px; }
  .card-val { font-size: 28px; font-weight: 700; }
  .card-delta { font-size: 13px; margin-top: 5px; }
  .delta-pos { color: var(--green); }
  .delta-neg { color: var(--red); }
  .section-title { font-size: 18px; margin: 25px 0 15px 0; color: #e2e8f0; border-left: 4px solid var(--accent); padding-left: 10px; }
  table { width: 100%; border-collapse: collapse; margin-top: 10px; }
  th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); font-size: 14px; }
  th { background: #0f172a; color: var(--muted); font-weight: 600; }
  .bar-container { background: #334155; border-radius: 4px; height: 12px; width: 100%; overflow: hidden; }
  .bar-fill { background: var(--accent); height: 100%; }
  .cm-table td { text-align: center; font-weight: 600; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1>Experiment {{ exp_id }}: {{ model_name }}</h1>
      <p style="color: var(--muted); margin: 5px 0 0 0;">{{ hypothesis }}</p>
    </div>
    <div>
      <span class="badge {{ 'passed' if gate_passed else 'failed' }}">
        GATE: {{ 'PASSED' if gate_passed else 'REJECTED' }}
      </span>
      <div style="font-size: 12px; color: var(--muted); margin-top: 5px; text-align: right;">{{ timestamp }}</div>
    </div>
  </div>

  <div class="grid-kpi">
    <div class="card">
      <div class="card-title">OOF Multi-Class Log Loss</div>
      <div class="card-val" style="color: var(--accent);">{{ diagnostics.overall_log_loss }}</div>
      <div class="card-delta {{ 'delta-pos' if (delta is not none and delta <= 0) else ('delta-neg' if delta is not none else '') }}">
        vs Baseline: {{ '%+.5f' % delta if delta is not none else 'Baseline' }}
      </div>
    </div>
    <div class="card">
      <div class="card-title">Multi-Class Brier Score</div>
      <div class="card-val">{{ diagnostics.brier_score }}</div>
      <div class="card-delta" style="color: var(--muted);">Mean Squared Prob Error</div>
    </div>
    <div class="card">
      <div class="card-title">Expected Calibration Error (ECE)</div>
      <div class="card-val">{{ diagnostics.ece }}</div>
      <div class="card-delta" style="color: var(--muted);">Confidence alignment</div>
    </div>
    <div class="card">
      <div class="card-title">Symmetry Divergence</div>
      <div class="card-val">{{ symmetry_div }}</div>
      <div class="card-delta" style="color: var(--muted);">Swap Invariance Score</div>
    </div>
  </div>

  <div class="card" style="margin-bottom: 25px;">
    <div class="card-title">5-Fold Cross-Validation Scores</div>
    <table>
      <thead>
        <tr>
          {% for score in diagnostics.fold_scores %}
          <th>Fold {{ loop.index }}</th>
          {% endfor %}
          <th>Mean Log Loss</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          {% for score in diagnostics.fold_scores %}
          <td>{{ score }}</td>
          {% endfor %}
          <td style="font-weight: 700; color: var(--accent);">{{ diagnostics.overall_log_loss }}</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px;">
    <div class="card">
      <div class="card-title">Normalized Confusion Matrix</div>
      <table class="cm-table">
        <thead>
          <tr>
            <th>Actual \\ Pred</th>
            <th>Pred Model A</th>
            <th>Pred Model B</th>
            <th>Pred Tie</th>
          </tr>
        </thead>
        <tbody>
          {% for row in diagnostics.confusion_matrix.normalized %}
          <tr>
            <th style="text-align: left;">{{ diagnostics.confusion_matrix.class_names[loop.index0] }}</th>
            {% for val in row %}
            <td style="background-color: rgba(56, 189, 248, {{ val * 0.5 }});">{{ '%.1f' % (val * 100) }}%</td>
            {% endfor %}
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <div class="card">
      <div class="card-title">Top Feature Importances</div>
      {% if diagnostics.top_features %}
      <table>
        <thead>
          <tr>
            <th>Feature</th>
            <th>Relative Importance</th>
          </tr>
        </thead>
        <tbody>
          {% for feat in diagnostics.top_features %}
          <tr>
            <td style="font-family: monospace; font-size: 13px;">{{ feat.name }}</td>
            <td style="width: 50%;">
              <div style="display: flex; align-items: center; gap: 10px;">
                <div class="bar-container">
                  <div class="bar-fill" style="width: {{ [100, (feat.importance / diagnostics.top_features[0].importance * 100)]|min }}%;"></div>
                </div>
                <span style="font-size: 12px; color: var(--muted);">{{ feat.importance }}</span>
              </div>
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% else %}
      <p style="color: var(--muted);">Feature importances not applicable for this architecture.</p>
      {% endif %}
    </div>
  </div>

  <div class="card">
    <div class="card-title">Verbosity & Length Bias Analysis</div>
    <p style="font-size: 13px; color: var(--muted); margin-bottom: 15px;">
      Examines win rate behavior across difference bins: (len(Response A) - len(Response B)).
    </p>
    <table>
      <thead>
        <tr>
          <th>Char Diff Bin</th>
          <th>Samples</th>
          <th>Actual Win A</th>
          <th>Pred Win A</th>
          <th>Actual Win B</th>
          <th>Pred Win B</th>
          <th>Actual Tie</th>
          <th>Pred Tie</th>
        </tr>
      </thead>
      <tbody>
        {% for b in diagnostics.verbosity_curve %}
        <tr>
          <td style="font-family: monospace;">{{ b.bin_range }}</td>
          <td>{{ b.sample_count }}</td>
          <td style="color: {{ '#38bdf8' if b.actual_win_a > 0.4 else 'inherit' }}">{{ '%.1f' % (b.actual_win_a * 100) }}%</td>
          <td>{{ '%.1f' % (b.pred_win_a * 100) }}%</td>
          <td style="color: {{ '#38bdf8' if b.actual_win_b > 0.4 else 'inherit' }}">{{ '%.1f' % (b.actual_win_b * 100) }}%</td>
          <td>{{ '%.1f' % (b.pred_win_b * 100) }}%</td>
          <td>{{ '%.1f' % (b.actual_tie * 100) }}%</td>
          <td>{{ '%.1f' % (b.pred_tie * 100) }}%</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
</body>
</html>
"""


class HtmlReportGenerator:
    """Generates standalone visual HTML diagnostic reports."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        exp_id: str,
        model_name: str,
        hypothesis: str,
        diagnostics: dict[str, Any],
        delta: float | None = None,
        symmetry_div: float = 0.0,
        gate_passed: bool = True,
    ) -> Path:
        """Renders HTML template and writes report to disk."""
        from jinja2 import Template

        template = Template(HTML_TEMPLATE)
        rendered = template.render(
            exp_id=exp_id,
            model_name=model_name,
            hypothesis=hypothesis,
            diagnostics=diagnostics,
            delta=delta,
            symmetry_div=symmetry_div,
            gate_passed=gate_passed,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        out_file = self.output_dir / f"{exp_id}_diagnostic.html"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(rendered)

        return out_file
