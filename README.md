# Intimate Partner Homicide Risk Assessment

Synthetic ML demo for identifying escalation patterns in fragmented intimate partner violence case histories.

## What This Is

- Decision-support demo for case prioritization
- Pattern recognition on synthetic event histories
- Explainable risk scoring with visible factors and human-review recommendations
- A GitHub portfolio project, not an operational public-safety system

## What This Is Not

- Not a prediction that a person will harm another person
- Not an automatic action trigger
- Not a guilt, innocence, arrest, or surveillance system
- Not trained on real personal data

Every CLI and API prediction includes this disclaimer: synthetic demo only, decision support only, and human review required.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Run one prediction:

```bash
python -m src.cli.predict examples/case_high_risk.json --format json
```

Run batch prediction:

```bash
python -m src.cli.batch_predict examples/cases_batch.json
```

Start the API:

```bash
python -m src.api.app
```

Try the API:

```bash
curl http://127.0.0.1:5000/health
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d @examples/case_high_risk.json
```

## Project Structure

```text
src/data/            Synthetic cases and feature extraction
src/models/          Ensemble model and shared prediction service
src/cli/             Single and batch prediction CLIs
src/api/             Flask REST API
src/analysis/        Trend, trajectory, and pattern analysis
src/evaluation/      Metrics and fairness audit helpers
src/explainability/  SHAP explainer
scripts/             Training and evaluation entry points
examples/            Runnable demo inputs and examples
docs/                API and deployment notes
tests/               Unit and integration tests
```

## Model Contract

The feature extractor returns 15 normalized, interpretable features plus `risk_label`.

Important examples:

- `violence_escalation_rate`
- `threat_frequency`
- `weapon_mention`
- `restraining_order_breaches`
- `multiple_agencies_involved`
- `breach_of_protection`

The demo model trains on generated synthetic cases at runtime and keeps the model in memory.

## Development

```bash
pytest -q
python scripts/train_model.py
python scripts/evaluate_model.py
```

Optional API key protection for prediction endpoints:

```bash
export DEMO_API_KEY=local-demo-key
python -m src.api.app
```

Then send `x-api-key: local-demo-key` on `/predict` and `/batch-predict`.

## Ethics

See [ETHICS.md](ETHICS.md). This project is intentionally framed as synthetic decision support with human oversight. The safe use case is prioritization, safety planning, and professional review in a demo or research setting.
