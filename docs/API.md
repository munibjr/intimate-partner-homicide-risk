# API

This project exposes a local Flask API for synthetic decision-support demos.

## Run

```bash
python -m src.api.app
```

The API listens on `http://127.0.0.1:5000` by default.

## Health

```bash
curl http://127.0.0.1:5000/health
```

## Predict

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d @examples/case_high_risk.json
```

Responses include `risk_score`, `risk_level`, top factors, recommended human-review actions, and a disclaimer.

## Batch Predict

```bash
curl -X POST http://127.0.0.1:5000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{"cases":[{"case_id":"demo","events":[]}]}'
```

## Optional Demo API Key

Set `DEMO_API_KEY` to require an `x-api-key` header for prediction endpoints.

This API is a synthetic demo. It must not be used as an automatic action trigger or as a prediction about a person.

