# Deployment

This repository is designed as a local GitHub portfolio demo.

## Local

```bash
pip install -r requirements.txt
python -m src.api.app
```

## Docker

```bash
docker build -t intimate-partner-homicide-risk .
docker run -p 5000:5000 intimate-partner-homicide-risk
```

## Configuration

- `config/model_config.yaml` controls synthetic training size and seed.
- `config/deployment_config.yaml` documents API runtime defaults.
- `DEMO_API_KEY` enables simple demo API-key protection for prediction endpoints.

## Safety

Keep the disclaimer visible in any deployed demo. The model is trained on synthetic data and is decision support only.

