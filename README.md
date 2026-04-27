# Intimate Partner Homicide Risk Assessment

## Problem Statement

71% of Norwegian intimate partner homicides had prior registered partner violence.

Yet cases are often treated as isolated incidents. The system has the data but fails to connect the signals.

## Solution

ML-powered escalation risk detection: identify dangerous patterns in fragmented case histories.

**What this is:**
- Decision support tool for case prioritization
- Pattern recognition on existing data
- Explainable risk scoring

**What this is NOT:**
- Automatic action trigger
- Guilt/innocence determination
- Predictive policing

## Quick Start

```bash
cat > METHODOLOGY.md << 'EOF'
# Methodology

## Research Foundation

**Norwegian Data:**
- 71% of intimate partner homicides had prior registered partner violence
- Cases fragmented across police, prosecutors, health, child welfare
- System misses patterns due to siloed data

## Approach

### 1. Synthetic Data Generation
- Create realistic case histories based on research patterns
- 500 cases: 70% low-risk, 30% high-risk
- Temporal dynamics (events over months/years)

### 2. Feature Engineering
15 domain-expert features:
- Violence escalation trajectory
- Threat patterns
- Isolation indicators
- Restraining order breaches
- Multi-agency interaction frequency

### 3. Ensemble Model
- XGBoost + LightGBM (soft voting)
- Class imbalance handling
- 5-fold cross-validation

### 4. Explainability
- SHAP integration
- Each prediction shows why
- Interpretable output for police

### 5. Fairness Audit
- Precision/recall by demographics
- Bias detection and mitigation

## Validation

- Target: >0.85 AUC
- Target: <5% demographic variance in precision

## Limitations

- Synthetic data (not real)
- Norwegian-specific patterns
- Requires human oversight
