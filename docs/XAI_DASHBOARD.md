# Explainable AI Dashboard

## Overview

The XAI layer explains why the PPO-based recommendation engine selected a charging station without modifying PPO training or inference logic. It is implemented as an additive explanation module that wraps the existing recommendation output and publishes the explanation into the live visualization and exported JSON artifacts.

## What is included

- Decision explanation panel
- Top-K station ranking
- Feature importance
- Reward decomposition
- Battery, distance, waiting time, queue, electricity price, and traffic contribution
- Confidence score
- Policy probability visualization
- Action probability chart
- Observation visualization
- Decision timeline
- Episode playback
- Historical decision comparison
- Explainability API metadata
- Live explanation updates through the existing WebSocket visualization stream
- JSON export for explanations

## Integration points

- Recommendation explanations are generated in [src/xai/explainer.py](../src/xai/explainer.py)
- Explanations are attached to recommendation events in [src/simulation/controller.py](../src/simulation/controller.py)
- The live dashboard payload exposes XAI data via [src/visualization/services/visualization_builder.py](../src/visualization/services/visualization_builder.py)
- The exported explanation artifact is written to outputs/latest_explanation.json

## Usage

The explanation is automatically generated when a vehicle receives a recommendation. Existing PPO logic remains unchanged.

## Verification

Run:

```bash
pytest tests/test_xai.py -q
```
