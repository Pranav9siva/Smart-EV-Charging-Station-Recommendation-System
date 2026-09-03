# User Manual

Use the existing `/three` dashboard for live simulation, XAI explanations, station highlighting, and replay controls. Use the research API for durable studies.

A recording is created for each websocket snapshot. Select a scenario ID when querying recordings to isolate an experiment. Replay uses zero-based `start` and exclusive `end` indexes. Analytics returns compact maps suitable for heatmap rendering.

For a study, record the scenario name, seed, model version, policy configuration, environment version, and evaluation window. Export JSON for archival, CSV for statistical analysis, and the video manifest for deterministic frame capture.
