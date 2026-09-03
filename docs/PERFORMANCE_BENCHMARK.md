# Performance Benchmark Report

The research service uses append-only SQLite recording and bounded queries (`limit` up to 10,000 frames). Analytics is linear in the selected recording window and emits compact maps rather than duplicating raw frames.

The renderer remains responsible for visual performance. Use browser performance tooling to measure frame time, draw calls, and memory for the target hardware. For large scenarios, prefer scenario-filtered replay, lower recording windows, and offline report generation. The CI job runs the full Python test suite; browser FPS checks should be added to the deployment environment with a real GPU/browser runner.
