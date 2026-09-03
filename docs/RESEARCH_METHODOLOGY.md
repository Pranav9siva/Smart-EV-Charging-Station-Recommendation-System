# Research Methodology

Use fixed scenario seeds and identical SUMO route inputs for policy comparisons. Separate calibration, validation, and held-out evaluation episodes. Report average reward, success rate, waiting time, queue length, charging cost, energy consumption, station utilization, and grid load with episode counts and timestamps.

For A/B tests, assign episodes rather than individual frames to control and variant policies. Store model version and configuration in experiment metadata. Use the baseline comparison endpoint for deltas, then export the raw recording and report so results remain reproducible.
