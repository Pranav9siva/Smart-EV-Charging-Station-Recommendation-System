# Behavioral A/B Evaluation

## Evaluation Setup

- Checkpoints: original=ppo_ev_final.zip, fine_tuned=ppo_ev_finetuned_50k.zip
- Database: D:\Smart_EV_Station_Recommandadtion_System\scripts\data\stations\stations.sqlite
- Seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
- Scenarios: [10.0, 20.0, 30.0, 40.0, 50.0]
- Episodes per seed: 1
- Tracked vehicles: 1
- Candidate stations: 1
- Placeholder `_empty_candidate` usage: original=0, fine_tuned=0

## Overall Results

| Metric | Original PPO | Fine-tuned PPO | Difference | Improvement % |
| --- | --- | --- | --- | --- |
| mean_reward | None | None | None | None |
| mean_soc_improvement | None | None | None | None |
| mean_range_improvement | None | None | None | None |
| charging_decision_success_rate | None | None | None | None |
| charging_event_rate | None | None | None | None |
| charging_completion_rate | None | None | None | None |
| invalid_action_rate | None | None | None | None |
| safety_violation_rate | None | None | None | None |

## Scenario-Level Results

| Scenario | Original success | Fine-tuned success | Original reward | Fine-tuned reward |
| --- | --- | --- | --- | --- |
| SOC_10 | 0.0 | 0.0 | None | None |
| SOC_20 | 0.0 | 0.0 | None | None |
| SOC_30 | 0.0 | 0.0 | None | None |
| SOC_40 | 0.0 | 0.0 | None | None |
| SOC_50 | 0.0 | 0.0 | None | None |