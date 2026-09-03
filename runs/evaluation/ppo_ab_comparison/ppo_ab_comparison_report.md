# PPO A/B Evaluation Report

## Checkpoints
- Original: D:\Smart_EV_Station_Recommandadtion_System\runs\ppo_ckpt\ppo_ev_final.zip
- Fine-tuned: D:\Smart_EV_Station_Recommandadtion_System\runs\ppo_ckpt\ppo_ev_finetuned_50k.zip

## Evaluation Setup
- Database: D:\Smart_EV_Station_Recommandadtion_System\data\stations\stations.sqlite
- Seeds: [42, 43, 44]
- Episodes per seed: 3
- Steps per episode: 10
- Tracked vehicles: 10
- Candidate stations: 8
- Placeholder `_empty_candidate` usage verified: {'original': 0, 'fine_tuned': 0}

## Aggregate Metrics
- Original reward mean: -1277.5105
- Fine-tuned reward mean: -1277.5105
- Reward absolute delta: 0.0
- Reward percentage delta: -0.0%
- Successful charging decisions delta: 0
- Completed episodes delta: 0

## Episode-Level Summary

### original
- Episode 1 seed 42: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False
- Episode 2 seed 42: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False
- Episode 3 seed 42: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False
- Episode 1 seed 43: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False
- Episode 2 seed 43: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False
- Episode 3 seed 43: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False
- Episode 1 seed 44: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False
- Episode 2 seed 44: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False
- Episode 3 seed 44: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False

### fine_tuned
- Episode 1 seed 42: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False
- Episode 2 seed 42: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False
- Episode 3 seed 42: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False
- Episode 1 seed 43: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False
- Episode 2 seed 43: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False
- Episode 3 seed 43: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False
- Episode 1 seed 44: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False
- Episode 2 seed 44: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False
- Episode 3 seed 44: reward_total=-1277.5105, reward_mean=-127.75105, reward_std=0.0, charging_events=0, successful_charging_decisions=0, invalid_actions=0, completed=False

## Validation
- Same seeds: [42, 43, 44]
- Same environment conditions: True
- Placeholder candidate usage: {'original': 0, 'fine_tuned': 0}
- Errors: none
