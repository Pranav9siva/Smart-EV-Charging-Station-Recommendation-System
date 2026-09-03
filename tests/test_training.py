import math

from src.rl_env.training import train_and_evaluate_policy, train_and_evaluate_ppo_for_test


def test_train_and_evaluate_policy_returns_metrics() -> None:
    result = train_and_evaluate_policy(episodes=2, steps_per_episode=3)

    assert result["episodes"] == 2
    assert result["avg_reward"] >= 0.0
    assert result["best_reward"] >= result["avg_reward"]
    assert result["last_selected_station"] is not None


def test_train_and_evaluate_ppo_returns_metrics() -> None:
    result = train_and_evaluate_ppo_for_test(steps=100, num_stations=2, num_vehicles=2)

    assert result["training_steps"] == 100
    assert "average_eval_reward" in result
    assert isinstance(result["average_eval_reward"], (int, float))
    assert math.isfinite(float(result["average_eval_reward"]))
    assert result["max_eval_reward"] >= result["min_eval_reward"]
