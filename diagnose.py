import numpy as np
import json
from fiveg_env import FiveGEnv
if __name__ == "__main__":
    config = json.load(open("scenarios/dense_urban.json", 'r'))
    env = FiveGEnv(env_config=config, max_cells=57)
    env.test_action_impact()

    # 2. Run one step and diagnose
    obs, _ = env.reset()
    action = np.random.uniform(0, 1, env.action_space.shape)
    obs, reward, done, _, info = env.step(action)
    env.diagnose_connection_issue()

    # 3. Check what the analyze function shows now
    env.analyze_reward_distribution(num_episodes=3)

    # compliance_data = env.monitor_qos_compliance(num_episodes=10)

    print("ANALYZE CONSTRAINT FEASIBILITY REPORT:")
        # 1. Check if constraints are achievable
    results = env.analyze_constraint_feasibility(
        num_episodes=10,
        power_levels=[0.3, 0.5, 0.7, 0.9]
    )

    # Expected output:
    # Testing with constant power = 0.7
    # Compliance rate: 87.3%  ← Should be >80%
    # Avg drop: 0.45% (threshold: 1.0%)  ✓
    # Avg latency: 42ms (threshold: 50ms)  ✓
    # ✅ Constraints are ACHIEVABLE

    # 2. Check reward distribution
    stats = env.get_reward_statistics(num_episodes=20)

    # Expected output:
    # When constraints satisfied (2143 steps):
    #   Mean: 0.234  ← Positive
    #   Range: [0.154, 0.478]
    # When constraints violated (1857 steps):
    #   Mean: -0.456  ← Negative
    #   Range: [-0.892, -0.123]
