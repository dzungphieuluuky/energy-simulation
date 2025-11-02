import numpy as np
import json
from fiveg_env_chatgpt import FiveGEnv
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

    compliance_data = env.monitor_qos_compliance(num_episodes=10)
