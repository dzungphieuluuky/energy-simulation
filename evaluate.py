# evaluate.py
import json
import torch
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from datetime import datetime

from fiveg_env import FiveGEnv

# Helper class for defining simple baseline policies
class BaselinePolicy:
    """A simple class to represent baseline policies for comparison."""
    def __init__(self, action_type: str, action_space):
        self.action_type = action_type
        self.action_space = action_space

    def predict(self, obs, deterministic=True):
        """Return a fixed or random action based on the policy type."""
        if self.action_type == "random":
            return self.action_space.sample(), None
        elif self.action_type == "max_power":
            # Action '1.0' corresponds to max power
            action = np.ones(self.action_space.shape, dtype=np.float32)
            return action, None
        elif self.action_type == "min_power":
            # Action '0.0' corresponds to min power
            action = np.zeros(self.action_space.shape, dtype=np.float32)
            return action, None
        else:
            raise ValueError(f"Unknown baseline action type: {self.action_type}")

def run_evaluation(env: VecNormalize, model, num_episodes: int) -> pd.DataFrame:
    """
    Runs a full evaluation for a given model (or baseline) over multiple episodes.
    
    Args:
        env: The vectorized and normalized environment.
        model: The agent or baseline policy to evaluate.
        num_episodes: The number of episodes to run.

    Returns:
        A pandas DataFrame containing all collected metrics from all steps.
    """
    all_episode_infos = []

    for episode in range(num_episodes):
        print(f"  - Running Episode {episode + 1}/{num_episodes}...")
        obs = env.reset()
        terminated = False
        
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, info = env.step(action)
            
            # The info dict is wrapped in a list for vec envs
            # We also flatten the nested dictionaries for easier analysis
            flat_info = pd.json_normalize(info[0], sep='_')
            all_episode_infos.append(flat_info)
    
    if not all_episode_infos:
        return pd.DataFrame()

    return pd.concat(all_episode_infos, ignore_index=True)


def analyze_results(policy_name: str, metrics_df: pd.DataFrame) -> pd.Series:
    """
    Analyzes the collected metrics DataFrame and computes summary statistics.

    Args:
        policy_name: The name of the policy being analyzed (e.g., "PPO Agent").
        metrics_df: The DataFrame of collected metrics.

    Returns:
        A pandas Series containing the summarized results.
    """
    if metrics_df.empty:
        return pd.Series(name=policy_name, dtype=float)

    # Key Performance Indicators (KPIs)
    kpis = {
        'Avg Reward': metrics_df['normalized_reward'].mean(),
        'Std Reward': metrics_df['normalized_reward'].std(),
        'Connection Rate (%)': metrics_df['metrics_connection_rate_pct'].mean(),
        'Drop Rate (%)': metrics_df['metrics_avg_drop_rate'].mean(),
        'Latency (ms)': metrics_df['metrics_avg_latency'].mean(),
        'Total Energy (W)': metrics_df['totalEnergy'].mean(),
        'CPU Violations': metrics_df['cpuViolations'].mean(),
        'PRB Violations': metrics_df['prbViolations'].mean(),
    }

    # Detailed Reward Component Analysis
    reward_components = {
        'Score_Connection': metrics_df['reward_components_connection_score'].mean(),
        'Score_Drop': metrics_df['reward_components_drop_score'].mean(),
        'Score_Latency': metrics_df['reward_components_latency_score'].mean(),
        'Score_Energy': metrics_df['reward_components_energy_score'].mean(),
        'Score_LoadBalance': metrics_df['reward_components_load_balance_score'].mean(),
        'Score_SINR': metrics_df['reward_components_sinr_score'].mean(),
        'Score_Stability': metrics_df['reward_components_stability_score'].mean(),
    }
    
    summary = pd.Series(kpis, name=policy_name)
    reward_summary = pd.Series(reward_components, name=policy_name)
    
    return pd.concat([summary, reward_summary])


def main():
    print("--- 5G RL Agent Evaluation Script ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device} ---")
    
    MAX_CELLS_SYSTEM_WIDE = 57 # This MUST match the value used during training

    # --- User Input with Auto-Detection ---
    model_dir = "sb3_models/"
    scenario_folder = "scenarios/"

    scenario_files = {str(i+1): f for i, f in enumerate(sorted(os.listdir(scenario_folder))) if f.endswith('.json')}
    print("\nSelect SCENARIO to evaluate on:")
    for key, name in scenario_files.items(): print(f"{key}. {name}")
    eval_scenario_key = input(f"Enter choice (1-{len(scenario_files)}): ").strip()
    scenario_file = scenario_files.get(eval_scenario_key, "dense_urban.json")
    scenario_path = os.path.join(scenario_folder, scenario_file)

    # Auto-detect latest model and stats
    try:
        all_models = sorted([f for f in os.listdir(model_dir) if f.endswith('.zip')], key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
        latest_model = all_models[-1] if all_models else ""
        all_stats = sorted([f for f in os.listdir(model_dir) if f.endswith('.pkl')], key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
        latest_stats = all_stats[-1] if all_stats else ""
    except FileNotFoundError:
        latest_model, latest_stats = "", ""

    env_path_name = input(f"Enter path to VecNormalize stats file (press Enter for latest: {latest_stats}): ").strip() or latest_stats
    model_path_name = input(f"Enter path to model checkpoint file (press Enter for latest: {latest_model}): ").strip() or latest_model

    # --- Environment and Model Loading ---
    with open(scenario_path, 'r') as f:
        scenario_config = json.load(f)
    print(f"\nLoaded evaluation scenario: {scenario_config['name']}")

    env_kwargs = dict(env_config=scenario_config, max_cells=MAX_CELLS_SYSTEM_WIDE)
    base_env = make_vec_env(env_id=FiveGEnv, n_envs=1, seed=42, vec_env_cls=DummyVecEnv, env_kwargs=env_kwargs)
    
    env_stats_path = os.path.join(model_dir, env_path_name)
    if not os.path.exists(env_stats_path):
        print(f"ERROR: Normalization stats file not found at {env_stats_path}")
        return
        
    env = VecNormalize.load(env_stats_path, base_env)
    env.training = False
    env.norm_reward = False
    print("Normalization environment loaded in evaluation mode.")

    model_path = os.path.join(model_dir, model_path_name)
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return
        
    model = PPO.load(model_path, env=env, device=device)
    print(f"Model loaded from {model_path}")

    # --- Evaluation Setup ---
    num_episodes = input("\nEnter number of evaluation episodes per policy (default 3): ").strip()
    num_episodes = int(num_episodes) if num_episodes else 3
    
    # Define all policies to be evaluated
    policies_to_evaluate = {
        "PPO Agent": model,
        "Max Power": BaselinePolicy("max_power", env.action_space),
        "Min Power": BaselinePolicy("min_power", env.action_space),
        "Random": BaselinePolicy("random", env.action_space),
    }

    all_results = []

    # --- Main Evaluation Loop ---
    for name, policy in policies_to_evaluate.items():
        print(f"\n--- Evaluating Policy: {name} ---")
        policy_metrics_df = run_evaluation(env, policy, num_episodes)
        
        if policy_metrics_df.empty:
            print(f"  Skipping analysis for {name} due to no completed steps.")
            continue
            
        policy_summary = analyze_results(name, policy_metrics_df)
        all_results.append(policy_summary)
        
    env.close()

    # --- Aggregate and Display Final Results ---
    if not all_results:
        print("\nEvaluation did not yield any results.")
        return

    final_df = pd.DataFrame(all_results)
    
    # Separate KPIs and reward scores for clearer presentation
    kpi_cols = ['Avg Reward', 'Std Reward', 'Connection Rate (%)', 'Drop Rate (%)', 'Latency (ms)', 'Total Energy (W)', 'CPU Violations', 'PRB Violations']
    score_cols = [col for col in final_df.columns if col.startswith('Score_')]
    
    pd.set_option('display.float_format', '{:.3f}'.format)
    
    print("\n\n" + "="*80)
    print(" " * 25 + "AGENT PERFORMANCE EVALUATION")
    print("="*80)
    print("\n--- Key Performance Indicators (KPIs) ---\n")
    print(final_df[kpi_cols].T)
    
    print("\n\n--- Average Reward Component Scores ---\n")
    print(final_df[score_cols].T)
    print("="*80)

    # --- Save Results to CSV ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_short = os.path.splitext(model_path_name)[0].replace('model_', '')
    scenario_name_short = os.path.splitext(scenario_file)[0]
    output_filename = f"evaluation_results_{model_name_short}_on_{scenario_name_short}_{timestamp}.csv"
    
    final_df.to_csv(output_filename)
    print(f"\nFull evaluation summary saved to: {output_filename}")


if __name__ == "__main__":
    main()