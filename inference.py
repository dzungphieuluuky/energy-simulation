"""
Inference Script for 5G Network RL Agent
=========================================
Loads a trained model and runs inference episodes on all scenarios.
Now consistent with StateNormalizerWrapper approach (no VecNormalize).
"""

import numpy as np
import torch
import os
import json
import argparse
from pathlib import Path
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# Import your custom environment and wrappers
from fiveg_env import FiveGEnv
from wrapper import StateNormalizerWrapper
from stable_baselines3.common.monitor import Monitor

# --- Simulation Bounds Features (Indices 0 - 16) ---
TOTAL_CELLS           = 0  # number of cells
TOTAL_UES             = 1  # number of UEs
SIM_TIME              = 2  # simulation time
TIME_STEP             = 3  # time step
TIME_PROGRESS         = 4  # progress ratio
CARRIER_FREQUENCY     = 5  # frequency Hz
ISD                   = 6  # inter-site distance
MIN_TX_POWER          = 7  # dBm
MAX_TX_POWER          = 8  # dBm
BASE_POWER            = 9  # watts
IDLE_POWER            = 10 # watts
DROP_CALL_THRESHOLD   = 11 # percentage
LATENCY_THRESHOLD     = 12 # ms
CPU_THRESHOLD         = 13 # percentage
PRB_THRESHOLD         = 14 # percentage
TRAFFIC_LAMBDA        = 15 # traffic rate
PEAK_HOUR_MULTIPLIER  = 16 # multiplier

# --- Network Bounds Features (Indices 17 - 30) ---
TOTAL_ENERGY          = 17 # kWh
ACTIVE_CELLS          = 18 # number of cells
AVG_DROP_RATE         = 19 # percentage
AVG_LATENCY           = 20 # ms
TOTAL_TRAFFIC         = 21 # traffic units
CONNECTED_UES         = 22 # number of UEs
CONNECTION_RATE       = 23 # percentage
CPU_VIOLATIONS        = 24 # number of violations
PRB_VIOLATIONS        = 25 # number of violations
MAX_CPU_USAGE         = 26 # percentage
MAX_PRB_USAGE         = 27 # percentage
KPI_VIOLATIONS        = 28 # number of violations
TOTAL_TX_POWER        = 29 # total power
AVG_POWER_RATIO       = 30 # ratio

def load_scenarios(scenarios_dir: str = "scenarios") -> list:
    """
    Load all scenario configurations from the scenarios directory.
    
    Args:
        scenarios_dir: Path to the scenarios directory
        
    Returns:
        List of scenario configurations
    """
    scenarios_path = Path(scenarios_dir)
    if not scenarios_path.exists():
        raise FileNotFoundError(f"Scenarios directory not found: {scenarios_dir}")
    
    scenarios = []
    scenario_files = sorted(scenarios_path.glob("*.json"))
    
    if not scenario_files:
        raise FileNotFoundError(f"No scenario files found in {scenarios_dir}")
    
    for scenario_file in scenario_files:
        with open(scenario_file, 'r') as f:
            scenario = json.load(f)
            scenarios.append({
                'name': scenario_file.stem,
                'config': scenario
            })
            print(f"✓ Loaded {scenario_file.name}")
    
    print(f"\nTotal scenarios loaded: {len(scenarios)}\n")
    return scenarios


def load_trained_model(model_path: str, algorithm: str, scenario_config: dict):
    """
    Loads a trained model with the correct environment setup.

    Args:
        model_path: Path to the trained model's .zip file.
        algorithm: Algorithm type ('ppo' or 'sac').
        scenario_config: The configuration for the FiveGEnv.

    Returns:
        A tuple of (trained_model, environment).
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Create environment with the same wrappers as training
    def make_env():
        base_env = FiveGEnv(scenario_config, max_cells=57)
        normalized_env = StateNormalizerWrapper(base_env)
        monitored_env = Monitor(normalized_env)
        return monitored_env
    
    # Wrap in DummyVecEnv for compatibility with SB3
    env = DummyVecEnv([make_env])
    
    # Load the model
    model_class = {'ppo': PPO, 'sac': SAC}[algorithm.lower()]
    model = model_class.load(model_path, env=env)
    
    return model, env


def run_inference_episode(model, env, episode_num: int, verbose: bool = True):
    """
    Runs a single inference episode and logs the results.
    
    Args:
        model: Trained RL model
        env: Environment (wrapped with StateNormalizerWrapper)
        episode_num: Episode number for logging
        verbose: Whether to print detailed step information
    """
    print(f"  Episode #{episode_num}")
    
    # Reset the environment
    obs = env.reset()
    
    # Access the underlying FiveGEnv to get info
    if hasattr(env.envs[0], 'unwrapped'):
        base_env = env.envs[0].unwrapped
        if verbose:
            print(f"    Active cells: {obs[0][ACTIVE_CELLS]}, UEs: {obs[0][CONNECTED_UES]}")
    
    done = False
    total_reward = 0
    step_count = 0
    compliant_steps = 0
    
    # Track metrics
    rewards_history = []
    actions_history = []
    violations_by_constraint = {}
    
    while not done:
        # Get the agent's action (deterministic for inference)
        action, _states = model.predict(obs, deterministic=True)
        
        # Perform the action in the environment
        obs, reward, done, info = env.step(action)
        
        # Log step information
        step_count += 1
        total_reward += reward[0]
        rewards_history.append(reward[0])
        actions_history.append(action[0][0])
        
        # Extract compliance info from the nested dictionary
        reward_info = info[0].get('reward_info', {})
        is_compliant = reward_info.get('constraints_satisfied', False)
        
        if is_compliant:
            compliant_steps += 1
        else:
            # Track violations by constraint type
            if 'constraint_violations' in reward_info:
                for constraint, violation in reward_info['constraint_violations'].items():
                    if violation > 0:
                        violations_by_constraint[constraint] = violations_by_constraint.get(constraint, 0) + 1
        
        if verbose and step_count % 50 == 0:  # Print every 50 steps to reduce clutter
            status_icon = "✅" if is_compliant else "❌"
            print(f"    Step {step_count:3d}: Action={action[0][0]:.3f}, Reward={reward[0]:+.3f} {status_icon}")

    compliance_rate = (compliant_steps / step_count) * 100 if step_count > 0 else 0
    avg_reward = np.mean(rewards_history)
    avg_action = np.mean(actions_history)
    
    if verbose:
        print(f"    → Steps: {step_count}, Total Reward: {total_reward:.2f}, Compliance: {compliance_rate:.1f}%")
        if violations_by_constraint:
            print(f"    → Violations: {violations_by_constraint}")
    
    return {
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'compliance_rate': compliance_rate,
        'total_steps': step_count,
        'avg_action': avg_action,
        'violations': violations_by_constraint
    }


def run_scenario_inference(model_path: str, algorithm: str, scenario: dict, 
                          episodes_per_scenario: int, verbose: bool = True):
    """
    Run inference on a single scenario across multiple episodes.
    
    Args:
        model_path: Path to trained model
        algorithm: Algorithm type
        scenario: Scenario configuration dictionary
        episodes_per_scenario: Number of episodes to run
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with aggregated results
    """
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario['name']}")
    print(f"{'='*60}")
    print(f"Config: Sites={scenario['config']['numSites']}, "
          f"UEs={scenario['config']['numUEs']}, "
          f"SimTime={scenario['config']['simTime']}")
    print(f"Running {episodes_per_scenario} episodes...\n")
    
    # Load model with this scenario's config
    model, env = load_trained_model(model_path, algorithm, scenario['config'])
    
    # Run episodes
    episode_results = []
    for i in range(1, episodes_per_scenario + 1):
        result = run_inference_episode(model, env, episode_num=i, verbose=verbose)
        episode_results.append(result)
    
    # Clean up
    env.close()
    
    # Aggregate results
    aggregated = {
        'scenario_name': scenario['name'],
        'avg_total_reward': np.mean([r['total_reward'] for r in episode_results]),
        'std_total_reward': np.std([r['total_reward'] for r in episode_results]),
        'avg_compliance_rate': np.mean([r['compliance_rate'] for r in episode_results]),
        'std_compliance_rate': np.std([r['compliance_rate'] for r in episode_results]),
        'avg_action': np.mean([r['avg_action'] for r in episode_results]),
        'total_violations': {}
    }
    
    # Aggregate violations across all episodes
    for result in episode_results:
        for constraint, count in result['violations'].items():
            aggregated['total_violations'][constraint] = aggregated['total_violations'].get(constraint, 0) + count
    
    # Print scenario summary
    print(f"\n{'-'*60}")
    print(f"SCENARIO SUMMARY: {scenario['name']}")
    print(f"{'-'*60}")
    print(f"Average Total Reward: {aggregated['avg_total_reward']:.2f} ± {aggregated['std_total_reward']:.2f}")
    print(f"Average Compliance Rate: {aggregated['avg_compliance_rate']:.1f}% ± {aggregated['std_compliance_rate']:.1f}%")
    print(f"Average Action (Power Ratio): {aggregated['avg_action']:.3f}")
    if aggregated['total_violations']:
        print(f"Total Violations by Constraint:")
        for constraint, count in aggregated['total_violations'].items():
            print(f"  {constraint}: {count}")
    else:
        print("No constraint violations! ✨")
    print(f"{'-'*60}\n")
    
    return aggregated


# =================================================================
# MAIN INFERENCE EXECUTION
# =================================================================

if __name__ == "__main__":
    
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="5G Network RL Inference on All Scenarios")
    parser.add_argument("--model", "-m", type=str, required=True, 
                        help="Path to trained model .zip file")
    parser.add_argument("--algorithm", "-a", type=str, default="sac", 
                        choices=['ppo', 'sac'], help="Algorithm used for training")
    parser.add_argument("--episodes", "-e", type=int, default=3, 
                        help="Number of episodes per scenario")
    parser.add_argument("--scenarios-dir", type=str, default="scenarios",
                        help="Path to scenarios directory")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Print detailed step-by-step information")
    
    args = parser.parse_args()
    
    try:
        print("="*60)
        print("5G NETWORK RL INFERENCE - ALL SCENARIOS")
        print("="*60)
        print(f"Model: {args.model}")
        print(f"Algorithm: {args.algorithm.upper()}")
        print(f"Episodes per scenario: {args.episodes}")
        print("="*60 + "\n")
        
        # --- Load All Scenarios ---
        scenarios = load_scenarios(args.scenarios_dir)
        
        # --- Run Inference on Each Scenario ---
        all_scenario_results = []
        for scenario in scenarios:
            result = run_scenario_inference(
                model_path=args.model,
                algorithm=args.algorithm,
                scenario=scenario,
                episodes_per_scenario=args.episodes,
                verbose=args.verbose
            )
            all_scenario_results.append(result)
        
        # --- Print Overall Statistics Across All Scenarios ---
        print("\n" + "="*60)
        print("OVERALL STATISTICS (ALL SCENARIOS)")
        print("="*60)
        
        overall_avg_reward = np.mean([r['avg_total_reward'] for r in all_scenario_results])
        overall_avg_compliance = np.mean([r['avg_compliance_rate'] for r in all_scenario_results])
        overall_avg_action = np.mean([r['avg_action'] for r in all_scenario_results])
        
        print(f"Average Total Reward (across scenarios): {overall_avg_reward:.2f}")
        print(f"Average Compliance Rate (across scenarios): {overall_avg_compliance:.1f}%")
        print(f"Average Action (across scenarios): {overall_avg_action:.3f}")
        
        print("\nPer-Scenario Breakdown:")
        for result in all_scenario_results:
            print(f"  {result['scenario_name']}: "
                  f"Reward={result['avg_total_reward']:.2f}, "
                  f"Compliance={result['avg_compliance_rate']:.1f}%")
        
        print("="*60 + "\n")
            
    except FileNotFoundError as e:
        print(f"❌ FATAL ERROR: {e}")
        print("Please ensure your model file and scenarios directory are correct.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Inference script finished.")