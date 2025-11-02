# finetune.py
import os
import json
import argparse
import torch
import pandas as pd
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from train_with_model_selection import make_env_thunk
from fiveg_env_chatgpt import FiveGEnv
from custom_models_sb3 import EnhancedAttentionNetwork

def transfer_ppo_weights_to_sac(ppo_model: PPO, sac_model: SAC):
    """
    Performs a partial weight transfer from PPO to SAC.
    - Transfers the shared Feature Extractor.
    - Transfers the compatible Actor (policy) network.
    - SKIPS the incompatible Critic (value) network.
    """
    print("--- [Policy Transfer] Starting PARTIAL weight transfer from PPO to SAC ---")

    ppo_state_dict = ppo_model.policy.state_dict()
    sac_actor_state_dict = sac_model.actor.state_dict()

    # 1. Transfer Feature Extractor weights to BOTH Actor and Critic
    for key, value in ppo_state_dict.items():
        if key.startswith('features_extractor.'):
            if key in sac_model.policy.state_dict():
                sac_model.policy.state_dict()[key].copy_(value)

    # 2. Transfer compatible Actor MLP weights
    for key, value in ppo_state_dict.items():
        if key.startswith('mlp_extractor.policy_net.'):
            new_key = key.replace('mlp_extractor.policy_net.', 'latent_pi.')
            if new_key in sac_actor_state_dict:
                # This will now succeed because both MLPs are [64, 64]
                sac_actor_state_dict[new_key].copy_(value)
    
    sac_model.actor.load_state_dict(sac_actor_state_dict)
    print("[Policy Transfer] Feature Extractor and Actor weights transferred successfully.")

    print("[Policy Transfer] SKIPPED Critic MLP transfer due to V(s) vs Q(s,a) incompatibility.")

    sac_model.critic_target.load_state_dict(sac_model.critic.state_dict())
    print("[Policy Transfer] Critic Target network synchronized.")
    
    print("--- [Policy Transfer] Partial weight transfer complete! ---")


def main(args):
    # --- 1. Load Scenarios and Define Constants ---
    scenario_folder = "scenarios/"
    scenario_files = [f for f in os.listdir(scenario_folder) if f.endswith('.json')]
    if not scenario_files:
        raise FileNotFoundError(f"No scenario files found in '{scenario_folder}'.")
        
    scenario_configs = []
    print("\nLoading all scenarios for mixed fine-tuning:")
    for sf in sorted(scenario_files):
        with open(os.path.join(scenario_folder, sf), 'r') as f:
            config = json.load(f)
            scenario_configs.append(config)
            print(f"  - Loaded: {config.get('name', sf)}")

    num_cpu = 5
    MAX_CELLS_SYSTEM_WIDE = 57
    N_CELL_FEATURES_CONSISTENT = 25

    # --- 2. Create Vectorized Environment ---
    env_thunks = [make_env_thunk(scenario_configs[i % len(scenario_configs)], MAX_CELLS_SYSTEM_WIDE) for i in range(num_cpu)]
    env = SubprocVecEnv(env_thunks)

    model_dir = "sb3_models/"
    os.makedirs(model_dir, exist_ok=True)
    
    # --- 3. User Input ---
    print("\nAvailable PPO model checkpoints to fine-tune:")
    for f in sorted(os.listdir(model_dir)):
        if f.endswith('.zip'): print(f"  - {f}")
    model_path_name = input("Enter path to PPO model checkpoint file: ").strip()
    model_full_path = os.path.join(model_dir, model_path_name)

    print("\nAvailable normalization stats files:")
    for f in sorted(os.listdir(model_dir)):
        if f.endswith('.pkl'): print(f"  - {f}")
    env_path_name = input("Enter path to VecNormalize stats file: ").strip()
    stats_full_path = os.path.join(model_dir, env_path_name)

    # --- 4. Load Normalization Stats ---
    print(f"\nLoading VecNormalize stats from: {stats_full_path}")
    if not os.path.exists(stats_full_path):
         raise FileNotFoundError(f"Normalization file not found at {stats_full_path}.")
    env = VecNormalize.load(stats_full_path, env)
    
    # --- 5. Configure SAC Model with EXPLICIT and CONSISTENT Architecture ---
    policy_kwargs = dict(
        features_extractor_class=EnhancedAttentionNetwork,
        features_extractor_kwargs=dict(
            features_dim=256,
            max_cells=MAX_CELLS_SYSTEM_WIDE,
            n_cell_features=N_CELL_FEATURES_CONSISTENT,
        ),
        # THE FIX: Explicitly define the MLP architecture to match PPO's default [64, 64].
        # Use 'qf' for the critic, as expected by SAC.
        net_arch=dict(pi=[64, 64], qf=[64, 64])
    )

    # --- 6. Initialize SAC Model and Perform Partial Transfer ---
    print(f"\nInitializing SAC model for fine-tuning with learning rate: {args.lr}")
    sac_model = SAC(
        "MlpPolicy", env, verbose=1,
        buffer_size=1_000_000, batch_size=256,
        learning_rate=args.lr, gamma=0.99, ent_coef='auto',
        train_freq=(1, "step"), tensorboard_log="sb3_logs/",
        policy_kwargs=policy_kwargs,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"Loading pre-trained PPO model from {model_full_path}...")
    # To be perfectly safe, we tell the loader what the PPO's original architecture was.
    ppo_policy_kwargs = policy_kwargs.copy()
    ppo_policy_kwargs['net_arch'] = dict(pi=[64, 64], vf=[64, 64]) # Use 'vf' for PPO

    ppo_model = PPO.load(
        model_full_path, device='cpu',
    )
    
    transfer_ppo_weights_to_sac(ppo_model, sac_model)
    del ppo_model

    # --- 7. Setup Callbacks and Train ---
    run_name_prefix = f"sac_finetuned_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // env.num_envs, 1),
        save_path=model_dir,
        name_prefix=run_name_prefix,
    )
    
    try:
        sac_model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=True,
            tb_log_name=run_name_prefix
        )
    finally:
        # --- 8. Save Final Model and Env Stats ---
        time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        network_name = policy_kwargs['features_extractor_class'].__name__
        final_model_path = os.path.join(model_dir, f"sac_finetuned_{network_name}_{time_stamp}.zip")
        sac_model.save(final_model_path)

        stats_path = os.path.join(model_dir, f"vec_normalize_finetuned_{time_stamp}.pkl")
        env.save(stats_path)
        
        print(f"\nFine-tuning finished or interrupted.")
        print(f"Final fine-tuned model saved to: {final_model_path}")
        print(f"Normalization stats saved to: {stats_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a PPO agent using SAC.")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total timesteps for fine-tuning.")
    parser.add_argument("--lr", type=float, default=1e-5, help="A SMALL learning rate for fine-tuning.")
    
    args = parser.parse_args()
    
    main(args)