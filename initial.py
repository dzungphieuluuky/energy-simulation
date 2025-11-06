from stable_baselines3.common.base_class import BaseAlgorithm
import torch
import torch.nn as nn
import numpy as np

def bias_policy_output(model: BaseAlgorithm, target_action: float = 0.9):
    """
    For SAC: Bias the actor to output actions around target_action (0-1 range)
    Since SAC uses tanh, we need to compute the inverse: arctanh
    """
    if hasattr(model.policy, "actor") and hasattr(model.policy.actor, "mu"):
        print(f"Biasing SAC actor to output ~{target_action:.2f}")
        final_layer = model.policy.actor.mu
        
        # Compute required logit: logit = arctanh(2*action - 1)
        # For action=0.7: arctanh(0.4) ≈ 0.424
        logit = np.arctanh(2 * target_action - 1)
        
        torch.nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(final_layer.bias, logit)
        
        print(f"  Set bias to {logit:.4f} (→ action ≈ {target_action:.2f})")
