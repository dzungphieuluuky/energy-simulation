import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import logging
import glob
import os
from datetime import datetime
from collections import namedtuple
from typing import Tuple, Dict, Optional

from .models import Actor
from .models import Critic
from .state_normalizer import StateNormalizer

SIM_TIME_IDX = 2
TIME_PROGRESS_IDX = 4
DROP_THRESHOLD = 11
LATENCY_THRESHOLD = 12
CPU_THRESHOLD_IDX = 13
PRB_THRESHOLD_IDX = 14

# Network features start at index 17
TOTAL_ENERGY_IDX = 17 + 0   
ACTIVE_CELLS_IDX = 17 + 1  
AVG_DROP_RATE_IDX = 17 + 2  
AVG_LATENCY_IDX = 17 + 3    
TOTAL_TRAFFIC_IDX = 17 + 4  
CONNECTED_UES_IDX = 17 + 5  
CONNECTION_RATE_IDX = 17 + 6 
CPU_VIOLATIONS_IDX = 17 + 7  
PRB_VIOLATIONS_IDX = 17 + 8  
MAX_CPU_USAGE_IDX = 17 + 9   
MAX_PRB_USAGE_IDX = 17 + 10  
KPI_VIOLATIONS_IDX = 17 + 11 
TOTAL_TX_POWER_IDX = 17 + 12 
AVG_POWER_RATIO_IDX = 17 + 13 

CELL_START_IDX = 17 + 14  # Cell features start index

class ConstraintPredictor(nn.Module):
    """
    Neural network that predicts future constraint violations
    Input: current state + proposed action
    Output: predicted costs in next N steps
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, prediction_horizon=5):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.drop_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, prediction_horizon), nn.Softplus()
        )
        
        self.latency_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, prediction_horizon), nn.Softplus()
        )
        
        self.resource_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, prediction_horizon), nn.Softplus()
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        features = self.encoder(x)
        drop_pred = self.drop_predictor(features)
        latency_pred = self.latency_predictor(features)
        resource_pred = self.resource_predictor(features)
        return drop_pred, latency_pred, resource_pred
    
    def predict_total_cost(self, state, action, gamma=0.99):
        """Compute discounted sum of predicted costs"""
        drop_pred, latency_pred, resource_pred = self.forward(state, action)
        horizon = self.prediction_horizon
        discount_weights = torch.tensor([gamma ** i for i in range(horizon)], device=state.device).unsqueeze(0)
        
        drop_cost = (drop_pred * discount_weights).sum(dim=1, keepdim=True)
        latency_cost = (latency_pred * discount_weights).sum(dim=1, keepdim=True)
        resource_cost = (resource_pred * discount_weights).sum(dim=1, keepdim=True)
        return drop_cost, latency_cost, resource_cost

class AdaptiveLagrangeController:
    """
    Manages Lagrange multipliers with adaptive learning and safety margins.
    """
    def __init__(self, initial_lambda=1.0, lr=0.01, target_constraint_value=0.0, 
                 safety_margin=0.2, min_lambda=0.0, max_lambda=100.0):
        self.lambda_val = initial_lambda
        self.lr = lr
        self.target = target_constraint_value
        self.safety_margin = safety_margin
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.violation_history = deque(maxlen=100)
        self.lambda_history = deque(maxlen=100)
        self.avg_violation = 0.0
        self.violation_trend = 0.0
    
    def update(self, current_cost, predicted_future_cost=None):
        effective_target = self.target * (1.0 - self.safety_margin)
        violation = current_cost - effective_target
        
        if predicted_future_cost is not None:
            violation = 0.7 * violation + 0.3 * (predicted_future_cost - effective_target)
            
        self.violation_history.append(violation)
        alpha = 0.1
        self.avg_violation = alpha * violation + (1 - alpha) * self.avg_violation
        
        if len(self.violation_history) >= 20:
            recent = np.mean(list(self.violation_history)[-10:])
            older = np.mean(list(self.violation_history)[-20:-10])
            self.violation_trend = recent - older
        
        effective_lr = self.lr * 1.5 if self.violation_trend > 0 else self.lr * 0.8
        
        self.lambda_val = np.clip(self.lambda_val + effective_lr * violation, self.min_lambda, self.max_lambda)
        self.lambda_history.append(self.lambda_val)
        return self.lambda_val

    def get_lambda(self):
        return self.lambda_val
        
    def is_converged(self, tolerance=0.01, window=50):
        if len(self.lambda_history) < window: return False
        return np.std(list(self.lambda_history)[-window:]) < tolerance
    
    def get_state(self):
        """Returns the full internal state as standard Python types."""
        return {
            'lambda_val': self.lambda_val,
            # Convert deques to lists for safe saving
            'violation_history': list(self.violation_history),
            'lambda_history': list(self.lambda_history),
            'avg_violation': self.avg_violation,
            'violation_trend': self.violation_trend,
        }

    def load_state(self, state_dict):
        """Loads the state from a saved dictionary."""
        self.lambda_val = state_dict['lambda_val']
        # Convert lists back to deques, preserving maxlen if applicable
        self.violation_history = deque(state_dict['violation_history'], maxlen=100)
        self.lambda_history = deque(state_dict['lambda_history'], maxlen=100)
        self.avg_violation = state_dict['avg_violation']
        self.violation_trend = state_dict['violation_trend']

# CHANGE 1: Add 'mask' to the Transition tuple to store the cell mask.
Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done', 
                         'log_prob', 'mask', 'value_R', 'value_C_drop', 'value_C_latency', 'value_C_resources',
                         'cost_drop', 'cost_latency', 'cost_resources', 'predicted_costs'))

class TransitionBuffer:
    """
    Experience replay buffer for PPO agent
    Stores transitions and provides batch sampling for training
    """
    
    def __init__(self, capacity=2048):
        """
        Initialize transition buffer
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def add(self, transition):
        """
        Add a transition to the buffer
        
        Args:
            transition (Transition): Transition tuple to add
        """
        if not isinstance(transition, Transition):
            raise TypeError("Expected Transition namedtuple")
        
        self.buffer.append(transition)
        self.position = (self.position + 1) % self.capacity
    
    def get_all(self):
        """
        Get all transitions in the buffer
        
        Returns:
            list: List of all transitions
        """
        return list(self.buffer)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions randomly
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            list: List of sampled transitions
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def get_last_n(self, n):
        """
        Get the last n transitions
        
        Args:
            n (int): Number of recent transitions to get
            
        Returns:
            list: List of last n transitions
        """
        if n >= len(self.buffer):
            return list(self.buffer)
        
        return list(self.buffer)[-n:]
    
    def clear(self):
        """Clear all transitions from buffer"""
        self.buffer.clear()
        self.position = 0
    
    def __len__(self):
        """Get number of transitions in buffer"""
        return len(self.buffer)
    
    def is_full(self):
        """Check if buffer is at capacity"""
        return len(self.buffer) == self.capacity
    
    def get_statistics(self):
        """
        Get buffer statistics
        
        Returns:
            dict: Buffer statistics including reward stats
        """
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'is_full': False,
                'avg_reward': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0
            }
        
        rewards = [t.reward for t in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'is_full': self.is_full(),
            'avg_reward': np.mean(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'std_reward': np.std(rewards)
        }

class RLAgent:
    def __init__(self, n_cells, n_ues, max_time, log_file='rl_agent.log', use_gpu=False, config: Optional[Dict] = None):
        """
        Initialize Predictive Lagrangian Optimization (PLO) agent.
        """
        print("Initializing Predictive Lagrangian Optimization (PLO) Agent")
        
        self.config = config or {}
        self.max_cells = 57
        self.n_cells = n_cells
        self.n_ues = n_ues
        self.max_time = max_time
        
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        self.state_dim = 17 + 14 + (self.max_cells * 12)
        self.action_dim = self.max_cells
        self.global_state_dim = 17 + 14 

        self.state_normalizer = StateNormalizer(self.state_dim, n_cells=self.max_cells)
        
        self._init_hyperparameters()
        
        hidden_dim = self.config.get('hidden_dim', 512)
        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.critic_R = Critic(self.state_dim, hidden_dim).to(self.device)
        self.critic_C_drop = Critic(self.state_dim, hidden_dim).to(self.device)
        self.critic_C_latency = Critic(self.state_dim, hidden_dim).to(self.device)
        self.critic_C_resources = Critic(self.state_dim, hidden_dim).to(self.device)
        
        self.constraint_predictor = ConstraintPredictor(
            self.state_dim, self.action_dim, 
            hidden_dim=128, prediction_horizon=self.prediction_horizon
        ).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizers = {
            'R': optim.Adam(self.critic_R.parameters(), lr=self.critic_lr),
            'drop': optim.Adam(self.critic_C_drop.parameters(), lr=self.critic_lr),
            'latency': optim.Adam(self.critic_C_latency.parameters(), lr=self.critic_lr),
            'resources': optim.Adam(self.critic_C_resources.parameters(), lr=self.critic_lr)
        }
        self.predictor_optimizer = optim.Adam(self.constraint_predictor.parameters(), lr=self.predictor_lr)
        
        self.lambda_controllers = {
            'drop': AdaptiveLagrangeController(lr=self.lambda_lr, target_constraint_value=0.0, safety_margin=0.2, max_lambda=50.0),
            'latency': AdaptiveLagrangeController(lr=self.lambda_lr, target_constraint_value=50.0, safety_margin=0.1, max_lambda=50.0),
            'resources': AdaptiveLagrangeController(initial_lambda=0.5, lr=self.lambda_lr, target_constraint_value=95.0, safety_margin=0.05, max_lambda=20.0)
        }
        
        self.buffer = TransitionBuffer(self.buffer_size)
        self.training_mode = True
        self.total_episodes = 0
        self.total_steps = 0
        
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        self.last_predictions = None
        self.last_mask = None
        self.current_episode_reward = 0.0

        self.episode_rewards = deque(maxlen=100)
        self.episode_costs = {'drop': deque(maxlen=100), 'latency': deque(maxlen=100), 'resources': deque(maxlen=100)}
        self.predictor_losses = deque(maxlen=100)
        self.action_history = deque(maxlen=1000)
        
        self.setup_logging(log_file)
        self.logger.info(f"PLO Agent initialized on device: {self.device}")
        self.logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")

    def _init_hyperparameters(self):
        self.actor_lr = self.config.get('actor_lr', 1e-5)
        self.critic_lr = self.config.get('critic_lr', 5e-5)
        self.predictor_lr = self.config.get('predictor_lr', 1e-3)
        self.lambda_lr = self.config.get('lambda_lr', 0.05)
        
        self.gamma = self.config.get('gamma', 0.99)
        self.lambda_gae = self.config.get('lambda_gae', 0.95)
        self.clip_epsilon = self.config.get('clip_epsilon', 0.1)
        self.entropy_coeff = self.config.get('entropy_coeff', 0.02)
        self.ppo_epochs = self.config.get('ppo_epochs', 5)
        self.batch_size = self.config.get('batch_size', 128)
        self.buffer_size = self.config.get('buffer_size', 4096)
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
        
        # PLO-specific
        self.prediction_horizon = self.config.get('prediction_horizon', 5)
        self.predictor_update_freq = self.config.get('predictor_update_freq', 10)

    def setup_logging(self, log_file):
        """Setup logging configuration"""
        self.logger = logging.getLogger('PLOAgent')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def normalize_state(self, state):
        """Normalize state vector to [0, 1] range"""
        return self.state_normalizer.normalize(state)
    
    def start_scenario(self):
        self.total_episodes += 1
        self.current_episode_reward = 0.0
        self.current_episode_costs = {'drop': 0.0, 'latency': 0.0, 'resources': 0.0}
        self.episode_steps = 0

        list_of_files = glob.glob('plo_agent_model_*.pth') # Get all model files
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            self.logger.info(f"Found latest model: {latest_file}. Loading...")
            try:
                # Add weights_only=False to handle the PyTorch 2.6+ warning
                self.load_model(latest_file)
            except Exception as e:
                self.logger.error(f"Could not load model from {latest_file}. Starting fresh. Error: {e}")
        else:
            self.logger.info("No existing model found. Starting from scratch.")


        self.logger.info(f"Starting episode {self.total_episodes}")

    def end_scenario(self):
        self.episode_rewards.append(self.current_episode_reward)
        for key in self.episode_costs:
            self.episode_costs[key].append(self.current_episode_costs[key])
        
        avg_cost_drop = self.current_episode_costs['drop'] / self.episode_steps if self.episode_steps > 0 else 0
        self.logger.info(f"Episode {self.total_episodes} ended: Reward={self.current_episode_reward:.2f}, AvgCostDrop={avg_cost_drop:.2f}")
        
        if self.training_mode and len(self.buffer) >= self.batch_size:
            self.train()
        self.save_model()

    def _pad_state(self, compact_state):
        """Pads a compact state vector to the full state_dim size."""
        compact_state_np = np.array(compact_state).flatten()
        current_len = len(compact_state_np)

        if current_len == self.state_dim:
            return compact_state_np # Already full size, no padding needed

        if current_len > self.state_dim:
            # This is an error condition, but we can handle it by truncating
            self.logger.warning(f"Received state of length {current_len}, expected {self.state_dim}. Truncating.")
            return compact_state_np[:self.state_dim]

        # Create a full-sized zero vector and copy the compact state into it
        padded_state = np.zeros(self.state_dim)
        padded_state[:current_len] = compact_state_np
        
        return padded_state

    def calculate_reward(self, prev_state, current_state):
        """IMPROVED reward calculation with better shaping."""
        if prev_state is None:
            return 0.0, 0.0, 0.0, 0.0

        prev_state = np.array(prev_state).flatten()
        current_state = np.array(current_state).flatten()
        
        curr_power = current_state[TOTAL_TX_POWER_IDX]
        curr_energy, prev_energy = current_state[TOTAL_ENERGY_IDX], prev_state[TOTAL_ENERGY_IDX]
        curr_active_cells = current_state[ACTIVE_CELLS_IDX]
        
        # Reward: Energy savings and cell reduction
        energy_saving = (prev_energy - curr_energy) / prev_energy if prev_energy > 1e-6 else 0.0
        cell_reduction = (self.n_cells - curr_active_cells) / self.n_cells
        reward = (5.0 * energy_saving) + (2.0 * cell_reduction) - (0.005 * curr_power)
        
        # Costs: Linear penalties for violations
        curr_drop = current_state[AVG_DROP_RATE_IDX]
        curr_latency = current_state[AVG_LATENCY_IDX]
        drop_limit = current_state[DROP_THRESHOLD]
        latency_limit = current_state[LATENCY_THRESHOLD]

        cost_drop = max(0.0, curr_drop - drop_limit)
        cost_latency = max(0.0, curr_latency - latency_limit)
        drop_penalty = 1000.0 * cost_drop
        reward -= drop_penalty

        curr_cpu, curr_prb = current_state[MAX_CPU_USAGE_IDX], current_state[MAX_PRB_USAGE_IDX]
        cpu_threshold = current_state[CPU_THRESHOLD_IDX]
        prb_threshold = current_state[PRB_THRESHOLD_IDX]
        cost_resources = max(0.0, curr_cpu - cpu_threshold) + max(0.0, curr_prb - prb_threshold)
        
        # Bonuses: Staying within safe margins
        if curr_drop <= drop_limit and curr_latency <= latency_limit: reward += 5.0
        if curr_cpu <= cpu_threshold and curr_prb <= prb_threshold: reward += 2.0
        
        return float(reward), float(cost_drop), float(cost_latency), float(cost_resources)

    def get_action(self, state):
        # CHANGE 2: Generate a mask based on the current number of active cells.
        # It's crucial that 'state' here is the raw, unnormalized state.
        padded_state = self._pad_state(state)
        state_flat = np.array(padded_state).flatten()

        # Create a mask for the active cells.
        mask_np = np.zeros(self.max_cells)
        mask_np[:self.n_cells] = 1.0
        mask_tensor = torch.FloatTensor(mask_np).to(self.device).unsqueeze(0) # Add batch dim

        # Normalize state for the networks
        normalized_state = self.normalize_state(state_flat)
        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean, action_logstd = self.actor(state_tensor)
            
            if self.training_mode:
                action_std = torch.exp(action_logstd.clamp(-20, 2))
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                
                # CHANGE 3: Calculate log probability using the mask. This ensures we only
                # sum the log probs of actions for the active cells.
                log_prob_full = dist.log_prob(action)
                log_prob = (log_prob_full * mask_tensor).sum(dim=-1)

                # Predictive constraint check
                # Mask the action before sending it to the predictor
                masked_action = action * mask_tensor
                drop_pred, lat_pred, res_pred = self.constraint_predictor.predict_total_cost(state_tensor, masked_action)
                self.last_predictions = {
                    'drop': drop_pred.item(), 'latency': lat_pred.item(), 'resources': res_pred.item()
                }
            else:
                action = action_mean
                log_prob = torch.zeros(1, device=self.device)
                self.last_predictions = None
        
        # CHANGE 4: Apply the mask to the final action to ensure we don't act on inactive cells.
        action_np = action.clamp(0.0, 1.0).cpu().numpy().flatten()
        masked_action_np = action_np * mask_np

        # Store necessary info for the 'update' step
        self.last_state = normalized_state
        self.last_action = masked_action_np # Store the masked action
        self.last_log_prob = log_prob.item()
        self.last_mask = mask_np # Store the mask for the transition
        self.action_history.append(masked_action_np.copy())
        
        return masked_action_np[:self.n_cells]
    
    def update(self, state, action, next_state, done):
        if not self.training_mode: return

        padded_next_state = self._pad_state(next_state)
        padded_state = self._pad_state(state)
        reward, cost_drop, cost_latency, cost_resources = self.calculate_reward(padded_state, padded_next_state)
        
        self.episode_steps += 1
        self.total_steps += 1
        self.current_episode_reward += reward
        self.current_episode_costs['drop'] += cost_drop
        self.current_episode_costs['latency'] += cost_latency
        self.current_episode_costs['resources'] += cost_resources

        # The 'last_state' is already normalized from get_action
        state_norm = self.last_state 
        next_state_norm = self.normalize_state(np.array(next_state).flatten())
        
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value_R = self.critic_R(state_tensor).item()
            value_C_drop = self.critic_C_drop(state_tensor).item()
            value_C_latency = self.critic_C_latency(state_tensor).item()
            value_C_resources = self.critic_C_resources(state_tensor).item()
        
        # CHANGE 5: Add the 'last_mask' to the Transition object when saving to the buffer.
        transition = Transition(
            state=state_norm, action=np.array(action).flatten(), reward=reward,
            next_state=next_state_norm, done=done, log_prob=self.last_log_prob,
            mask=self.last_mask, # Add mask here
            value_R=value_R, value_C_drop=value_C_drop, value_C_latency=value_C_latency,
            value_C_resources=value_C_resources, cost_drop=cost_drop,
            cost_latency=cost_latency, cost_resources=cost_resources,
            predicted_costs=self.last_predictions
        )
        self.buffer.add(transition)

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.lambda_gae * (1.0 - dones[t]) * last_advantage
        return advantages, advantages + values

    def train(self):
        if len(self.buffer) < self.batch_size: return
        
        transitions = self.buffer.get_all()
        self.buffer.clear()
        
        # Unpack transitions, including the new 'mask'
        states = np.array([t.state for t in transitions])
        actions = np.array([t.action for t in transitions])
        # CHANGE 6: Unpack the masks from the buffer.
        masks = np.array([t.mask for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        costs = {k: np.array([t.cost_drop if k == 'drop' else t.cost_latency if k == 'latency' else t.cost_resources for t in transitions]) for k in self.lambda_controllers}
        predicted_costs = {k: np.array([t.predicted_costs[k] for t in transitions if t.predicted_costs]) for k in self.lambda_controllers}
        next_states = np.array([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions])
        old_log_probs = np.array([t.log_prob for t in transitions])
        values = {'R': np.array([t.value_R for t in transitions]), 'drop': np.array([t.value_C_drop for t in transitions]),
                  'latency': np.array([t.value_C_latency for t in transitions]), 'resources': np.array([t.value_C_resources for t in transitions])}

        # Update Lagrange multipliers based on this batch's performance
        lambdas = self.update_lagrange_multipliers(costs, predicted_costs)

        # Compute next values and advantages
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        with torch.no_grad():
            next_values_R = self.critic_R(next_states_tensor).cpu().numpy().flatten()
            next_values_C_drop = self.critic_C_drop(next_states_tensor).cpu().numpy().flatten()
            next_values_C_latency = self.critic_C_latency(next_states_tensor).cpu().numpy().flatten()
            next_values_C_resources = self.critic_C_resources(next_states_tensor).cpu().numpy().flatten()

        adv_R, ret_R = self.compute_gae(rewards, values['R'], next_values_R, dones)
        adv_C_drop, ret_C_drop = self.compute_gae(costs['drop'], values['drop'], next_values_C_drop, dones)
        adv_C_latency, ret_C_latency = self.compute_gae(costs['latency'], values['latency'], next_values_C_latency, dones)
        adv_C_resources, ret_C_resources = self.compute_gae(costs['resources'], values['resources'], next_values_C_resources, dones)

        # Lagrangian Advantage
        adv_L = (adv_R - lambdas['drop'] * adv_C_drop - lambdas['latency'] * adv_C_latency - lambdas['resources'] * adv_C_resources) / \
                (1.0 + lambdas['drop'] + lambdas['latency'] + lambdas['resources'])
        adv_L = (adv_L - np.mean(adv_L)) / (np.std(adv_L) + 1e-8)

        # Convert to tensors, including the new 'masks' tensor
        tensors = {
            "states": torch.FloatTensor(states).to(self.device), 
            "actions": torch.FloatTensor(actions).to(self.device),
            "masks": torch.FloatTensor(masks).to(self.device), # Add masks tensor
            "old_log_probs": torch.FloatTensor(old_log_probs).to(self.device), 
            "adv_L": torch.FloatTensor(adv_L).to(self.device),
            "ret_R": torch.FloatTensor(ret_R).to(self.device), 
            "ret_C_drop": torch.FloatTensor(ret_C_drop).to(self.device),
            "ret_C_latency": torch.FloatTensor(ret_C_latency).to(self.device), 
            "ret_C_resources": torch.FloatTensor(ret_C_resources).to(self.device),
        }
        
        # PPO training loop
        for _ in range(self.ppo_epochs):
            for i in range(0, len(states), self.batch_size):
                batch_indices = slice(i, i + self.batch_size)
                batch_tensors = {k: v[batch_indices] for k, v in tensors.items()}
                
                # Actor update
                mean, logstd = self.actor(batch_tensors["states"])
                dist = torch.distributions.Normal(mean, logstd.exp())
                
                # CHANGE 7: Use the mask when recalculating log probabilities to ensure
                # the ratio is computed correctly over active cells only.
                new_log_probs_full = dist.log_prob(batch_tensors["actions"])
                new_log_probs = (new_log_probs_full * batch_tensors["masks"]).sum(dim=-1)

                ratio = (new_log_probs - batch_tensors["old_log_probs"]).exp()
                surr1 = ratio * batch_tensors["adv_L"]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_tensors["adv_L"]
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * dist.entropy().mean()
                
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Critics update
                critics = {'R': self.critic_R, 'drop': self.critic_C_drop, 'latency': self.critic_C_latency, 'resources': self.critic_C_resources}
                for key, critic in critics.items():
                    ret_tensor = batch_tensors[f"ret_{key.upper()}" if key == 'R' else f"ret_C_{key}"]
                    critic_loss = nn.MSELoss()(critic(batch_tensors["states"]).squeeze(), ret_tensor)
                    self.critic_optimizers[key].zero_grad(set_to_none=True)
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
                    self.critic_optimizers[key].step()
        
        # Train predictor if it's time
        if self.total_episodes % self.predictor_update_freq == 0:
            self.train_predictor(states, actions, costs)

        self.logger.info(
            f"Training completed: Avg Reward={np.mean(rewards):.2f}, "
            f"Avg Cost Drop={np.mean(costs['drop']):.4f} (lambda={lambdas['drop']:.2f}), "
            f"Avg Cost Latency={np.mean(costs['latency']):.4f} (lambda={lambdas['latency']:.2f})"
        )

    def update_lagrange_multipliers(self, costs_batch, predicted_costs_batch):
        lambdas = {}
        for key, controller in self.lambda_controllers.items():
            avg_cost = np.mean(costs_batch[key])
            avg_pred_cost = np.mean(predicted_costs_batch[key]) if key in predicted_costs_batch and len(predicted_costs_batch[key]) > 0 else None
            lambdas[key] = controller.update(avg_cost, avg_pred_cost)
        return lambdas

    def train_predictor(self, states, actions, next_costs):
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        
        pred_drop, pred_lat, pred_res = self.constraint_predictor(states_tensor, actions_tensor)
        
        # Use first-step prediction for simplicity
        target_drop = torch.FloatTensor(next_costs['drop']).unsqueeze(1).to(self.device)
        target_lat = torch.FloatTensor(next_costs['latency']).unsqueeze(1).to(self.device)
        target_res = torch.FloatTensor(next_costs['resources']).unsqueeze(1).to(self.device)
        
        loss = nn.MSELoss()(pred_drop[:, 0:1], target_drop) + \
               nn.MSELoss()(pred_lat[:, 0:1], target_lat) + \
               nn.MSELoss()(pred_res[:, 0:1], target_res)
        
        self.predictor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.constraint_predictor.parameters(), self.max_grad_norm)
        self.predictor_optimizer.step()
        
        self.predictor_losses.append(loss.item())
        self.logger.info(f"Constraint predictor trained with loss: {loss.item():.4f}")
        return loss.item()

    def save_model(self, filepath=None):
        """
        Save model parameters to a file, ensuring data types are safe for loading.
        
        FIXES: 
        1. Added saving of all critic optimizers via the self.critic_optimizers dictionary.
        2. Added saving of state_normalizer state.
        3. Added saving of self.total_steps, self.episode_costs, and self.predictor_losses.
        4. Added saving of critic optimizers and state normalizer.
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"plo_agent_model_{timestamp}.pth"
        
        # Ensure all non-tensor data is a standard Python type
        safe_lambdas = {key: controller.get_state() for key, controller in self.lambda_controllers.items()}
        
        # Package critic optimizer states into a single dictionary
        critic_opt_state_dicts = {
            k: opt.state_dict() for k, opt in self.critic_optimizers.items()
        }

        # Convert deques of dictionaries (episode_costs) to dict of lists
        safe_costs = {k: list(v) for k, v in self.episode_costs.items()}
        
        checkpoint = {
            # --- NETWORKS ---
            'actor_state_dict': self.actor.state_dict(),
            'critic_R_state_dict': self.critic_R.state_dict(),
            'critic_C_drop_state_dict': self.critic_C_drop.state_dict(),
            'critic_C_latency_state_dict': self.critic_C_latency.state_dict(),
            'critic_C_resources_state_dict': self.critic_C_resources.state_dict(),
            'predictor_state_dict': self.constraint_predictor.state_dict(),
            
            # --- OPTIMIZERS ---
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'predictor_optimizer_state_dict': self.predictor_optimizer.state_dict(),
            'critic_optimizers_state_dict': critic_opt_state_dicts, # FIXED: Saving all critic optimizers
            
            # --- STATE ---
            'lambdas': safe_lambdas,
            'total_episodes': int(self.total_episodes),
            'total_steps': int(self.total_steps), # FIXED: Saving total steps
            
            # --- TRACKING ---
            'episode_rewards': list(self.episode_rewards), 
            'episode_costs': safe_costs, # FIXED: Saving episode costs
            'predictor_losses': list(self.predictor_losses), # FIXED: Saving predictor losses
            # Note: action_history is usually optional to save, but included here for completeness:
            'action_history': list(self.action_history),
        }
        
        # Save using the default, safe method
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load model parameters from a file, mapping to the current device.
        
        FIXES:
        1. Added loading of all critic optimizers.
        2. Added loading of state_normalizer state.
        3. Added loading of total_steps and all tracking deques.
        4. Explicitly map location to self.device.
        """
        # Load the checkpoint dictionary
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # --- NETWORKS ---
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_R.load_state_dict(checkpoint['critic_R_state_dict'])
        self.critic_C_drop.load_state_dict(checkpoint['critic_C_drop_state_dict'])
        self.critic_C_latency.load_state_dict(checkpoint['critic_C_latency_state_dict'])
        self.critic_C_resources.load_state_dict(checkpoint['critic_C_resources_state_dict'])
        self.constraint_predictor.load_state_dict(checkpoint['predictor_state_dict'])

        # --- OPTIMIZERS ---
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.predictor_optimizer.load_state_dict(checkpoint['predictor_optimizer_state_dict'])
        
        # FIXED: Loading all critic optimizers
        if 'critic_optimizers_state_dict' in checkpoint:
            for key, state_dict in checkpoint['critic_optimizers_state_dict'].items():
                if key in self.critic_optimizers:
                    self.critic_optimizers[key].load_state_dict(state_dict)

        # Loading Adaptive Lagrange Controllers
        if 'lambdas' in checkpoint:
            for key, state_dict in checkpoint['lambdas'].items():
                if key in self.lambda_controllers:
                    self.lambda_controllers[key].load_state(state_dict)
        
        # FIXED: Loading total steps and episodes
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        
        # --- TRACKING (Deques) ---
        # FIXED: Loading tracking deques (converting lists back to deques)
        if 'episode_rewards' in checkpoint:
            self.episode_rewards.clear()
            self.episode_rewards.extend(checkpoint['episode_rewards'])

        if 'episode_costs' in checkpoint:
            for key, cost_list in checkpoint['episode_costs'].items():
                if key in self.episode_costs:
                    self.episode_costs[key].clear()
                    self.episode_costs[key].extend(cost_list)

        if 'predictor_losses' in checkpoint:
            self.predictor_losses.clear()
            self.predictor_losses.extend(checkpoint['predictor_losses'])

        if 'action_history' in checkpoint:
            self.action_history.clear()
            self.action_history.extend(checkpoint['action_history'])

        self.logger.info(f"Model loaded from {filepath}. Resumed at Episode {self.total_episodes}, Step {self.total_steps}")
    
    def set_training_mode(self, training):
        self.training_mode = training
        self.actor.train(training)
        self.critic_R.train(training)
        self.critic_C_drop.train(training)
        self.critic_C_latency.train(training)
        self.critic_C_resources.train(training)
        self.constraint_predictor.train(training)
        self.logger.info(f"Training mode set to {training}")

    def get_stats(self):
        return {
            'total_episodes': self.total_episodes,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'avg_cost_drop': np.mean(self.episode_costs['drop']) if self.episode_costs['drop'] else 0.0,
            'avg_cost_latency': np.mean(self.episode_costs['latency']) if self.episode_costs['latency'] else 0.0,
            'avg_cost_resources': np.mean(self.episode_costs['resources']) if self.episode_costs['resources'] else 0.0,
            'lambda_drop': self.lambda_controllers['drop'].get_lambda(),
            'lambda_latency': self.lambda_controllers['latency'].get_lambda(),
            'lambda_resources': self.lambda_controllers['resources'].get_lambda(),
            'predictor_loss': np.mean(self.predictor_losses) if self.predictor_losses else 0.0
        }