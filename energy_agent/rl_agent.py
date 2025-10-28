import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import logging
import os
from datetime import datetime
from collections import namedtuple
# from .transition import Transition, TransitionBuffer
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
CELL_CPU_USAGE_IDX = 0
CELL_PRB_USAGE_IDX = 1
CELL_CURRENT_LOAD_IDX = 2
CELL_MAX_CAPACITY_IDX = 3
CELL_NUM_CONNECTED_UES_IDX = 4
CELL_TX_POWER_IDX = 5
CELL_ENERGY_CONSUMPTION_IDX = 6
CELL_AVG_RSRP_IDX = 7
CELL_AVG_RSRQ_IDX = 8
CELL_AVG_SINR_IDX = 9
CELL_TOTAL_TRAFFIC_IDX = 10
CELL_LOAD_RATIO_IDX = 11

Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done', 
                         'log_prob', 'value_R', 'value_C_drop', 'value_C_latency', 'value_C_resources',
                         'cost_drop', 'cost_latency', 'cost_resources'))
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
    def __init__(self, n_cells, n_ues, max_time, log_file='rl_agent.log', use_gpu=False):
        """
        Initialize PPO agent for 5G energy saving
        
        Args:
            n_cells (int): Number of cells to control
            n_ues (int): Number of UEs in network
            max_time (int): Maximum simulation time steps
            log_file (str): Path to log file
            use_gpu (bool): Whether to use GPU acceleration
        """
        print("Initializing RL Agent")
        self.n_cells = n_cells
        self.n_ues = n_ues
        self.max_time = max_time
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        self.state_dim = 17 + 14 + (n_cells * 12)
        self.action_dim = n_cells
        self.state_normalizer = StateNormalizer(self.state_dim, n_cells=n_cells)
        
        self.actor_lr = 1e-5
        self.critic_lr = 1e-5
        
        # define actor
        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim=512).to(self.device)

        # define critics for reward, drop cost, latency cost
        self.critic_R = Critic(self.state_dim, hidden_dim=512).to(self.device) # Cho Reward
        self.critic_C_drop = Critic(self.state_dim, hidden_dim=512).to(self.device) # Cho Cost Drop Rate
        self.critic_C_latency = Critic(self.state_dim, hidden_dim=512).to(self.device) # Cho Cost Latency
        self.critic_C_resources = Critic(self.state_dim, hidden_dim=512).to(self.device) # Cho Cost resources

        # define optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_R_optimizer = optim.Adam(self.critic_R.parameters(), lr=self.critic_lr)
        self.critic_C_drop_optimizer = optim.Adam(self.critic_C_drop.parameters(), lr=self.critic_lr)
        self.critic_C_latency_optimizer = optim.Adam(self.critic_C_latency.parameters(), lr=self.critic_lr)
        self.critic_C_resources_optimizer = optim.Adam(self.critic_C_resources.parameters(), lr=self.critic_lr)
        
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.clip_epsilon = 0.2
        self.ppo_epochs = 5
        self.batch_size = 128
        self.buffer_size = 4096
        self.entropy_coeff = 0.02
        self.max_grad_norm = 0.5

        # learning rate for lagrangian
        self.lambda_lr = 0.05

        # Lagrangian multipliers for costs
        self.lambda_drop = torch.nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.lambda_latency = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.lambda_resources = torch.nn.Parameter(torch.tensor(5.0), requires_grad=True)

        # Optimizers to update lagrangians
        self.lambda_optimizer = optim.Adam([self.lambda_drop, self.lambda_latency, self.lambda_resources], lr=self.lambda_lr)

        self.buffer = TransitionBuffer(self.buffer_size)
        
        self.training_mode = True
        self.total_episodes = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_costs_drop = deque(maxlen=100)
        self.episode_costs_latency = deque(maxlen=100)
        self.episode_steps = 0
        self.current_episode_reward = 0.0
        self.current_episode_cost_drop = 0.0
        self.current_episode_cost_latency = 0.0
        self.action_history = deque(maxlen=1000)
        
        # Store action history
        self.action_history = deque(maxlen=1000)
        
        # Exploration 
        self.exploration_noise = 0.2  # Stddev for action noise during training
        self.exploration_min = 0.02
        self.exploration_decay = 0.99  # Decay rate per episode

        # Curriculum learning
        self.use_curriculum = True
        self.curriculum_stage = 1
        self.curriculum_transition = {
            1: 50, 
            2: 150
        }

        # Reward and violation tracking
        self.actor_losses = deque(maxlen=100)
        self.critic_losses = deque(maxlen=100)
        self.entropy_values = deque(maxlen=100)
        self.kl_divergences = deque(maxlen=100)
        self.energy_savings_history = deque(maxlen=100)
        self.drop_rate_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        self.kpi_violation_count = 0
        self.successful_episodes = 0

        # Logging setup
        self.setup_logging(log_file)
        
        self.logger.info(f"PPO Agent initialized: {n_cells} cells, {n_ues} UEs")
        self.logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        self.logger.info(f"Device: {self.device}")
    
    def normalize_state(self, state):
        """Normalize state vector to [0, 1] range"""
        return self.state_normalizer.normalize(state)
    
    def setup_logging(self, log_file):
        """Setup logging configuration"""
        self.logger = logging.getLogger('PPOAgent')
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
    
    def start_scenario(self):
        # self.load_model("ppo_model_20251026_142333.pth")
        self.total_episodes += 1
        self.episode_steps = 0
        self.current_episode_reward = 0.0
        self.action_history.clear()
        self.logger.info(f"Starting episode {self.total_episodes}")
    
    def end_scenario(self):
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_costs_drop.append(self.current_episode_cost_drop)
        self.episode_costs_latency.append(self.current_episode_cost_latency)
        
        self.logger.info(
            f"Episode {self.total_episodes} ended: "
            f"Reward={self.current_episode_reward:.2f}, "
            f"AvgCostDrop={self.current_episode_cost_drop / self.episode_steps:.2f}, "
            f"AvgCostLatency={self.current_episode_cost_latency / self.episode_steps:.2f}"
        )

        # Reset
        self.total_episodes += 1
        self.episode_steps = 0
        self.current_episode_reward = 0.0
        self.current_episode_cost_drop = 0.0
        self.current_episode_cost_latency = 0.0
        
        if self.training_mode and len(self.buffer) >= self.batch_size:
            self.train()
        
        self.save_model()

    # NOT REMOVED FOR INTERACTING WITH SIMULATION (CAN BE MODIFIED)
    def get_action(self, state):
        """
        Get action from policy network
        
        Args:
            state: State vector from MATLAB interface
            
        Returns:
            action: Power ratios for each cell [0, 1]
        """
        state = self.normalize_state(np.array(state).flatten())  # make sure it’s 1D
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean, action_logstd = self.actor(state_tensor)
            
            if self.training_mode:
                # Sample from policy during training
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
            else:
                # Use mean during evaluation
                action = action_mean
                log_prob = torch.zeros(1).to(self.device)
        
        # Clamp actions to [0, 1]
        action = torch.clamp(action, 0.0, 1.0)
        action_np = action.cpu().numpy().flatten()
        
        # Store for experience replay
        self.last_state = state_tensor.cpu().numpy().flatten()
        self.last_action = action_np
        self.last_log_prob = log_prob.cpu().numpy().flatten()

        # Add action to history
        self.action_history.append(action_np.copy())
        return action_np
    
    ## OPTIONAL: Modify reward calculation as needed
    def calculate_reward(self, prev_state, current_state):
        """Calculate reward based on energy savings and KPI constraints"""
        if prev_state is None:
            return 0.0, 0.0, 0.0, 0.0

        prev_state = np.array(prev_state).flatten()
        current_state = np.array(current_state).flatten()
        
        current_tx_power = current_state[TOTAL_TX_POWER_IDX]

        # Minimize total transmission power
        reward = -current_tx_power
        
        # Add rewards for meeting constraints
        current_drop_rate = current_state[AVG_DROP_RATE_IDX]
        current_latency = current_state[AVG_LATENCY_IDX]
        self.current_drop_limit = current_state[DROP_THRESHOLD]
        self.current_latency_limit = current_state[LATENCY_THRESHOLD]

        if current_drop_rate < self.current_drop_limit and current_latency < self.current_latency_limit:
            reward += 1.0 

        # Quality constraints
        cost_drop = max(0, (current_drop_rate - self.current_drop_limit) ** 2)
        cost_latency = max(0, (current_latency - self.current_latency_limit) ** 2)

        # Resource constraints
        cpu_threshold = current_state[CPU_THRESHOLD_IDX]
        prb_threshold = current_state[PRB_THRESHOLD_IDX]
        
        max_cpu_usage = current_state[MAX_CPU_USAGE_IDX]
        max_prb_usage = current_state[MAX_PRB_USAGE_IDX]
        
        cost_cpu = max(0, max_cpu_usage - cpu_threshold)
        cost_prb = max(0, max_prb_usage - prb_threshold)

        cost_resources = cost_cpu + cost_prb
        return float(reward), float(cost_drop), float(cost_latency), float(cost_resources)
    
    # NOT REMOVED FOR INTERACTING WITH SIMULATION (CAN BE MODIFIED)
    def update(self, state, action, next_state, done):
        """
        Update agent with experience
        
        Args:
            state: Previous state
            action: Action taken
            next_state: Next state
            done: Whether episode is done
        """
        if not self.training_mode:
            return

        reward, cost_drop, cost_latency, cost_resources = self.calculate_reward(state, next_state)

        self.episode_steps += 1
        self.total_steps += 1
        self.current_episode_reward += reward
        self.current_episode_cost_drop += cost_drop
        self.current_episode_cost_latency += cost_latency
        
        state_norm = self.normalize_state(np.array(state).flatten())
        next_state_norm = self.normalize_state(np.array(next_state).flatten())
        
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value_R = self.critic_R(state_tensor).item()
            value_C_drop = self.critic_C_drop(state_tensor).item()
            value_C_latency = self.critic_C_latency(state_tensor).item()
            value_C_resources = self.critic_C_resources(state_tensor).item()

        transition = Transition(
            state=state_norm,
            action=np.array(action).flatten(),
            reward=reward,
            next_state=next_state_norm,
            done=done,
            log_prob=getattr(self, 'last_log_prob', np.array([0.0]))[0],
            value_R=value_R,
            value_C_drop=value_C_drop,
            value_C_latency=value_C_latency,
            value_C_resources=value_C_resources,
            cost_drop=cost_drop,
            cost_latency=cost_latency,
            cost_resources=cost_resources
        )
        self.buffer.add(transition)
    
    
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            next_val = next_values[t]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.lambda_gae * next_non_terminal * last_advantage
        returns = advantages + values
        return advantages, returns
    
    def train(self):
        """Train the PPO agent"""
        if len(self.buffer) < self.batch_size:
            return
        
        # Get all transitions
        transitions = self.buffer.get_all()
        self.buffer.clear()
        
        states = np.array([t.state for t in transitions], dtype=np.float32)
        actions = np.array([t.action for t in transitions], dtype=np.float32)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        costs_drop = np.array([t.cost_drop for t in transitions], dtype=np.float32)
        costs_latency = np.array([t.cost_latency for t in transitions], dtype=np.float32)
        costs_resources = np.array([t.cost_resources for t in transitions], dtype=np.float32)
        next_states = np.array([t.next_state for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)
        old_log_probs = np.array([t.log_prob for t in transitions], dtype=np.float32)

        values_R = np.array([t.value_R for t in transitions], dtype=np.float32)
        values_C_drop = np.array([t.value_C_drop for t in transitions], dtype=np.float32)
        values_C_latency = np.array([t.value_C_latency for t in transitions], dtype=np.float32)
        values_C_resources = np.array([t.value_C_resources for t in transitions], dtype=np.float32)

        # Compute next values
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        with torch.no_grad():
            next_values_R = self.critic_R(next_states_tensor).cpu().numpy().flatten()
            next_values_C_drop = self.critic_C_drop(next_states_tensor).cpu().numpy().flatten()
            next_values_C_latency = self.critic_C_latency(next_states_tensor).cpu().numpy().flatten()
            next_values_C_resources = self.critic_C_resources(next_states_tensor).cpu().numpy().flatten()
        
        # Compute advantages for rewards and costs
        advantages_R, returns_R = self.compute_gae(rewards, values_R, next_values_R, dones)
        advantages_C_drop, returns_C_drop = self.compute_gae(costs_drop, values_C_drop, next_values_C_drop, dones)
        advantages_C_latency, returns_C_latency = self.compute_gae(costs_latency, values_C_latency, next_values_C_latency, dones)
        advantages_C_resources, returns_C_resources = self.compute_gae(costs_resources, values_C_resources, next_values_C_resources, dones)

        # Combine advantages to create Lagrangian advantage
        lambda_d = self.lambda_drop.item()
        lambda_l = self.lambda_latency.item()
        lambda_v = self.lambda_resources.item()

        advantages_L = (advantages_R - lambda_d * advantages_C_drop - lambda_l * advantages_C_latency - lambda_v * advantages_C_resources) / (1.0 + lambda_d + lambda_l + lambda_v)

        # Normalize combined advantage
        advantages_L = (advantages_L - np.mean(advantages_L)) / (np.std(advantages_L) + 1e-8)
                    
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_L_tensor = torch.FloatTensor(advantages_L).to(self.device)

        returns_R_tensor = torch.FloatTensor(returns_R).to(self.device)
        returns_C_drop_tensor = torch.FloatTensor(returns_C_drop).to(self.device)
        returns_C_latency_tensor = torch.FloatTensor(returns_C_latency).to(self.device)
        returns_C_resources_tensor = torch.FloatTensor(returns_C_resources).to(self.device)

        # PPO training loop
        for epoch in range(self.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_entropies = []
            epoch_kl_divs = []

            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                # Create mini-batch
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages_L = advantages_L_tensor[batch_indices]
                batch_returns_R = returns_R_tensor[batch_indices]
                batch_returns_C_drop = returns_C_drop_tensor[batch_indices]
                batch_returns_C_latency = returns_C_latency_tensor[batch_indices]
                batch_returns_C_resources = returns_C_resources_tensor[batch_indices]

                # Compute action mean and logstd
                action_mean, action_logstd = self.actor(batch_states)

                # Clip action logstd 
                action_logstd = torch.clamp(action_logstd, -20, 2)
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                with torch.no_grad():
                    kl_divs = (batch_old_log_probs - new_log_probs).mean()
                    epoch_kl_divs.append(kl_divs.item())


                # Compute between old policy and new policy
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages_L
                surr2 = torch.clamp(ratio, 
                                    1 - self.clip_epsilon, 
                                    1 + self.clip_epsilon
                                    ) * batch_advantages_L
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 
                    self.max_grad_norm
                )
                self.actor_optimizer.step()

                # Update reward critic
                current_values_R = self.critic_R(batch_states).squeeze()
                critic_loss_R = nn.MSELoss()(current_values_R, batch_returns_R)
                self.critic_R_optimizer.zero_grad()
                critic_loss_R.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic_R.parameters(), 
                    self.max_grad_norm
                )
                self.critic_R_optimizer.step()

                # Update drop cost critic
                current_values_C_drop = self.critic_C_drop(batch_states).squeeze()
                critic_loss_C_drop = nn.MSELoss()(current_values_C_drop, batch_returns_C_drop)
                self.critic_C_drop_optimizer.zero_grad()
                critic_loss_C_drop.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic_C_drop.parameters(), 
                    self.max_grad_norm
                )
                self.critic_C_drop_optimizer.step()

                # Update latency cost critic
                current_values_C_latency = self.critic_C_latency(batch_states).squeeze()
                critic_loss_C_latency = nn.MSELoss()(current_values_C_latency, batch_returns_C_latency)
                self.critic_C_latency_optimizer.zero_grad()
                critic_loss_C_latency.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic_C_latency.parameters(), 
                    self.max_grad_norm
                )
                self.critic_C_latency_optimizer.step()

                # Update resources cost critic
                current_values_C_resources = self.critic_C_resources(batch_states).squeeze()
                critic_loss_C_resources = nn.MSELoss()(current_values_C_resources, batch_returns_C_resources)
                self.critic_C_resources_optimizer.zero_grad()
                critic_loss_C_resources.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic_C_resources.parameters(), 
                    self.max_grad_norm
                )
                self.critic_C_resources_optimizer.step()

            
            # Check if KL divergence is too high
            mean_kl_div = np.mean(epoch_kl_divs)
            kl_div_threshold = 0.015
            if mean_kl_div > kl_div_threshold:
                self.logger.warning(f"Early stopping at epoch {epoch} due to high KL divergence: {mean_kl_div:.4f}")
                break

        # In train()
        avg_cost_drop = np.mean(costs_drop)
        avg_cost_latency = np.mean(costs_latency)
        avg_cost_resource = np.mean(costs_resources) # Rename to costs_resource

        self.lambda_optimizer.zero_grad()

        # The goal is to drive the average cost (which represents the violation amount) to zero.
        cost_limit_drop = 2.0
        cost_limit_latency = 0.0
        cost_limit_resource = 0.0

        # The loss is -(lambda * (avg_cost - limit)). Maximizing this is the same as 
        # doing gradient ascent on lambda based on the violation.
        lambda_loss = -(self.lambda_drop * (avg_cost_drop - cost_limit_drop) + 
                        self.lambda_latency * (avg_cost_latency - cost_limit_latency) +
                        self.lambda_resources * (avg_cost_resource - cost_limit_resource)) # Use the renamed lambda

        lambda_loss.backward()
        self.lambda_optimizer.step()

        # Ensure lambdas non negative
        self.lambda_drop.data.clamp_(0)
        self.lambda_latency.data.clamp_(0)
        self.lambda_resources.data.clamp_(0) # Use the renamed lambda

        self.logger.info(
            f"Training completed: "
            f"Avg Reward={np.mean(rewards):.2f}, "
            f"Avg Cost Drop={avg_cost_drop:.4f}, "
            f"Avg Cost Latency={avg_cost_latency:.4f}, "
            f"Lambda Drop={self.lambda_drop.item():.4f}, "
            f"Lambda Latency={self.lambda_latency.item():.4f}, "
            f"Lambda resources={self.lambda_resources.item():.4f}"
        )
        
    
    def save_model(self, filepath=None):
        """
        Save model parameters to a file.
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ppo_model_{timestamp}.pth"
        
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_R_state_dict': self.critic_R.state_dict(),
            'critic_C_drop_state_dict': self.critic_C_drop.state_dict(),
            'critic_C_latency_state_dict': self.critic_C_latency.state_dict(),
            'critic_C_resources_state_dict': self.critic_C_resources.state_dict(), # FIXED: Consistent naming

            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_R_optimizer_state_dict': self.critic_R_optimizer.state_dict(),
            'critic_C_drop_optimizer_state_dict': self.critic_C_drop_optimizer.state_dict(),
            'critic_C_latency_optimizer_state_dict': self.critic_C_latency_optimizer.state_dict(),
            'critic_C_resources_optimizer_state_dict': self.critic_C_resources_optimizer.state_dict(), # FIXED: Consistent naming

            'lambda_optimizer_state_dict': self.lambda_optimizer.state_dict(), # ADDED: Save lambda optimizer state
            
            'lambda_drop': self.lambda_drop.item(),
            'lambda_latency': self.lambda_latency.item(),
            'lambda_resources': self.lambda_resources.item(), # FIXED: Consistent naming
            
            'total_episodes': self.total_episodes, # ADDED: Save episode count
            'total_steps': self.total_steps,
            'exploration_noise': self.exploration_noise,
            'episode_rewards': list(self.episode_rewards),
            'episode_costs_drop': list(self.episode_costs_drop),
            'episode_costs_latency': list(self.episode_costs_latency),
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load model parameters from a file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_R.load_state_dict(checkpoint['critic_R_state_dict'])
        self.critic_C_drop.load_state_dict(checkpoint['critic_C_drop_state_dict'])
        self.critic_C_latency.load_state_dict(checkpoint['critic_C_latency_state_dict'])
        self.critic_C_resources.load_state_dict(checkpoint['critic_C_resources_state_dict']) # FIXED: Consistent naming
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_R_optimizer.load_state_dict(checkpoint['critic_R_optimizer_state_dict'])
        self.critic_C_drop_optimizer.load_state_dict(checkpoint['critic_C_drop_optimizer_state_dict'])
        self.critic_C_latency_optimizer.load_state_dict(checkpoint['critic_C_latency_optimizer_state_dict'])
        self.critic_C_resources_optimizer.load_state_dict(checkpoint['critic_C_resources_optimizer_state_dict']) # FIXED: Consistent naming

        # ADDED: Load lambda optimizer state (check for existence for backward compatibility)
        if 'lambda_optimizer_state_dict' in checkpoint:
            self.lambda_optimizer.load_state_dict(checkpoint['lambda_optimizer_state_dict'])
        
        self.lambda_drop = torch.nn.Parameter(torch.tensor(checkpoint.get('lambda_drop', 1.0)), requires_grad=True)
        self.lambda_latency = torch.nn.Parameter(torch.tensor(checkpoint.get('lambda_latency', 1.0)), requires_grad=True)
        self.lambda_resources = torch.nn.Parameter(torch.tensor(checkpoint.get('lambda_resources', 1.0)), requires_grad=True) # FIXED: Consistent naming
        
        self.total_episodes = checkpoint.get('total_episodes', 0) # ADDED: Load episode count
        self.total_steps = checkpoint.get('total_steps', 0)
        self.exploration_noise = checkpoint.get('exploration_noise', 0.1)
        
        if 'episode_rewards' in checkpoint:
            self.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=100)
        if 'episode_costs_drop' in checkpoint:
            self.episode_costs_drop = deque(checkpoint['episode_costs_drop'], maxlen=100)
        if 'episode_costs_latency' in checkpoint:
            self.episode_costs_latency = deque(checkpoint['episode_costs_latency'], maxlen=100)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def set_training_mode(self, training):
        """Set training mode"""
        self.training_mode = training
        self.actor.train(training)
        self.critic_R.train(training)
        self.critic_C_drop.train(training)
        self.critic_C_latency.train(training)
        self.critic_C_resources.train(training)
        self.logger.info(f"Training mode set to {training}")
    
    def get_stats(self):
        """Get training statistics"""
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        avg_actor_loss = np.mean(self.actor_losses) if self.actor_losses else 0.0
        avg_critic_loss = np.mean(self.critic_losses) if self.critic_losses else 0.0
        avg_entropy = np.mean(self.entropy_values) if self.entropy_values else 0.0
        
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'avg_reward': avg_reward,
            'avg_actor_loss': avg_actor_loss,
            'avg_critic_loss': avg_critic_loss,
            'avg_entropy': avg_entropy,
            'buffer_size': len(self.buffer),
            'training_mode': self.training_mode,
            'episode_steps': self.episode_steps,
            'current_episode_reward': self.current_episode_reward,
            'exploration_noise': self.exploration_noise,
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_R_optimizer.param_groups[0]['lr']
        }