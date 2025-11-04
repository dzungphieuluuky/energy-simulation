from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
class ConstraintMonitorCallback(BaseCallback):
    """Enhanced monitoring with violation tracking and adaptive penalties."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.violation_history = []
        self.compliance_history = []
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # Check if episode just finished
        if self.locals.get('dones', [False])[0]:
            # Get final metrics from info
            info = self.locals.get('infos', [{}])[0]
            
            # Track violations
            kpi_violations = info.get('kpi_violations', 0)
            self.violation_history.append(kpi_violations)
            
            # Track compliance
            drop_ok = info.get('avg_drop_rate', 999) <= self.training_env.get_attr('sim_params')[0].drop_call_threshold
            latency_ok = info.get('avg_latency', 999) <= self.training_env.get_attr('sim_params')[0].latency_threshold
            cpu_ok = info.get('cpu_violations', 999) == 0
            prb_ok = info.get('prb_violations', 999) == 0
            
            all_ok = drop_ok and latency_ok and cpu_ok and prb_ok
            self.compliance_history.append(all_ok)
            
            # Log every 10 episodes
            if len(self.violation_history) % 10 == 0:
                recent_compliance = np.mean(self.compliance_history[-10:]) * 100
                recent_violations = np.mean(self.violation_history[-10:])
                
                print(f"\n[Step {self.num_timesteps}] Last 10 Episodes:")
                print(f"  Compliance Rate: {recent_compliance:.1f}%")
                print(f"  Avg Violations: {recent_violations:.2f}")
                
                if recent_compliance < 50:
                    print("  ⚠️  WARNING: Low compliance rate! Agent struggling with constraints.")
                elif recent_compliance > 80:
                    print("  ✅ Good compliance! Agent learning constraint satisfaction.")
        
        return True

class AlgorithmComparisonCallback(BaseCallback):
    """Callback to track training progress and compare algorithm performance."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        # Track episode rewards
        if len(self.locals.get('rewards', [])) > 0:
            self.current_episode_reward += sum(self.locals['rewards'])
            
        dones = self.locals.get('dones', [])
        if any(dones):
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            
            if len(self.episode_rewards) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {len(self.episode_rewards)}, Last 10 eps mean reward: {mean_reward:.3f}")
        
        return True