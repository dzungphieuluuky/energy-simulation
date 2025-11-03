from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
class ConstraintMonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_compliance = []
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # Log every episode
        if self.locals.get('dones')[0]:
            info = self.locals['infos'][0]
            
            if 'reward_info' in info:
                compliance = info['reward_info']['metrics']['qos_compliant_steps']
                total_steps = info['reward_info']['metrics'].get('total_steps', 500)
                compliance_rate = 100 * compliance / total_steps
                
                self.episode_compliance.append(compliance_rate)
                self.episode_rewards.append(info['reward_info']['episode_rewards'])
                
                # Print every 50 episodes
                if len(self.episode_compliance) % 50 == 0:
                    recent_compliance = np.mean(self.episode_compliance[-50:])
                    recent_reward = np.mean(self.episode_rewards[-50:])
                    
                    print(f"\n[Episodes {len(self.episode_compliance)-49}-{len(self.episode_compliance)}]")
                    print(f"  Avg compliance: {recent_compliance:.1f}%")
                    print(f"  Avg reward: {recent_reward:.3f}")
        
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