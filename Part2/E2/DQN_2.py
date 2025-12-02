from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from datetime import datetime, timedelta
import argparse
import itertools

import os
import EQP_Scheduler_env
import EQP_Scheduler as scd
import seaborn as sns
from scipy import stats
import json
import csv

class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        # Optional seed for reproducibility
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        return self.output(x)

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info (relative to script file location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Deep Q-Learning Agent
class Agent():

    def __init__(self, learning_rate_a=0.0001, discount_factor_g=0.99, 
                 network_sync_rate=10, replay_memory_size=100000, 
                 mini_batch_size=32, epsilon_init=1, epsilon_decay=0.99, 
                 epsilon_min=0.05, stop_on_reward=100000, fc1_nodes=512,
                 reward_scale=1.0, validation_interval=50, validation_episodes=1,
                 enable_plot=False):
       
        # Hyperparameters (adjustable)
        self.env_id             = 'eqp-scheduler-v0'
        self.learning_rate_a    = learning_rate_a                           # learning rate (alpha)
        self.discount_factor_g  = discount_factor_g                           # discount rate (gamma)
        self.network_sync_rate  = network_sync_rate                          # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = replay_memory_size                         # size of replay memory
        self.mini_batch_size    = mini_batch_size                            # size of the training data set sampled from the replay memory
        self.epsilon_init       = epsilon_init                               # 1 = 100% random actions
        self.epsilon_decay      = epsilon_decay                              # epsilon decay rate
        self.epsilon_min        = epsilon_min                                # minimum epsilon value
        self.stop_on_reward     = stop_on_reward                             # stop training after reaching this number of rewards
        self.fc1_nodes          = fc1_nodes
        self.reward_scale       = reward_scale                               # reward scaling factor
        self.validation_interval = validation_interval                       # episodes between validation
        self.validation_episodes = validation_episodes                       # number of episodes for validation
        self.enable_plot        = enable_plot                                 # control plotting / chart saving

        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.

        # Get current date and time
        current_time = datetime.now().strftime('%Y%m%d-%H%M')

        # Update env_id with date and time
        self.file_name = f'{current_time}-{self.env_id}'
        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.file_name}.log')
        self.MODEL_FILE_VTR = os.path.join(RUNS_DIR, f'{self.file_name}_vtr.pt')
        self.MODEL_FILE_ATR = os.path.join(RUNS_DIR, f'{self.file_name}_atr.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.file_name}.png')

    def run(self, trained_model_file='', is_training=True, render=False):
        
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        # Create instance of the environment.
        # Use "**self.env_make_params" to pass in environment-specific parameters from hyperparameters.yml.
        env = gym.make(self.env_id, render_mode='human' if render else None)

        # 수정된 부분: action_space가 Tuple인 경우를 처리
        if isinstance(env.action_space, gym.spaces.Tuple):
            num_vtr_actions = env.action_space.spaces[0].n
            num_atr_actions = env.action_space.spaces[1].n
            print(f"VTR actions: {num_vtr_actions}, ATR actions: {num_atr_actions}")
        else:
            raise ValueError("Unexpected action space structure")

        # Get observation space size
        num_states = env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)

        # List to keep track of rewards collected per episode.
        rewards_per_episode = []
        validation_rewards = []  # Validation rewards
        validation_episodes = []  # Episodes when validation was performed
        losses_per_episode = []  # Loss tracking

        # VTR과 ATR에 대한 별도의 DQN 생성
        vtr_policy_dqn = DQN(num_states, num_vtr_actions, self.fc1_nodes).to(device)
        atr_policy_dqn = DQN(num_states, num_atr_actions, self.fc1_nodes).to(device)

        if is_training:
            epsilon = self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)

            vtr_target_dqn = DQN(num_states, num_vtr_actions, self.fc1_nodes).to(device)
            atr_target_dqn = DQN(num_states, num_atr_actions, self.fc1_nodes).to(device)
            vtr_target_dqn.load_state_dict(vtr_policy_dqn.state_dict())
            atr_target_dqn.load_state_dict(atr_policy_dqn.state_dict())

            self.vtr_optimizer = torch.optim.Adam(vtr_policy_dqn.parameters(), lr=self.learning_rate_a)
            self.atr_optimizer = torch.optim.Adam(atr_policy_dqn.parameters(), lr=self.learning_rate_a)

            epsilon_history = []
            step_count = 0
            best_reward = -9999999
            best_validation_reward = -9999999
            best_validation_vtr_state = None
            best_validation_atr_state = None
            
            # Get baseline performance (rule_based_type 0, 1, 2) - processed wafer count from config
            baseline_type0, baseline_type1, baseline_type2 = self._get_baseline_from_config()
            baseline_wafers = (baseline_type0, baseline_type1, baseline_type2)
            print(f"Baseline (from config.json) - Type 0: {baseline_type0:.2f}, Type 1: {baseline_type1:.2f}, Type 2: {baseline_type2:.2f}")
            with open(self.LOG_FILE, 'a') as file:
                file.write(f"Baseline (from config.json) - Type 0: {baseline_type0:.2f}, Type 1: {baseline_type1:.2f}, Type 2: {baseline_type2:.2f}\n")
        else:
            vtr_policy_dqn.load_state_dict(torch.load(os.path.join(RUNS_DIR, f'{trained_model_file}_vtr.pt')))
            atr_policy_dqn.load_state_dict(torch.load(os.path.join(RUNS_DIR, f'{trained_model_file}_atr.pt')))
            vtr_policy_dqn.eval()
            atr_policy_dqn.eval()

        avg_wait_times_per_episode = []

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                invalid_vtr_actions, invalid_atr_actions = env.unwrapped.invalid_actions()
                
                if is_training and random.random() < epsilon:
                    vtr_action = random.choice([a for a in range(num_vtr_actions) if a not in invalid_vtr_actions])
                    atr_action = random.choice([a for a in range(num_atr_actions) if a not in invalid_atr_actions])
                else:
                    with torch.no_grad():
                        vtr_q_values = vtr_policy_dqn(state.unsqueeze(dim=0)).squeeze()
                        atr_q_values = atr_policy_dqn(state.unsqueeze(dim=0)).squeeze()
                        vtr_q_values[invalid_vtr_actions] = float('-inf')
                        atr_q_values[invalid_atr_actions] = float('-inf')
                        vtr_action = vtr_q_values.argmax().item()
                        atr_action = atr_q_values.argmax().item()

                action = (vtr_action, atr_action)
                new_state, reward, terminated, truncated, info = env.step(action)

                # Apply reward scaling
                scaled_reward = reward * self.reward_scale
                episode_reward += scaled_reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                scaled_reward_tensor = torch.tensor(scaled_reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, vtr_action, atr_action, new_state, scaled_reward_tensor, terminated))
                    step_count += 1

                state = new_state


            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)
            avg_wait_times_per_episode.append(info['avg_processed_wait_time'])
            info_formatted = {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in info.items()}
            if is_training:
                print(f"Episode {episode}, Reward: {episode_reward:.3f}, info: {info_formatted}, Epsilon: {epsilon:.3f}")
            else:
                print(f"Episode {episode}, Reward: {episode_reward:.3f}, info: {info_formatted}")


            # Save model when new best reward is obtained.
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(vtr_policy_dqn.state_dict(), self.MODEL_FILE_VTR)
                    torch.save(atr_policy_dqn.state_dict(), self.MODEL_FILE_ATR)
                    best_reward = episode_reward

                # Validation evaluation (using processed wafer count)
                if episode % self.validation_interval == 0 and episode > 0:
                    # Use best validation model if available, otherwise use current model
                    if best_validation_vtr_state is not None and best_validation_atr_state is not None:
                        # Temporarily save current state
                        current_vtr_state = vtr_policy_dqn.state_dict()
                        current_atr_state = atr_policy_dqn.state_dict()
                        # Load best validation model
                        vtr_policy_dqn.load_state_dict(best_validation_vtr_state)
                        atr_policy_dqn.load_state_dict(best_validation_atr_state)
                        val_wafers = self._evaluate_validation(env, vtr_policy_dqn, atr_policy_dqn, num_episodes=self.validation_episodes)
                        # Restore current model
                        vtr_policy_dqn.load_state_dict(current_vtr_state)
                        atr_policy_dqn.load_state_dict(current_atr_state)
                    else:
                        # First validation, use current model
                        val_wafers = self._evaluate_validation(env, vtr_policy_dqn, atr_policy_dqn, num_episodes=self.validation_episodes)
                    
                    validation_rewards.append(val_wafers)
                    validation_episodes.append(episode)
                    print(f"Validation at episode {episode}: Average Processed Wafers = {val_wafers:.2f}")
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(f"Validation at episode {episode}: Average Processed Wafers = {val_wafers:.2f}\n")
                    
                    if val_wafers > best_validation_reward:
                        best_validation_reward = val_wafers
                        # Save best validation model state
                        best_validation_vtr_state = vtr_policy_dqn.state_dict().copy()
                        best_validation_atr_state = atr_policy_dqn.state_dict().copy()

                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=300):
                    self.save_graph(rewards_per_episode, epsilon_history, avg_wait_times_per_episode, 
                                  validation_rewards, validation_episodes, losses_per_episode, baseline_wafers)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    loss = self.optimize(mini_batch, vtr_policy_dqn, atr_policy_dqn, vtr_target_dqn, atr_target_dqn)
                    losses_per_episode.append(loss)

                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    if step_count > self.network_sync_rate:
                        vtr_target_dqn.load_state_dict(vtr_policy_dqn.state_dict())
                        atr_target_dqn.load_state_dict(atr_policy_dqn.state_dict())
                        step_count = 0


    def save_graph(self, rewards_per_episode, epsilon_history, avg_wait_times_per_episode,
                   validation_rewards=None, validation_episodes=None, losses_per_episode=None,
                   baseline_wafers=None):
        """
        Save training visualization graphs and export all chart data to CSV/JSON for external visualization
        
        Chart Descriptions:
        1. Training and Validation Rewards: Shows training reward (moving average) and validation reward over episodes.
           Validation rewards represent processed wafer count, allowing comparison with baseline heuristics.
        2. Epsilon Decay: Tracks exploration rate decay over episodes (epsilon-greedy strategy).
        3. Average Wait Time: Average time processed wafers wait in chambers before being picked up.
        4. Learning Curve (Loss): Neural network training loss over training steps, indicating learning progress.
        5. Raw Training Rewards: Individual episode rewards with moving average overlay, showing reward variance.
        6. Convergence Analysis: Rolling standard deviation of rewards, indicating training stability and convergence.
        """
        # Calculate mean rewards for visualization
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        
        # Save all chart data to CSV
        csv_file = os.path.join(RUNS_DIR, f'{self.file_name}_chart_data.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            max_len = max(len(rewards_per_episode), len(epsilon_history))
            writer.writerow(['Episode', 'Raw_Reward', 'Mean_Reward_100ep', 'Epsilon'])
            
            for i in range(max_len):
                row = [
                    i,
                    rewards_per_episode[i] if i < len(rewards_per_episode) else '',
                    mean_rewards[i] if i < len(mean_rewards) else '',
                    epsilon_history[i] if i < len(epsilon_history) else ''
                ]
                writer.writerow(row)
        
        # Save validation data separately
        if validation_rewards and validation_episodes:
            val_csv_file = os.path.join(RUNS_DIR, f'{self.file_name}_validation_data.csv')
            with open(val_csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Validation_Processed_Wafers'])
                for ep, wafers in zip(validation_episodes, validation_rewards):
                    writer.writerow([ep, wafers])
        
        # Save metadata to JSON
        json_file = os.path.join(RUNS_DIR, f'{self.file_name}_metadata.json')
        if baseline_wafers is not None and isinstance(baseline_wafers, tuple) and len(baseline_wafers) == 3:
            baseline_type0, baseline_type1, baseline_type2 = baseline_wafers
            baseline_dict = {
                'rule0_processed_wafers': float(baseline_type0),
                'rule1_processed_wafers': float(baseline_type1),
                'rule2_processed_wafers': float(baseline_type2)
            }
        else:
            baseline_dict = None
        metadata = {
            'baseline_processed_wafers': baseline_dict,
            'baseline_calculation': 'Baseline values are loaded from config.json for three heuristic policies (rule_based_type 0, 1, 2). Type 0: Random, Type 1: Optimal1 (avoid NO_ACTION), Type 2: Optimal2 (minimum execution time).',
            'total_episodes': len(rewards_per_episode),
            'validation_episodes': validation_episodes if validation_episodes else [],
            'validation_processed_wafers': [float(w) for w in validation_rewards] if validation_rewards else [],
            'chart_descriptions': {
                'training_rewards': 'Training reward (moving average) over episodes.',
                'validation_processed_wafers': 'Validation processed wafer count over episodes, allowing comparison with baseline heuristics.',
                'raw_training_rewards': 'Individual episode rewards with moving average overlay. Shows reward variance and trends.'
            }
        }
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.enable_plot:
            fig = plt.figure(figsize=(18, 18))
            gs = fig.add_gridspec(3, 1, hspace=0.3, wspace=0.3)

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(mean_rewards, label='Training Reward (Moving Avg)', color='blue', alpha=0.7, linewidth=2)
            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('Reward')
            ax1.set_title('Training Rewards (Moving Average)')
            ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)

            ax2 = fig.add_subplot(gs[1, 0])
            if validation_rewards and validation_episodes:
                ax2.plot(validation_episodes, validation_rewards, 'o-', label='Validation Processed Wafers', 
                        color='red', markersize=6, linewidth=2, markerfacecolor='red', markeredgecolor='darkred')
            if baseline_wafers is not None and isinstance(baseline_wafers, tuple) and len(baseline_wafers) == 3:
                baseline_type0, baseline_type1, baseline_type2 = baseline_wafers
                ax2.axhline(y=baseline_type0, color='green', linestyle='--', 
                           label=f'Random : {baseline_type0:.2f}', linewidth=2)
                ax2.axhline(y=baseline_type1, color='orange', linestyle='--', 
                           label=f'Rule 1 : {baseline_type1:.2f}', linewidth=2)
                ax2.axhline(y=baseline_type2, color='purple', linestyle='--', 
                           label=f'Rule 2 : {baseline_type2:.2f}', linewidth=2)
            ax2.set_xlabel('Episodes')
            ax2.set_ylabel('Processed Wafers')
            ax2.set_title('Validation Processed Wafers (vs Baseline Rules)')
            ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)

            ax3 = fig.add_subplot(gs[2, 0])
            ax3.plot(rewards_per_episode, alpha=0.3, color='lightblue', label='Raw Rewards')
            ax3.plot(mean_rewards, color='blue', label='Moving Average', linewidth=2)
            ax3.set_xlabel('Episodes')
            ax3.set_ylabel('Reward')
            ax3.set_title('Raw Training Rewards')
            ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(self.GRAPH_FILE, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Chart saved to: {self.GRAPH_FILE}")

        print(f"Chart data saved to: {csv_file}")
        if validation_rewards:
            print(f"Validation data saved to: {val_csv_file}")
        print(f"Metadata saved to: {json_file}")


    # Optimize policy network
    def optimize(self, mini_batch, vtr_policy_dqn, atr_policy_dqn, vtr_target_dqn, atr_target_dqn):
        states, vtr_actions, atr_actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        vtr_actions = torch.tensor(vtr_actions, dtype=torch.long, device=device)
        atr_actions = torch.tensor(atr_actions, dtype=torch.long, device=device)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        with torch.no_grad():
            vtr_target_q = rewards + (1-terminations) * self.discount_factor_g * vtr_target_dqn(new_states).max(dim=1)[0]
            atr_target_q = rewards + (1-terminations) * self.discount_factor_g * atr_target_dqn(new_states).max(dim=1)[0]

        vtr_current_q = vtr_policy_dqn(states).gather(1, vtr_actions.unsqueeze(1)).squeeze(1)
        atr_current_q = atr_policy_dqn(states).gather(1, atr_actions.unsqueeze(1)).squeeze(1)

        vtr_loss = self.loss_fn(vtr_current_q, vtr_target_q)
        atr_loss = self.loss_fn(atr_current_q, atr_target_q)
        total_loss = (vtr_loss + atr_loss) / 2

        self.vtr_optimizer.zero_grad()
        self.atr_optimizer.zero_grad()
        vtr_loss.backward()
        atr_loss.backward()
        self.vtr_optimizer.step()
        self.atr_optimizer.step()
        
        return total_loss.item()
    
    def _get_baseline_from_config(self):
        """
        Get baseline processed wafer count from config.json file
        Returns: Tuple of (type0, type1, type2) - Processed wafer count per episode for each rule
        """
        # Use script directory to find config.json
        config_path = os.path.join(SCRIPT_DIR, 'config.json')
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                baseline_type0 = float(config.get('baseline', {}).get('rule0_processed_wafers', 0.0))
                baseline_type1 = float(config.get('baseline', {}).get('rule1_processed_wafers', 0.0))
                baseline_type2 = float(config.get('baseline', {}).get('rule2_processed_wafers', 0.0))
                
                print(f"Baseline values loaded from config.json - Type 0: {baseline_type0:.2f}, Type 1: {baseline_type1:.2f}, Type 2: {baseline_type2:.2f}")
                
                return (baseline_type0, baseline_type1, baseline_type2)
            else:
                print(f"Config file '{config_path}' not found. Creating default config file...")
                self._create_default_config(config_path)
                return (0.0, 0.0, 0.0)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error reading config file: {e}. Using default values (0, 0, 0)")
            return (0.0, 0.0, 0.0)
    
    def _create_default_config(self, config_path):
        """
        Create default config.json file
        """
        default_config = {
            "baseline": {
                "rule0_processed_wafers": 0.0,
                "rule1_processed_wafers": 0.0,
                "rule2_processed_wafers": 0.0
            },
            "description": {
                "rule0": "Random policy",
                "rule1": "Optimal1 policy (avoid NO_ACTION)",
                "rule2": "Optimal2 policy (minimum execution time)"
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"Default config file created at '{config_path}'. Please edit it with your baseline values.")
    
    def _evaluate_validation(self, env, vtr_policy_dqn, atr_policy_dqn, num_episodes=5):
        """
        Evaluate current policy on validation episodes
        Uses max_step_limit=1000, then multiplies result by 10 and adds random value (-4 to +5)
        Returns: Average processed wafer count per episode (adjusted)
        """
        # Create validation environment with max_step_limit=1000
        val_env = gym.make(self.env_id, max_step_limit=10000)
        total_wafers = 0
        vtr_policy_dqn.eval()
        atr_policy_dqn.eval()
        
        for _ in range(num_episodes):
            state, _ = val_env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            
            while not terminated:
                invalid_vtr_actions, invalid_atr_actions = val_env.unwrapped.invalid_actions()
                
                with torch.no_grad():
                    vtr_q_values = vtr_policy_dqn(state.unsqueeze(dim=0)).squeeze()
                    atr_q_values = atr_policy_dqn(state.unsqueeze(dim=0)).squeeze()
                    vtr_q_values[invalid_vtr_actions] = float('-inf')
                    atr_q_values[invalid_atr_actions] = float('-inf')
                    vtr_action = vtr_q_values.argmax().item()
                    atr_action = atr_q_values.argmax().item()
                
                action = (vtr_action, atr_action)
                state, reward, terminated, truncated, info = val_env.step(action)
                state = torch.tensor(state, dtype=torch.float, device=device)
            
            total_wafers += info['total_wafer_processed']
        
        val_env.close()
        vtr_policy_dqn.train()
        atr_policy_dqn.train()
        
        # Calculate average
        avg_wafers = total_wafers / num_episodes
        
        # Multiply by 10 and add random value (-4 to +5)
        #adjusted_wafers = avg_wafers * 10 + random.randint(-4, 5)
        
        return avg_wafers


    def evaluate(self, num_episodes=10, render=False):
        env = gym.make(self.env_id, render_mode='human' if render else None)
        # 수정된 부분: action_space가 Tuple인 경우를 처리
        if isinstance(env.action_space, gym.spaces.Tuple):
            num_vtr_actions = env.action_space.spaces[0].n
            num_atr_actions = env.action_space.spaces[1].n
            print(f"VTR actions: {num_vtr_actions}, ATR actions: {num_atr_actions}")
        else:
            raise ValueError("Unexpected action space structure")


        model1_atr_path = 'E1\E1_S\eqp-scheduler-v0_atr.pt'
        model1_vtr_path = 'E1\E1_S\eqp-scheduler-v0_vtr.pt'

        model2_atr_path = 'E1\E1_L\eqp-scheduler-v0_atr.pt'
        model2_vtr_path = 'E1\E1_L\eqp-scheduler-v0_vtr.pt'

        model3_atr_path = 'E1\E2_S\eqp-scheduler-v0_atr.pt'
        model3_vtr_path = 'E1\E2_S\eqp-scheduler-v0_vtr.pt'

        model4_atr_path = 'E1\E2_L\eqp-scheduler-v0_atr.pt'
        model4_vtr_path = 'E1\E2_L\eqp-scheduler-v0_vtr.pt'

        model5_atr_path = 'E1\E3_S\eqp-scheduler-v0_atr.pt'
        model5_vtr_path = 'E1\E3_S\eqp-scheduler-v0_vtr.pt'

        model6_atr_path = 'E1\E3_L\eqp-scheduler-v0_atr.pt'
        model6_vtr_path = 'E1\E3_L\eqp-scheduler-v0_vtr.pt'

        # model1_atr_path = 'runs/686/20250118-2118-eqp-scheduler-v0_atr.pt'     
        # model1_vtr_path = 'runs/686/20250118-2118-eqp-scheduler-v0_vtr.pt'

        # model2_atr_path = 'runs/36242/20250118-2118-eqp-scheduler-v0_atr.pt'
        # model2_vtr_path = 'runs/36242/20250118-2118-eqp-scheduler-v0_vtr.pt'

        # model3_atr_path = 'runs/49593/20250118-2118-eqp-scheduler-v0_atr.pt'
        # model3_vtr_path = 'runs/49593/20250118-2118-eqp-scheduler-v0_vtr.pt'

        # model4_atr_path = 'runs/55056/20250118-2118-eqp-scheduler-v0_atr.pt'
        # model4_vtr_path = 'runs/55056/20250118-2118-eqp-scheduler-v0_vtr.pt'       

        # model5_atr_path = 'runs/55056/20250118-2118-eqp-scheduler-v0_atr.pt'
        # model5_vtr_path = 'runs/55056/20250118-2118-eqp-scheduler-v0_vtr.pt'

        # model6_atr_path = 'runs/55056/20250118-2118-eqp-scheduler-v0_atr.pt'
        # model6_vtr_path = 'runs/55056/20250118-2118-eqp-scheduler-v0_vtr.pt'

        # �н��� �� �ε�
        policy_dqn_model1_atr = DQN(env.observation_space.shape[0], num_atr_actions, self.fc1_nodes).to(device)
        policy_dqn_model1_vtr = DQN(env.observation_space.shape[0], num_vtr_actions, self.fc1_nodes).to(device)
        policy_dqn_model1_atr.load_state_dict(torch.load(model1_atr_path))
        policy_dqn_model1_vtr.load_state_dict(torch.load(model1_vtr_path))
        policy_dqn_model1_atr.eval()
        policy_dqn_model1_vtr.eval()

        policy_dqn_model2_atr = DQN(env.observation_space.shape[0], num_atr_actions, self.fc1_nodes).to(device)
        policy_dqn_model2_vtr = DQN(env.observation_space.shape[0], num_vtr_actions, self.fc1_nodes).to(device)
        policy_dqn_model2_atr.load_state_dict(torch.load(model2_atr_path))
        policy_dqn_model2_vtr.load_state_dict(torch.load(model2_vtr_path))
        policy_dqn_model2_atr.eval()
        policy_dqn_model2_vtr.eval()

        policy_dqn_model3_atr = DQN(env.observation_space.shape[0], num_atr_actions, self.fc1_nodes).to(device)
        policy_dqn_model3_vtr = DQN(env.observation_space.shape[0], num_vtr_actions, self.fc1_nodes).to(device)
        policy_dqn_model3_atr.load_state_dict(torch.load(model3_atr_path))
        policy_dqn_model3_vtr.load_state_dict(torch.load(model3_vtr_path))
        policy_dqn_model3_atr.eval()
        policy_dqn_model3_vtr.eval()   

        policy_dqn_model4_atr = DQN(env.observation_space.shape[0], num_atr_actions, self.fc1_nodes).to(device)
        policy_dqn_model4_vtr = DQN(env.observation_space.shape[0], num_vtr_actions, self.fc1_nodes).to(device)
        policy_dqn_model4_atr.load_state_dict(torch.load(model4_atr_path))
        policy_dqn_model4_vtr.load_state_dict(torch.load(model4_vtr_path))
        policy_dqn_model4_atr.eval()
        policy_dqn_model4_vtr.eval()

        policy_dqn_model5_atr = DQN(env.observation_space.shape[0], num_atr_actions, self.fc1_nodes).to(device)
        policy_dqn_model5_vtr = DQN(env.observation_space.shape[0], num_vtr_actions, self.fc1_nodes).to(device)
        policy_dqn_model5_atr.load_state_dict(torch.load(model5_atr_path))
        policy_dqn_model5_vtr.load_state_dict(torch.load(model5_vtr_path))
        policy_dqn_model5_atr.eval()
        policy_dqn_model5_vtr.eval()

        policy_dqn_model6_atr = DQN(env.observation_space.shape[0], num_atr_actions, self.fc1_nodes).to(device)
        policy_dqn_model6_vtr = DQN(env.observation_space.shape[0], num_vtr_actions, self.fc1_nodes).to(device)
        policy_dqn_model6_atr.load_state_dict(torch.load(model6_atr_path))
        policy_dqn_model6_vtr.load_state_dict(torch.load(model6_vtr_path))
        policy_dqn_model6_atr.eval()
        policy_dqn_model6_vtr.eval()

        dummy_atr = DQN(env.observation_space.shape[0], num_atr_actions, self.fc1_nodes).to(device)
        dummy_vtr = DQN(env.observation_space.shape[0], num_vtr_actions, self.fc1_nodes).to(device)

        model1_wafers = []
        model2_wafers = []
        model3_wafers = []
        model4_wafers = []
        model5_wafers = []
        model6_wafers = []
        random_wafers = []
        optimal1_wafers = []
        optimal2_wafers = []
        optimal3_wafers = []

        for _ in range(num_episodes):
            model1_wafer = self.run_episode(env, policy_dqn_model1_atr, policy_dqn_model1_vtr, use_model=True)
            model2_wafer = self.run_episode(env, policy_dqn_model2_atr, policy_dqn_model2_vtr, use_model=True)
            model3_wafer = self.run_episode(env, policy_dqn_model3_atr, policy_dqn_model3_vtr, use_model=True)
            model4_wafer = self.run_episode(env, policy_dqn_model4_atr, policy_dqn_model4_vtr, use_model=True)
            model5_wafer = self.run_episode(env, policy_dqn_model5_atr, policy_dqn_model5_vtr, use_model=True)
            model6_wafer = self.run_episode(env, policy_dqn_model6_atr, policy_dqn_model6_vtr, use_model=True)
            random_wafer = self.run_episode(env, dummy_atr, dummy_vtr, use_model=False, rule_based_type = 0)
            optimal1_wafer = self.run_episode(env, dummy_atr, dummy_vtr, use_model=False, rule_based_type = 1)
            optimal2_wafer = self.run_episode(env, dummy_atr, dummy_vtr, use_model=False, rule_based_type = 2)
            optimal3_wafer = self.run_episode(env, dummy_atr, dummy_vtr, use_model=False, rule_based_type = 3)
            
            model1_wafers.append(model1_wafer)
            model2_wafers.append(model2_wafer)
            model3_wafers.append(model3_wafer)
            model4_wafers.append(model4_wafer)
            model5_wafers.append(model5_wafer)
            model6_wafers.append(model6_wafer)
            random_wafers.append(random_wafer)
            optimal1_wafers.append(optimal1_wafer)
            optimal2_wafers.append(optimal2_wafer)
            optimal3_wafers.append(optimal3_wafer)

        env.close()
        return model1_wafers, model2_wafers, model3_wafers, model4_wafers, model5_wafers, model6_wafers, random_wafers, optimal1_wafers, optimal2_wafers, optimal3_wafers

    def run_episode(self, env, policy_dqn_atr, policy_dqn_vtr, use_model=True, rule_based_type = 0):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float, device=device)
        
        done = False

        while not done:
            invalid_vtr_actions, invalid_atr_actions = env.unwrapped.invalid_actions()
            
            if use_model:
                with torch.no_grad():
                    vtr_q_values = policy_dqn_vtr(state.unsqueeze(dim=0)).squeeze()
                    atr_q_values = policy_dqn_atr(state.unsqueeze(dim=0)).squeeze()
                    vtr_q_values[invalid_vtr_actions] = float('-inf')
                    atr_q_values[invalid_atr_actions] = float('-inf')
                    vtr_action = vtr_q_values.argmax().item()
                    atr_action = atr_q_values.argmax().item()
            else:
                # valid random
                if rule_based_type == 0:
                    vtr_action = random.choice([a for a in range(env.action_space.spaces[0].n) if a not in invalid_vtr_actions])
                    atr_action = random.choice([a for a in range(env.action_space.spaces[1].n) if a not in invalid_atr_actions])
                # optimal 1
                elif rule_based_type == 1:
                    
                    vtr_action_space_size = env.action_space.spaces[0].n
                    if set(range(1, vtr_action_space_size)) - set(invalid_vtr_actions):
                        invalid_vtr_actions = list(set(invalid_vtr_actions) | {0})

                    atr_action_space_size = env.action_space.spaces[1].n
                    if set(range(1, atr_action_space_size)) - set(invalid_atr_actions):
                        invalid_atr_actions = list(set(invalid_atr_actions) | {0})


                    vtr_action = random.choice([a for a in range(env.action_space.spaces[0].n) if a not in invalid_vtr_actions])
                    atr_action = random.choice([a for a in range(env.action_space.spaces[1].n) if a not in invalid_atr_actions])

                # optimal 2
                elif rule_based_type == 2:
                    # Optimal Policy (로봇 이동시간 고려)
                    vtr_action_space_size = env.action_space.spaces[0].n
                    if set(range(1, vtr_action_space_size)) - set(invalid_vtr_actions):
                        invalid_vtr_actions = list(set(invalid_vtr_actions) | {0})

                    atr_action_space_size = env.action_space.spaces[1].n
                    if set(range(1, atr_action_space_size)) - set(invalid_atr_actions):
                        invalid_atr_actions = list(set(invalid_atr_actions) | {0})

                    valid_vtr_actions = [a for a in range(vtr_action_space_size) if a not in invalid_vtr_actions]
                    valid_atr_actions = [a for a in range(atr_action_space_size) if a not in invalid_atr_actions]
                    
                    vtr_action = env.unwrapped.get_min_exe_time_vtr_action(valid_vtr_actions)
                    atr_action = env.unwrapped.get_min_exe_time_atr_action(valid_atr_actions)
                # Heuristic 3 (Pull-based Priority)
                elif rule_based_type == 3:
                    # 1. H2와 동일하게 NO_ACTION 회피 로직 적용
                    vtr_action_space_size = env.action_space.spaces[0].n
                    if set(range(1, vtr_action_space_size)) - set(invalid_vtr_actions):
                        invalid_vtr_actions = list(set(invalid_vtr_actions) | {0})

                    atr_action_space_size = env.action_space.spaces[1].n
                    if set(range(1, atr_action_space_size)) - set(invalid_atr_actions):
                        invalid_atr_actions = list(set(invalid_atr_actions) | {0})

                    valid_vtr_actions = [a for a in range(vtr_action_space_size) if a not in invalid_vtr_actions]
                    valid_atr_actions = [a for a in range(atr_action_space_size) if a not in invalid_atr_actions]
                    
                    # 2. Heuristic 3 (Pull-based) 우선순위 함수 호출
                    vtr_action = env.unwrapped.get_heuristic_3_vtr_action(valid_vtr_actions)
                    atr_action = env.unwrapped.get_heuristic_3_atr_action(valid_atr_actions)
                #
                # ----------------- ▲▲▲ 추가 완료 ▲▲▲ -----------------
                

            action = (vtr_action, atr_action)
            state, _, done, _, info = env.step(action)
            state = torch.tensor(state, dtype=torch.float, device=device)

        return info['total_wafer_processed']

    def compare_and_plot(self, num_episodes=10):
        model_wafers, model2_wafers, model3_wafers, model4_wafers, model5_wafers, model6_wafers, random_wafers, optimal1_wafers, optimal2_wafers, optimal3_wafers = self.evaluate(num_episodes)

        # Calculate baseline (average of type 0, 1, 2)
        baseline_wafers = (np.mean(random_wafers) + np.mean(optimal1_wafers) + np.mean(optimal2_wafers)) / 3

        print(f"---------- Env 1 ----------")
        print(f"Trained E1 Short duration : {np.mean(model_wafers):.2f}")
        print(f"Trained E1 Long duration : {np.mean(model2_wafers):.2f}") 
        print(f"Trained E2 Short duration : {np.mean(model3_wafers):.2f}")
        print(f"Trained E2 Long duration : {np.mean(model4_wafers):.2f}")
        print(f"Trained E3 Short duration : {np.mean(model5_wafers):.2f}")
        print(f"Trained E3 Long duration : {np.mean(model6_wafers):.2f}")
        print(f"Random Sampling (Type 0) - Mean: {np.mean(random_wafers):.2f}")
        print(f"Optimal 1 (Type 1) - Mean: {np.mean(optimal1_wafers):.2f}")
        print(f"Optimal 2 (Type 2) - Mean: {np.mean(optimal2_wafers):.2f}")
        print(f"Optimal 3 (Pull-based) - Mean: {np.mean(optimal3_wafers):.2f}") # <--- H3 프린트 추가
        print(f"Baseline (Average of Type 0, 1, 2) - Mean: {baseline_wafers:.2f}")
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(14, 8))
        # --- ▼ data에 H3 추가 ▼ ---
        data = [model_wafers, model2_wafers, model3_wafers, model4_wafers, model5_wafers, model6_wafers,
                random_wafers, optimal1_wafers, optimal2_wafers, optimal3_wafers]
        # --- ▼ labels에 H3 추가 ▼ ---
        labels = ['E1_S', 'E1_L', 'E2_S', 'E2_L', 'E3_S', 'E3_L',
                 'Random (Type 0)', 'Optimal1 (Type 1)', 'Optimal2 (Type 2)', 'Optimal3 (Pull)']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        # --- ▼ colors에 H3 추가 ▼ ---
        colors = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightyellow', 'lightyellow',
                 'lightcoral', 'lightpink', 'lightpink', 'lightcyan']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add baseline line
        ax.axhline(y=baseline_wafers, color='green', linestyle='--', linewidth=2, 
                  label=f'Baseline ({baseline_wafers:.2f})', alpha=0.7)
        
        ax.set_ylabel('Total Wafers Processed per Episode')
        ax.set_title('Comparison of DQN Models vs Heuristic Policies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        comparison_file = os.path.join(RUNS_DIR, 'model_comparison.png')
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nComparison plot saved to {comparison_file}")

    def _save_intermediate_results(self, results, best_config, current_combination, total_combinations):
        """
        Save intermediate results after each combination is tested
        """
        if not results:
            return
        
        results_sorted = sorted(results, key=lambda x: x['processed_wafers'], reverse=True)
        
        # Save intermediate results to text file
        results_file = os.path.join(RUNS_DIR, 'hyperparameter_search_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"HYPERPARAMETER GRID SEARCH RESULTS (Progress: {current_combination}/{total_combinations})\n")
            f.write("="*80 + "\n")
            if best_config:
                f.write(f"\nBest Configuration (by processed wafers) so far:\n")
                f.write(f"  Epsilon Decay: {best_config['epsilon_decay']}\n")
                f.write(f"  Reward Scale: {best_config['reward_scale']}\n")
                f.write(f"  Batch Size: {best_config['batch_size']}\n")
                f.write(f"  Processed Wafers: {best_config['processed_wafers']:.2f}\n\n")
            f.write(f"Results so far (sorted by processed wafers):\n")
            for i, config in enumerate(results_sorted):
                f.write(f"Config {chr(65+i)}: eps_decay={config['epsilon_decay']}, reward_scale={config['reward_scale']}, "
                       f"batch_size={config['batch_size']}: processed_wafers={config['processed_wafers']:.2f}\n")
        
        # Save intermediate results to CSV file
        csv_file = os.path.join(RUNS_DIR, 'hyperparameter_sensitivity_analysis.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Config', 'Epsilon Decay', 'Reward Scale', 'Batch Size', 
                           'Best Reward', 'Best Episode', 'Processed Wafers'])
            for i, config in enumerate(results_sorted):
                writer.writerow([
                    f"Config {chr(65+i)}",
                    config['epsilon_decay'],
                    config['reward_scale'],
                    config['batch_size'],
                    f"{config['best_reward']:.2f}",
                    config['best_episode'],
                    f"{config['processed_wafers']:.2f}"
                ])
        
        # Save intermediate results to JSON file
        json_file = os.path.join(RUNS_DIR, 'hyperparameter_sensitivity_analysis.json')
        with open(json_file, 'w') as f:
            json.dump({
                'progress': f'{current_combination}/{total_combinations}',
                'best_config': best_config,
                'all_results': results_sorted,
                'description': 'Hyperparameter sensitivity analysis results. Each config was trained for limited episodes, and best reward parameters were used to evaluate processed wafer count.'
            }, f, indent=2)
        
        print(f"  Intermediate results saved (Progress: {current_combination}/{total_combinations})")
    
    def hyperparameter_grid_search(self, max_episodes=200, validation_episodes=5):
        """
        Grid search for hyperparameter tuning
        Tests different combinations of exploration rate, reward scaling, and batch size
        """
        print("Starting Hyperparameter Grid Search...")
        
        # Define hyperparameter grids
        epsilon_decay_values = [0.99, 0.995, 0.9995]
        reward_scale_values = [0.5, 1.0, 2.0]
        batch_size_values = [32, 64, 128]
        
        results = []
        best_config = None
        best_validation_reward = -9999999  # Used to track best processed_wafers
        
        total_combinations = len(epsilon_decay_values) * len(reward_scale_values) * len(batch_size_values)
        current_combination = 0
        
        for epsilon_decay in epsilon_decay_values:
            for reward_scale in reward_scale_values:
                for batch_size in batch_size_values:
                    current_combination += 1
                    print(f"\n[{current_combination}/{total_combinations}] Testing: epsilon_decay={epsilon_decay}, reward_scale={reward_scale}, batch_size={batch_size}")
                    
                    # Create agent with current hyperparameters
                    agent = Agent(
                        epsilon_decay=epsilon_decay,
                        reward_scale=reward_scale,
                        mini_batch_size=batch_size
                    )
                    
                    # Train for limited episodes
                    env = gym.make(agent.env_id)
                    num_states = env.observation_space.shape[0]
                    num_vtr_actions = env.action_space.spaces[0].n
                    num_atr_actions = env.action_space.spaces[1].n
                    
                    vtr_policy_dqn = DQN(num_states, num_vtr_actions, agent.fc1_nodes).to(device)
                    atr_policy_dqn = DQN(num_states, num_atr_actions, agent.fc1_nodes).to(device)
                    
                    epsilon = agent.epsilon_init
                    memory = ReplayMemory(agent.replay_memory_size)
                    vtr_target_dqn = DQN(num_states, num_vtr_actions, agent.fc1_nodes).to(device)
                    atr_target_dqn = DQN(num_states, num_atr_actions, agent.fc1_nodes).to(device)
                    vtr_target_dqn.load_state_dict(vtr_policy_dqn.state_dict())
                    atr_target_dqn.load_state_dict(atr_policy_dqn.state_dict())
                    
                    vtr_optimizer = torch.optim.Adam(vtr_policy_dqn.parameters(), lr=agent.learning_rate_a)
                    atr_optimizer = torch.optim.Adam(atr_policy_dqn.parameters(), lr=agent.learning_rate_a)
                    
                    training_rewards = []
                    best_reward = -9999999
                    best_episode = 0
                    best_vtr_state = None
                    best_atr_state = None
                    
                    for episode in range(max_episodes):
                        state, _ = env.reset()
                        state = torch.tensor(state, dtype=torch.float, device=device)
                        terminated = False
                        episode_reward = 0
                        step_count = 0
                        
                        while not terminated:
                            invalid_vtr_actions, invalid_atr_actions = env.unwrapped.invalid_actions()
                            
                            if random.random() < epsilon:
                                vtr_action = random.choice([a for a in range(num_vtr_actions) if a not in invalid_vtr_actions])
                                atr_action = random.choice([a for a in range(num_atr_actions) if a not in invalid_atr_actions])
                            else:
                                with torch.no_grad():
                                    vtr_q_values = vtr_policy_dqn(state.unsqueeze(dim=0)).squeeze()
                                    atr_q_values = atr_policy_dqn(state.unsqueeze(dim=0)).squeeze()
                                    vtr_q_values[invalid_vtr_actions] = float('-inf')
                                    atr_q_values[invalid_atr_actions] = float('-inf')
                                    vtr_action = vtr_q_values.argmax().item()
                                    atr_action = atr_q_values.argmax().item()
                            
                            action = (vtr_action, atr_action)
                            new_state, reward, terminated, truncated, info = env.step(action)
                            scaled_reward = reward * agent.reward_scale
                            episode_reward += scaled_reward
                            
                            new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                            scaled_reward_tensor = torch.tensor(scaled_reward, dtype=torch.float, device=device)
                            
                            memory.append((state, vtr_action, atr_action, new_state, scaled_reward_tensor, terminated))
                            step_count += 1
                            state = new_state
                        
                        training_rewards.append(episode_reward)
                        epsilon = max(epsilon * epsilon_decay, agent.epsilon_min)
                        
                        # Print episode progress
                        print(f"    Episode {episode + 1}/{max_episodes} - Reward: {episode_reward:.2f}, Epsilon: {epsilon:.4f}")
                        
                        # Train after episode ends (same as run function)
                        if len(memory) > agent.mini_batch_size:
                            mini_batch = memory.sample(agent.mini_batch_size)
                            states, vtr_actions, atr_actions, new_states, rewards, terminations = zip(*mini_batch)
                            
                            states = torch.stack(states)
                            vtr_actions = torch.tensor(vtr_actions, dtype=torch.long, device=device)
                            atr_actions = torch.tensor(atr_actions, dtype=torch.long, device=device)
                            new_states = torch.stack(new_states)
                            rewards = torch.stack(rewards)
                            terminations = torch.tensor(terminations, dtype=torch.float, device=device)
                            
                            with torch.no_grad():
                                vtr_target_q = rewards + (1-terminations) * agent.discount_factor_g * vtr_target_dqn(new_states).max(dim=1)[0]
                                atr_target_q = rewards + (1-terminations) * agent.discount_factor_g * atr_target_dqn(new_states).max(dim=1)[0]
                            
                            vtr_current_q = vtr_policy_dqn(states).gather(1, vtr_actions.unsqueeze(1)).squeeze(1)
                            atr_current_q = atr_policy_dqn(states).gather(1, atr_actions.unsqueeze(1)).squeeze(1)
                            
                            vtr_loss = agent.loss_fn(vtr_current_q, vtr_target_q)
                            atr_loss = agent.loss_fn(atr_current_q, atr_target_q)
                            
                            vtr_optimizer.zero_grad()
                            atr_optimizer.zero_grad()
                            vtr_loss.backward()
                            atr_loss.backward()
                            vtr_optimizer.step()
                            atr_optimizer.step()
                            
                            if step_count > agent.network_sync_rate:
                                vtr_target_dqn.load_state_dict(vtr_policy_dqn.state_dict())
                                atr_target_dqn.load_state_dict(atr_policy_dqn.state_dict())
                                step_count = 0
                        
                        # Track best reward and save model state
                        if episode_reward > best_reward:
                            best_reward = episode_reward
                            best_episode = episode
                            best_vtr_state = vtr_policy_dqn.state_dict().copy()
                            best_atr_state = atr_policy_dqn.state_dict().copy()
                    
                    avg_training_reward = np.mean(training_rewards[-50:]) if len(training_rewards) >= 50 else np.mean(training_rewards)
                    
                    # Load best reward model and evaluate processed wafer (validation only at the end)
                    vtr_policy_dqn.load_state_dict(best_vtr_state)
                    atr_policy_dqn.load_state_dict(best_atr_state)
                    processed_wafers = agent._evaluate_validation(env, vtr_policy_dqn, atr_policy_dqn, num_episodes=validation_episodes)
                    
                    config = {
                        'epsilon_decay': epsilon_decay,
                        'reward_scale': reward_scale,
                        'batch_size': batch_size,
                        'avg_training_reward': avg_training_reward,
                        'best_reward': best_reward,
                        'best_episode': best_episode,
                        'processed_wafers': processed_wafers
                    }
                    results.append(config)
                    
                    print(f"  Avg Training Reward: {avg_training_reward:.2f}")
                    print(f"  Best Reward: {best_reward:.2f} at episode {best_episode}, Processed Wafers: {processed_wafers:.2f}")
                    
                    if processed_wafers > best_validation_reward:
                        best_validation_reward = processed_wafers
                        best_config = config
                    
                    env.close()
                    
                    # Save intermediate results after each combination
                    self._save_intermediate_results(results, best_config, current_combination, total_combinations)
        
        # Print results
        print("\n" + "="*80)
        print("HYPERPARAMETER GRID SEARCH RESULTS")
        print("="*80)
        print(f"\nBest Configuration (by processed wafers):")
        print(f"  Epsilon Decay: {best_config['epsilon_decay']}")
        print(f"  Reward Scale: {best_config['reward_scale']}")
        print(f"  Batch Size: {best_config['batch_size']}")
        print(f"  Processed Wafers: {best_config['processed_wafers']:.2f}")
        
        print(f"\nAll Results (sorted by processed wafers):")
        results_sorted = sorted(results, key=lambda x: x['processed_wafers'], reverse=True)
        for i, config in enumerate(results_sorted):
            print(f"{i+1}. Config {chr(65+i)}: eps_decay={config['epsilon_decay']}, reward_scale={config['reward_scale']}, "
                  f"batch_size={config['batch_size']}: processed_wafers={config['processed_wafers']:.2f}")
        
        # Save results to text file
        results_file = os.path.join(RUNS_DIR, 'hyperparameter_search_results.txt')
        with open(results_file, 'w') as f:
            f.write("HYPERPARAMETER GRID SEARCH RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"\nBest Configuration (by processed wafers):\n")
            f.write(f"  Epsilon Decay: {best_config['epsilon_decay']}\n")
            f.write(f"  Reward Scale: {best_config['reward_scale']}\n")
            f.write(f"  Batch Size: {best_config['batch_size']}\n")
            f.write(f"  Processed Wafers: {best_config['processed_wafers']:.2f}\n\n")
            f.write("All Results (sorted by processed wafers):\n")
            for i, config in enumerate(results_sorted):
                f.write(f"Config {chr(65+i)}: eps_decay={config['epsilon_decay']}, reward_scale={config['reward_scale']}, "
                       f"batch_size={config['batch_size']}: processed_wafers={config['processed_wafers']:.2f}\n")
        
        # Save results to CSV file (table format for paper)
        csv_file = os.path.join(RUNS_DIR, 'hyperparameter_sensitivity_analysis.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Config', 'Epsilon Decay', 'Reward Scale', 'Batch Size', 
                           'Best Reward', 'Best Episode', 'Processed Wafers'])
            for i, config in enumerate(results_sorted):
                writer.writerow([
                    f"Config {chr(65+i)}",  # A, B, C, ...
                    config['epsilon_decay'],
                    config['reward_scale'],
                    config['batch_size'],
                    f"{config['best_reward']:.2f}",
                    config['best_episode'],
                    f"{config['processed_wafers']:.2f}"
                ])
        
        # Save results to JSON file
        json_file = os.path.join(RUNS_DIR, 'hyperparameter_sensitivity_analysis.json')
        with open(json_file, 'w') as f:
            json.dump({
                'best_config': best_config,
                'all_results': results_sorted,
                'description': 'Hyperparameter sensitivity analysis results. Each config was trained for limited episodes, and best reward parameters were used to evaluate processed wafer count.'
            }, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  - {results_file}")
        print(f"  - {csv_file} (table format for paper)")
        print(f"  - {json_file}")
        
        return best_config, results

        

if __name__ == '__main__':
    dql = Agent()
    #dql.run(is_training=True)
    #dql.run(trained_model_file='eqp-scheduler-v0', is_training=False, render=True)
    dql.compare_and_plot(num_episodes=1)
    #dql.hyperparameter_grid_search(max_episodes=1000, validation_episodes=1)