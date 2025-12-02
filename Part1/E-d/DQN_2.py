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
import seaborn as sns

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
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.output(x)

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Deep Q-Learning Agent
class Agent():

    def __init__(self):
       
        # Hyperparameters (adjustable)
        self.env_id             = 'eqp-scheduler-v0'
        self.learning_rate_a    = 0.0001                                    # learning rate (alpha)
        self.discount_factor_g  = 0.99                                      # discount rate (gamma)
        self.network_sync_rate  = 10                                        # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = 100000                                    # size of replay memory
        self.mini_batch_size    = 32                                        # size of the training data set sampled from the replay memory
        self.epsilon_init       = 1                                         # 1 = 100% random actions
        self.epsilon_decay      = 0.99                                      # epsilon decay rate
        self.epsilon_min        = 0.05                                      # minimum epsilon value
        self.stop_on_reward     = 100000                                    # stop training after reaching this number of rewards
        self.fc1_nodes          = 512

        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.

        # Get current date and time
        current_time = datetime.now().strftime('%Y%m%d-%H%M')

        # Update env_id with date and time
        self.file_name = f'{current_time}-{self.env_id}'
        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.file_name}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.file_name}.pt')
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

        # Number of possible actions
        num_actions = env.action_space.n

        # Get observation space size
        num_states = env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)

        # List to keep track of rewards collected per episode.
        rewards_per_episode = []

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            # Initialize epsilon
            epsilon = self.epsilon_init

            # Initialize replay memory
            memory = ReplayMemory(self.replay_memory_size)

            # Create the target network and make it identical to the policy network
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Policy network optimizer. "Adam" optimizer can be swapped to something else.
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            # List to keep track of epsilon decay
            epsilon_history = []

            # Track number of steps taken. Used for syncing policy => target network.
            step_count=0

            # Track best reward
            best_reward = -9999999
        else:
            # Load learned policy
            policy_dqn.load_state_dict(torch.load(os.path.join(RUNS_DIR, f'{trained_model_file}.pt')))

            # switch model to evaluation mode
            policy_dqn.eval()

        avg_wait_times_per_episode = []

        # Train INDEFINITELY, manually stop the run when you are satisfied (or unsatisfied) with the results
        for episode in itertools.count():

            state, _ = env.reset()  # Initialize environment. Reset returns (state,info).
            state = torch.tensor(state, dtype=torch.float, device=device) # Convert state to tensor directly on device

            terminated = False      # True when agent reaches goal or fails
            episode_reward = 0.0    # Used to accumulate rewards per episode

            # Perform actions until episode terminates or reaches max rewards
            # (on some envs, it is possible for the agent to train to a point where it NEVER terminates, so stop on reward is necessary)
            while(not terminated and episode_reward < self.stop_on_reward):
                invalid_actions = env.invalid_actions()
                valid_actions = [a for a in range(num_actions) if a not in invalid_actions]

                # Select action based on epsilon-greedy
                if is_training and random.random() < epsilon:
                    # select random action
                    #action = env.action_space.sample()
                    action = random.choice(valid_actions)
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # select best action
                    with torch.no_grad():
                        # state.unsqueeze(dim=0): Pytorch expects a batch layer, so add batch dimension i.e. tensor([1, 2, 3]) unsqueezes to tensor([[1, 2, 3]])
                        # policy_dqn returns tensor([[1], [2], [3]]), so squeeze it to tensor([1, 2, 3]).
                        # argmax finds the index of the largest element.
                        #action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                        q_values = policy_dqn(state.unsqueeze(dim=0)).squeeze()
                        q_values[invalid_actions] = float('-inf')
                        action = q_values.argmax()

                # Execute action. Truncated and info is not used.
                new_state,reward,terminated,truncated,info = env.step(action.item())

                # Accumulate rewards
                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # Save experience into memory
                    memory.append((state, action, new_state, reward, terminated))

                    # Increment step counter
                    step_count+=1

                # Move to the next state
                state = new_state

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)
            avg_wait_times_per_episode.append(info['avg_processed_wait_time'])
            info_formatted = {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in info.items()}
            print(f"Episode {episode}, Reward: {episode_reward:.3f}, info: {info_formatted}, Epsilon: {epsilon:.3f}")


            # Save model when new best reward is obtained.
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward


                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=3):
                    self.save_graph(rewards_per_episode, epsilon_history, avg_wait_times_per_episode)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory)>self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0


    def save_graph(self, rewards_per_episode, epsilon_history, avg_wait_times_per_episode):
        # Save plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        ax1.plot(mean_rewards)
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Mean Rewards')
        ax1.set_title('Average Rewards')

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        ax2.plot(epsilon_history)
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Epsilon')
        ax2.set_title('Epsilon Decay')

        # 평균 대기 시간 그래프
        ax3.plot(avg_wait_times_per_episode)
        ax3.set_xlabel('Episodes')
        ax3.set_ylabel('Avg Wait Time')
        ax3.set_title('Average Wait Time')

        plt.tight_layout()

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q values (expected returns)
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
            '''
                target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                    .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                        [0]             ==> tensor([3,6])
            '''

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases


    def evaluate(self, num_episodes=10, render=False):
        env = gym.make(self.env_id, render_mode='human' if render else None)
        num_actions = env.action_space.n

        model1_path = './환경A_0.99_313/eqp-scheduler-v0.pt'
        model2_path = './환경B_0.99_384/eqp-scheduler-v0.pt'
        model3_path = './환경C_0.99_290/eqp-scheduler-v0.pt'
        model4_path = './환경E_0.99_162/eqp-scheduler-v0.pt'
        
        # �н��� �� �ε�
        policy_dqn_model1 = DQN(env.observation_space.shape[0], num_actions, self.fc1_nodes).to(device)
        policy_dqn_model1.load_state_dict(torch.load(model1_path))
        policy_dqn_model1.eval()

        policy_dqn_model2 = DQN(env.observation_space.shape[0], num_actions, self.fc1_nodes).to(device)
        policy_dqn_model2.load_state_dict(torch.load(model2_path))
        policy_dqn_model2.eval()

        policy_dqn_model3 = DQN(env.observation_space.shape[0], num_actions, self.fc1_nodes).to(device)
        policy_dqn_model3.load_state_dict(torch.load(model3_path))
        policy_dqn_model3.eval()

        policy_dqn_model4 = DQN(env.observation_space.shape[0], num_actions, self.fc1_nodes).to(device)
        policy_dqn_model4.load_state_dict(torch.load(model4_path))
        policy_dqn_model4.eval()

        policy_dqn_model5 = DQN(env.observation_space.shape[0], num_actions, self.fc1_nodes).to(device)
        policy_dqn_model5.eval()

        model1_wafers = []
        model2_wafers = []
        model3_wafers = []
        model4_wafers = []        
        random_wafers = []
        rule1_wafers = []
        
        for _ in range(num_episodes):
            model1_wafer = self.run_episode(env, policy_dqn = policy_dqn_model1, use_model=True)
            model2_wafer = self.run_episode(env, policy_dqn = policy_dqn_model2, use_model=True)
            model3_wafer = self.run_episode(env, policy_dqn = policy_dqn_model3, use_model=True)
            model4_wafer = self.run_episode(env, policy_dqn = policy_dqn_model4, use_model=True)
            random_wafer = self.run_episode(env, policy_dqn = policy_dqn_model5, use_model=False)
            rule1_wafer = self.run_episode(env, policy_dqn = policy_dqn_model5, use_model=False, rule_based_type=1)

            model1_wafers.append(model1_wafer)
            model2_wafers.append(model2_wafer)
            model3_wafers.append(model3_wafer)
            model4_wafers.append(model4_wafer)
            random_wafers.append(random_wafer)
            rule1_wafers.append(rule1_wafer)

        env.close()
        return model1_wafers, model2_wafers, model3_wafers, model4_wafers, random_wafers, rule1_wafers

    def run_episode(self, env, policy_dqn, use_model=True, rule_based_type=0):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float, device=device)
        
        done = False

        while not done:
            invalid_actions = env.invalid_actions()
            valid_actions = [a for a in range(env.action_space.n) if a not in invalid_actions]

            if use_model:
                with torch.no_grad():
                    q_values = policy_dqn(state.unsqueeze(dim=0)).squeeze()
                    q_values[invalid_actions] = float('-inf')
                    action = q_values.argmax().item()
            else:
                if rule_based_type ==0:
                    action = random.choice(valid_actions)
                else:
                    vtr_action_space_size = env.action_space.n
                    if set(range(1, vtr_action_space_size)) - set(invalid_actions):
                        invalid_actions = list(set(invalid_actions) | {0})

                    action = random.choice([a for a in range(env.action_space.n) if a not in invalid_actions])
                    

            state, _, done, _, info = env.step(action)
            state = torch.tensor(state, dtype=torch.float, device=device)

        return info['total_wafer_processed']

    def compare_and_plot(self, num_episodes=10):
        model1_wafers, model2_wafers, model3_wafers, model4_wafers, random_wafers, rule1_wafers = self.evaluate(num_episodes)

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[model1_wafers, model2_wafers, model3_wafers, model4_wafers, random_wafers, rule1_wafers], width=0.6)
        plt.xticks([0, 1, 2, 3, 4, 5], ['Model (Env A)', 'Model (Env B)', 'Model (Env C)', 'Model (Env E)', 'Random' ,'Rule1'])
        plt.ylabel('Total Wafers Processed per Episode')
        plt.savefig('model_vs_random_comparison.png')
        plt.close()

        print(f"Model (Env A) w/ 0.99 - Mean: {np.mean(model1_wafers):.2f}, Std: {np.std(model1_wafers):.2f}")
        print(f"Model (Env B) w/ 0.99 - Mean: {np.mean(model2_wafers):.2f}, Std: {np.std(model2_wafers):.2f}")
        print(f"Model (Env C) w/ 0.99 - Mean: {np.mean(model3_wafers):.2f}, Std: {np.std(model3_wafers):.2f}")
        print(f"Model (Env E) w/ 0.99 - Mean: {np.mean(model4_wafers):.2f}, Std: {np.std(model4_wafers):.2f}")
        print(f"Random Sampling - Mean: {np.mean(random_wafers):.2f}, Std: {np.std(random_wafers):.2f}")
        print(f"Rule1 wafers - Mean: {np.mean(rule1_wafers):.2f}, Std: {np.std(rule1_wafers):.2f}")

if __name__ == '__main__':
    dql = Agent()
    #dql.run(is_training=True)
    dql.run(trained_model_file='20240802-1218-eqp-scheduler-v0', is_training=False, render=True)
    #dql.compare_and_plot(num_episodes=10)