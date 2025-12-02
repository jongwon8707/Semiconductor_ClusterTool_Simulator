import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import EQP_Scheduler_env

# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class SCD_DQN():
    # Hyperparameters
    learning_rate_a = 0.0001  # 학습률 감소
    discount_factor_g = 0.99
    network_sync_rate = 1000  # 타겟 네트워크 업데이트 주기 증가
    replay_memory_size = 100000  # 메모리 크기 증가
    mini_batch_size = 128  # 배치 크기 증가
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995  # 엡실론 감소 속도 조절

    # Neural Network
    loss_fn = nn.MSELoss()
    optimizer = None

    def train(self, episodes, render=False):
        # Check for GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create EQP_Scheduler environment
        env = gym.make('eqp-scheduler-v0', render_mode='human' if render else None)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        epsilon = self.epsilon_start
        total_steps = 0
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN(in_states=num_states, h1_nodes=128, out_actions=num_actions).to(device)
        target_dqn = DQN(in_states=num_states, h1_nodes=128, out_actions=num_actions).to(device)

        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []
        step_count = 0
            
        for i in range(episodes):
            state, _ = env.reset()
            state = torch.FloatTensor(state).to(device)
            terminated = False
            truncated = False
            episode_reward = 0

            while not (terminated or truncated):
                invalid_actions = env.invalid_actions()
                valid_actions = [a for a in range(num_actions) if a not in invalid_actions]

                if random.random() < epsilon:
                    action = random.choice(valid_actions)
                else:
                    with torch.no_grad():
                        q_values = policy_dqn(state)
                        q_values[invalid_actions] = float('-inf')
                        action = q_values.argmax().item()

                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = torch.FloatTensor(next_state).to(device)
                reward = torch.FloatTensor([reward]).to(device)
                terminated = torch.FloatTensor([terminated]).to(device)

                # 보상 스케일링
                reward = reward / 100.0  # 보상 스케일 조정

                memory.append((state, action, next_state, reward, terminated))
                state = next_state
                episode_reward += reward
                total_steps += 1

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn, device)

                if total_steps % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            rewards_per_episode[i] = episode_reward
            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)

            #if (i + 1) % 10 == 0:
            print(f"Episode {i+1}/{episodes}, Reward: {episode_reward}, info: {info}, Epsilon: {epsilon:.2f}")


        env.close()

        torch.save(policy_dqn.state_dict(), "eqp_scheduler_dqn.pt")

        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot(rewards_per_episode)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(122)
        plt.plot(epsilon_history)
        plt.title('Epsilon Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        plt.savefig('eqp_scheduler_dqn_training.png')

    def optimize(self, mini_batch, policy_dqn, target_dqn, device):
        state_batch = torch.stack([t[0] for t in mini_batch]).to(device)
        action_batch = torch.LongTensor([t[1] for t in mini_batch]).unsqueeze(1).to(device)
        next_state_batch = torch.stack([t[2] for t in mini_batch]).to(device)
        reward_batch = torch.cat([t[3] for t in mini_batch]).to(device)
        done_batch = torch.cat([t[4] for t in mini_batch]).to(device)

        # Double DQN
        next_state_values = target_dqn(next_state_batch).gather(1, policy_dqn(next_state_batch).argmax(1, keepdim=True)).squeeze(1)
        expected_q_values = reward_batch + self.discount_factor_g * next_state_values * (1 - done_batch)

        q_values = policy_dqn(state_batch).gather(1, action_batch)

        loss = self.loss_fn(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=1)
        self.optimizer.step()

    def test(self, episodes, render=True):
        # Check for GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        env = gym.make('eqp-scheduler-v0', render_mode='human' if render else None)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(in_states=num_states, h1_nodes=128, out_actions=num_actions).to(device)
        policy_dqn.load_state_dict(torch.load("eqp_scheduler_dqn.pt"))
        policy_dqn.eval()

        for i in range(episodes):
            state, _ = env.reset()
            state = torch.FloatTensor(state).to(device)
            terminated = False
            truncated = False
            total_reward = 0

            while not (terminated or truncated):
                invalid_actions = env.invalid_actions()
                with torch.no_grad():
                    q_values = policy_dqn(state)
                    q_values[invalid_actions] = float('-inf')
                    action = q_values.argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = torch.FloatTensor(next_state).to(device)
                total_reward += reward

                state = next_state

            print(f"Test Episode {i+1}: Total Reward = {total_reward}")

        env.close()

if __name__ == '__main__':
    scd_dqn = SCD_DQN()
    scd_dqn.train(episodes=1000, render=False)
    scd_dqn.test(episodes=10, render=True)