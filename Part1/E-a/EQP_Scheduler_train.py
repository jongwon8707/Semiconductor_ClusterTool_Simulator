import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from stable_baselines3 import A2C, PPO, DQN
import os
import EQP_Scheduler_env
import shutil
from stable_baselines3.common.callbacks import BaseCallback
import time
from datetime import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import logging

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='환경3.log')


def mask_fn(env):
    mask = np.ones(env.action_space.n, dtype=bool)
    invalid_actions = env.invalid_actions()
    mask[invalid_actions] = False
    return mask

# Train using StableBaseline3. Lots of hardcoding for simplicity i.e. use of the A2C (Advantage Actor Critic) algorithm.
def train_sb3(algo=0):
    # Where to store trained model and logs
    model_dir = "models"
    log_dir = "logs"
    model_name = "A2C"
    delete_and_recreate_folder(log_dir)
    delete_and_recreate_folder(model_dir)

    env = gym.make('eqp-scheduler-v0')

    # Use Advantage Actor Critic (A2C) algorithm.
    # Use MlpPolicy for observation space 1D vector.
    if algo == 0:
        print('A2C Start~!')
        model = A2C('MlpPolicy', env, verbose=0, device='cuda', tensorboard_log=log_dir)       #학습됨
    elif algo == 1:
        print('PPO Start~!')
        model_name = "PPO"
        model = PPO('MlpPolicy', env, verbose=0, device='cuda', tensorboard_log=log_dir)       #학습됨
    elif algo == 2:
        print('MaskPPO Start~!')
        env = ActionMasker(env, mask_fn)
        model_name = "MaskPPO"
        model = MaskablePPO('MlpPolicy', env, verbose=0, device='cuda', tensorboard_log=log_dir)
    else:
        print('DQN Start~!')
        model_name = "DQN"
        model = DQN(
                    policy="MlpPolicy",
                    env=env,
                    learning_rate=6.3e-4,
                    batch_size=128,
                    buffer_size=50000,
                    learning_starts=0,
                    gamma=0.99,
                    target_update_interval=250,
                    train_freq=4,
                    gradient_steps=-1,
                    exploration_fraction=0.12,
                    exploration_final_eps=0.1,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    verbose=0,
                    device="cuda",  # GPU 사용 설정, GPU가 없는 경우 "cpu"로 변경
                    tensorboard_log=log_dir  # TensorBoard 로그 경로
                )
    
    # This loop will keep training until you stop it with Ctr-C.
    # Start another cmd prompt and launch Tensorboard: tensorboard --logdir logs
    # Once Tensorboard is loaded, it will print a URL. Follow the URL to see the status of the training.
    # Stop the training when you're satisfied with the status.
    TIMESTEPS = 10000
    iters = 100
        
    # Format the current date and time as a string 'YearMonthDay-HourMinuteSecond', e.g., '20230401-153042'
    for i in range(iters):
        
        #Train
        current_datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        model.learn(total_timesteps=TIMESTEPS, 
                    reset_num_timesteps=False, 
                    tb_log_name=current_datetime_str,
                    ) # train
        model.save(f"{model_dir}/{model_name}_{TIMESTEPS*(i+1)}") # Save a trained model every TIMESTEPS

        #Test per 10000 time steps
        for episode in range(10):
            test_sb3(False, algo, i+1, TIMESTEPS, episode+1, model_name)



# Test using StableBaseline3. Lots of hardcoding for simplicity.
def test_sb3(render=True, algo=0, iters = 0, n_timesteps = 0, episode = 0, model_name=""):

    env = gym.make('eqp-scheduler-v0', render_mode='human' if render else None)

    # Load model
    if algo == 0:
        model = A2C.load(f'models/A2C_{n_timesteps*iters}', env=env)
    elif algo == 1:
        model = PPO.load(f'models/PPO_{n_timesteps*iters}', env=env)
    elif algo == 2:
        env = ActionMasker(env, mask_fn)
        model = MaskablePPO.load(f'models/MaskPPO_{n_timesteps*iters}', env=env)
    else:
        model = DQN.load(f'models/DQN_{n_timesteps*iters}', env=env)

    # Run a test
    obs = env.reset()[0]
    terminated = False
    while True:
        if algo == 2:
            action, _ = model.predict(observation=obs, action_masks=mask_fn(env))
        else:
            action, _ = model.predict(observation=obs, deterministic=True) # Turn on deterministic, so predict always returns the same behavior

        obs, _, terminated, _, info = env.step(action)

        if terminated:
            if 'log_message' in info:
                logging.info(f'model : {model_name}, episode : {episode}, timestep : {iters * n_timesteps}, {info["log_message"]}')
                print(f'model : {model_name}, episode : {episode}, timestep : {iters * n_timesteps}, {info["log_message"]}')
            break

# Test using StableBaseline3. Lots of hardcoding for simplicity.
def real_test_sb3(render=True, n_timesteps = 0):

    env = gym.make('eqp-scheduler-v0', render_mode='human' if render else None)

  
    model = DQN.load(f'models/DQN_{n_timesteps}', env=env)

    # Run a test
    obs = env.reset()[0]
    terminated = False
    while True:
        action, _ = model.predict(observation=obs, deterministic=True) # Turn on deterministic, so predict always returns the same behavior
        obs, _, terminated, _, info = env.step(action)


def delete_and_recreate_folder(folder_path):
    # 폴더가 존재하면 삭제
    # if os.path.exists(folder_path):
    #     shutil.rmtree(folder_path)
    # 폴더 재생성
    if os.path.exists(folder_path):
        pass
    else:
        os.makedirs(folder_path)

# 'logs'와 'models' 폴더 내부를 삭제하고 다시 생성

if __name__ == '__main__':
    # Train/test using Q-Learning
    #run_q(4000, is_training=True, render=False)
    #run_q(1, is_training=False, render=True)

    # Train/test using StableBaseline3
    # for i in range(4):
        # train_sb3(algo=3)
    train_sb3(algo=3)
    #real_test_sb3(True, 100000)