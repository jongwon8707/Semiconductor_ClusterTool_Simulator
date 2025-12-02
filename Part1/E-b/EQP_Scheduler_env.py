import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import numpy as np
import EQP_Scheduler as scd
import logging

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='eqp-scheduler-v0',                                   # call it whatever you want
    entry_point='EQP_Scheduler_env:EQP_Scheduler_Env',       # module_name:class_name
)

MAX_STEP_LIMIT = 1000
MAX_WAFER_NO = 1000
MAX_CHAMBER_COUNT = 6
class EQP_Scheduler_Env(gym.Env):

    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.scd = scd.EQP_Scheduler(MAX_WAFER_NO)
        self.action_space = spaces.Discrete(len(scd.VTRAction))

        self.observation_space = spaces.Box(
            low=0,
            high=np.array([
                1,  # Entry wafer present
                1,  # Exit wafer present
                1,  # Left arm wafer present
                1,  # Right arm wafer present
                *[1 for _ in range(MAX_CHAMBER_COUNT)],  # Chamber wafers present (6 chambers)
                1,  # Entry wafer process state
                1,  # Exit wafer process state
                1,  # Left arm wafer process state
                1,  # Right arm wafer process state
                *[1 for _ in range(MAX_CHAMBER_COUNT)],  # Chamber wafers process states (6 chambers)
                1,  # Left arm available
                1,  # Right arm available
                *[1 for _ in range(MAX_CHAMBER_COUNT)],  # Chamber process times (6 chambers)
            ]),
            dtype=np.int32
        )

        self.reset_cnt = 0
        self.cumulative_reward = 0
        self.step_cnt = 0
        self.total_wafer_processed = 0
        self.no_action_count = 0

        logging.basicConfig(filename='6cb_2vtr_A2C_PPO_MPPO_DQN_2MSTEP_100WF.log', level=logging.INFO)

        
    def get_observation(self):
        observation = []
        # Wafer presence (0 or 1)
        observation.append(1 if self.scd.components[scd.ComponentType.ENTRY].holding_wafer else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.EXIT].holding_wafer else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.LEFT_ARM].holding_wafer else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.RIGHT_ARM].holding_wafer else 0)
        observation.extend([1 if chamber.holding_wafer else 0 for chamber in self.scd.components[scd.ComponentType.CHAMBER]])

        # Wafer process states (0 or 1)
        observation.append(1 if self.scd.components[scd.ComponentType.ENTRY].holding_wafer and self.scd.components[scd.ComponentType.ENTRY].holding_wafer.process_state == scd.WaferProcessState.PROCESSED else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.EXIT].holding_wafer and self.scd.components[scd.ComponentType.EXIT].holding_wafer.process_state == scd.WaferProcessState.PROCESSED else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.LEFT_ARM].holding_wafer and self.scd.components[scd.ComponentType.LEFT_ARM].holding_wafer.process_state == scd.WaferProcessState.PROCESSED else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.RIGHT_ARM].holding_wafer and self.scd.components[scd.ComponentType.RIGHT_ARM].holding_wafer.process_state == scd.WaferProcessState.PROCESSED else 0)
        observation.extend([1 if chamber.holding_wafer and chamber.holding_wafer.process_state == scd.WaferProcessState.PROCESSED else 0 for chamber in self.scd.components[scd.ComponentType.CHAMBER]])

        # Availability (0=N/A or 1)
        observation.append(1 if self.scd.components[scd.ComponentType.LEFT_ARM].process_time == 0 else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.RIGHT_ARM].process_time == 0 else 0)
        observation.extend([1 if chamber.process_time == 0 else 0 for chamber in self.scd.components[scd.ComponentType.CHAMBER]])

        return np.array(observation)
    
    def invalid_actions(self):
        # 마스킹된 액션 리스트를 반환
        invalid_actions = []
        for action in range(self.action_space.n):
            if self.scd.is_impossible_action(scd.VTRAction(action)):
                invalid_actions.append(action)
        return invalid_actions

    def step(self, action):

        # Get valid actions
        invalid_actions = self.invalid_actions()
        valid_actions = [a for a in range(self.action_space.n) if a not in invalid_actions]
        # Perform action
        action_result, wafer_processed, busy_chamber_cnt = self.scd.perform_action(scd.VTRAction(action))
        # if(busy_chamber_cnt == MAX_CHAMBER_COUNT):
        #     print("All Chambers are busy!!")
            
        # Determine reward and termination
        reward = 0  # 각 스텝별로 새로운 보상을 설정
        terminated = False

        # if action_result == -1:  # Impossible action
        #     reward = -100
        #     terminated = True
        # elif action_result == 0 and len(valid_actions) > 1:  # No action but other valid actions are available
        #     #self.no_action_count += 1
        #     reward = -5
        #     #reward = -1 
        # elif action_result == 0 and len(valid_actions) == 1:
        #     #self.no_action_count = 0
        #     reward = -1
        # else:
        #     #self.no_action_count = 0
        #     reward = 5  # Right Action
        
        #reward = -1
        if action_result == 1:
            reward = 0.1
        else:
            reward = -10
        
        if wafer_processed:
            self.total_wafer_processed += 1
            reward = 1
            if self.total_wafer_processed == MAX_WAFER_NO:
                reward = 100
                terminated = True

        obs = self.get_observation()
        
        info = {}

        self.cumulative_reward += reward
        self.step_cnt += 1
        #reward -= self.step_cnt

        if self.step_cnt > MAX_STEP_LIMIT:
            terminated = True

        if self.render_mode == 'human':
            self.render()

        log_message = (
            f"__STEPCNT : {self.step_cnt} "
            f"__WF_PROCESSED : {self.total_wafer_processed} "
            f"__TOTAL_REWARDS : {self.cumulative_reward}"
        )
        info = {"log_message": log_message, "total_wafer_processed": self.total_wafer_processed}
        return obs, reward, terminated, False, info 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # if self.reset_cnt > 0:
        #     log_message = (
        #         f"EPISODE : {self.reset_cnt} "
        #         f"__STEPCNT : {self.step_cnt} "
        #         f"__WF_PROCESSED : {self.total_wafer_processed} "
        #         f"__TOTAL_REWARDS : {self.cumulative_reward}"
        #     )
        #     logging.info(log_message)
            
        self.reset_cnt += 1

        self.cumulative_reward = 0
        self.step_cnt = 0
        self.total_wafer_processed = 0
        self.no_action_count = 0
        self.scd.reset(seed=seed)

        obs = self.get_observation()
        
        info = {}

        # Render environment
        if self.render_mode == 'human':
            self.render()

        # Return observation and info
        return obs, info

    def render(self):
        self.scd.render(self.cumulative_reward)

    
# For unit testing
if __name__=="__main__":
    env = gym.make('eqp-scheduler-v0', render_mode='human')

    # Use this to check our custom environment
    print("Check environment begin")
    check_env(env.unwrapped)
    print("Check environment end")

    # Reset environment
    obs = env.reset()[0]

    # Take some random actions
    for i in range(10000):
        rand_action = env.action_space.sample()  # 유효한 액션 샘플링
        obs, reward, terminated, _, _ = env.step(rand_action)
