import gymnasium as gym
from envs import MultiAgentEnv
import random
import numpy as np

from envs.tiger.rewards import OPEN_TIGER_REWARD, OPEN_MONEY_REWARD, LISTEN_REWARD
from envs.tiger.translator import Translator


# Observation space: {0, 1} x {0, 1}
NULL_OBS = np.array([0, 0])
LEFT_OBS = np.array([1, 0])
RIGHT_OBS = np.array([0, 1])
NOISE_OBS = np.array([1, 1])

# Action space: {0, 1, 2, 3}
OPEN_LEFT = 0
OPEN_RIGHT = 1
LISTEN_LEFT = 2
LISTEN_RIGHT = 3

class DecTiger(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):

        # Open left, open right, listen_left, listen_right
        self.observation_space = gym.spaces.MultiDiscrete(
            [2, 2])  # {0, 1} x {0, 1}

        # Open left, open right, listen_left, listen_right
        self.action_space = gym.spaces.Discrete(4)  # {0, 1, 2 ,3}

        # Initialize state and other variables
        self.state = NULL_OBS
        self._agent_ids = ['agent_0', 'agent_1']

        self.current_observation = [NULL_OBS, NULL_OBS]
        self.translator = Translator()
        super().__init__()
        self.n_agents = 2
        self.episode_limit = 6

    def reset(self):
        self.current_step = 0

        self.state = random.choice([LEFT_OBS, RIGHT_OBS])

        # Observations for each agent (initially, both agents have the same observation)
        self.current_observation = [NULL_OBS, NULL_OBS]
        return self.get_obs(), self.get_state()

    def step(self, actions):
        self.current_step += 1

        observations, reward= [], 0.

        done =  OPEN_LEFT in actions or OPEN_RIGHT in actions or self.current_step == self.episode_limit

        for i, agent_id in enumerate(self._agent_ids):
            # get the reward for the agent's action
            reward += self._agent_reward(agent_id, actions[i])
            # get the observation for the agent's action
            observations.append(self._agent_obs(agent_id, actions[i]))

        #convert this back to a list
        
        self.current_observation = observations

        return reward, done, {}

    def render(self):
        pass

    def _agent_reward(self, agent_id, action):
        if self.translator.is_open_left(action):
            if self._is_tiger_left():
                return OPEN_TIGER_REWARD
            else:
                return OPEN_MONEY_REWARD
            
        if self.translator.is_open_right(action):
            if self._is_tiger_right():
                return OPEN_TIGER_REWARD
            else:
                return OPEN_MONEY_REWARD
            
        if self.translator.is_listen_left(action):
            return LISTEN_REWARD

        if self.translator.is_listen_right(action):
            return LISTEN_REWARD

    def _is_tiger_left(self):
        return np.array_equal(self.state, LEFT_OBS)

    def _is_tiger_right(self):
        return np.array_equal(self.state, RIGHT_OBS)

    def _agent_obs(self, agent_id, action):
        if self.translator.is_open_action(action):
            return self.state
        return self._listen_obs(action)

    def _listen_obs(self, action):
        # this function assumes the action is either listen left or listen right
        p = 0
        if (self.translator.is_listen_left(action) and self._is_tiger_left()) or (self.translator.is_listen_right(action) and self._is_tiger_right()):
            p = 0.85

        if random.random() < p:
            return self.state
        return NULL_OBS    

    def get_state(self):
        return self.state
    
    def get_state_size(self):
        return self.get_obs_size()
    
    def get_obs_size(self):
        return 2
    
    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions
        
    def get_avail_agent_actions(self, agent_id):
        return [1] * self.action_space.n
    
    def get_total_actions(self):
        return self.action_space.n
    
    def close(self):
        pass
    
    def seed(self):
        pass
    
    def save_replay(self):
        pass
    
    def get_obs(self):
        return self.current_observation
    
    def get_obs_agent(self, agent_id):
        return self.current_observation[agent_id]
    
    def get_stats(self):
        pass

    def get_env_info(self):
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }