from envs.rock_sampling.rock_sampling import DecRockSampling
from gymnasium import Env as GymEnv
from gymnasium import spaces

class RockSampling(GymEnv):
    def __init__(self, env_config={}):
        self._env = DecRockSampling(env_config)

        self.observation_space = spaces.Dict({
            agent_id: self._env.observation_space for agent_id in self._env._agent_ids
        })

        self.action_space = spaces.Dict({
            agent_id: self._env.action_space for agent_id in self._env._agent_ids
        })

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)         


    def step(self, action):
        observations, rewards, dones, truncs, infos = self._env.step(action)
        done = dones['__all__']
        trunc = truncs['__all__']
        reward = sum(rewards.values())

        return observations, reward, done, trunc, infos
    
    def render(self, mode='human'):
        return self._env.render(mode)