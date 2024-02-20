from ..multiagentenv import MultiAgentEnv
import numpy as np
import subprocess

def log(func):
    #this decorator will log the result of the function
    # into logs.txt

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        with open("logs.txt", "a+") as f:
            f.write(f"{func.__name__} returned {result}\n")
        return result
    return wrapper

#@log
def get_mapping(env):
    ACTION2ACTIONS = {}
    ACTIONS2ACTION = {}

    # we need to add the mapping (x) -> (x_1,....,x_n) where n is the number of agents
    # and each x_i is the action of agent i
    
    for j in range(env.action_space.n ** env.n_agents):
        
        actions = []
        k = j
        for _ in range(env.n_agents):
            actions.append(k % env.action_space.n)
            k = k // env.action_space.n
        
        ACTION2ACTIONS[j] = actions
        actions = tuple(actions)
        ACTIONS2ACTION[actions] = j
    

    return ACTION2ACTIONS, ACTIONS2ACTION

class TeamEnv(MultiAgentEnv):
        
        def __init__(self, env:MultiAgentEnv):
                self.env = env
                self.action2actions, self.actions2action = get_mapping(env)
                self.n_agents = 1
                self.episode_limit = env.episode_limit
        
        #@log
        def reset(self):
            self.env.reset()
            return self.get_obs(), self.get_state()
        
        #@log
        def step(self, actions):
            action = actions[0]
            actions = self.action2actions[action]
            return self.env.step(actions)
        
        #@log
        def get_obs(self):
            # gets a list of observations for each agent, flattenes
            return np.array(self.env.get_obs()).flatten()
        
        #@log
        def get_obs_agent(self, agent_id):
            return self.env.get_obs().flatten()
        
        #@log
        def get_obs_size(self):
            return self.env.get_obs_size() * self.env.n_agents
        
        #@log
        def get_state(self):
            return self.env.get_state()
        
        #@log
        def get_state_size(self):
            return self.env.get_state_size()
        
        #@log
        def get_avail_agent_actions(self, agent_id):
            return [1] * self.get_total_actions()
        
        #@log
        def get_avail_actions(self):
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        
        #@log
        def get_total_actions(self):
            return len(self.action2actions.keys())
        
        #@log
        def get_stats(self):
            return self.env.get_stats()
        
        #@log
        def close(self):
             return self.env.close()
        
        #@log
        def seed(self):
            return self.env.seed()
        
        #@log
        def save_replay(self):
            return self.env.save_replay()
        
        #@log
        def get_env_info(self):
            return {
                "state_shape": self.get_state_size(),
                "obs_shape": self.get_obs_size(),
                "n_actions": self.get_total_actions(),
                "n_agents": self.n_agents,
                "episode_limit": self.episode_limit
            }

        def render(self):
            self.env.render()
