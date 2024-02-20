import gymnasium as gym
from envs.multiagentenv import MultiAgentEnv
import numpy as np
from collections import OrderedDict
from envs.box_pushing.translator import Translator
from envs.box_pushing.grid import Grid

from envs.box_pushing.rewards import IDLE_REWARD, MOVE_AGENT_REWARD, SENSE_LIGHT_BOX_REWARD, SENSE_HEAVY_BOX_REWARD, PUSH_LIGHT_BOX_REWARD, PUSH_HEAVY_BOX_REWARD, PUSH_LIGHT_BOX_SUCC_REWARD, PUSH_HEAVY_BOX_SUCC_REWARD


class DecBoxPushing(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        env_config = kwargs.get('env_args', {})
        num_agents:int = env_config.get('num_agents', 2) # number of agents
        grid_size = env_config.get('grid_size', (2, 2)) # of the form (x, y)
        num_light_boxes:int = env_config.get('num_light_boxes', 1)
        num_heavy_boxes:int = env_config.get('num_heavy_boxes', 1)
        self.p_push:float = env_config.get('p_push', 0.8) # probability of pushing a box in the intended direction
        self.horizon:int = env_config.get('horizon', 300)
        self.p_sense:float = env_config.get('p_sense', 1.0) # probability of sensing a box in the intended direction

        self.grid = Grid(num_agents, num_light_boxes, num_heavy_boxes, grid_size)
        self.translator = Translator(num_agents, num_light_boxes, num_heavy_boxes)

        self.current_step = 0
        self.n_agents = num_agents
        self.episode_limit = self.horizon

        self._agent_ids = [f'agent_{i}' for i in range(num_agents)]

        self.action_space = gym.spaces.Discrete(
            1 + # idle
            4 + # move in any direction
            num_light_boxes + # sense b_i
            num_heavy_boxes + # sense B_i
            num_light_boxes*4 + # push b_i in any direction
            num_heavy_boxes*4 # collab push B_i in any direction
        )
        
        self.observation_space = gym.spaces.Dict({
            'location': gym.spaces.MultiDiscrete([*grid_size]),
            'sensed_box': gym.spaces.Discrete(2),  # Boolean (True/False)
            'target_location': gym.spaces.MultiDiscrete([*grid_size])
        })
        
        self.reset()
        super().__init__()

    def reset(self):

        self.current_step = 0
        self.grid.reset_board()
        self.translator.reset_box_pushers()

        observations = {
            agent_id: OrderedDict({
                'location': self.grid.get_agent_location(agent_id),
                'sensed_box': 0,
                'target_location': self.grid.get_target_location(),
            }) for agent_id in self._agent_ids
        }

        self._update_observations(observations)
        self._update_state()

        return self.get_obs(), self.get_state()

    def push_light_boxes_reward(self, boxes):
        rewards = {}
        for box_id, directions in boxes.items():
            done_state_start = self.grid.is_light_box_done(box_id)
            for direction in directions:
                if np.random.rand() < self.p_push:
                    self.grid.move_light_box(box_id, direction)
            done_state_end = self.grid.is_light_box_done(box_id)

        if done_state_start != done_state_end:
            rewards[box_id] = PUSH_LIGHT_BOX_SUCC_REWARD
        else:
            rewards[box_id] = 0
        return rewards
            
    def push_heavy_boxes_reward(self, boxes):
        rewards = {}
        for box_id, directions in boxes.items():
            done_state_start = self.grid.is_heavy_box_done(box_id)
            for direction in directions:
                if np.random.rand() < self.p_push:
                    self.grid.move_heavy_box(box_id, direction)
            done_state_end = self.grid.is_heavy_box_done(box_id)

            if done_state_start != done_state_end:
                rewards[box_id] = PUSH_HEAVY_BOX_SUCC_REWARD
            else:
                rewards[box_id] = 0
        return rewards
    
    def step(self, actions):

        self.current_step += 1

        action_dict = {self._agent_ids[i]: actions[i] for i in range(len(actions))}
        observations, rewards, dones, truncs, infos = {}, {}, {}, {}, {}
        
        for agent_id, action in action_dict.items():
            observations[agent_id] = OrderedDict({
                                        'location': self.grid.get_agent_location(agent_id),
                                        'sensed_box': 0,
                                        'target_location': self.grid.get_target_location(),
                                    })
            
            if self.translator.is_idle_action(action):
                rewards[agent_id] = IDLE_REWARD

            if self.translator.is_move_agent_action(action):
                direction = self.translator.get_move_agent_direction(action)
                self.grid.move_agent(agent_id, direction)
                observations[agent_id]['location'] = self.grid.get_agent_location(agent_id)
                rewards[agent_id] = MOVE_AGENT_REWARD
            
            if self.translator.is_sense_light_box_action(action):
                box_num = self.translator.get_sense_light_box_num(action)
                box_id = f'light_box_{box_num}'
                if np.random.rand() < self.p_sense:
                    observations[agent_id]['sensed_box'] = self.grid.sense_light_box(agent_id, box_id)
                rewards[agent_id] = SENSE_LIGHT_BOX_REWARD
            
            if self.translator.is_sense_heavy_box_action(action):
                box_num = self.translator.get_sense_heavy_box_num(action)
                box_id = f'heavy_box_{box_num}'
                if np.random.rand() < self.p_sense:
                    observations[agent_id]['sensed_box'] = self.grid.sense_heavy_box(agent_id, box_id)
                rewards[agent_id] = SENSE_HEAVY_BOX_REWARD
            
            if self.translator.is_push_light_box_action(action):
                rewards[agent_id] = PUSH_LIGHT_BOX_REWARD
                box_num = self.translator.get_push_light_box_num(action)
                box_id = f'light_box_{box_num}'
                direction = self.translator.get_push_light_box_direction(action)

                if self.grid.can_push_light_box(agent_id, box_id):
                    self.translator.add_light_box_pusher(box_id, direction, agent_id)
            
            if self.translator.is_push_heavy_box_action(action):
                rewards[agent_id] = PUSH_HEAVY_BOX_REWARD
                box_num = self.translator.get_push_heavy_box_num(action)
                box_id = f'heavy_box_{box_num}'
                direction = self.translator.get_push_heavy_box_direction(action)

                if self.grid.can_push_heavy_box(agent_id, box_id):
                    self.translator.add_heavy_box_pusher(box_id, direction, agent_id)
            
        pushing_results = self.translator.get_box_pushing_directions()

        light_boxes_rewards = self.push_light_boxes_reward(pushing_results['light_boxes_directions'])
        heavy_boxes_rewards = self.push_heavy_boxes_reward(pushing_results['heavy_boxes_directions'])

        light_boxes_succ_agents = pushing_results['light_boxes_succ_pushers']
        heavy_boxes_succ_agents = pushing_results['heavy_boxes_succ_pushers']

        for box_id, agents in light_boxes_succ_agents.items():
            for agent_id in agents:
                rewards[agent_id] += light_boxes_rewards[box_id]

        for box_id, agents in heavy_boxes_succ_agents.items():
            for agent_id in agents:
                rewards[agent_id] += heavy_boxes_rewards[box_id]

        dones['__all__'] = self.grid.is_game_over() or self.current_step >= self.horizon
        truncs['__all__'] = False

        self._update_observations(observations)
        self._update_state()

        return sum(rewards.values()), dones['__all__'], {}


    def _update_observations(self, observations):
        
        # convert the dict observations to a list of observations
        # each observation consists of 3 elements: location, sensed_box, target_location
        # location is a tuple (x, y)
        # sensed_box is a boolean
        # target_location is a tuple (x, y)
        # we are to flatten the observations to a list of numbers

        self.current_observation = []
        for agent_id, obs in observations.items():
            obs = [
                *obs['location'],
                obs['sensed_box'],
                *obs['target_location']
            ]
            self.current_observation.append(obs)

    def _update_state(self):
        # sets the state to be a list of all agent locations, box locations, and target location

        agent_locs = [self.grid.get_agent_location(agent_id) for agent_id in self._agent_ids]
        light_box_locs = [self.grid.get_light_box_location(f'light_box_{box_id}') for box_id in range(self.grid.get_num_light_boxes())]
        heavy_box_locs = [self.grid.get_heavy_box_location(f'heavy_box_{box_id}') for box_id in range(self.grid.get_num_heavy_boxes())]
        target_loc = self.grid.get_target_location()

        self.current_state = []
        self.current_state.extend(agent_locs)
        self.current_state.extend(light_box_locs)
        self.current_state.extend(heavy_box_locs)
        # now flatten current state
        self.current_state.extend([np.array(target_loc)])

        self.current_state = np.array(self.current_state)


    def get_obs(self):
        return self.current_observation

    def get_obs_agent(self, agent_id):
        return self.current_observation[agent_id]
    
    def get_obs_size(self):
        return len(self.current_observation[0])
    
    def get_state(self):
        return self.current_state.flatten()
    
    def get_state_size(self):
        return self.current_state.size
    
    def get_avail_actions(self):
        return [self.get_avail_agent_actions(agent_id) for agent_id in self._agent_ids]
    
    def get_avail_agent_actions(self, agent_id):
        return [1] * self.action_space.n
    
    def get_total_actions(self):
        return self.action_space.n
    
    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        pass

    def get_stats(self):
        pass
                