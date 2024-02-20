import gymnasium as gym
from envs.multiagentenv import MultiAgentEnv
import numpy as np
from envs.rock_sampling.grid import Grid, NULL_QUALITY, BAD_QUALITY, GOOD_QUALITY
from envs.rock_sampling.translator import Translator
from collections import OrderedDict
from envs.rock_sampling.constants import DEFAULT_CONFIG


from envs.rock_sampling.rewards import IDLE_REWARD, MOVE_REWARD, SENSE_REWARD, GOOD_SAMPLE_REWARD, BAD_SAMPLE_REWARD, ROVER_AREA_CLEAR_REWARD


class DecRockSampling(MultiAgentEnv):

    def __init__(self, batch_size=None, **kwargs):
        
        env_args = kwargs.get('env_args', {})
        grid_config = env_args.get('grid_config', DEFAULT_CONFIG['grid_config'])
        self.horizon = env_args.get('horizon', DEFAULT_CONFIG['horizon'])
        self.current_step = 0
        self.grid = Grid(grid_config)

        self.translator = Translator(self.grid.get_num_rocks())
        self._agent_ids = ['agent_0', 'agent_1']

        self.action_space = gym.spaces.Discrete(1 + # Idle action
                                                4 + # Move actions
                                                self.grid.get_num_rocks() + # Sense actions
                                                self.grid.get_num_rocks()   # Sample actions
                                                )
        
        self.n_agents = kwargs.get('n_agents', 2)
        self.episode_limit = self.horizon
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.grid.reset_board()
        observations = []

        for agent_id in self._agent_ids:
            # agent_obs = {
            #     'position': self.grid.get_rover_position(agent_id),
            #     'rock_quality': NULL_QUALITY
            # }

            # agent_obs.update({
            #     f'rock_{i}_position': self.grid.get_rock_position(i) for i in range(self.grid.get_num_rocks())
            # })

            # turning agent obs into a long list of numbers

            agent_obs = [
                *self.grid.get_rover_position(agent_id),
                NULL_QUALITY
            ]

            for i in range(self.grid.get_num_rocks()):
                agent_obs.extend(self.grid.get_rock_position(i))

            for i in range(self.grid.get_num_rocks()):
                agent_obs.append(self.grid.get_rock_quality(i))

            # agent_obs = OrderedDict(agent_obs)
            observations.append(agent_obs)

        self._update_current_observation(observations)
        self._update_current_state()

        return self.get_obs(), self.get_state()
    
    def step(self, actions):

        action_dict = {self._agent_ids[i]: actions[i] for i in range(len(actions))}
        self.current_step += 1
        observations, rewards, dones, truncs, infos = {}, {}, {}, {}, {}

        for agent_id, action in action_dict.items():

            observations[agent_id] = [
                *self.grid.get_rover_position(agent_id),
                NULL_QUALITY
            ]

            for i in range(self.grid.get_num_rocks()):
                observations[agent_id].extend(self.grid.get_rock_position(i))

            for i in range(self.grid.get_num_rocks()):
                observations[agent_id].append(self.grid.get_rock_quality(i))
            

            if self.translator.is_idle_action(action):
                rewards[agent_id] = IDLE_REWARD
            
            if self.translator.is_move_action(action):
                direction = self.translator.get_move_direction(action)
                self.grid.move_rover(agent_id, direction)
                # observations[agent_id]['position'] = self.grid.get_rover_position(agent_id)

                observations[agent_id][0] = self.grid.get_rover_position(agent_id)[0]
                observations[agent_id][1] = self.grid.get_rover_position(agent_id)[1]

                rewards[agent_id] = MOVE_REWARD
            
            if self.translator.is_sense_action(action):
                rock_id = self.translator.get_sensed_rock_id(action)
                rock_quality = self.grid.sense_rock(agent_id, rock_id)
                # observations[agent_id]['rock_quality'] = rock_quality
                observations[agent_id][2] = rock_quality
                rewards[agent_id] = SENSE_REWARD
            
            if self.translator.is_sample_action(action):
                rover_1_area_clear_before = self.grid.is_rover1_area_clear()
                rover_2_area_clear_before = self.grid.is_rover2_area_clear()
                shared_area_clear_before = self.grid.is_shared_area_clear()

                rock_id = self.translator.get_sampled_rock_id(action)
                rock_quality = self.grid.sample_rock(agent_id, rock_id)

                if rock_quality is None or rock_quality == 'Bad':
                    rewards[agent_id] = BAD_SAMPLE_REWARD
                    continue

                rover_1_area_clear_after = self.grid.is_rover1_area_clear()
                rover_2_area_clear_after = self.grid.is_rover2_area_clear()
                shared_area_clear_after = self.grid.is_shared_area_clear()

                if rover_1_area_clear_before != rover_1_area_clear_after:
                    rewards['rover1'] = ROVER_AREA_CLEAR_REWARD
                    continue
                if rover_2_area_clear_before != rover_2_area_clear_after:
                    rewards['rover2'] = ROVER_AREA_CLEAR_REWARD
                    continue
                
                if shared_area_clear_before != shared_area_clear_after:
                    rewards['rover1'] = ROVER_AREA_CLEAR_REWARD
                    rewards['rover2'] = ROVER_AREA_CLEAR_REWARD
                    continue
            
        dones['__all__'] = self.grid.is_game_over() or self.current_step >= self.horizon
        truncs['__all__'] = False


        observations = [observations[agent_id] for agent_id in self._agent_ids]
        terminates = dones['__all__']
        self._update_current_observation(observations)
        self._update_current_state()

        return sum(rewards.values()), terminates, {}


    def _update_current_observation(self, observations):
        # create the state, it consists of agent locations, rock locations and rock qualities
        self.current_observations = observations

    
    def _update_current_state(self):
        # # self.state = np.array([self.grid.get_rover_position(agent_id) for agent_id in self._agent_ids]) + \
        # #         [self.grid.get_rock_position(i) for i in range(self.grid.get_num_rocks())] + \
        # #         np.array([self.grid.get_rock_quality(i) for i in range(self.grid.get_num_rocks())])
        
        pos = np.array([ self.grid.get_rover_position(agent_id) for agent_id in self._agent_ids])
        rock_pos = np.array([self.grid.get_rock_position(i) for i in range(self.grid.get_num_rocks())])
        rock_quality = np.array([self.grid.get_rock_quality(i) for i in range(self.grid.get_num_rocks())])
        

        self.state = np.concatenate((pos.flatten(), rock_pos.flatten(), rock_quality.flatten()))
        
    
    def get_obs(self):
        return self.current_observations
    
    def get_state(self):
        return self.state
    
    def get_obs_agent(self, agent_id):
        return self.current_observations[agent_id]
    
    def get_obs_size(self):
        return len(self.current_observations[0])
    
    def get_state_size(self):
        return len(self.state)
    
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
