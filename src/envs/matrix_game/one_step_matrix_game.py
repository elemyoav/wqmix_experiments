from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np
import torch as th

# this non-monotonic matrix can be solved by qmix
payoff_values = [[12, -0.1, -0.1],
                    [-0.1, 0, 0],
                    [-0.1, 0, 0]]

# payoff_values = [[12, -12, -12],
#                     [-12, 0, 0],
#                     [-12, 0, 0]]

# payoff_values = [[12, 0, 10],
#                     [0, 0, 10],
#                     [10, 10, 10]]


# payoff_values = [[1, 0], [0, 1]]
# n_agents = 3
# payoff_values = np.zeros((n_agents, n_agents))
# for i in range(n_agents):
#     payoff_values[i, i] = 1

class OneStepMatrixGame(MultiAgentEnv):



    action_names = ["right" , "down" , "left" , "up" , "stay" ,
                      "push-right" , "push-down", "push-left", "push-up" ,
                      "collaborative-push-right", "collaborative-push-down", "collaborative-push-left" , "collaborative-push-up", "sense-box"]
                     
    
    width = 4
    height = 3
    n_agents = 2
    n_light = 2
    n_heavy = 1
    n_boxes = n_light + n_heavy
    cur_actions = []
    episode_limit = 100
    n_actions = len(action_names)
    #rnd = np.random.default_rng(12345)
    rnd = np.random.RandomState(12345)




    def __init__(self, batch_size=None, **kwargs):
        # Define the agents
        self.n_agents = 2

        # Define the internal state
        self.steps = 0
        self.n_actions = len(payoff_values[0])
        self.episode_limit = 1


    def reset(self):
        """ Returns initial observations and states"""
        
        self.agent_position = {"a0": [0,0], "a1": [self.width-1, 0]}
        self.light_box_position = {"lb0" : [0,1], "lb1": [self.width-1, 1]}
        self.heavy_box_position = {"hb0" : [self.width / 2, 1]}
        
        for key in self.light_box_position :
            if self.rnd.random() < 0.5 :
                self.light_box_position[key] = [self.width - 1, self.height - 1]
        
        for key in self.heavy_box_position :
            if self.rnd.random() < 0.5 :
                self.heavy_box_position[key] = [self.width - 1, self.height - 1]

        
        
        self.steps = 0
        return self.get_obs(), self.get_state()

        
    def move(self, pos, action):
        new_pos = pos
        
        if "right" in action:
            if pos[0] < self.width - 1:
                new_pos[0] = pos[0] + 1
        if "left" in action:
            if pos[0] > 0:
                new_pos[0] = pos[0] - 1
        if "up" in action:
            if pos[1] < self.height - 1:
                new_pos[1] = pos[1] + 1
        if "down" in action:
            if pos[1] > 0:
                new_pos[1] = pos[1] - 1
            
        return new_pos


    def step(self, actions):
        """ Returns reward, terminated, info """
        
        #print("step " + str(actions) +        " \n")
        self.cur_actions = actions
        reward = 0
        for i in range(0, self.n_agents):
            action_name = self.action_names[actions[i]]
            #print("Step: " + action_name + ", " + str("push" in action_name) + "\n")
            if action_name != "stay":
                cur_pos = self.agent_position["a"+str(i)]
                new_pos = self.move(cur_pos, action_name)
                if "collaborative" in action_name:
                    reward -= 3
                elif "push" in action_name:
                    reward -= 2
                else:
                    reward -= 1
                if "push" in action_name:
                    if "collaborative" in action_name:
                        #heavy box pushing here
                        for box in self.heavy_box_position:
                            if np.array_equal(self.heavy_box_position[box],cur_pos):
                                for j in range(0, self.n_agents):
                                    if i != j:
                                        other_action_name = self.action_names[actions[j]]
                                        if action_name == other_action_name: 
                                            other_pos =  self.agent_position["a"+str(j)]
                                            if np.array_equal(other_pos, cur_pos):
                                                self.heavy_box_position[box] = new_pos
                                                break
                                        
                    else:
                        #light box pushing - pushes the first box only
                        for box in self.light_box_position:
                            if np.array_equal(self.light_box_position[box],cur_pos):
                                self.light_box_position[box] = new_pos
                                break
                                
                else:
                    self.agent_position["a"+str(i)] = new_pos
        
        terminated = False
        for box in self.light_box_position:
            if np.array_equal(self.light_box_position[box], [self.width - 1, self.height - 1]):
                reward += 10
            else:
                terminated = False
               
        for box in self.heavy_box_position:
            if np.array_equal(self.heavy_box_position[box], [self.width - 1, self.height - 1]):
                reward += 20
            else:
                terminated = False
        
        if terminated:
            reward += 100
        
        
        
        #reward = payoff_values[actions[0]][actions[1]]

        self.steps += 1
        
        #terminated = True

        info = {}
        return reward, terminated, info

    def step1(self, actions):
        """ Returns reward, terminated, info """
        reward = payoff_values[actions[0]][actions[1]]

        self.steps = 1
        terminated = True

        info = {}
        return reward, terminated, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        one_hot_step = np.zeros(2)
        one_hot_step[self.steps] = 1
        return [np.copy(one_hot_step) for _ in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.get_obs_agent(0))

    def get_state(self):
        return self.get_obs_agent(0)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.get_obs_size()

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_stats(self):
        return None

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self):
        raise NotImplementedError
    
    
# for mixer methods
def print_matrix_status(batch, mixer, mac_out):
    batch_size = batch.batch_size
    matrix_size = len(payoff_values)
    results = th.zeros((matrix_size, matrix_size))       
        
    with th.no_grad():
        for i in range(results.shape[0]):
            for j in range(results.shape[1]):
                actions = th.LongTensor([[[[i], [j]]]]).to(device=mac_out.device).repeat(batch_size, 1, 1, 1)
                if len(mac_out.size()) == 5: # n qvals
                    actions = actions.unsqueeze(-1).repeat(1, 1, 1, 1, mac_out.size(-1)) # b, t, a, actions, n
                qvals = th.gather(mac_out[:batch_size, 0:1], dim=3, index=actions).squeeze(3)
                
                global_q = mixer(qvals, batch["state"][:batch_size, 0:1]).mean()
                results[i][j] = global_q.item()
                
    th.set_printoptions(1, sci_mode=False)
    print(results)
    if len(mac_out.size()) == 5:
        mac_out = mac_out.mean(-1)
    print(mac_out.mean(dim=(0, 1)).detach().cpu())
    th.set_printoptions(4)