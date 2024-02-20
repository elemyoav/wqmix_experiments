from envs.multiagentenv import MultiAgentEnv
import torch as th
import numpy as np
import random
import pygame
import math
from utils.dict2namedtuple import convert

int_type = np.int16
float_type = np.float32


class BoxPushing(MultiAgentEnv):

    action_labels = {"right" : 0, "down" : 1, "left" : 2, "up" : 3, "stay" : 4,
                      "push-right" : 5, "push-down": 6, "push-left" : 7, "push-up" : 8,
                      "collaborative-push-right" : 9, "collaborative-push-down" : 10, "collaborative-push-left" : 11, "collaborative-push-up" : 12, "sense-box" : 13, "done" : 14}
    action_names = ["right" , "down" , "left" , "up" , "stay" ,
                      "push-right" , "push-down", "push-left", "push-up" ,
                      "collaborative-push-right", "collaborative-push-down", "collaborative-push-left" , "collaborative-push-up", "sense-box", "done"]
                     
    n=0
    width = 4
    height = 3
    c_agents = 2
    n_light = 2
    n_heavy = 0
    n_boxes = n_light + n_heavy
    cur_actions = []
    episode_limit_base = 50
    n_actions = len(action_names)
    rnd = np.random.default_rng(12345)
    debug = False
    n_get_state = 0
    test_mode = False
    action_list=[]
    heavy_box_position = {}
    light_box_position = {}
    SingleAgent = True
    

    def __init__(self, batch_size=None, **kwargs): 
        # Unpack arguments from sacred
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args
        self.print_push_box = getattr(args, "print_push_box", False)

        self.reset()

        self.made_screen = False
        self.scaling = 5
        
        #self.test()


    def test(self):
        #actions = [[3,3], [13,13], [8,8], [5,1], [5,4], [5,4], [1,4], [2,2], [13,13], [12,12], [11,11]] #for 4X3
        actions = [[3,3], [13,13], [8,8], [5,5], [5,1], [5,4], [1,4], [2,2], [13,13], [12,12], [9,9], [14,14]] #for 3X3
        for i in range(0,5):
            print("\n")
            print("reset " + str(i))
            self.reset()
            for a in actions:
                print("s=" + self.get_state_str())
                s = ""
                for i in range(0, self.c_agents):
                    s += ", " + self.action_names[a[i]]
                print("a=" + str(a) + ", " + s)
                r, t, i = self.step(a)
                print("s'=" + self.get_state_str())
                print("r = " + str(r) + ", t=" + str(t) + ", i=" + str(i))
                o = self.get_obs()
                print("o = " + str(o))

        exit()
        



    # ---------- INTERACTION METHODS -----------------------------------------------------------------------------------
    def reset(self):
        #print("reset\n")
        # Reset old episode
        self.steps = 0
        self.sum_rewards = 0

        self.action_list=[]
   
        #self.agent_position = {"a0": [0,0], "a1": [self.width-1, 0]}
        self.agent_position = {}
        step = math.floor(self.width / self.c_agents)
        for i in range(0, self.c_agents):
            self.agent_position["a" + str(i)] = [i * step, 0]
            
        #self.light_box_position = {"lb0" : [0,1], "lb1": [self.width-1, 1]}
        if self.n_light > 0:
            step = math.floor(self.width / self.n_light)
            self.light_box_position = {}
            for i in range(0, self.n_light):
                self.light_box_position["lb" + str(i)] = [i * step, math.floor(self.height / 2)]
            for key in self.light_box_position :
                if self.rnd.random() < 0.5 :
                    self.light_box_position[key] = [self.width - 1, self.height - 1]
            
        #self.heavy_box_position = {"hb0" : [math.floor(self.width / 2), 1]}
        if self.n_heavy > 0:
            step = math.floor(self.width / self.n_heavy)
            self.heavy_box_position = {}
            for i in range(0, self.n_heavy):
                self.heavy_box_position["hb" + str(i)] = [i * step + 1, math.floor(self.height / 2)]
            for key in self.heavy_box_position :
                if self.rnd.random() < 0.5 :
                    self.heavy_box_position[key] = [self.width - 1, self.height - 1]
        
        

        # self.step(th.zeros(self.c_agents).fill_(self.action_labels['stay']))
        return self.get_obs(), self.get_state()


    def step(self, actions):
        if self.SingleAgent:
            a = actions[0]
            i = a // len(self.action_names)
            j = a % len(self.action_names)
            real_actions = [i,j]
            reward, terminated, info = self.MultiAgentStep(real_actions)
        else:
            reward, terminated, info = self.MultiAgentStep(actions)
        return reward, terminated, info
    
    def collaborative(self, action_name):
        return "collaborative" in action_name
    
    def MultiAgentStep(self, actions):
        #print("step " + str(actions) +        " \n")

        self.action_list.append("Step: " + self.get_state_str())
        s = ""
        for i in range(0, self.c_agents):
            s += ", " + self.action_names[actions[i]]
        self.action_list.append("Step: " + s)

        self.cur_actions = actions
        reward = 0
        
        all_done = True
        
        for i in range(0, self.c_agents):
            action_name = self.action_names[actions[i]]
            
            if action_name != "done":
                all_done = False
            
            #print("Step: " + action_name + ", " + str("push" in action_name) + "\n")
            if action_name != "stay" and action_name != "done":
                cur_pos = self.agent_position["a"+str(i)]
                #print("s=" + str(cur_pos))
                new_pos = self.move(cur_pos, action_name)
                #print("s'=" + str(cur_pos))
                reward -= 1
                if "push" in action_name:
                    bEmptyPush = True
                    #print("push")
                    if self.collaborative(action_name):
                        #heavy box pushing here
                        for box in self.heavy_box_position:
                            #print("compare: " + str(cur_pos) + "==" + str(self.heavy_box_position[box]) + ":" + str(np.array_equal(self.heavy_box_position[box],cur_pos)))
                            if np.array_equal(self.heavy_box_position[box],cur_pos):
                                for j in range(0, self.c_agents):
                                    if i != j:
                                        other_action_name = self.action_names[actions[j]]
                                        if action_name == other_action_name: 
                                            other_pos =  self.agent_position["a"+str(j)]
                                            if np.array_equal(other_pos, cur_pos):
                                                self.heavy_box_position[box] = new_pos
                                                bEmptyPush = False
                                                break
                                        
                    else:
                        #light box pushing - pushes the first box only
                        for box in self.light_box_position:
                            #print("compare: " + str(cur_pos) + "==" + str(self.light_box_position[box]) + ":" + str(np.array_equal(self.light_box_position[box],cur_pos)))
                            if np.array_equal(self.light_box_position[box],cur_pos):
                                self.light_box_position[box] = new_pos
                                bEmptyPush = False
                                break
                    if bEmptyPush:
                        reward -= 3
                                
                #else:
                self.agent_position["a"+str(i)] = new_pos
        
        
        self.steps += 1
        info={}
        info["episode_limit"] = False
        
        if all_done:
            terminated = True
            success = True
            for box in self.light_box_position:
                if np.array_equal(self.light_box_position[box], [self.width - 1, self.height - 1]) == False:
                    success = False
                   
            for box in self.heavy_box_position:
                if np.array_equal(self.heavy_box_position[box], [self.width - 1, self.height - 1]) == False:
                    success = False
            if success:
                if self.test_mode and len(self.action_list) > 5:
                    for i in range(0,len(self.action_list)):
                        print(str(i) + ": " + self.action_list[i])
                    print("***********Goal***************")
                reward += 100
        else:
            terminated = False
        
        if self.steps >= self.episode_limit_base * self.c_agents:
            #if self.test_mode:
            #    print("***********Fail***************")
            terminated = True
            info["episode_limit"] = True #self.truncate_episodes
            self.action_list = []

        if terminated:
            self.action_list = []
                

        
        
        #print("Step end: " + str(self.get_state()))
        
        return reward, terminated, info
        
        
        
    def move(self, pos, action):
        new_pos = pos.copy()
        
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

    # ---------- OBSERVATION METHODS -----------------------------------------------------------------------------------
    def get_obs_agent(self, agent_id, batch=0):
        obs = [0]
        if len(self.cur_actions)>agent_id:
            action_name = self.action_names[self.cur_actions[agent_id]]
            if "sense" in action_name:
                pos = self.agent_position["a" + str(agent_id)]
                for box in self.light_box_position:
                    if np.array_equal(self.light_box_position[box], pos):
                       obs = [1]
                for box in self.heavy_box_position:
                    if np.array_equal(self.heavy_box_position[box], pos):
                       obs = [1]
        return obs

    def get_obs(self):
        #print("get_obs\n")
        if self.SingleAgent:
            agents_obs = np.array([self.get_obs_agent(0)])
        else:
            agents_obs = np.array([self.get_obs_agent(i) for i in range(self.c_agents)])
        return agents_obs

    def pos2idx(self, pos):
        idx = pos[1] * self.width + pos[0]
        return idx


    def get_state_str(self):
        s = str(self.get_state())
        s += ", agents: " + str(self.agent_position)
        s += ", light: " + str(self.light_box_position)
        s += ", heavy: " + str(self.heavy_box_position)
        
        return s


    def get_state(self):
    
        self.n_get_state = self.n_get_state + 1
        #if self.n_get_state % 100 == 0:
        #    print("get_state " + str(self.n_get_state))
    
        #print("get_state\n")
        state = np.zeros(self.c_agents + self.n_light + self.n_heavy)
        # Enqueue all agents
        
        if True:
            i = 0
            for a in self.agent_position:
                state[i] = self.pos2idx(self.agent_position[a])
                i = i + 1
            # Enqueue all light boxes
            for box in self.light_box_position:
                state[i] = self.pos2idx(self.light_box_position[box])
                i = i + 1
            # Enqueue all heavy boxes
            for box in self.heavy_box_position:
                state[i] = self.pos2idx(self.heavy_box_position[box])
                i = i + 1
        else:
            state[0] = self.get_obs_agent(0)[0]
            state[1] = self.get_obs_agent(1)[0]
        
        
        
        
        return state



    # ---------- GETTERS -----------------------------------------------------------------------------------------------
    def get_total_actions(self):
        if self.SingleAgent:
            return len(self.action_names) ** self.c_agents
        else:
            return len(self.action_names)

    def get_avail_agent_actions(self, agent_id):
        avail_actions = [1 for _ in range(self.n_actions)]
        return avail_actions

    def get_avail_actions(self):
        #print("get_avail_actions\n")
        avail_actions = []
        if self.SingleAgent:
            c_collaborative = 0
            for action_name in self.action_names:
                if self.collaborative(action_name):
                    c_collaborative = c_collaborative + 1
            c_joint_actions = self.n_actions ** self.c_agents
            all_actions = [1 for _ in range(c_joint_actions)]
            for i in range(self.n_actions):
                for j in range(self.n_actions):
                    if self.collaborative(self.action_names[i]):
                        if i != j :
                            all_actions[i*self.n_actions + j] = 0
                    if self.collaborative(self.action_names[j]):
                        if i != j :
                            all_actions[i*self.n_actions + j] = 0
            avail_actions.append(all_actions)
        else:
            for agent_id in range(self.c_agents):
                avail_actions.append(self.get_avail_agent_actions(agent_id))
        return np.array(avail_actions)


    def get_obs_size(self):
        return 1

    def get_state_size(self):
        return self.c_agents + self.n_boxes

    def get_stats(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "episode_limit": self.episode_limit_base * self.c_agents}
        if self.SingleAgent:
            env_info["n_agents"] = 1
        else:
            env_info["n_agents"] = self.c_agents
        return env_info 
    # --------- RENDER METHODS -----------------------------------------------------------------------------------------
    def close(self):
        if self.made_screen:
            pygame.quit()
        print("Closing box pushing")

    def render_array(self):
        # Return an rgb array of the frame
        return None

    def render(self):
        # TODO!
        pass

    def seed(self):
        raise NotImplementedError
