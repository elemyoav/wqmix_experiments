from envs.multiagentenv import MultiAgentEnv
import torch as th
import numpy as np
import random
import pygame
import math
import random
import threading
from utils.dict2namedtuple import convert

int_type = np.int16
float_type = np.float32


class MAPF(MultiAgentEnv):

    action_labels = {"right" : 0, "down" : 1, "left" : 2, "up" : 3, "stay" : 4 #currently without pinging specific beacons
                      , "done" : 5}
    action_names = ["right" , "down" , "left" , "up" , "stay" 
                      , "done"]
                     
    n=0
    width = 5
    height = 5
    grid = [width,height]
    c_agents = 2
    c_success = 0
    cur_actions = []
    episode_limit_base = 10
    n_actions = len(action_names)
    debug = False
    n_get_state = 0
    test_mode = False
    action_list=[]
    agent_start = {}
    agent_goal = {}
    beacons = []
    SingleAgent = False
    ImplicitSensing = True
    SensorError = 0.0
    MoveError = 0.3
    CollisionCost = True
    ID = 0
    

    def __init__(self, batch_size=None, **kwargs): 
        # Unpack arguments from sacred
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args
        #self.print_push_box = getattr(args, "print_push_box", False)

        self.made_screen = False
        self.scaling = 5
        
        random.seed()
        ID = random.randint(0,10000)
        
        #random.seed(333)
        
        
        #code here for loading a map
        
        
        
        #default map
        
        self.grid = [[0]*self.width]*self.height
        for x in range(0,self.width):
            for y in range(0,self.height):
                self.grid[x][y] = 0 #open space
        
        
        self.agent_start["a0"] = [1,1]
        self.agent_goal["a0"] = [3,3]
        
        self.agent_start["a1"] = [3,1]
        self.agent_goal["a1"] = [1,3]
        
        
        self.beacons.append([3,2,2])
        
        


        self.reset()
        
        #self.test()


    def test(self):
        actions = [[3,3],[3,3], [5,5]] #for 4X3
        #actions = [[3,3], [13,13], [8,8], [5,5], [5,1], [5,4], [1,4], [2,2], [13,13], [12,12], [9,9], [14,14]] #for 3X3
        
        if self.SingleAgent:
            actions1 = []
            for action in actions:
                actions1.append([action[0]])
                if action[0] < 9 or action[0] > 12: #non-collaborative actions
                    actions1.append([action[1]+self.n_actions])
            actions = actions1
        
        for i in range(0,5):
            print("\n")
            print("reset " + str(i))
            self.reset()
            for a in actions:
                print("s=" + self.get_state_str())
                s = ""
                if self.SingleAgent:
                    if a[0] >= 9 and a[0] <= 12:
                        s += ", joint:" + self.action_names[a[0]]
                    else:
                        if a[0] < self.n_actions:
                            s += ", a1:" + self.action_names[a[0]]
                        else:
                            s += ", a2:" + self.action_names[a[0] - self.n_actions]
                else:
                    for j in range(0, self.c_agents):
                        s += ", " + self.action_names[a[j]]
                print("a=" + str(a) + ", " + s)
                r, t, k = self.step(a)
                print("s'=" + self.get_state_str())
                print("r = " + str(r) + ", t=" + str(t) + ", k=" + str(k))
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
   
   
   
        self.agent_position = {}
        for i in range(0, self.c_agents):
            agent = "a" + str(i)
            self.agent_position[agent] = self.agent_start[agent]
            
      
        

        # self.step(th.zeros(self.c_agents).fill_(self.action_labels['stay']))
        return self.get_obs(), self.get_state()


    def step(self, actions):
        if self.SingleAgent:
            a = actions[0]
            if a < len(self.action_names):
                i = a
            else:
                i = a - len(self.action_names)
            if self.collaborative(self.action_names[i]) or self.action_names[i] == "done":
                j = i
            else:
                j = self.action_labels["stay"]
            if a < len(self.action_names):
                real_actions = [i,j]
            else:
                real_actions = [j,i]
            reward, terminated, info = self.MultiAgentStep(real_actions)
        else:
            reward, terminated, info = self.MultiAgentStep(actions)
        return reward, terminated, info
    
    
    def collaborative(self, action_name):
        return "collaborative" in action_name
    
    
    def MultiAgentStep(self, actions):
        #print("step " + str(actions) +        " \n")

        o = self.get_obs()
        self.action_list.append("Step: " + self.get_state_str() + "; " + str(o[0]) + ", " + str(o[1]))
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
            if action_name != "stay" and action_name != "done" and "sense" not in action_name:
                cur_pos = self.agent_position["a"+str(i)]
                #print("s=" + str(cur_pos))
                new_pos = self.move(cur_pos, action_name)
                #print("s'=" + str(cur_pos))
                reward -= 1
                                
                #else:
                for other in self.agent_position:
                    if other[0] == new_pos[0] and other[1] == new_pos[1]:
                        if self.CollisionCost:
                            reward -= 1000
                        else:
                            reward -= 1
                self.agent_position["a"+str(i)] = new_pos
        
        
        self.steps += 1
        info={}
        info["episode_limit"] = False
        
        success = True
        for agent in self.agent_position:
            if self.agent_position[agent][0] != self.agent_goal[agent][0] or self.agent_position[agent][1] != self.agent_goal[agent][1]:
                success = False
                    
        if success:
            #if self.test_mode and len(self.action_list) > 1:
             #   for i in range(0,len(self.action_list)):
             #       print(str(i) + ": " + self.action_list[i])
             #   print("***********Goal***************")
            reward += 100
            self.c_success = self.c_success + 1
            if self.c_success % 50 == 0:
               print("############### " + str(self.ID) + " Goals: " + str(self.c_success))
               if self.c_success > 500:
                    for i in range(0,len(self.action_list)):
                        print(str(self.ID) + ": " + str(i) + ": " + self.action_list[i])
                    print("-----------------------------------------------")
                    if self.c_success > 505:
                        exit()
        
        
        if all_done:
            terminated = True
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
        
        
    def moveOld(self, pos, action):
        new_pos = pos.copy()
        success = random.random() < self.MoveError
        error = random.randint(0,2)
        
        if "right" in action:
            if pos[0] < self.width - 1:
                if success:
                    new_pos[0] = pos[0] + 1
                else:
                    new_pos[0] = pos[0]
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
        
    def move(self, pos, action):
        new_pos = pos.copy()
        success = random.random() > self.MoveError
        error = random.randint(0,2)
        #print("move: " + str(success) + ", " + str(error))
        
        if "right" in action:
            if success:
                new_pos[0] = pos[0] + 1
            else:
                if error == 0:
                    new_pos[1] = pos[1] + 1
                if error == 1:
                    new_pos[1] = pos[1] - 1
                    
        if "left" in action:
            if success:
                new_pos[0] = pos[0] - 1
            else:
                if error == 0:
                    new_pos[1] = pos[1] + 1
                if error == 1:
                    new_pos[1] = pos[1] - 1
                    
        if "up" in action:
            if success:
                new_pos[1] = pos[1] + 1
            else:
                if error == 0:
                    new_pos[0] = pos[0] + 1
                if error == 1:
                    new_pos[0] = pos[0] - 1
               
        if "down" in action:
            if success:
                new_pos[1] = pos[1] - 1
            else:
                if error == 0:
                    new_pos[0] = pos[0] + 1
                if error == 1:
                    new_pos[0] = pos[0] - 1
            
        if new_pos[0] == self.width:
            new_pos[0] = self.width - 1
        if new_pos[0] < 0:
            new_pos[0] = 0
        if new_pos[1] == self.height:
            new_pos[1] = self.height - 1
        if new_pos[1] < 0:
            new_pos[1] = 0
           
        return new_pos

    # ---------- OBSERVATION METHODS -----------------------------------------------------------------------------------
    def get_obs_agent(self, agent_id, batch=0):
        if self.ImplicitSensing:
            obs = []
            for beacon in self.beacons:
                max_distance = beacon[2];
                pos = self.agent_position["a" + str(agent_id)]
                true_distance = self.distance_d1(pos,beacon)
                if true_distance == 0:
                    obs.append(0)
                else:
                    if true_distance > max_distance:
                        obs.append(max_distance + 1)
                    else:
                        if random.random() < self.SensorError:
                            obs.append(true_distance + 1)
                        else:
                            obs.append(true_distance)
        else:
            obs = [0]
            #print(self.cur_actions)
            #print(agent_id)
            
            if len(self.cur_actions)>agent_id:
                action_name = self.action_names[self.cur_actions[agent_id]]
                if "sense" in action_name:
                    #print( "Sensing!!! ")
                    pos = self.agent_position["a" + str(agent_id)]
                    #print( pos)
                    for box in self.light_box_position:
                        if np.array_equal(self.light_box_position[box], pos):
                           obs = [1]
                    for box in self.heavy_box_position:
                        if np.array_equal(self.heavy_box_position[box], pos):
                           obs = [1]
        return obs

 #manhatan
    def distance_d1(self, pos1, pos2):
        dX = abs(pos1[0] - pos2[0])
        dY = abs(pos1[1] - pos2[1])
        return dX + dY
        

    def get_obs(self):
        #print("get_obs\n")
        if self.SingleAgent:
            sum_o = 0
            #need to improve to handle arrays of observations
            for i in range(self.c_agents):
                sum_o = sum_o + self.get_obs_agent(i)[0]
            agents_obs = np.array([[sum_o]])
        else:
            agents_obs = np.array([self.get_obs_agent(i) for i in range(self.c_agents)])
        return agents_obs

    def pos2idx(self, pos):
        idx = pos[1] * self.width + pos[0]
        return idx


    def get_state_str(self):
        s = str(self.agent_position)
        
        return s


    def get_state(self):
    
        self.n_get_state = self.n_get_state + 1
        #if self.n_get_state % 100 == 0:
        #    print("get_state " + str(self.n_get_state))
    
        #print("get_state\n")
        state = np.zeros(self.c_agents)
        # Enqueue all agents
        
        if True:
            i = 0
            for a in self.agent_position:
                #state[i] = self.pos2idx(self.agent_position[a])
                state[i] = self.pos2idx(self.agent_position[a])
                i = i + 1
        else:
            state[0] = self.get_obs_agent(0)[0]
            state[1] = self.get_obs_agent(1)[0]
        
        
        
        
        return state



    # ---------- GETTERS -----------------------------------------------------------------------------------------------
    def get_total_actions(self):
        if self.SingleAgent:
            return len(self.action_names) * self.c_agents
        else:
            return len(self.action_names)


    def get_avail_actions(self):
        #print("get_avail_actions\n")
        avail_actions = []
        if self.SingleAgent:
            avail_actions.append([1 for _ in range(self.n_actions * self.c_agents)])
        else:
            for agent_id in range(self.c_agents):
                avail_actions.append([1 for _ in range(self.n_actions)])
        return np.array(avail_actions)


    def get_obs_size(self):
        return 1

    def get_state_size(self):
        return self.c_agents

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
