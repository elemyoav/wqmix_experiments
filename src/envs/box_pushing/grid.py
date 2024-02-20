import numpy as np
import copy

class Grid:

    def __init__(self, num_agents, num_light_boxes, num_heavy_boxes, grid_size):
        self.num_agents = num_agents
        self.num_light_boxes = num_light_boxes
        self.num_heavy_boxes = num_heavy_boxes
        self.grid_size = grid_size
        self.agent_ids = [ f'agent_{i}' for i in range(self.num_agents) ]
        self.light_box_ids = [ f'light_box_{i}' for i in range(self.num_light_boxes) ]
        self.heavy_box_ids = [ f'heavy_box_{i}' for i in range(self.num_heavy_boxes) ]

        self.grid = [ np.array([x, y]) for x in range(self.grid_size[0]) for y in range(self.grid_size[1]) ]

        self.init_board()

    def init_board(self):
        self.init_agent_locations = {
            agent_id: self.get_random_location() for agent_id in self.agent_ids
        }

        self.init_light_box_locations = {
            light_box_id: {
                            'location': self.get_random_location(),
                            'done': False
                          } 
                        for light_box_id in self.light_box_ids
        }

        self.init_heavy_box_locations = {
            heavy_box_id: {
                            'location': self.get_random_location(),
                            'done': False
                          } 
                        for heavy_box_id in self.heavy_box_ids
        }

        self.init_target_location = self.get_random_location_not_on_boxes()

        self.reset_board()

    def reset_board(self):
        self.agent_locations = copy.deepcopy(self.init_agent_locations)

        self.light_box_locations = copy.deepcopy(self.init_light_box_locations)

        self.heavy_box_locations = copy.deepcopy(self.init_heavy_box_locations)
        self.target_location = copy.deepcopy(self.init_target_location)

    def get_target_location(self):
        return np.array(self.target_location)
    
    def is_target_location(self, location):
        return np.array_equal(location, self.target_location)
    
    def is_light_box_done(self, box_id):
        return self.get_light_box_location(box_id)['done']
    
    def is_heavy_box_done(self, box_id):
        return self.get_heavy_box_location(box_id)['done']
    
    def get_agent_location(self, agent_id):
        return self.agent_locations[agent_id]

    def set_agent_location(self, agent_id, location):
        self.agent_locations[agent_id] = location

    def get_light_box_location(self, light_box_id):
        return self.light_box_locations[light_box_id]['location']
    
    def set_light_box_location(self, light_box_id, location):
        self.light_box_locations[light_box_id]['location'] = location

    def get_heavy_box_location(self, heavy_box_id):
        return self.heavy_box_locations[heavy_box_id]['location']
    
    def set_heavy_box_location(self, heavy_box_id, location):
        self.heavy_box_locations[heavy_box_id]['location'] = location
        
    def get_random_location(self):
        return self.grid[np.random.randint(len(self.grid))]
    
    def get_random_location_not_on_boxes(self):
        loc = self.get_random_location()
        for i in range(self.num_light_boxes):
            if np.array_equal(loc, self.init_light_box_locations[f'light_box_{i}']['location']):
                return self.get_random_location_not_on_boxes()
        for i in range(self.num_heavy_boxes):
            if np.array_equal(loc, self.init_heavy_box_locations[f'heavy_box_{i}']['location']):
                return self.get_random_location_not_on_boxes()
        return loc
    
    def can_push_light_box(self, agent_id, box_id):
        return int(np.array_equal(self.get_agent_location(agent_id), self.get_light_box_location(box_id)))
    
    def can_push_heavy_box(self, agent_id, box_id):
        return int(np.array_equal(self.get_agent_location(agent_id), self.get_heavy_box_location(box_id)))
    
    def move_agent(self, agent_id, direction):

        agent_location = self.get_agent_location(agent_id)
        agent_x, agent_y = agent_location[0], agent_location[1]

        if direction == 'Left':
            agent_x -= 1
        elif direction == 'Right':
            agent_x += 1
        elif direction == 'Up':
            agent_y += 1
        elif direction == 'Down':
            agent_y -= 1

        if agent_x < 0 or agent_x >= self.grid_size[0]:
            agent_x = agent_location[0]
        if agent_y < 0 or agent_y >= self.grid_size[1]:
            agent_y = agent_location[1]

        self.set_agent_location(agent_id, np.array([agent_x, agent_y]))
    
    def move_light_box(self, light_box_id, direction):
            
            light_box_location = self.get_light_box_location(light_box_id)

            if self.is_light_box_done(light_box_id):
                return
            
            light_box_x, light_box_y = light_box_location[0], light_box_location[1]
    
            if direction == 'Left':
                light_box_x -= 1
            elif direction == 'Right':
                light_box_x += 1
            elif direction == 'Up':
                light_box_y += 1
            elif direction == 'Down':
                light_box_y -= 1
    
            if light_box_x < 0 or light_box_x >= self.grid_size[0]:
                light_box_x = light_box_location[0]
            if light_box_y < 0 or light_box_y >= self.grid_size[1]:
                light_box_y = light_box_location[1]

            self.set_light_box_location(light_box_id, np.array([light_box_x, light_box_y]))

            if self.is_target_location(self.get_light_box_location(light_box_id)):
                self.light_box_locations[light_box_id]['done'] = True
    
    def move_heavy_box(self, heavy_box_id, direction):

        heavy_box_location = self.get_heavy_box_location(heavy_box_id)

        if self.is_heavy_box_done(heavy_box_id):
            return
        
        heavy_box_x, heavy_box_y = heavy_box_location[0], heavy_box_location[1]

        if direction == 'Left':
            heavy_box_x -= 1
        elif direction == 'Right':
            heavy_box_x += 1
        elif direction == 'Up':
            heavy_box_y += 1
        elif direction == 'Down':
            heavy_box_y -= 1

        if heavy_box_x < 0 or heavy_box_x >= self.grid_size[0]:
            heavy_box_x = heavy_box_location[0]
        if heavy_box_y < 0 or heavy_box_y >= self.grid_size[1]:
            heavy_box_y = heavy_box_location[1]

        self.set_heavy_box_location(heavy_box_id, np.array([heavy_box_x, heavy_box_y]))

        if self.is_target_location(self.get_heavy_box_location(heavy_box_id)):
            self.heavy_box_locations[heavy_box_id]['done'] = True
    
    def sense_light_box(self, agent_id, box_id):
        return int(np.array_equal(self.get_agent_location(agent_id), self.get_light_box_location(box_id)))

    def sense_heavy_box(self, agent_id, box_id):
        return int(np.array_equal(self.get_agent_location(agent_id), self.get_heavy_box_location(box_id)))
    
    def is_game_over(self):
        return all([self.is_light_box_done(light_box_id) for light_box_id in self.light_box_ids]) and all([self.is_heavy_box_done(heavy_box_id) for heavy_box_id in self.heavy_box_ids])
    
    def is_light_box_done(self, box_id):
        return self.light_box_locations[box_id]['done']
    
    def is_heavy_box_done(self, box_id):
        return self.heavy_box_locations[box_id]['done']
    
    def get_num_light_boxes(self):
        return self.num_light_boxes
    
    def get_num_heavy_boxes(self):
        return self.num_heavy_boxes