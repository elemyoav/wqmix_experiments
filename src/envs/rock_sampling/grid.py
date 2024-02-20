
import random
from numpy import array

NULL_QUALITY = 0
BAD_QUALITY = 1
GOOD_QUALITY = 2

class Grid:

    def __init__(self, grid_config):

        self.num_rocks_rover_1 = grid_config['num_rocks_rover_1']
        self.num_rocks_rover_2 = grid_config['num_rocks_rover_2']
        self.num_rocks_shared = grid_config['num_rocks_shared']
        self.num_rocks = self.num_rocks_rover_1 + self.num_rocks_rover_2 + self.num_rocks_shared
        
        self.area_width = grid_config['width'] # This is shared across all agents
        self.rover1_area_height = grid_config['rover1_height']
        self.shared_area_height = grid_config['shared_height']
        self.rover2_area_height = grid_config['rover2_height']
        self.area_height = self.rover1_area_height + self.shared_area_height + self.rover2_area_height
        self.observation_quality_function = grid_config['observation_quality_function']

        self.grid = [ (x,y) for x in range(self.area_width) for y in range(self.area_height) ]
        self.rover_1_grid = [ (x,y) for x in range(self.area_width) for y in range(self.rover1_area_height) ]
        self.shared_grid = [ (x,y) for x in range(self.area_width) for y in range(self.rover1_area_height, self.rover1_area_height + self.shared_area_height) ]
        self.rover_2_grid = [ (x,y) for x in range(self.area_width) for y in range(self.rover1_area_height + self.shared_area_height, self.area_height) ]
        self.sample_rock_positions() # Rock positions are constants for all episodes
        self.sample_quality() # Rock quality is constant for all episodes
        self.original_rock_quality = self.rock_quality.copy()
        self.reset_board()

    
    def sample_rock_positions(self):
        self.rock_positions = random.sample(self.rover_1_grid, self.num_rocks_rover_1) + \
                              random.sample(self.rover_2_grid, self.num_rocks_rover_2) + \
                              random.sample(self.shared_grid, self.num_rocks_shared)
        
    def sample_quality(self):
        # make 1 rock in each grid bad quality
        self.rock_quality = [GOOD_QUALITY] * self.num_rocks
        self.rock_quality[random.choice(range(self.num_rocks_rover_1))] = BAD_QUALITY
        self.rock_quality[random.choice(range(self.num_rocks_rover_1, self.num_rocks_rover_1 + self.num_rocks_rover_2))] = BAD_QUALITY
        self.rock_quality[random.choice(range(self.num_rocks_rover_1 + self.num_rocks_rover_2, self.num_rocks_rover_1 + self.num_rocks_rover_2 + self.num_rocks_shared))] = BAD_QUALITY
    
    def reset_rock_quality(self):
        self.rock_quality = self.original_rock_quality.copy()

    def sample_rover1_position(self):
        self.rover1_position = random.choice(self.rover_1_grid)
    
    def sample_rover2_position(self):
        self.rover2_position = random.choice(self.rover_2_grid)

    def reset_board(self):

        self.sample_rover1_position()
        self.sample_rover2_position()
        self.reset_rock_quality()

    def get_rover1_position(self):
        return self.rover1_position
    
    def get_rover2_position(self):
        return self.rover2_position

    def get_num_rocks(self):
        return self.num_rocks
    
    def move_rover_1(self, direction):
        new_position = self.rover1_position

        if direction == 'Up':
            new_position = (self.rover1_position[0], self.rover1_position[1] + 1)
        elif direction == 'Down':
            new_position = (self.rover1_position[0], self.rover1_position[1] - 1)
        elif direction == 'Left':
            new_position = (self.rover1_position[0] - 1, self.rover1_position[1])
        elif direction == 'Right':
            new_position = (self.rover1_position[0] + 1, self.rover1_position[1])

        if new_position in self.rover_1_grid or new_position in self.shared_grid:
            self.rover1_position = new_position
    
    def move_rover_2(self, direction):
        new_position = self.rover2_position

        if direction == 'Up':
            new_position = (self.rover2_position[0], self.rover2_position[1] + 1)
        elif direction == 'Down':
            new_position = (self.rover2_position[0], self.rover2_position[1] - 1)
        elif direction == 'Left':
            new_position = (self.rover2_position[0] - 1, self.rover2_position[1])
        elif direction == 'Right':
            new_position = (self.rover2_position[0] + 1, self.rover2_position[1])

        if new_position in self.rover_2_grid or new_position in self.shared_grid:
            self.rover2_position = new_position

    
    # def sense_rock(self, rock_id):
    #     rock_position = self.rock_positions[rock_id]
    #     prob = self.observation_quality_function(self.rover1_position, rock_position)

    #     if random.random() < prob:
    #         return self.rock_quality[rock_id]
    #     else:
    #         return NULL_QUALITY
    
    def sample_rock(self, rover_id, rock_id):
        rock_position = self.rock_positions[rock_id]
        rover_pos = self.rover1_position if rover_id == 'agent_0' else self.rover2_position
        
        if rover_pos != rock_position:
            return BAD_QUALITY
        
        rock_quality = self.rock_quality[rock_id]

        if rock_quality == GOOD_QUALITY:
            self.rock_quality[rock_id] = BAD_QUALITY
            return GOOD_QUALITY
        else:
            return BAD_QUALITY
    
    def sense_rock(self, rover_id, rock_id):
        rock_position = self.rock_positions[rock_id]
        rover_pos = self.rover1_position if rover_id == 'agent_0' else self.rover2_position
        prob = self.observation_quality_function(rover_pos, rock_position)

        if random.random() < prob:
            return self.rock_quality[rock_id]
        else:
            return NULL_QUALITY
        
    def is_game_over(self):
        return self.is_rover1_area_clear() and self.is_rover2_area_clear() and self.is_shared_area_clear()
    
    def move_rover(self, rover_id, direction):
        if rover_id == 'agent_0':
            self.move_rover_1(direction)
        elif rover_id == 'agent_1':
            self.move_rover_2(direction)

    def get_rover_position(self, rover_id):
        if rover_id == 'agent_0':
            return array(self.rover1_position)
        elif rover_id == 'agent_1':
            return array(self.rover2_position)
        
    def is_rover1_area_clear(self):
        # check if all rocks in rover1 area are bad
        for rock_id, rock_position in enumerate(self.rock_positions):
            if rock_position in self.rover_1_grid:
                if self.rock_quality[rock_id] == GOOD_QUALITY:
                    return False
        return True
    
    def is_rover2_area_clear(self):
        # check if all rocks in rover2 area are bad
        for rock_id, rock_position in enumerate(self.rock_positions):
            if rock_position in self.rover_2_grid:
                if self.rock_quality[rock_id] == GOOD_QUALITY:
                    return False
        return True
    
    def is_shared_area_clear(self):
        # check if all rocks in shared area are bad
        for rock_id, rock_position in enumerate(self.rock_positions):
            if rock_position in self.shared_grid:
                if self.rock_quality[rock_id] == GOOD_QUALITY:
                    return False
        return True
    
    def get_rock_area(self, rock_id):
        rock_position = self.rock_positions[rock_id]
        if rock_position in self.rover_1_grid:
            return 'rover1'
        elif rock_position in self.rover_2_grid:
            return 'rover2'
        elif rock_position in self.shared_grid:
            return 'shared'
    
    def get_rock_position(self, rock_id):
        return array(self.rock_positions[rock_id])
    
    def get_rock_quality(self, rock_id):
        return self.rock_quality[rock_id]