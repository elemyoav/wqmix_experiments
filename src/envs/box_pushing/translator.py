import copy

DIRECTIONS2ACTIONS = {
    'Left': 0,
    'Right': 1,
    'Up': 2,
    'Down': 3,
}

ACTIONS2DIRECTIONS = {
    0: 'Left',
    1: 'Right',
    2: 'Up',
    3: 'Down',
}

BOXPUSHERSINITVALUE = {
    'Left': [],
    'Right': [],
    'Up': [],
    'Down': []
}

class Translator:

    def __init__(self,num_agents, num_light_boxes, num_heavy_boxes):
        self.num_agents = num_agents
        self.num_light_boxes = num_light_boxes
        self.num_heavy_boxes = num_heavy_boxes
        self.num_boxes = num_light_boxes + num_heavy_boxes
        self.light_boxes_ids = [f'light_box_{i}' for i in range(self.num_light_boxes)]
        self.heavy_boxes_ids = [f'heavy_box_{i}' for i in range(self.num_heavy_boxes)]

        self.reset_box_pushers()

    def reset_box_pushers(self):
        self.light_box_pushers = {
            f'light_box_{i}': copy.deepcopy(BOXPUSHERSINITVALUE) for i in range(self.num_light_boxes)
        }

        self.heavy_box_pushers = {
            f'heavy_box_{i}': copy.deepcopy(BOXPUSHERSINITVALUE) for i in range(self.num_heavy_boxes)
        }

    def get_idle_action(self):
        return 0
    
    def get_move_agent_action(self, direction):
        return DIRECTIONS2ACTIONS[direction] + 1
    
    def get_move_agent_direction(self, action):
        return ACTIONS2DIRECTIONS[action - 1] 
    
    def get_sense_light_box_action(self, box_num):
        return 5 + box_num
    
    def get_sense_light_box_num(self, action):
        return action - 5
    
    def get_sense_heavy_box_action(self, box_num):
        return 5 + self.num_light_boxes + box_num
    
    def get_sense_heavy_box_num(self, action):
        return action - 5 - self.num_light_boxes
    
    def get_push_light_box_action(self, box_num, direction):
        return 5 + self.num_boxes + 4 * box_num + DIRECTIONS2ACTIONS[direction]
    
    def get_push_light_box_num(self, action):
        return (action - 5 - self.num_boxes) // 4
    
    def get_push_light_box_direction(self, action):
        return ACTIONS2DIRECTIONS[(action - 5 - self.num_boxes) % 4]
    
    def get_push_heavy_box_action(self, box_num, direction):
        return 5 + self.num_boxes + 4 * self.num_light_boxes + 4 * box_num + DIRECTIONS2ACTIONS[direction]
    
    def get_push_heavy_box_num(self, action):
        return (action - 5 - self.num_boxes - 4 * self.num_light_boxes) // 4
    
    def get_push_heavy_box_direction(self, action):
        return ACTIONS2DIRECTIONS[(action - 5 - self.num_boxes - 4 * self.num_light_boxes) % 4]
    
    def is_idle_action(self, action):
        return action == 0
    
    def is_move_agent_action(self, action):
        return action >= 1 and action <= 4
    
    def is_sense_light_box_action(self, action):
        return action >= 5 and action < 5 + self.num_light_boxes
    
    def is_sense_heavy_box_action(self, action):
        return action >= 5 + self.num_light_boxes and action < 5 + self.num_boxes
    
    def is_push_light_box_action(self, action):
        return action >= 5 + self.num_boxes and action < 5 + self.num_boxes + 4 * self.num_light_boxes
    
    def is_push_heavy_box_action(self, action):
        return action >= 5 + self.num_boxes + 4 * self.num_light_boxes and action < 5 + self.num_boxes + 4 * self.num_boxes
    
    def add_light_box_pusher(self, box_id, direction, agent_id):
        self.light_box_pushers[box_id][direction].append(agent_id)
    
    def add_heavy_box_pusher(self, box_id, direction, agent_id):
        self.heavy_box_pushers[box_id][direction].append(agent_id)

    
    def get_box_pushing_directions(self):
        light_boxes_directions = {light_box_id:[] for light_box_id in self.light_boxes_ids} 
        heavy_boxes_directions = {heavy_box_id:[] for heavy_box_id in self.heavy_boxes_ids}
        light_boxes_succ_pushers = {light_box_id:[] for light_box_id in self.light_boxes_ids} 
        heavy_boxes_succ_pushers = {heavy_box_id:[] for heavy_box_id in self.heavy_boxes_ids}

        for light_box_id in self.light_boxes_ids:
            light_boxes_directions[light_box_id] = []

            num_left_pushers = len(self.light_box_pushers[light_box_id]['Left'])
            num_right_pushers = len(self.light_box_pushers[light_box_id]['Right'])
            num_up_pushers = len(self.light_box_pushers[light_box_id]['Up'])
            num_down_pushers = len(self.light_box_pushers[light_box_id]['Down'])

            push_left = num_left_pushers - num_right_pushers

            if push_left >= 1:
                light_boxes_directions[light_box_id].append('Left')
                light_boxes_succ_pushers[light_box_id] += self.light_box_pushers[light_box_id]['Left']
            elif push_left <= -1:
                light_boxes_directions[light_box_id].append('Right')
                light_boxes_succ_pushers[light_box_id] += self.light_box_pushers[light_box_id]['Right']
            push_up = num_up_pushers - num_down_pushers

            if push_up >= 1:
                light_boxes_directions[light_box_id].append('Up')
                light_boxes_succ_pushers[light_box_id] += self.light_box_pushers[light_box_id]['Up']
            elif push_up <= -1:
                light_boxes_directions[light_box_id].append('Down')
                light_boxes_succ_pushers[light_box_id] += self.light_box_pushers[light_box_id]['Down']
        
        for heavy_box_id in self.heavy_boxes_ids:
            heavy_boxes_directions[heavy_box_id] = []

            num_left_pushers = len(self.heavy_box_pushers[heavy_box_id]['Left'])
            num_right_pushers = len(self.heavy_box_pushers[heavy_box_id]['Right'])
            num_up_pushers = len(self.heavy_box_pushers[heavy_box_id]['Up'])
            num_down_pushers = len(self.heavy_box_pushers[heavy_box_id]['Down'])

            push_left = num_left_pushers - num_right_pushers

            if push_left >= 2:
                heavy_boxes_directions[heavy_box_id].append('Left')
                heavy_boxes_succ_pushers[heavy_box_id] += self.heavy_box_pushers[heavy_box_id]['Left']
            elif push_left <= -2:
                heavy_boxes_directions[heavy_box_id].append('Right')
                heavy_boxes_succ_pushers[heavy_box_id] += self.heavy_box_pushers[heavy_box_id]['Right']
            
            push_up = num_up_pushers - num_down_pushers

            if push_up >= 2:
                heavy_boxes_directions[heavy_box_id].append('Up')
                heavy_boxes_succ_pushers[heavy_box_id] += self.heavy_box_pushers[heavy_box_id]['Up']
            elif push_up <= -2:
                heavy_boxes_directions[heavy_box_id].append('Down')
                heavy_boxes_succ_pushers[heavy_box_id] += self.heavy_box_pushers[heavy_box_id]['Down']

        self.reset_box_pushers()
        return {
            'light_boxes_directions': light_boxes_directions,
            'heavy_boxes_directions': heavy_boxes_directions,
            'light_boxes_succ_pushers': light_boxes_succ_pushers,
            'heavy_boxes_succ_pushers': heavy_boxes_succ_pushers
        }
    
    def get_all_actions(self):
        actions = [ 'IDLE', 'MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP', 'MOVE_DOWN']
        actions += [f'SENSE_LIGHT_BOX_{i}' for i in range(self.num_light_boxes)]
        actions += [f'SENSE_HEAVY_BOX_{i}' for i in range(self.num_heavy_boxes)]

        for i in range(self.num_light_boxes):
            actions += [f'PUSH_LIGHT_BOX_{i}_LEFT', f'PUSH_LIGHT_BOX_{i}_RIGHT', f'PUSH_LIGHT_BOX_{i}_UP', f'PUSH_LIGHT_BOX_{i}_DOWN']
        
        for i in range(self.num_heavy_boxes):
            actions += [f'PUSH_HEAVY_BOX_{i}_LEFT', f'PUSH_HEAVY_BOX_{i}_RIGHT', f'PUSH_HEAVY_BOX_{i}_UP', f'PUSH_HEAVY_BOX_{i}_DOWN']
        
        return actions