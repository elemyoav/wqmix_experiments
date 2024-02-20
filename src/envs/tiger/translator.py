OPEN_LEFT = 0
OPEN_RIGHT = 1
LISTEN_LEFT = 2
LISTEN_RIGHT = 3


class Translator:

    def __init__(self):
        pass
    
    def is_open_left(self, action):
        return action == OPEN_LEFT
    
    def is_open_right(self, action):
        return action == OPEN_RIGHT
    
    def is_listen_left(self, action):
        return action == LISTEN_LEFT
    
    def is_listen_right(self, action):
        return action == LISTEN_RIGHT

    def is_listen_action(self, action):
        return self.is_listen_left(action) or self.is_listen_right(action)
    
    def is_open_action(self, action):
        return self.is_open_left(action) or self.is_open_right(action)
    
    def are_listen_actions(self, actions):
        return all([self.is_listen_action(action) for action in actions])
    
    def are_open_actions(self, actions):
        return all([self.is_open_action(action) for action in actions])
    
    def get_open_left_action(self):
        return OPEN_LEFT
    
    def get_open_right_action(self):
        return OPEN_RIGHT
    
    def get_listen_left_action(self):
        return LISTEN_LEFT
    
    def get_listen_right_action(self):
        return LISTEN_RIGHT
    
    def get_all_actions(self):
        return ['OPEN_LEFT', 'OPEN_RIGHT', 'LISTEN_LEFT', 'LISTEN_RIGHT']