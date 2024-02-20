import numpy as np

def default_observation_quality_function(pos1, pos2):
    """
    summary: Default observation quality function. Gives a more accurate observation if the rover is closer to the rock.
    input: pos1, pos2: tuple of (x, y) coordinates
    output: float between 0 and 1
    """
    
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    return np.exp(-np.linalg.norm(pos1 - pos2))

DEFAULT_CONFIG = {
        'grid_config': {
            'width': 5,
            'rover1_height': 3,
            'shared_height': 2,
            'rover2_height': 3,
            'num_rocks_rover_1': 2,
            'num_rocks_rover_2': 2,
            'num_rocks_shared': 2,
            'observation_quality_function': default_observation_quality_function
        },
        'horizon': 3000
    }