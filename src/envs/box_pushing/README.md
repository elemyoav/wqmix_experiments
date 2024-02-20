# Box Pushing Environment

## Overview
"Box Pushing" is a grid-based environment where agents cooperate to push boxes to a target location. Boxes vary in weight, with some requiring collaboration between agents to move.

## Environment Setup
- **Grid Size**: Customizable grid dimensions.
- **Box Types**: Light (movable by one agent) and heavy (requires at least two agents).
- **Initialization**: Agent positions, box locations, and the target location are randomized at the start and remain constant.

## Actions
Agents can:
- **Do nothing**.
- **Move**: Up, down, left, or right.
- **Sense**: Check if a box is in the same location (sensing is infallible).
- **Push**: Attempt to move a box in their location.

## Constraints
- Sensing and pushing are only possible if the agent is in the same location as the box.
- Heavy boxes require a minimum of two agents to push.

## Success Probabilities
- `p_push`: Probability of successfully pushing a box.
- `p_sense`: Probability of successfully sensing a box (default is 1.0, meaning sensing always succeeds).

## Observations
Each agent receives:
- Their location on the grid.
- A boolean indicating if they sensed a box in the last turn.
- The target location.

## State
Includes all agent positions, box locations, and the target location.

## Parameters
Environment configuration (`env_args`) can be customized with:
- `num_agents`: Number of agents (default 2).
- `grid_size`: Grid dimensions (default `(2,2)`).
- `num_light_boxes`: Number of light boxes (default 1).
- `num_heavy_boxes`: Number of heavy boxes (default 1).
- `p_push`: Push success probability (default 0.8).
- `p_sense`: Sense success probability (default 1.0).
- `horizon`: Maximum number of steps per episode (default 300).

## Rewards
- `IDLE_REWARD = 0`
- `MOVE_AGENT_REWARD = -10`
- `SENSE_BOX_REWARD = -1` (applies to sensing any box)
- `PUSH_LIGHT_BOX_REWARD = -30`
- `PUSH_HEAVY_BOX_REWARD = -20`
- `PUSH_LIGHT_BOX_SUCC_REWARD = 500`
- `PUSH_HEAVY_BOX_SUCC_REWARD = 1000`

## Objective
Maximize rewards by efficiently pushing all boxes to the target location, balancing the need for action with the imperative of cooperation.
