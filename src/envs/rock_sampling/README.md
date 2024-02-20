# Rock Sampling Environment

## Introduction
In the Rock Sampling environment, two rovers are tasked with identifying and sampling good quality rocks within a given grid. Each rock in the grid has a quality level, and the rovers must sample good quality rocks while avoiding bad quality ones. The grid is divided into three areas: Rover1 Area, Rover2 Area, and the Shared Area. Each rover can operate in all areas except the one designated for the other rover.

## Configuration
- **Grid Configuration (`grid_config`)**:
  - **Width**: Width of the entire grid.
  - **Rover1 Height**: Height of the Rover1 Area.
  - **Shared Height**: Height of the Shared Area.
  - **Rover2 Height**: Height of the Rover2 Area.
  - **Number of Rocks**: `num_rocks`.
  - **Observation Quality Function**: A function that accepts two coordinate inputs (positions) and returns a probability (between 0 and 1) indicating the likelihood of correctly observing the actual rock quality.
- **Horizon**: `horizon` (integer).

## Action Space
Each rover can perform the following actions:
1. Remain idle.
2. Move in any direction (up, down, left, right).
3. Sense any rock.
4. Try to sample any rock.

## Rewards
- **Idle Action**: 0 points.
- **Move Action**: -1 point.
- **Sense Action**: -5 points.
- **Sampling a Bad Rock or Sampling at a Wrong Position**: -500 points.
- **Sampling a Good Rock**: 0 points.
- **Rover1 Completes Sampling All Good Quality Rocks in Its Area**: +750 points.
- **Rover2 Completes Sampling All Good Quality Rocks in Its Area**: +750 points.
- **Both Rovers Complete Sampling All Good Quality Rocks in the Shared Area**: +750 points for each.
> **Note**: All the reward function values can be changed in the [`rewards.py`](rewards.py) file in this directory

## Observation Space
- **Position**: The rover's current position on the grid.
- **Rock Quality**: The quality of the sampled rock; 0 for no sample, 1 for bad quality, 2 for good quality.

