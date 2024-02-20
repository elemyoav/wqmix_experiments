# Tiger Environment

## Overview
The "Tiger" environment simulates a scenario with two agents and two doors. Behind one door lies a treasure, and behind the other, a tiger. The agents must decide whether to open a door or attempt to listen for the tiger's presence behind the doors.

## Actions
Agents can perform one of the following actions:
- `open left`: Open the left door.
- `open right`: Open the right door.
- `listen left`: Listen for the tiger behind the left door.
- `listen right`: Listen for the tiger behind the right door.

## Action Details
- **Listening is noisy**: Listening for the tiger has a 75% success rate of correctly identifying the tiger's location.

## States
The state of the environment is represented as:
- `[1 0]`: Tiger is behind the left door.
- `[0 1]`: Tiger is behind the right door.

## Observation Space
Each agent's observation space is:
- `[0 0]`: No belief about the tiger's location (default).
- `[1 0]`: Belief that the tiger is behind the left door.
- `[0 1]`: Belief that the tiger is behind the right door.

## Rewards
- **+10**: For opening a door with gold.
- **-50**: For opening a door with a tiger.
- **-1**: For each listen action.

## Episode Horizon
- Each episode is fixed to a horizon of 6 actions.

## Objective
The goal for each agent is to maximize their total reward by strategically choosing actions based on their beliefs about the tiger's location.
