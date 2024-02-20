from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env
from .matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt
from .box_pushing.box_pushing import DecBoxPushing
from .MAPF import MAPF
from .tiger.tiger import DecTiger
from .rock_sampling.rock_sampling import DecRockSampling
from .team_envs.tiger import Tiger
from .team_envs.box_pushing import BoxPushing
from .team_envs.rock_sampling import RockSampling

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except:
    gfootball = False

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["box_pushing"] = partial(env_fn, env=DecBoxPushing)
REGISTRY["MAPF"] = partial(env_fn, env=MAPF)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)
REGISTRY["tiger"] = partial(env_fn, env=DecTiger)
REGISTRY["rock_sampling"] = partial(env_fn, env=DecRockSampling)
REGISTRY["team_tiger"] = partial(env_fn, env=Tiger)
REGISTRY["team_box_pushing"] = partial(env_fn, env=BoxPushing)
REGISTRY["team_rock_sampling"] = partial(env_fn, env=RockSampling)

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
