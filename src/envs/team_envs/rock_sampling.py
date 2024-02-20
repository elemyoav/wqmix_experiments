from ..rock_sampling.rock_sampling import DecRockSampling
from .to_team import TeamEnv


class RockSampling(TeamEnv):
    def __init__(self, batch_size=None, **kwargs):
        super().__init__(DecRockSampling(batch_size, **kwargs))