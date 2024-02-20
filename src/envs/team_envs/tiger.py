from ..tiger.tiger import DecTiger
from .to_team import TeamEnv

class Tiger(TeamEnv):
    def __init__(self, batch_size=None, **kwargs):
        super().__init__(DecTiger(batch_size, **kwargs))