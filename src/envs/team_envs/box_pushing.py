from ..box_pushing.box_pushing import DecBoxPushing
from .to_team import TeamEnv


class BoxPushing(TeamEnv):
    
    def __init__(self, batch_size=None, **kwargs):
        super().__init__(DecBoxPushing(batch_size, **kwargs))