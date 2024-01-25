from typing import Optional

import lightweaver as lw

from .bc_provider import PromBcProvider


class PromBc(lw.BoundaryCondition):
    """
    Abstract base class for boundary conditions
    """

    def __init__(
        self,
        projection: str,
        bc_provider: PromBcProvider,
        altitude_m: float,
        prom_upper_lower: Optional[str] = None,
        **kwargs
    ):
        pass
