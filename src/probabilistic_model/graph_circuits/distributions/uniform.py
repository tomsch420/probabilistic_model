from ...distributions.uniform import UniformDistribution as PMUniformDistribution
from .distributions import ContinuousDistribution


class UniformDistribution(ContinuousDistribution, PMUniformDistribution):
    ...
