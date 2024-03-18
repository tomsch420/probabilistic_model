from ..distributions.distributions import UniformDistribution, GaussianDistribution, TruncatedGaussianDistribution
from ..probabilistic_circuit import ProbabilisticCircuitMixin
from ..distributions.distributions import DiracDeltaDistribution


class Convolution:
    """
    Abstract class to implement convolutions from distributions of independent variables."""

    distribution: ProbabilisticCircuitMixin
    """
    The distribution that is the left side of the addition
    """

    def __init__(self, distribution: ProbabilisticCircuitMixin):
        self.distribution = distribution

    def convolve_with_dirac_delta(self, other: DiracDeltaDistribution) -> ProbabilisticCircuitMixin:
        """
        Convolve this distribution with a dirac delta.

        :param other: The dirac delta distribution to convolve with.
        """
        raise NotImplementedError

    def convolve_with_gaussian(self, other: GaussianDistribution) -> ProbabilisticCircuitMixin:
        """
        Convolve this distribution with a gaussian.

        :param other: The gaussian distribution to convolve with.
        """
        raise NotImplementedError


class UniformDistributionConvolution(Convolution):

    distribution: UniformDistribution

    def convolve_with_dirac_delta(self, other: DiracDeltaDistribution) -> UniformDistribution:
        new_interval = self.distribution.interval.replace(lower=self.distribution.interval.lower + other.location,
                                                          upper=self.distribution.interval.upper + other.location)
        return UniformDistribution(self.distribution.variable, new_interval)


class DiracDeltaDistributionConvolution(Convolution):

    distribution: DiracDeltaDistribution

    def convolve_with_dirac_delta(self, other: DiracDeltaDistribution) -> DiracDeltaDistribution:
        return DiracDeltaDistribution(self.distribution.variable, self.distribution.location + other.location,
                                      self.distribution.density_cap)


class GaussianDistributionConvolution(Convolution):

    distribution: GaussianDistribution

    def convolve_with_dirac_delta(self, other: DiracDeltaDistribution) -> GaussianDistribution:
        return GaussianDistribution(self.distribution.variable, self.distribution.mean + other.location,
                                    self.distribution.scale)

    def convolve_with_gaussian(self, other: GaussianDistribution) -> GaussianDistribution:
        return GaussianDistribution(self.distribution.variable, self.distribution.mean + other.mean,
                                    self.distribution.scale + other.scale)


class TruncatedGaussianDistributionConvolution(Convolution):

    distribution: TruncatedGaussianDistribution

    def convolve_with_dirac_delta(self, other: DiracDeltaDistribution) -> TruncatedGaussianDistribution:
        new_interval = self.distribution.interval.replace(lower=self.distribution.interval.lower + other.location,
                                                          upper=self.distribution.interval.upper + other.location)
        return TruncatedGaussianDistribution(self.distribution.variable, new_interval,
                                             self.distribution.mean + other.location, self.distribution.scale)
