#############
Distributions
#############

At the core of each probabilistic models are known distributions. This package implements the following distributions

-  :class:`probabilistic_model.distributions.distributions.SymbolicDistribution`
-  :class:`probabilistic_model.distributions.distributions.IntegerDistribution`
-  :class:`probabilistic_model.distributions.multinomial.MultinomialDistribution`
-  :class:`probabilistic_model.distributions.distributions.DiracDeltaDistribution`
-  :class:`probabilistic_model.distributions.uniform.UniformDistribution`
-  :class:`probabilistic_model.distributions.gaussian.GaussianDistribution`
-  :class:`probabilistic_model.distributions.gaussian.TruncatedGaussianDistribution`

There is plenty of literature for each of those distributions, hence I will present only the interface to univariate
distributions in general.

Univariate Distributions
========================

.. autoapi-inheritance-diagram:: probabilistic_model.distributions.distributions.UnivariateDistribution
    :parts: 1

The :class:`probabilistic_model.distributions.distributions.UnivariateDistribution` class extends probabilistic models
by the following properties/methods

- :attr:`probabilistic_model.distributions.distributions.UnivariateDistribution.representation`
- :meth:`probabilistic_model.distributions.distributions.UnivariateDistribution.pdf`
- :meth:`probabilistic_model.distributions.distributions.UnivariateDistribution.plot`

Continuous Distributions
************************

.. autoapi-inheritance-diagram:: probabilistic_model.distributions.distributions.ContinuousDistribution
    :parts: 1

The :class:`probabilistic_model.distributions.distributions.ContinuousDistribution` class extends univariate
distributions by the straight forward method
:meth:`probabilistic_model.distributions.distributions.ContinuousDistribution.cdf`. Also a default implementation of
:meth:`probabilistic_model.distributions.distributions.ContinuousDistribution.plot` is provided that uses samples to
plot the pdf, cdf, expectation and mode.

A bit more interesting are the following methods:

- :meth:`probabilistic_model.distributions.distributions.ContinuousDistribution.conditional`
- :meth:`probabilistic_model.distributions.distributions.ContinuousDistribution.conditional_from_singleton`
- :meth:`probabilistic_model.distributions.distributions.ContinuousDistribution.conditional_from_simple_interval`
- :meth:`probabilistic_model.distributions.distributions.ContinuousDistribution.conditional_from_complex_interval`

These methods handle the creation of conditional distributions on the real line. The first one is a general handling
mechanism and will result in either of the latter three methods. The second creates a
:class:`probabilistic_model.distributions.distributions.DiracDeltaDistribution` and the last two have to be implemented
by the respective subclasses.

Note that for many distributions such as the Gaussian distribution it is, mathematically speaking, quite complicated to
provide a fully functional conditional implementation.
See `this example`_ to get an idea of what I am talking about.

.. _this example: examples/truncated_gaussians.ipynb


Discrete Distributions
**********************

.. autoapi-inheritance-diagram::
    probabilistic_model.distributions.distributions.IntegerDistribution
    probabilistic_model.distributions.distributions.SymbolicDistribution
    :parts: 1

The final part are discrete distributions such as the Symbolic and Integer distributions. This can be thought of as
tabular distributions over discrete variables.