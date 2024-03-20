import unittest

from random_events.variables import Continuous

from probabilistic_model.distributions.exponential import ExponentialDistribution
import plotly.graph_objects as go


class ExponentialDistributionTestCase(unittest.TestCase):

    x = Continuous('x')
    model = ExponentialDistribution(x, location=1, scale=2)

    def test_pdf(self):
        self.assertAlmostEqual(self.model.pdf(1), 0.5)

    def test_cdf(self):
        self.assertEqual(self.model.cdf(1), 0)
        self.assertEqual(self.model.cdf(float("inf")), 1)

    def test_sampling(self):
        samples = self.model.sample(100)
        for sample in samples:
            self.assertTrue(self.model.pdf(sample[0]) > 0)

    def test_plot(self):
        fig = go.Figure(self.model.plot(), self.model.plotly_layout())
        # fig.show()


if __name__ == '__main__':
    unittest.main()
