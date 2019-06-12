import unittest
from kelly import bernoulli

class Test(unittest.TestCase):

    def test_bernoulli(self):
        x = bernoulli(3, .5)
        self.assertAlmostEqual(x, 0.25)

        x = bernoulli(3, .5, .1)
        self.assertAlmostEqual(x, 0.19463667820069203)

        x = bernoulli(5, .33)
        self.assertAlmostEqual(x, 0.1625)

        x = bernoulli(5, .33, .1)
        self.assertAlmostEqual(x, 0.10515584316128047)


if __name__ == "__main__":
    unittest.main()
