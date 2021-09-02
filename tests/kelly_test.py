import unittest
from kelly import bernoulli

class Test(unittest.TestCase):

    def test_bernoulli(self):
        x = bernoulli(3, 0.5)
        self.assertAlmostEqual(x, 0.25)

        x = bernoulli(3, 0.5, err=0.1)
        self.assertAlmostEqual(x, 0.1838235294117647)

        x = bernoulli(5, 0.33)
        self.assertAlmostEqual(x, 0.1625)

        x = bernoulli(5, 0.33, err=0.1)
        self.assertAlmostEqual(x, 0.10209107806691455)


if __name__ == "__main__":
    unittest.main()
