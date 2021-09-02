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

    def test_bernoulli_exp(self):
        x = bernoulli_exp(3, 0.5)
        self.assertAlmostEqual(x, 0.23104906018664842)

        x = bernoulli_exp(5, 0.33)
        self.assertAlmostEqual(x, 0.13562186063908097)

        x = bernoulli_exp(3, 0.5, 0.9)
        self.assertAlmostEqual(x, 0.2567211779851649)

        x = bernoulli_exp(3, 0.5, 1.1)
        self.assertAlmostEqual(x, 0.21004460016968038)

        x = bernoulli_exp(2, 0.5, q=0.4)
        self.assertAlmostEqual(x, 0.11157177565710485)

        x = bernoulli_exp(2, 0.5, 0.9, q=0.4)
        self.assertAlmostEqual(x, 0.1239686396190054)

        x = bernoulli_exp(2, 0.5, 1.1, q=0.4)
        self.assertAlmostEqual(x, 0.10142888696100441)

    def test_bernoulli_pow(self):
        x = bernoulli_pow(3, 0.5, 1)
        self.assertAlmostEqual(x, 0.25)

        x = bernoulli_pow(5, 0.33, 1)
        self.assertAlmostEqual(x, 0.1625)

        x = bernoulli_pow(3, 0.5)
        self.assertAlmostEqual(x, 0.5)

        x = bernoulli_pow(3, 0.5, 0.9)
        self.assertAlmostEqual(x, 0.27886686523782495)

        x = bernoulli_pow(2, 0.5, q=0.4)
        self.assertAlmostEqual(x, 0.21951219512195114)

        x = bernoulli_pow(2, 0.5, 0.9, q=0.4)
        self.assertAlmostEqual(x, 0.12333746012026381)


if __name__ == "__main__":
    unittest.main()
