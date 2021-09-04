import unittest
import numpy as np
from kelly import exclusive, exclusive_exp

class Test(unittest.TestCase):

    def test_exclusive(self):
        p = np.array([0.25, .40, 0.35])
        q = np.array([0.23, .40, 0.37])

        # check normalization
        p /= p.sum()
        q /= q.sum()
        
        o = 1 / q
        b = exclusive(o, p)
        b_expected = [0.03243243243243242, 0.021621621621621623, 0.]
        np.testing.assert_allclose(b, b_expected, atol=1e-8)

        o = 0.95 / q
        b = exclusive(o, p)
        b_expected = [0.010416666666666614, 0., 0.]
        np.testing.assert_allclose(b, b_expected, atol=1e-8)

        # zero investment
        o = 0.90 / q
        b = exclusive(o, p)
        b_expected = [0., 0., 0.]
        np.testing.assert_allclose(b, b_expected, atol=1e-8)

        # arbitrageable
        o = 1.05 / q
        self.assertRaises(ValueError, exclusive, o, p)

    def test_exclusive_exp(self):
        p = np.array([0.25, .40, 0.35])
        q = np.array([0.23, .40, 0.37])

        # check normalization
        p /= p.sum()
        q /= q.sum()
        
        o = 1 / q

        b = exclusive_exp(o, p, .5)
        print(b[0], b[1])
        b_expected = [0.06391767164317638,0.04445588092384863, 0.]
        np.testing.assert_allclose(b, b_expected, atol=1e-8)

        b = exclusive_exp(o, p, 1)
        b_expected = [0.03195883582158819, 0.022227940461924316, 0.]
        np.testing.assert_allclose(b, b_expected, atol=1e-8)

        o = 0.95 / q
        b = exclusive_exp(o, p, 1)
        b_expected = [0.010303906648761062, 0., 0.]
        np.testing.assert_allclose(b, b_expected, atol=1e-8)

        # zero investment
        o = 0.90 / q
        b = exclusive_exp(o, p, 1)
        b_expected = [0., 0., 0.]
        np.testing.assert_allclose(b, b_expected, atol=1e-8)

        # arbitrageable
        o = 1.05 / q
        self.assertRaises(ValueError, exclusive_exp, o, p, 1)

if __name__ == "__main__":
    unittest.main()
