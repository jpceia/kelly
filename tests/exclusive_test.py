import unittest
import numpy as np
from kelly import exclusive

class Test(unittest.TestCase):

    def test_exclusive1(self):
        p = np.array([0.25, .40, 0.35])
        q = np.array([0.23, .40, 0.37])

        # check normalization
        p /= p.sum()
        q /= q.sum()
        
        o = 1 / q
        b = exclusive(o, p)
        b_expected = [0.02597403, 0.01038961, 0.]
        np.testing.assert_allclose(b, b_expected, atol=1e-8)

        o = 0.95 / q
        b = exclusive(o, p)
        b_expected = [0.00789474, 0., 0.]
        np.testing.assert_allclose(b, b_expected, atol=1e-8)

        # zero investment
        o = 0.90 / q
        b = exclusive(o, p)
        b_expected = [0., 0., 0.]
        np.testing.assert_allclose(b, b_expected, atol=1e-8)

        # arbitrageable
        o = 1.05 / q
        self.assertRaises(ValueError, exclusive, o, p)


if __name__ == "__main__":
    unittest.main()
