import unittest
import numpy as np
from kelly import exclusive, exclusive_exp, exclusive_pow

class Test(unittest.TestCase):

    p = np.array([
        0.113200854,
        0.068957507,
        0.096069411,
        0.156693717,
        0.013001072,
        0.013012409,
        0.158214192,
        0.209907307,
        0.170443998,
        0.000499532
    ])

    o = np.array([
        8.53751818,
        13.31713933,
        10.03821446,
        6.075078932,
        64.33368976,
        63.40641769,
        6.224278224,
        5.145337772,
        5.993057062,
        1518.997446
    ])

    def test_exclusive(self):
        o = self.o
        x = exclusive(o, self.p)
        np.testing.assert_allclose(x, [
            0.00021243,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.00323386,
            0.02242871,
            0.00948429,
            0.
        ], atol=1e-8)

        o = 0.95 * self.o
        x = exclusive(o, self.p)
        np.testing.assert_allclose(x, [
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.00669787,
            0.,
            0.
        ], atol=1e-8)

        # zero investment
        o = 0.90 * self.o
        x = exclusive(o, self.p)
        np.testing.assert_allclose(x, np.zeros(10), atol=1e-8)

        # arbitrageable
        o = 1.05 * self.o
        self.assertRaises(ValueError, exclusive, o, self.p)

    def test_exclusive_exp(self):
        o = self.o
        x = exclusive_exp(o, self.p)
        np.testing.assert_allclose(x, [
            0.00022001,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.0033179,
            0.02196189,
            0.00955317,
            0.
        ], atol=1e-8)

        o = 0.95 * self.o
        x = exclusive_exp(o, self.p)
        np.testing.assert_allclose(x, [
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.00663429,
            0.,
            0.
        ], atol=1e-8)

        # zero investment
        o = 0.90 * self.o
        x = exclusive_exp(o, self.p)
        np.testing.assert_allclose(x, np.zeros(10), atol=1e-8)

        # arbitrageable
        o = 1.05 * self.o
        self.assertRaises(ValueError, exclusive_exp, o, self.p)

    def test_exclusive_pow_sqrt(self):
        o = self.o
        x = exclusive_pow(o, self.p, 0.5)
        np.testing.assert_allclose(x, [
            0.00040943,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.0062919,
            0.04577072,
            0.01880043,
            0.        
        ], atol=1e-8)

        o = 0.95 * self.o
        x = exclusive_pow(o, self.p, 0.5)
        np.testing.assert_allclose(x, [
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.01352295,
            0.,
            0.
        ], atol=1e-8)

        # zero investment
        o = 0.90 * self.o
        x = exclusive_pow(o, self.p, 0.5)
        np.testing.assert_allclose(x, np.zeros(10), atol=1e-8)

        # arbitrageable
        o = 1.05 * self.o
        self.assertRaises(ValueError, exclusive_pow, o, self.p, 0.5)

    def test_exclusive_pow_square(self):
        o = self.o
        x = exclusive_pow(o, self.p, 2)
        np.testing.assert_allclose(x, [
            0.00010812,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.00163819,
            0.0110982,
            0.00476031,
            0.        
        ], atol=1e-8)

        o = 0.95 * self.o
        x = exclusive_pow(o, self.p, 2)
        np.testing.assert_allclose(x, [
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.00333304,
            0.,
            0.
        ], atol=1e-8)

        # zero investment
        o = 0.90 * self.o
        x = exclusive_pow(o, self.p, 2)
        np.testing.assert_allclose(x, np.zeros(10), atol=1e-8)

        # arbitrageable
        o = 1.05 * self.o
        self.assertRaises(ValueError, exclusive_pow, o, self.p, 2)


if __name__ == "__main__":
    unittest.main()
