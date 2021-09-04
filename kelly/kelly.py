import numpy as np


def bernoulli(odd, p, err=0):
    """
    Solution for the classical case of the kelly criterion:
    utility function
        u(x) = log(x)
    """
    b = (odd * p - 1) / (odd - 1)
    k = 1
    if err > 0:
        s = (odd / (odd - 1)) * err
        k = b * b / (b * b + s * s)
    return k * b


def bernoulli_exp(odd, p, q=None):
    """
    Solution for the exponential case:
    utility function
        u(x) = -e^(-x)
    """
    if q is None:
        q = 1 - p
    return (np.log(odd - 1) + np.log(p) - np.log(q)) / odd


def bernoulli_pow(odd, p, a=0.5, q=None):
    """
    Solution for the power case:
    utility function:
        u(x) = (x^(1-a) - 1) / (1 - a)
    """
    if q is None:
        q = 1 - p
    P = np.power(p, 1 / a)
    Q = np.power(q, 1 / a)
    return 1 / (1 + Q * odd / (P * np.power(odd - 1, 1 / a) - Q))
