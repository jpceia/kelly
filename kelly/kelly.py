import numpy as np


def bernoulli(odd, prob, prob_err=None):
    b = (odd * prob - 1) / (odd - 1)
    k = 1
    if prob_err is not None:
        s = (1 + 1 / odd) * prob_err
        k = b * b / (b * b + s * s)
    return k * b
