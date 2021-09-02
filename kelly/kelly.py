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

def bernoulli_exp(odd, p, a=1, q=None):
    """
    Solution for the exponential case:
    utility function
        u(x) = (1 - e^(ax)) / a
    """
    if q is None:
        q = 1 - p
    return (np.log(odd - 1) + np.log(p) - np.log(q)) / (odd * a)

def bernoulli_pow(odd, p, a=0.5, q=None):
    """
    Solution for the power case:
    utility function:
        u(x) = x^(1-a) / (1 - a)
    """
    if q is None:
        q = 1 - p
    P = np.power(p, 1 / a)
    Q = np.power(q, 1 / a)
    return 1 / (1 + Q * odd / (P * np.power(odd - 1, 1 / a) - Q))


def exclusive(o, p):
    """
    Exclusive Kelly Algorithm:
    1.  Calculated expected revenues:
        E[r_i] = p_i * o_i
    2.  Reorder the indexes so that the sequence E[r_i] is nonincreasing
    3.  Set S = [], i = 1 and R = 1
    4.  Repeat
        if E[r_i] > R:
            insert i in S
            R := 1 - (sum_{not S} p_k) / (1 - sum_{S} (1 / o_i))
        else:
            break
    5.  f_i := p_i - (1 / o_i) * (sum_{not S} p_k) / (1 - sum_{S} (1 / o_i))
    """
    rev, q = p * o, 1 / o
    
    if (rev <= 1).all():
        return np.zeros_like(p)
    if q.sum() < 1:
        raise ValueError("Arbitrageable")

    idx = np.argsort(-rev)
    tmp_p = 1 - np.cumsum(p[idx][:-1])
    tmp_q = 1 - np.cumsum(q[idx][:-1])
    R = np.insert(tmp_p / tmp_q, 0, 1)
    i = np.argmin(rev[idx] > R)
    return np.maximum((rev - R[i]) * q, 0)
