import numpy as np


def bernoulli(odd, prob, prob_err=None):
    b = (odd * prob - 1) / (odd - 1)
    k = 1
    if prob_err is not None:
        s = (1 + 1 / odd) * prob_err
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
    i = np.argmin(rev[idx] > R) - 1
    return np.maximum((rev - R[i]) * q, 0)
