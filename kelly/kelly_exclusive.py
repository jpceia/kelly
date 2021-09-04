import numpy as np

"""
    Tries to find the Kelly criterion for multiple (exclusive) outcomes.
    This comes from maximizing the following function:
        
        L(x)    =   sum_i p_i * u(1 - sum_k x_k + o_i * x_i)

    subject to      x_i >= 0    for all i
                    sum_k x_k <= 1
"""

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
            R := 1 - (sum_{not S} p_k) / (1 - sum_{S} (1 / o_k))
        else:<
            break
    5.  x_i := p_i - (1 / o_i) * R
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


def exclusive_exp(o, p):
    """
    Exclusive kelly for exponential utility
    x_i = log(p_i * o_i / R) / o_i
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
    return np.maximum(np.log(o * p / R[i]) * q, 0)


def exclusive_pow(o, p, a=0.5):
    """
    Exclusive kelly for power utility
    """
    assert a > 0
    assert a != 1

    rev, q = p * o, 1 / o
    if (rev <= 1).all():
        return np.zeros_like(p)
    if q.sum() < 1:
        raise ValueError("Arbitrageable")

    idx = np.argsort(-rev)
    r = (rev ** (1 / a) * q)
    tmp_p = 1 - np.cumsum(p[idx][:-1])
    tmp_q = 1 - np.cumsum(q[idx][:-1])
    tmp_r = np.cumsum(r[idx][:-1])

    R = np.insert(tmp_p / tmp_q, 0, 1)
    i = np.argmin(rev[idx] > R)
    R = R[i]
    tmp_p = np.insert(tmp_p, 0, 1)[i]
    tmp_q = np.insert(tmp_q, 0, 1)[i]
    tmp_r = np.insert(tmp_r, 0, 0)[i]
    R_adj = 1 / (tmp_r * R ** (-1 / a) + tmp_q)
    x = R_adj * ((R / rev) ** (-1 / a) - 1) * q
    return (np.maximum(x, 0))
