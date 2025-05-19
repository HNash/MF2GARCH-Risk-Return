import numpy as np
from scipy.stats import norm
from scipy.sparse import csr_matrix
import estimation
from numba import njit

def hessianTwoSided(llfunc, params, y, m, proportional, components, D):
    n = params.size

    fx = llfunc(params, y, m, proportional, components, D)

    h = (np.finfo(float).eps)**(1/3)*np.maximum(abs(params), 1e-8)
    xh = params + h
    h = xh - params
    ee = csr_matrix((h, (range(n), range(n))), shape=(n, n))

    gp = np.zeros(n)
    gm = np.zeros(n)
    for i in range (n):
        gp[i] = llfunc(params + ee[:, i].toarray().ravel(), y, m, proportional, components, D)
        gm[i] = llfunc(params - ee[:, i].toarray().ravel(), y, m, proportional, components, D)

    hh = np.matmul(np.reshape(h, (h.size,1)), np.transpose(np.reshape(h,(h.size,1))))
    Hm = np.full((n, n), np.nan)
    Hp = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):
            Hp[i, j] = llfunc(params + ee[:, i].toarray().ravel() + ee[:, j].toarray().ravel(), y, m, proportional, components,  D)
            Hm[i, j] = llfunc(params - ee[:, i].toarray().ravel() - ee[:, j].toarray().ravel(), y, m, proportional, components,  D)
            Hp[j,i] = Hp[i,j]
            Hm[j,i] = Hm[i,j]

    H = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            H[i,j] = (Hp[i,j] - gp[i] - gp[j] + 2*fx - gm[i] - gm[j] + Hm[i,j]) / (2*hh[i,j])
            H[j,i] = H[i,j]
    return H


def stdErrors(params, y, e, h, tau, m, proportional, components, D):
    T = tau.size
    k = params.size

    # Two-sided finite differences for parameters
    hhh = np.maximum(np.abs(params) * (np.finfo(float).eps)**(1/3), 1e-8)
    hhh = np.diag(hhh)
    scores = np.zeros((T, k))

    for j in range(k):
        params_h_h = params + hhh[:, j]
        e_h, h_h, tau_h, V = estimation.mf2_execute(params_h_h, y, m, proportional, components, D)
        ll_sum_h = -0.5 * (np.log(2*np.pi) + np.log(np.multiply(h_h,tau_h)) + np.power(e_h,2))

        params_h_m = params - hhh[:, j]
        e_m, h_m, tau_m, V = estimation.mf2_execute(params_h_m, y, m, proportional, components, D)
        ll_sum_m = -0.5 * (np.log(2*np.pi) + np.log(np.multiply(h_m,tau_m)) + np.power(e_m,2))

        scores[:,j] = np.divide((ll_sum_h - ll_sum_m),(2*hhh[j,j]))

    S = (1/T)*np.matmul(np.transpose(scores),scores);

    # Hessian
    H = hessianTwoSided(estimation.negativeLogLikelihood, params, y, m, proportional, components, D)
    A = H/T
    mhess = np.linalg.inv(A)

    # Standard errors
    qmle_se = np.transpose(np.sqrt(np.diag(np.matmul(np.matmul(mhess, S), mhess)) / T))

    # P-values
    p_value_qmle = 2 * (1-norm.cdf(abs(np.divide(np.transpose(params),qmle_se)),0,1))

    return qmle_se, p_value_qmle
