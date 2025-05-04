import numpy as np
from scipy.optimize import minimize
import stderr
from numba import njit

# This function simply calculates MF2-GARCH components with given parameter values
@njit
def mf2_execute(param, y, m):
    alpha, gamma, beta = param[:3]
    lambda_0, lambda_1, lambda_2 = param[3:6]
    gamma_0 = param[6]
    gamma_1_s = param[7]
    #gamma_1_l = param[7]

    base = 1 - alpha - gamma/2 - beta

    n = y.size
    h = np.ones(n, dtype=y.dtype)
    tau = np.ones(n, dtype=y.dtype) * np.mean(np.power(y,2))
    V = np.ones(n, dtype=y.dtype)
    V_m = np.ones(n, dtype=y.dtype)
    cumsum_V = np.zeros(n+1, dtype=y.dtype)

    h[:2] = 1.0
    V[:2] = 1.0

    # This first for loop only calculates h values since tau requires m previous observations
    for t in range(2, n):
        # mu in MF2-GARCH is given here by the univariate risk-return spec from Maheu & McCurdy
        mu_prev = (gamma_1_s * h[t-1]) + gamma_0#+(gamma_1_l * tau[t-1])
        # If negative, leverage effect parameter (gamma) is included
        if((y[t-1]-mu_prev) < 0):
            h[t] = base + ((alpha+gamma)*(( (y[t-1]-mu_prev)**2)/tau[t-1])) + (beta*h[t-1])
        else:
            h[t] = base + (alpha*(((y[t-1]-mu_prev)**2)/tau[t-1])) + (beta*h[t-1])

        if (t>=m):
            V[t] = ((y[t] - mu_prev) ** 2) / h[t]

            cumsum_V[t + 1] = cumsum_V[t] + V[t]
            V_m[t] = (cumsum_V[t + 1] - cumsum_V[t + 1 - m]) / m

            tau[t] = lambda_0 + (lambda_1 * V_m[t-1]) + (lambda_2 * tau[t-1])

    mu = (gamma_1_s*h)+ gamma_0#+(gamma_1_l*tau)
    e = np.divide((y-mu), np.sqrt(np.multiply(h,tau)))

    # Ignoring first two years of data
    start_index = (2*252)+1
    return e[start_index:], h[start_index:], tau[start_index:], V_m

@njit
def totallikelihood(param, y, m):
    # Get component values to use in likelihood function
    e, h, tau, V_m = mf2_execute(param, y, m)
    # Likelihood function for MF2-GARCH specification
    ll_mf2 = -0.5 * (np.log(2*np.pi) + np.log(np.multiply(h,tau)) + np.power(e,2))
    return -np.sum(ll_mf2)

def estimate(y, m=63):
    # Initial guesses for parameters
    y = np.asarray(y)
    param0 = np.array([0.007, 0.14, 0.85, y.var() * (1 - 0.07 - 0.91), 0.07, 0.91, 0.0, 0.0])

    # Constraints are passed in the form Ax<b, --> b-Ax>0
    A = np.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    b = np.array([0.0, 1.0, 0.0, 1.0])
    cons = [{'type': 'ineq', 'fun': lambda x, A=A, b=b: b - np.dot(A, x)}]
    # Upper/lower bound pairs for each parameter
    bounds = ((0.0, 1.0), (-0.5, 0.5), (0.0, 1.0), (0.000001, 10.0), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    m = 63
    sol = minimize(lambda x: totallikelihood(x, y, m), param0, method='SLSQP', bounds=bounds, constraints=cons)

    param_solution = sol.x
    ll = sol.fun
    e, h, tau, V_m = mf2_execute(param_solution, y, m)
    qmle_se, p_value_qmle = stderr.stdErrors(param_solution, y, e, h, tau, m)

    return param_solution, qmle_se, p_value_qmle, m, ll