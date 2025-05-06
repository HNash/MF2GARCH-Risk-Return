import numpy as np
from scipy.optimize import minimize
import stderr
from numba import njit
from matplotlib import pyplot as plt

# This function simply calculates MF2-GARCH components with given parameter values
@njit
def mf2_execute(param, y, m, proportional, components):
    alpha, gamma, beta = param[:3]
    lambda_0, lambda_1, lambda_2 = param[3:6]

    # Initializing to 0.0 so that if the user doesn't want a param, it drops
    gamma_0, gamma_1_s, gamma_1_l = [0.0, 0.0, 0.0]

    if (components == 0):
        if (proportional == 0):
            gamma_0 = param[6]
            gamma_1_s = param[7]
        else:
            gamma_1_s = param[6]

    elif (components == 1):
        if (proportional == 0):
            gamma_0 = param[6]
            gamma_1_l = param[7]
        else:
            gamma_1_l = param[6]
    else:
        if (proportional == 0):
            gamma_0 = param[6]
            gamma_1_s = param[7]
            gamma_1_l = param[8]
        else:
            gamma_1_s = param[6]
            gamma_1_l = param[7]

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
        # Default value is 0.0, so the param drops if required
        mu_prev = gamma_0 + (gamma_1_s * h[t-1]) + (gamma_1_l * tau[t-1])
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

    mu = gamma_0 + (gamma_1_s*h) + (gamma_1_l*tau)
    e = np.divide((y-mu), np.sqrt(np.multiply(h,tau)))

    # Ignoring first two years of data
    start_index = (2*252)+1
    return e[start_index:], h[start_index:], tau[start_index:], V_m

@njit
def totallikelihood(param, y, m, proportional, components):
    # Get component values to use in likelihood function
    e, h, tau, V_m = mf2_execute(param, y, m, proportional, components)
    # Likelihood function for MF2-GARCH specification
    ll_mf2 = -0.5 * (np.log(2*np.pi) + np.log(np.multiply(h,tau)) + np.power(e,2))
    return -np.sum(ll_mf2)

def estimate(y, proportional, components, **kwargs):
    m = kwargs.get('m', 63)

    y = np.asarray(y)

    # DEFAULT CASE: PROPORTIONAL, ONE COMPONENT
    # Initial guesses for parameters
    param0 = np.array([0.007, 0.14, 0.85, y.var() * (1 - 0.07 - 0.91), 0.07, 0.91, 0.0])
    # Constraints are passed in the form Ax<b, --> b-Ax>0
    A = np.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]])
    b = np.array([0.0, 1.0, 0.0, 1.0])
    bounds = ((0.0, 1.0), (-0.5, 0.5), (0.0, 1.0), (0.000001, 10.0), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0))

    # ADD THE NECESSARY PARAMETERS TO param0 AND THE CONSTRAINTS
    for i in range(int(proportional==0) + int(components==2)):
        param0 = np.append(param0, 0.0)
        A = np.hstack([A, np.zeros((len(A), 1), dtype=float)])

        to_add = (-1.0,1.0)
        bounds = bounds + (to_add,)

    cons = [{'type': 'ineq', 'fun': lambda x, A=A, b=b: b - np.dot(A, x)}]

    BICs = np.zeros(130)
    for m in range(20, 150):
        param_solution = minimize(fun=lambda x: totallikelihood(x, y, m, proportional, components), x0=param0, method='SLSQP', bounds=bounds, constraints=cons).x
        ll = totallikelihood(param_solution, y, m, proportional, components)
        BICs[m-20] = (np.log(y.size)*param_solution.size)-(2*np.log(ll))
        #print(ll)

    m = np.argmin(BICs) + 21

    plt.plot(range(20,150), BICs)
    plt.show()

    sol = minimize(lambda x: totallikelihood(x, y, m, proportional, components), param0, method='SLSQP', bounds=bounds, constraints=cons)

    param_solution = sol.x
    ll = sol.fun
    e, h, tau, V_m = mf2_execute(param_solution, y, m, proportional, components)
    qmle_se, p_value_qmle = stderr.stdErrors(param_solution, y, e, h, tau, m, proportional, components)

    return param_solution, qmle_se, p_value_qmle, m, ll