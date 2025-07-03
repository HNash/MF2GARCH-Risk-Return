import numpy as np
from scipy.optimize import minimize
import stderr
from numba import njit
from matplotlib import pyplot as plt

# This code is based on:
# Conrad, Christian and Julius Schoelkopf. 2025. MF2-GARCH Toolbox for Matlab. Matlab package version 0.1.0.
# (github.com/juliustheodor/mf2garch/)
# which was originally designed for MF2-GARCH parameter estimation

# This function simply calculates MF2-GARCH components with given parameter values
@njit
def mf2_execute(param, y, m, proportional, components, D):
    # If dummy variable was not initialized, crisis_control is False
    crisis_control = np.sum(D)!=0

    alpha, gamma, beta = param[:3]
    lambda_0, lambda_1, lambda_2 = param[3:6]

    # Initializing to 0.0 so that if the user doesn't want a param, it drops
    delta_0, delta_1_s, delta_1_l = [0.0, 0.0, 0.0]
    theta_0, theta_1_s, theta_1_l = [0.0, 0.0, 0.0]

    # These parameters are only used in the "overall volatility" case (components = 3)
    delta_1, theta_1 = [0.0, 0.0]

    # Depending on specification, parameters are initialized from arguments
    if (components == 0):
        if (proportional == 0):
            delta_0 = param[6]
            delta_1_s = param[7]
        else:
            delta_1_s = param[6]

    elif (components == 1):
        if (proportional == 0):
            delta_0 = param[6]
            delta_1_l = param[7]
        else:
            delta_1_l = param[6]
    elif(components == 2):
        if (proportional == 0):
            delta_0 = param[6]
            delta_1_s = param[7]
            delta_1_l = param[8]
        else:
            delta_1_s = param[6]
            delta_1_l = param[7]
    else:
        if (proportional == 0):
            delta_0 = param[6]
            delta_1 = param[7]
        else:
            delta_1 = param[6]

    # If dummy variable is included, dummy parameters are initialized from arguments
    if(crisis_control):
        if(proportional == 1):
            if(components == 0):
                theta_1_s = param[7]
            elif(components == 1):
                theta_1_l = param[7]
            elif(components == 2):
                theta_1_s = param[8]
                theta_1_l = param[9]
            else:
                theta_1 = param[7]
        else:
            if (components == 0):
                theta_0 = param[8]
                theta_1_s = param[9]
            elif (components == 1):
                theta_0 = param[8]
                theta_1_l = param[9]
            elif (components == 2):
                theta_0 = param[9]
                theta_1_s = param[10]
                theta_1_l = param[11]
            else:
                theta_0 = param[8]
                theta_1 = param[9]

    # MF2-GARCH intercept
    base = 1 - alpha - gamma/2 - beta

    n = y.size
    h = np.ones(n, dtype=y.dtype)
    tau = np.ones(n, dtype=y.dtype) * np.mean(np.power(y,2))
    V = np.ones(n, dtype=y.dtype)
    V_m = np.ones(n, dtype=y.dtype)
    cumsum_V = np.zeros(n+1, dtype=y.dtype)


    for t in range(2, n):
        # mu in MF2-GARCH is given here by the univariate risk-return spec from Maheu & McCurdy
        if(crisis_control):
            # Default param value is 0.0, so the param drops if required
            mu_prev = (((delta_0 + theta_0) * D[t-1]) + ((delta_1 + theta_1 * D[t-1]) * h[t-1] * tau[t-1]) + ((delta_1_s + theta_1_s * D[t-1]) * h[t - 1]) + ((delta_1_l + theta_1_l * D[t-1]) * tau[t - 1]))
        else:
            # Default param value is 0.0, so the param drops if required
            mu_prev = delta_0 + delta_1*h[t-1]*tau[t-1] + delta_1_s*h[t-1] + delta_1_l*tau[t-1]

        # If negative, leverage effect parameter (gamma) is included
        if((y[t-1]-mu_prev) < 0):
            h[t] = base + ((alpha+gamma)*(((y[t-1]-mu_prev)**2)/tau[t-1])) + (beta*h[t-1])
        else:
            h[t] = base + (alpha*(((y[t-1]-mu_prev)**2)/tau[t-1])) + (beta*h[t-1])

        # Preventing rare division by zero error
        if (h[t] == 0):
            h[t] = h[t-1]
        if (tau[t] == 0):
            tau[t] = tau[t-1]

        V[t] = ((y[t] - mu_prev) ** 2) / h[t]
        cumsum_V[t + 1] = cumsum_V[t] + V[t]

        # V_m (moving average) needs window of m observations
        if (t>=m):
            V_m[t] = (cumsum_V[t + 1] - cumsum_V[t + 1 - m]) / m
            tau[t] = lambda_0 + (lambda_1 * V_m[t-1]) + (lambda_2 * tau[t-1])

    mu = np.zeros(len(h))
    # mu calculated with crisis dummy if controlling for crises
    if(crisis_control):
        for i in range(len(mu)):
            mu[i] = (delta_0+theta_0*D[i]) + ((delta_1+theta_1*D[i]) * h[i] * tau[i]) + ((delta_1_s+theta_1_s*D[i]) * h[i]) + ((delta_1_l+theta_1_l*D[i]) * tau[i])
    else:
        mu = delta_0 + delta_1*h*tau + delta_1_s*h + delta_1_l*tau

    e = np.divide((y-mu), np.sqrt(np.multiply(h,tau)))

    # Ignoring first two years of data
    start_index = (2*252)+1
    return e[start_index:], h[start_index:], tau[start_index:], V_m

@njit
def negativeLogLikelihood(param, y, m, proportional, components, D):
    e, h, tau, V_m = mf2_execute(param, y, m, proportional, components, D)
    # Likelihood function for MF2-GARCH specification
    ll = -0.5 * (np.log(2*np.pi) + np.log(np.multiply(h,tau)) + np.power(e,2))
    return -np.sum(ll)

def estimate(y, proportional, components, D, mchoice):
    # If dummy variable was not initialized, crisis_control is False
    crisis_control = np.sum(D)!=0

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

    # Based on the specification, add the necessary parameters (initial guesses and constraints)
    for i in range(int(proportional == 0) + int(components == 2)):
        param0 = np.append(param0, 0.0)
        A = np.hstack([A, np.zeros((len(A), 1), dtype=float)])

        to_add = (-1.0, 1.0)
        bounds = bounds + (to_add,)

    # If controlling for crises, add the parameters for the dummy variable (initial guesses and constraints)
    if(crisis_control):
        for i in range(1 + int(proportional == 0) + int(components == 2)):
            param0 = np.append(param0, 0.0)
            A = np.hstack([A, np.zeros((len(A), 1), dtype=float)])
            to_add = (-1.0, 1.0)
            bounds = bounds + (to_add,)

    # Constraints to be passed into SciPy minimization function
    cons = [{'type': 'ineq', 'fun': lambda x, A=A, b=b: b - np.dot(A, x)}]

    m = 0
    final_BIC = 0.0

    if(mchoice == 0):
        # Chooses the m that minimizes the Bayesian Info Criterion
        BICs = np.zeros(130)
        # Loop over possible values of m...
        for m in range(20, 150):
            # ...minimize negative log-likelihood for each...
            param_solution = minimize(fun=lambda x: negativeLogLikelihood(x, y, m, proportional, components, D), x0=param0,
                                          method='SLSQP', bounds=bounds, constraints=cons).x
            nll = negativeLogLikelihood(param_solution, y, m, proportional, components, D)
            # ...calculate and store BIC for this m...
            BICs[m-20] = (np.log(y.size)*param_solution.size)-(-2*nll)
        # ...find the value of m that minmizes the BIC
        m = np.argmin(BICs) + 20
        final_BIC = np.min(BICs)
        # Plots m on the x axis vs. BIC values on the y axis
        plt.plot(range(20, 150), BICs)
        plt.title("BIC Plot")
        plt.xlabel("Moving Avg. Window Size (m)")
        plt.ylabel("BIC Value")
        # Ths final plt.show() statement is in main.py, is/isn't executed based on user options

        # Get the solution for the optimal m
        sol = minimize(fun=lambda x: negativeLogLikelihood(x, y, m, proportional, components, D), x0=param0,
                       method='SLSQP', bounds=bounds, constraints=cons)
        param_solution = sol.x
        nll = sol.fun
        e, h, tau, V_m = mf2_execute(param_solution, y, m, proportional, components, D)
        qmle_se, p_value_qmle = stderr.stdErrors(param_solution, y, e, h, tau, m, proportional, components, D)
    else:
        m = mchoice

        sol = minimize(fun=lambda x: negativeLogLikelihood(x, y, m, proportional, components, D), x0=param0,
                       method='SLSQP', bounds=bounds, constraints=cons)
        param_solution = sol.x
        nll = sol.fun
        final_BIC = (np.log(y.size)*param_solution.size)-(-2*nll)
        e, h, tau, V_m = mf2_execute(param_solution, y, m, proportional, components, D)
        qmle_se, p_value_qmle = stderr.stdErrors(param_solution, y, e, h, tau, m, proportional, components, D)

    return param_solution, qmle_se, p_value_qmle, m, nll, final_BIC