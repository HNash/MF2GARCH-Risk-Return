import numpy as np
from scipy.optimize import minimize
import stderr
import statsmodels.api as sm

# This function simply calculates MF2-GARCH components with given parameter values
def mf2_execute(param, y, m):
    alpha = param[0]
    gamma = param[1]
    beta = param[2]

    lambda_0 = param[3]
    lambda_1 = param[4]
    lambda_2 = param[5]

    gamma_0 = param[6]
    gamma_1_s = param[7]
    #gamma_1_l = param[7]

    h = np.ones(y.size)
    tau = np.ones(y.size)*np.mean(np.power(y,2))
    V = np.ones(y.size)
    V_m = np.ones(y.size)

    # This first for loop only calculates h values since tau requires m previous observations
    for t in range(2, m+1):
        # mu in MF2-GARCH is given here by the univariate risk-return spec from Maheu & McCurdy
        #*****ADD TIME INDEX*****
        mu_prev = (gamma_1_s * h[t-1])+ gamma_0#+(gamma_1_l * tau[t-1])
        # If negative, leverage effect parameter (gamma) is included
        if((y[t-1]-mu_prev) < 0):
            h[t] = (1-alpha-(gamma/2)-beta) + ((alpha+gamma)*(( (y[t-1]-mu_prev)**2)/tau[t-1])) + (beta*h[t-1])
        else:
            h[t] = (1-alpha-(gamma/2)-beta) + (alpha*(((y[t-1]-mu_prev)**2)/tau[t-1])) + (beta*h[t-1])

    # Same as above except V, V_m and tau are now able to be calculated
    for t in range(m+1, y.size):
        # mu in MF2-GARCH is given here by the univariate risk-return spec from Maheu & McCurdy
        mu_prev = (gamma_1_s * h[t-1])+ gamma_0#+(gamma_1_l * tau[t-1])
        if((y[t-1]-mu_prev) < 0):
            h[t] = (1-alpha-(gamma/2)-beta) + ((alpha+gamma)*(((y[t-1]-mu_prev)**2)/tau[t-1])) + (beta*h[t-1])
        else:
            h[t] = (1-alpha-(gamma/2)-beta) + (alpha*(((y[t-1]-mu_prev)**2)/tau[t-1])) + (beta*h[t-1])
        V[t] = ((y[t]-mu)**2)/h[t]
        V_m[t] = np.sum(np.divide(V[t-(m-1):t],m))

        tau[t] = lambda_0 + (lambda_1 * V_m[t-1]) + (lambda_2 * tau[t-1])

    mu = (gamma_1_s*h)+ gamma_0#+(gamma_1_l*tau)
    e = np.divide((y-mu), np.sqrt(np.multiply(h,tau)))

    # Ignoring first two years of data
    start_index = (2*252)+1
    h = h[start_index:]
    tau = tau[start_index:]
    e = e[start_index:]
    
    return e, h, tau, V_m

def totallikelihood(param, y, m):
    # Get component values to use in likelihood function
    e, h, tau, V_m = mf2_execute(param, y, m)

    # Likelihood function for MF2-GARCH specification
    ll_mf2 = -0.5 * (np.log(2*np.pi) + np.log(np.multiply(h,tau)) + np.power(e,2))

    return -1.0 * np.sum(ll_mf2)

def estimate(y):
    # Initial guesses for parameters
    param_init = np.array([0.007, 0.14, 0.85, np.mean(np.square(y))*(1-0.07-0.91), 0.07, 0.91, 0.0,0.0]) #np.mean(y)

    # Constraints are passed in the form Ax<b, --> b-Ax>0
    A = np.array([[-1.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0],
                 [1.0,  0.5,  1.0,  0.0,  0.0,  0.0, 0.0, 0.0],
                 [0.0,  0.0,  0.0,  0.0,  -1.0,  0.0, 0.0, 0.0],
                 [0.0,  0.0,  0.0,  0.0,  1.0,  1.0, 0.0, 0.0]])
    b = np.array([0.0, 1.0, 0.0, 1.0])
    constraints = [{'type': 'ineq', 'fun': lambda x, A=A, b=b: b-np.dot(A,x)}]
    # Upper/lower bound pairs for each parameter
    bounds = ((0.0,1.0), (-0.5,0.5), (0.0,1.0), (0.000001,10.0), (0.0,1.0), (0.0,1.0),(-1.0,1.0),(-1.0,1.0))

    #BICs = np.zeros(130)
    #for m in range(20, 150):
    #    param_solution = minimize(fun=lambda x: totallikelihood(x, y, m), x0=param_init, method='SLSQP', bounds=bounds, constraints=constraints).x
    #    ll = totallikelihood(param_solution, y, m)
    #    BICs[m-20] = (np.log(y.size)*param_solution.size)-(2*np.log(ll))
    m = 63
    param_solution = minimize(fun=lambda x: totallikelihood(x, y, m), x0=param_init, method='SLSQP', bounds=bounds, constraints=constraints).x
    e, h, tau, V_m = mf2_execute(param_solution, y, m)
    qmle_se, p_value_qmle = stderr.stdErrors(param_solution, y, e, h, tau, m)
    ll = totallikelihood(param_solution, y, m)

    return param_solution, qmle_se, p_value_qmle, m, ll