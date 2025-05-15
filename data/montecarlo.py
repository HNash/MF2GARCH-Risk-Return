import numpy as np

def generate(proportional, components, length):
    r = np.zeros(length)

    #param0 = np.array([0.007, 0.14, 0.85, y.var() * (1 - 0.07 - 0.91), 0.07, 0.91, 0.0])
    alpha = 0.02
    gamma = 0.1
    beta = 0.8
    lambda_0 = 0.02
    lambda_1 = 0.05
    lambda_2 = 0.94

    gamma_0 = 0.0
    gamma_1_s = 0.0
    gamma_1_l = 0.0

    if (proportional == 0):
        gamma_0 = 0.02
    if (components == 0):
        gamma_1_s = -0.013
    elif (components == 1):
        gamma_1_l = 0.05
    else:
        gamma_1_s = -0.013
        gamma_1_l = 0.05

    tau = np.zeros(length)
    h = np.zeros(length)
    V = np.zeros(length)
    V_m = np.zeros(length)

    tau[0] = 0.1
    h[0] = 0.1
    V[0] = (0.02**2)/0.1
    V_m[0] = 0.8

    r[0] = 0.02

    m=63

    for t in range(1, length):
        if (t<150):
            V_m[t] = 0.8
        else:
            V_m[t] = np.average(V[t-m+1:t])
        tau[t] = lambda_0 + (lambda_1 * V[t-1]) + (lambda_2 * tau[t-1])

        if (r[t-1] < 0):
            h[t] = (1-alpha-(gamma/2)-beta) + ((alpha + gamma)*((r[t-1]**2)/tau[t-1])) + (beta*h[t-1])
        else:
            h[t] = (1-alpha-(gamma/2)-beta) + (alpha*((r[t-1]**2)/tau[t-1])) + (beta*h[t-1])

        r[t] = gamma_0 + (gamma_1_s*h[t]) + (gamma_1_l*tau[t]) + (np.sqrt(h[t]*tau[t])*np.random.standard_normal)

        V[t] = (r[t]**2)/tau[t]

    return r