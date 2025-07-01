import numpy as np

def generate(proportional, components, length, seed):
    r = np.zeros(length)

    # Initial values are average parameter estimates from real data
    alpha = 0.006
    gamma = 0.16
    beta = 0.842
    lambda_0 = 0.011
    lambda_1 = 0.085
    lambda_2 = 0.902

    # Default value is 0 to drop parameter if specification doesn't call for it
    delta_0 = 0.0
    delta_1_s = 0.0
    delta_1_l = 0.0

    # Initialize parameters if specification calls for them
    if (proportional == 0):
        delta_0 = 0.02
    if (components == 0):
        delta_1_s = -0.013
    elif (components == 1):
        delta_1_l = 0.049
    else:
        delta_1_s = -0.013
        delta_1_l = 0.05

    tau = np.zeros(length)
    h = np.zeros(length)
    V = np.zeros(length)
    V_m = np.zeros(length)
    mu = np.zeros(length)

    # Start at averages
    tau[0] = 0.83
    h[0] = 1.186
    V[0] = 0.0
    V_m[0] = 0.0

    r[0] = 0.028

    m = 63

    # For replicability
    np.random.seed(seed)

    for t in range(1, length):
        if (t<m):
            tau[t] = 0.83
            V_m[t] = np.average(V[:m])
        else:
            V_m[t] = np.average(V[t-m+1:t])

        shock = np.random.normal(0,1)

        mu[t] = delta_0 + (delta_1_s*h[t-1]) + (delta_1_l*tau[t-1]) + (np.sqrt(h[t-1]*tau[t-1])*shock)
        if (r[t-1]-mu[t-1] < 0):
            h[t] = (1-alpha-(gamma/2)-beta) + ((alpha + gamma)*(((r[t-1]-mu[t-1])**2)/tau[t-1])) + (beta*h[t-1])
        else:
            h[t] = (1-alpha-(gamma/2)-beta) + (alpha*(((r[t-1]-mu[t-1])**2)/tau[t-1])) + (beta*h[t-1])
        V[t] = ((r[t]-mu[t])**2)/h[t]
        if (t<m):
            tau[t] = 0.83
        else:
            tau[t] = lambda_0 + (lambda_1 * V_m[t-1]) + (lambda_2 * tau[t-1])
        r[t] = np.sqrt(h[t]*tau[t])*shock + mu[t]
    return r