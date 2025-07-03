import numpy as np

def generate(proportional, components, length, seed):
    # The first 2520 observations are for burn-in, will be discarded
    burnin = 2520
    r = np.zeros(length+burnin)

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
    delta_1 = 0.0

    # Initialize parameters if specification calls for them
    if (proportional):
        if(components==0):
            delta_1_s=0.027
        elif(components==1):
            delta_1_l=0.049
        elif(components==2):
            delta_1_s=-0.005
            delta_1_l=0.054
        else:
            delta_1=0.042
    else:
        if (components == 0):
            delta_0 = 0.033
            delta_1_s = -0.003
        elif (components == 1):
            delta_0 = 0.003
            delta_1_l = 0.045
        elif (components == 2):
            delta_0 = 0.008
            delta_1_s = -0.008
            delta_1_l = 0.046
        else:
            delta_0 = 0.02
            delta_1 = 0.023

    tau = np.zeros(length+burnin)
    h = np.zeros(length+burnin)
    V = np.zeros(length+burnin)
    V_m = np.zeros(length+burnin)
    mu = np.zeros(length+burnin)

    # Start at averages
    tau[0] = 0.83
    h[0] = 1.186
    V[0] = 0.0
    V_m[0] = 0.0
    r[0] = 0.028

    m = 63

    # For replicability
    np.random.seed(seed)

    for t in range(1, length+burnin):
        V_m[t] = V[:m].mean() if t<m else V[t-m+1:t].mean()
        shock = np.random.normal()
        mu[t] = delta_0 + (delta_1_s*h[t-1]) + (delta_1_l*tau[t-1]) + (delta_1*h[t-1]*tau[t-1])

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
    return r[burnin+1:]