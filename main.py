import numpy as np
import pandas
import estimation
import montecarlo
from tabulate import tabulate
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis
import warnings

# Suppressing warnings due to square rooting of negative h*tau values
warnings.filterwarnings("ignore", category=RuntimeWarning)

####################################
############ USER INPUT ############
####################################
print("MF2-GARCH risk-return specification")
print("User options")
print("------------------------------------------------------------")
plotBIC=False
proportional = int(input("Proportional specification (exclude intercept)? (1 = yes, 0 = no) "))
components = int(input("Which components of MF2-GARCH volatility should be included in the risk-return specification? (0 = short-term only, 1 = long-term only, 2 = both, 3 = overall volatility) "))
mchoice = int(input("Moving average window size / m? (0 = find the value that minimizes the BIC) "))
montecarlo_sim = int(input("Monte Carlo simulation? (1 = yes, 0 = no/real data) "))
if(montecarlo_sim):
    sim_length = int(input("Simulation length (T)? "))
    iterations = int(input("How many simulations should be performed? "))
    crisis_control = False
else:
    crisis_control = int(input("Should crisis periods be controlled for? (1 = yes, 0 = no) "))
    if (mchoice == 0):
        plotBIC = int(input("Plot BIC values for each m? (1 = yes, 0 = no) "))
print("------------------------------------------------------------")
####################################
####################################
####################################

param_count = 7 + int(proportional==0) + int(components==2)
##### PARAMETER NAMES #####
# These parameters are always included
param_names = ["alpha", "gamma", "beta", "lambda_0", "lambda_1", "lambda_2"]

# Description of specification and parameter name inclusion
# Depending on specification, the intercept and volatility component coefficients are/aren't included
if (proportional == 0):
    param_names=np.append(param_names, "delta_0")
    print("Non-Proportional,", end=" ")
else:
    print("Proportional,", end=" ")
if (components == 0):
    param_names=np.append(param_names, "delta_1_s")
    print("Short-Term Component")
elif (components == 1):
    param_names=np.append(param_names, "delta_1_l")
    print("Long-Term Component")
elif (components == 2):
    param_names = np.append(param_names, "delta_1_s")
    param_names = np.append(param_names, "delta_1_l")
    print("Both Components")
else:
    param_names = np.append(param_names, "delta_1")
    print("Overall volatility")

# If Monte Carlo simulation was NOT selected, use imported data
if(montecarlo_sim):
    print("Monte Carlo Simulation / Estimation Results")
    print("Simulation Length (T): ", sim_length)
    print("Number of Iterations: ", iterations)
    print("------------------------------------------------------------")
    # Array to store estimated parameters for each Monte Carlo iteration
    solutions = np.zeros((iterations, param_count))
    stderrs = np.zeros((iterations, param_count))
    ms = np.zeros(iterations)
    nlls = np.zeros(iterations)
    BICs = np.zeros(iterations)

    for s in range(iterations):
        # Generate data
        y = montecarlo.generate(proportional, components, sim_length, s)
        # Estimate parameters
        D = np.zeros(sim_length)
        solutions[s], stderrs[s], p_values, ms[s], nlls[s], BICs[s] = estimation.estimate(y, proportional, components, D, mchoice)

    # The parameter estimates are averaged over all iterations
    solution = solutions.mean(axis=0)
    avgstderrs = stderrs.mean(axis=0)
    stderrofparams = np.std(np.array(solutions), axis=0)

    avgm = ms.mean()
    avgnll = nlls.mean()
    avgBIC = BICs.mean()

    print("Average Log-likelihood: ", format(-avgnll, '.3f'))
    print("Average BIC: ", format(avgBIC, '.3f'))
    print("m = ", avgm)
    print("---------------------------RESULTS--------------------------")
    table = [[0] * 4 for i in range(len(param_names))]
    for i in range(len(param_names)):
        table[i][0] = param_names[i]
        table[i][1] = format(solution[i], '.5f')
        table[i][2] = format(avgstderrs[i], '.5f')
        table[i][3] = format(stderrofparams[i], '.5f')
    print(tabulate(table, headers=["", "Avg. Est.", "Avg. Std. Err.", "Std. Err. of Avg."]))

else:
    # Importing data (market premia)
    returns = pandas.read_excel('data/FF_DAILY_3_FACTORS.xlsx')
    # Market premia
    y = returns['Mkt-RF'].values
    rfs = returns['RF'].values
    # Dummy variable to control for crises. If not desired then remains an array of zeros
    D = np.zeros(len(y))
    if(crisis_control):
        D = returns['Crisis'].values
    solution, stderrs, p_values, m, nll, BIC = estimation.estimate(y, proportional, components, D, mchoice)

    # Adding parameters and description if dummy variable is included to control for crises
    if(crisis_control):
        print("Controlling for crises")
        if (proportional == 0):
            param_names = np.append(param_names, "theta_0")
        if (components == 0):
            param_names = np.append(param_names, "theta_1_s")
        elif (components == 1):
            param_names = np.append(param_names, "theta_1_l")
        elif (components == 2):
            param_names = np.append(param_names, "theta_1_s")
            param_names = np.append(param_names, "theta_1_l")
        else:
            param_names = np.append(param_names, "theta_1")

    else:
        print("Not controlling for crises")

    ##### OUTPUT FORMATTING #####
    significance=[
        "***" if p<0.01
        else "**" if p<0.05
        else "*" if p<0.10
        else ""
        for p in p_values
    ]

    print("Log-likelihood: ", format(-nll, '.3f'))
    print("BIC: ", format(BIC, '.3f'))
    print("m/argmin(BIC): ", m)
    print("---------------------------RESULTS--------------------------")
    table = [[0]*5 for i in range(len(param_names))]
    for i in range(len(param_names)):
        table[i][0] = param_names[i]
        table[i][1] = format(solution[i], '.3f')
        table[i][2] = format(stderrs[i], '.3f')
        table[i][3] = format(p_values[i], '.3f')
        table[i][4] = significance[i]
    print(tabulate(table, headers=["", "Coeff.", "Std. Err.", "P-Value","Significance"]))
    if(plotBIC):
        plt.show()

    print("---------------------------SUMMARY STATS--------------------------")

    e, h, tau, V_m = estimation.mf2_execute(solution, y, m, proportional, components, D)
    vol = np.multiply(h,tau)

    # Making them into pandas series to get autocorrelation easily
    y_pd = pandas.Series(y)
    h_pd = pandas.Series(h)
    tau_pd = pandas.Series(tau)
    vol_pd = pandas.Series(vol)

    print("Sample length (in days, after discarding 2*252 days): ", len(h))

    print("Correlation between risk-free rate and components:")
    print("Corr(RF, tau)=", format(np.corrcoef(rfs[len(rfs)-len(tau):], tau)[0][1], '.3f'))
    print("Corr(RF, h)=", format(np.corrcoef(rfs[len(rfs)-len(h):], h)[0][1], '.3f'))

    print("---y---")
    print("Mean: ", format(np.mean(y), '.3f'))
    print("Std. Dev: ", format(np.sqrt(np.var(y)), '.3f'))
    print("Skew: ", format(skew(y), '.3f'))
    print("Kurtosis: ", format(kurtosis(y), '.3f'))
    print("Min: ", format(np.min(y), '.3f'))
    print("Max: ", format(np.max(y), '.3f'))
    print("AC(1): ", format(y_pd.autocorr(lag=1), '.3f'))

    print("---h---")
    print("Mean: ", format(np.mean(h), '.3f'))
    print("Min: ", format(np.min(h), '.3f'))
    print("Max: ", format(np.max(h), '.3f'))
    print("AC(1): ", format(h_pd.autocorr(lag=1), '.9f'))

    print("---tau---")
    print("Mean: ", format(np.mean(tau), '.3f'))
    print("Min: ", format(np.min(tau), '.3f'))
    print("Max: ", format(np.max(tau), '.3f'))
    print("AC(1): ", format(tau_pd.autocorr(lag=1), '.9f'))

    print("---Vol---")
    print("Mean: ", format(np.mean(vol), '.3f'))
    print("Min: ", format(np.min(vol), '.3f'))
    print("Max: ", format(np.max(vol), '.3f'))
    print("AC(1): ", format(vol_pd.autocorr(lag=1), '.9f'))