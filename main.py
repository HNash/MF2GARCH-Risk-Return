import numpy as np
import pandas
import estimation
import montecarlo
from tabulate import tabulate
from matplotlib import pyplot as plt
import warnings

# Suppressing warnings due to square rooting of negative h*tau values
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Importing data (market premia)
returns = pandas.read_excel('data/Modern_FF_DAILY_3_FACTORS.xlsx')

####################################
############ USER INPUT ############
####################################
print("MF2-GARCH risk-return specification")
print("User options")
print("------------------------------------------------------------")
proportional = int(input("Proportional specification (exclude intercept)? (1 = yes, 0 = no) "))
components = int(input("Which components of MF2-GARCH volatility should be included in the risk-return specification? (0 = short-term only, 1 = long-term only, 2 = both) "))
montecarlo_sim = int(input("Monte Carlo simulation? (1 = yes, 0 = no/real data) "))
if(montecarlo_sim):
    sim_length = int(input("Simulation length (T)? "))
    iterations = int(input("How many simulations should be performed? "))
else:
    crisis_control = int(input("Should crisis periods be controlled for? (1 = yes, 0 = no) "))
plotBIC = int(input("Plot BIC values for each m? (1 = yes, 0 = no) "))
print("------------------------------------------------------------")
####################################
####################################
####################################

param_count = 8 + int(proportional==0) + int(components==2)

# If Monte Carlo simulation was NOT selected, use imported data
if (montecarlo_sim == 0):
    # Market premia
    y = returns['Log_Prem'].values
    # Dummy variable to control for crises. If not desired then remains an array of zeros
    D = np.zeros(len(y))
    if(crisis_control):
        D = returns['Crisis'].values
    solution, stderrs, p_values, m, nll = estimation.estimate(y, proportional, components, D)
else:
    print("Monte Carlo Simulation / Estimation Results")
    print("Simulation Length (T): ", sim_length)
    print("Number of Iterations: ", iterations)
    print("------------------------------------------------------------")
    # Standard errors, p-values, m and likelihood are ignored for Monte Carlo simulation
    stderrs, p_values, m, nll = np.zeros(param_count), np.zeros(param_count), 0, 0

    # Array to store estimated parameters for each Monte Carlo iteration
    solutions = np.zeros((iterations, param_count))
    for s in range(iterations):
        # Generate data
        y = montecarlo.generate(proportional, components, sim_length, s)
        # Estimate parameters
        solutions[s], stderrs, p_values, m, nll = estimation.estimate(y, proportional, components, D)
    # The parameter estimates are averaged over all iterations
    solution = solutions.mean(axis=0)

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
else:
    param_names = np.append(param_names, "delta_1_s")
    param_names = np.append(param_names, "delta_1_l")
    print("Both Components")

# Adding parameters and description if dummy variable is included to control for crises
if(crisis_control):
    print("Controlling for crises")
    if (proportional == 0):
        param_names = np.append(param_names, "theta_0")
    if (components == 0):
        param_names = np.append(param_names, "theta_1_s")
    elif (components == 1):
        param_names = np.append(param_names, "theta_1_l")
    else:
        param_names = np.append(param_names, "theta_1_s")
        param_names = np.append(param_names, "theta_1_l")
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
print("m/argmin(BIC): ", m)
print("---------------------------RESULTS--------------------------")
table = [[0]*5 for i in range(len(param_names))]
for i in range(len(param_names)):
    table[i][0] = param_names[i]
    table[i][1] = format(solution[i], '.4f')
    table[i][2] = format(stderrs[i], '.4f')
    table[i][3] = format(p_values[i], '.4f')
    table[i][4] = significance[i]
print(tabulate(table, headers=["", "Coeff.", "Std. Err.", "P-Value","Significance"]))
if(plotBIC):
    plt.show()