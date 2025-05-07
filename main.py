import numpy as np
import pandas
import estimation
from tabulate import tabulate
import warnings

# Suppressing warnings due to square rooting of negative h*tau values
warnings.filterwarnings("ignore", category=RuntimeWarning)

returns = pandas.read_csv('data/Modern_FF_DAILY_3_FACTORS.csv')
y = returns['Mkt-RF'].values

#######################################################
#######################################################
################## USER INPUT HERE ####################
#######################################################
#######################################################
# Proportional=1 --> don't include intercept gamma_0
proportional = 0
# Which components of MF2-GARCH volatility to include in the risk-return specification
# 0 --> short-term only. 1 --> long-term only. 2 --> both
components = 2
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################

solution, stderrs, p_values, m, ll = estimation.estimate(y, proportional, components)

significance=[
    "***" if p<0.01
    else "**" if p<0.05
    else "*" if p<0.10
    else ""
    for p in p_values
]

param_names = ["alpha", "gamma", "beta", "lambda_0", "lambda_1", "lambda_2"]

if (proportional == 0):
    param_names=np.append(param_names, "gamma_0")
    print("Non-Proportional,", end=" ")
else:
    print("Proportional,", end=" ")
if (components == 0):
    param_names=np.append(param_names, "gamma_1_s")
    print("Short-Term Component")
elif (components == 1):
    param_names=np.append(param_names, "gamma_1_l")
    print("Long-Term Component")
else:
    param_names = np.append(param_names, "gamma_1_s")
    param_names = np.append(param_names, "gamma_1_l")
    print("Both Components")

# Simple text formatting for results
print("Likelihood: ", format(ll, '.3f'))
print("m/argmin(BIC): ", m)
print("-----------------------------------------------------")

table = [[0]*5 for i in range(len(param_names))]
for i in range(len(param_names)):
    table[i][0] = param_names[i]
    table[i][1] = format(solution[i], '.4f')
    table[i][2] = format(stderrs[i], '.4f')
    table[i][3] = format(p_values[i], '.4f')
    table[i][4] = significance[i]
print(tabulate(table, headers=["", "Coeff.", "Std. Err.", "P-Value","Significance"]))