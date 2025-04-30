import numpy as np
import pandas
import estimation
from tabulate import tabulate
import warnings
# Suppressing warnings due to square rooting of negative h*tau values
warnings.filterwarnings("ignore", category=RuntimeWarning)

returns = pandas.read_csv('data/FF_DAILY_3_FACTORS.csv')
y = returns['Mkt-RF'].values

solution, stderrs, p_values, m, ll = estimation.estimate(y)

significance=[
    "***" if p<0.01
    else "**" if p<0.05
    else "*" if p<0.10
    else ""
    for p in p_values
]

# Simple text formatting for results
print("Likelihood: ", format(ll, '.3f'))
print("m/argmin(BIC): ", m)
print("-----------------------------------------------------")
param_names = ["alpha", "gamma", "beta", "lambda_0", "lambda_1", "lambda_2", "gamma_0", "gamma_1_s"]
table = [[0]*5 for i in range(8)]
for i in range(8):
    table[i][0] = param_names[i]
    table[i][1] = format(solution[i], '.3f')
    table[i][2] = format(stderrs[i], '.3f')
    table[i][3] = format(p_values[i], '.3f')
    table[i][4] = significance[i]
print(tabulate(table, headers=["", "Coeff.", "Std. Err.", "P-Value","Significance"]))