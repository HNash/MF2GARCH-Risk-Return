import numpy as np
import pandas
import estimation
from tabulate import tabulate
import warnings
# Suppressing warnings due to square rooting of negative h*tau values
warnings.filterwarnings("ignore", category=RuntimeWarning)

returns = pandas.read_excel('data/SP500_1971_2023_06_30_ret.xlsx')
y = returns['RET_SPX'].values
# Calculating conditional means of excess market returns
excess = returns['EXCESS'].values
cond_means = np.zeros(excess.size)

for i in range(cond_means.size):
    cond_means[i] = np.mean(excess[:i])

solution, stderrs, p_values = estimation.estimate(y, cond_means)

# Simple text formatting for results
print("-------------------------------------------")
param_names = ["mu", "alpha", "gamma", "beta", "lambda_0", "lambda_1", "lambda_2", "gamma_0", "gamma_1"]
table = [[0]*4 for i in range(9)]
for i in range(9):
    table[i][0] = param_names[i]
    table[i][1] = format(solution[i], '.3f')
    table[i][2] = format(stderrs[i], '.3f')
    table[i][3] = format(p_values[i], '.3f')
print(tabulate(table, headers=["", "Coeff.", "Std. Err.", "P-Value"]))