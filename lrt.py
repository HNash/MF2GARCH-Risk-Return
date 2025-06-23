import numpy as np
from scipy.stats import chi2

llf_null_short=-18566.385
llf_alt_short=-18563.889
stat_short= -2*(llf_null_short-llf_alt_short)

llf_null_long=-18558.193
llf_alt_long=-18556.766
stat_long= -2*(llf_null_long-llf_alt_long)

llf_null_2_comp=-18557.619
llf_alt_2_comp=-18556.707
stat_2_comp= -2*(llf_null_2_comp-llf_alt_2_comp)

llf_null_vol=-18562.105
llf_alt_vol=-18560.907
stat_vol= -2*(llf_null_vol-llf_alt_vol)

diff_df=1

print("stat, short-term component", format(stat_short,'.3f'))
print("p-value, short-term component", format(chi2.sf(stat_short,diff_df),'.3f'))
print("stat, long-term component", format(stat_long,'.3f'))
print("p-value, long-term component", format(chi2.sf(stat_long,diff_df),'.3f'))
print("stat, both components", format(stat_2_comp,'.3f'))
print("p-value, both components", format(chi2.sf(stat_2_comp,diff_df),'.3f'))
print("stat, both components", format(stat_vol,'.3f'))
print("p-value, both components", format(chi2.sf(stat_vol,diff_df),'.3f'))
