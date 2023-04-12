

import numpy as np
import scipy.stats as sc
import pandas as pd
import matplotlib.pyplot as plot

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# import dataset as pandas dataframe
homeEquity = pd.read_excel('HmEqty.xlsx')


# print correlation matrix
print(homeEquity.corr())
# highest correlations are between
# MORTDUE vs VALUE (0.939816) and
# DEBTINC vs YOJ (-0.413628)

# check for null values and fill with column mean
print(
'\nDEBTINC null?: ', homeEquity['DEBTINC'].isnull().values.any(), #false
'\nYOJ null?: ', homeEquity['YOJ'].isnull().values.any(), #true
'\nMORTDUE null?: ', homeEquity['MORTDUE'].isnull().values.any(), #false
'\nVALUE null?: ', homeEquity['VALUE'].isnull().values.any() #false
) 

homeEquity['YOJ'] = homeEquity['YOJ'].fillna(homeEquity['YOJ'].mean())
print('\nNaN filled',
	  '\nYOJ still null?: ',   # noqa: E101
	  homeEquity['YOJ'].isnull().values.any()) #true  # noqa: E101

# assign independent and dependent variables
x = homeEquity['VALUE']
y = homeEquity['MORTDUE']

x2 = homeEquity['YOJ']
y2 = homeEquity['DEBTINC']


# use linregress to perform regressions and gather
# slope (m), intercept(b), R value, P value and standard error
# create array of slope and intercept for us in polyval in plots
slope1, intercept1, rvalue1, pvalue1, stderr1 = sc.linregress(x, y)
fit1 = [slope1, intercept1]

slope2, intercept2, rvalue2, pvalue2, stderr2 = sc.linregress(x2, y2)
fit2 = [slope2, intercept2]


# print regression results and best fit model for each regression
print('\nFit 1 (MORTDUE vs VALUE)',
	  '\nslope:\t\t\t', slope1, 
	  '\ny-intercept:\t', intercept1, 
	  '\nR value:\t\t', rvalue1,
	  '\nR-squared:\t\t', pow(rvalue1, 2),
	  '\nP value:\t\t', pvalue1,
	  '\nstd err:\t\t', stderr1,
	  '\nRegress. Model:  MORTDUE = ', intercept1, ' + ', slope1, ' * VALUE', 
	  '\n')

print('\nFit 2 (DEBTINC vs YOJ)',
	  '\nslope:\t\t\t', slope2, 
	  '\ny-intercept:\t', intercept2, 
	  '\nR value:\t\t', rvalue2,
	  '\nR-squared:\t\t', pow(rvalue2, 2),
	  '\nP value:\t\t', pvalue2,
	  '\nstd err:\t\t', stderr2,
	  '\nRegress. Model:  DEBTINC = ', intercept2, ' + ', slope2, ' * YOJ',
	  '\n')

# plot given values and regression prediction line
plot.plot(x, y, 'x')
plot.plot(x, np.polyval(fit1, x), 'r-')
plot.xlabel('Home Values (VALUE)')
plot.ylabel('Mortage Due (MORTDUE)')
plot.title('Simple Linear Regression (MORTDUE vs VALUE)')
plot.show()

plot.plot(x2, y2, 'x')
plot.plot(x2, np.polyval(fit2, x2), 'r-')
plot.xlabel('Years At Present Job (YOJ)')
plot.ylabel('Debt / Income Ratio (DEBTINC)')
plot.title('Simple Linear Regression (DEBTINC vs YOJ)')
plot.show()

ticks = homeEquity.columns


corr_matrix = pd.plotting.scatter_matrix(homeEquity)
for subaxis in corr_matrix:
    for ax in subaxis:
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")

pd.plotting.xticks([], ticks)
plot.show()


