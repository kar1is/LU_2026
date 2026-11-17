import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from scipy.stats import f

data = np.array([[0.9, 0.08361204013377943], [1.0, 0.11705685618729111], [1.1, 0.1505016722408028], [1.2000000000000002, 0.16722408026755842], [1.3000000000000003, 0.21739130434782616], [1.4000000000000001, 0.25083612040133785], [1.5000000000000002, 0.28428093645484953], [1.6, 0.3177257525083612], [1.7000000000000002, 0.36789297658862896], [1.8000000000000003, 0.4013377926421402], [1.9000000000000001, 0.45150501672240795], [2.0, 0.5016722408026753], [2.1, 0.5351170568561869], [2.2, 0.5685618729096986], [2.3000000000000003, 0.6020066889632103], [2.4000000000000004, 0.635451505016722]])

'''
def linear(x, a, b):
    return a*x+b

def quadratic(x, a, b, c):
    return a*(x**2)+b*x+c

xval = data[:,0]

params_linear, _ = curve_fit(linear, data[:,0], data[:,1])
params_quadratic, _ = curve_fit(quadratic, data[:,0], data[:,1])

exp_linear = linear(xval, *params_linear)
exp_quadratic = quadratic(xval, *params_quadratic)

chi_linear, _ = chisquare(data[:,1], exp_linear)
chi_quadratic, _ = chisquare(data[:,1], exp_quadratic)

print(f"Chi2 for linear: {chi_linear}.")
print(f"Chi2 for quadratic: {chi_quadratic}.")
'''

def linear(x, a, b):
    return a*x + b

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

x = data[:, 0]
y = data[:, 1]

params_linear, _ = curve_fit(linear, x, y)
params_quadratic, _ = curve_fit(quadratic, x, y)

residuals_linear = y - linear(x, *params_linear)
residuals_quadratic = y - quadratic(x, *params_quadratic)

RSS_linear = np.sum(residuals_linear**2)
RSS_quadratic = np.sum(residuals_quadratic**2)

n = len(y)
p_linear = 2
p_quadratic = 3

F = ((RSS_linear - RSS_quadratic) /
     (p_quadratic - p_linear)) / (
     RSS_quadratic / (n - p_quadratic)
)

p_value = f.sf(F, p_quadratic - p_linear, n - p_quadratic)

print(f"RSS linear: {RSS_linear:.6g}")
print(f"RSS quadratic: {RSS_quadratic:.6g}")
print(f"F-statistic: {F:.4f}")
print(f"p-value: {p_value:.6g}")