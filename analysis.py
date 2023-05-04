import numpy as np
from statsmodels.stats.outliers_influence import reset_ramsey, variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

filename = 'random_data.csv'
data = pd.read_csv(filename)

# Построение модели
model = smf.ols(formula='rent ~ cpu + ram + ssd + cpu_freq + bw', data=data).fit()

# Вывод результатов
print(model.summary())

# Проверка значимости параметров на 5%-ном уровне
alpha = 0.05
print("\nParameters and their affect on the 5% level:")
for param, pvalue in model.pvalues.items():
    is_significant = "significant" if pvalue < alpha else "insignificant"
    print(f"{param}: p-significance = {pvalue:.4f}, {is_significant}")
# Создание DataFrame с коэффициентами регрессии
coef_df = pd.DataFrame({'coefficients': model.params.values[1:]}, index=model.params.index[1:])

# Сортировка по убыванию значений коэффициентов
coef_df = coef_df.sort_values('coefficients', ascending=False)

reset_test = reset_ramsey(model, degree=2)

print("\nRamsey RESET Test Results:")
print(reset_test)

pvalue = reset_test.pvalue
is_specified = "correctly specified" if pvalue > alpha else "misspecified"
print(f"RESET Test p-value = {pvalue:.4f}, the model is {is_specified}")

# Вычисление VIF
X = data[['cpu', 'ram', 'ssd', 'cpu_freq', 'bw']]
X = sm.add_constant(X)
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nVariance Inflation Factors:")
print(vif)
# Выполнение теста Бройша-Пагана
bp_test = het_breuschpagan(model.resid, model.model.exog)

labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
result = dict(zip(labels, bp_test))

print("\nBreusch-Pagan Test Results:")
for key, value in result.items():
    print(f"{key}: {value:.4f}")

alpha = 0.05
if result['p-value'] < alpha:
    print("The test indicates the presence of heteroskedasticity.")
else:
    print("The test indicates no heteroskedasticity.")