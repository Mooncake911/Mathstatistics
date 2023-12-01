import pandas as pd

import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor


def law_func(column, results):
    """ Вывод закона (уравнения) регрессии. """
    coefficients = results.params
    coefficients_names = results.params.index
    output = f'Law:\n{column} = '
    for i, c in enumerate(coefficients_names):
        output += f'({coefficients[i]}) * {c}'
        if i < len(coefficients_names) - 1:
            output += ' + '
        if (i % 2 != 0) and (i != len(coefficients_names) - 1):
            output += '\n'
    return output


def breuschpagan_test(residuals, model):
    """ Проводим тест Бройша-Пагана (Breusch-Pagan test) на гетероскедастичность. """
    het_test = sms.het_breuschpagan(residuals, model.exog)
    output = f'Breusch-Pagan test: \n' \
             f'LM statistic: {het_test[0]}      LM-Test p-value: {het_test[1]:} \n' \
             f'F-statistic: {het_test[2]}       F-Test p-value: {het_test[3]:}'
    return output


def vif_tol_test(df):
    """ Проверяем модель на мультиколлинеарность данных. """
    vif_tol_data = pd.DataFrame()
    vif_tol_data["Variable"] = df.columns
    vif_tol_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif_tol_data["Tolerance"] = 1 / vif_tol_data["VIF"]
    return vif_tol_data
