import pandas as pd
import numpy as np

import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor


def law_func(column, results):
    """ Вывод закона (уравнения) регрессии. """
    params = results.params
    params_names = results.params.index
    output = f'Law:\n{column} = '
    for i, c in enumerate(params_names):
        output += f'({params[i]}) * {c}'
        if i < len(params_names) - 1:
            output += ' + '
        if (i % 2 != 0) and (i != len(params_names) - 1):
            output += '\n'
    return output


def breuschpagan_test(residuals, exogenous):
    """ Проводим тест Бройша-Пагана (Breusch-Pagan test) на гетероскедастичность. """
    het_test = sms.het_breuschpagan(residuals, exogenous)
    output = f'Breusch-Pagan test: \n' \
             f'LM statistic: {het_test[0]}      LM-Test p-value: {het_test[1]:} \n' \
             f'F-statistic: {het_test[2]}       F-Test p-value: {het_test[3]:}'
    return output


def white_test(residuals, exogenous):
    het_test = sms.het_white(residuals, exogenous)
    output = f'White test: \n' \
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


def wald_test(results, use_f=True):
    wald_data = []
    params_names = list(results.params.index)
    len_pn = len(params_names)

    for p in range(len_pn + 1):
        if p == len_pn:
            r_matrix = np.eye(len_pn)
            r_matrix_values = list(r_matrix[:, i] for i in range(len_pn))
        else:
            r_matrix = np.eye(len_pn)[p]
            r_matrix_values = list(r_matrix[i] for i in range(len_pn))
        output = results.wald_test(r_matrix=r_matrix, use_f=use_f, scalar=False)
        wald_data.append(r_matrix_values + [output.fvalue[0][0], output.pvalue, output.df_denom, output.df_num])

    df = pd.DataFrame(wald_data, columns=params_names + ["F-statistic", "Prob (F-statistic)", "df_denom", "df_num"])
    return df
