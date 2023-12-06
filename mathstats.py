import os
import pandas as pd
import numpy as np

import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor

current_file_path = os.path.abspath(__file__)


def law_func(results, family='OLS'):
    """ Вывод закона (уравнения) регрессии. """
    params = results.params
    params_names = results.model.exog_names

    put = ''
    for i, c in enumerate(params_names):
        put += f'({params[i]:.4f}) * {c}'
        if i < len(params_names) - 1:
            put += ' + '
        if (i % 2 != 0) and (i != len(params_names) - 1):
            put += '\n'

    if family == 'OLS':
        output = f'Law:\n{results.model.endog_names} ~ {put}'
    else:
        output = f'Law:\nP({results.model.endog_names}) ~ G({put})'

    return output


def breuschpagan_test(results):
    """ Проводим тест Бройша-Пагана (Breusch-Pagan test) на гетероскедастичность. """
    het_test = sms.het_breuschpagan(results.resid, results.model.exog)
    output = f'Breusch-Pagan test: \n' \
             f'LM statistic: {het_test[0]:.3f} LM-Test p-value: {het_test[1]:.3f} \n' \
             f'F-statistic: {het_test[2]:.3f} F-Test p-value: {het_test[3]:.3f}'
    return output


def white_test(results):
    """ Проводим тест Уайта (White test) на гетероскедастичность. """
    het_test = sms.het_white(results.resid_deviance, results.model.exog)
    output = f'White test: \n' \
             f'LM statistic: {het_test[0]:.3f} LM-Test p-value: {het_test[1]:.3f} \n' \
             f'F-statistic: {het_test[2]:.3f} F-Test p-value: {het_test[3]:.3f}'
    return output


def wald_test(results, use_f: bool = True):
    """ Анализ дисперсии модели. """
    wald_data = []
    params_names = list(results.model.exog_names)
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


def vif_tol_test(results, exclude_const: bool = False):
    """ Проверяем модель на мультиколлинеарность данных. """
    results = pd.DataFrame(results.model.exog[:, exclude_const:], columns=results.model.exog_names[exclude_const:])
    if len(results.columns) < 2:
        return f"{current_file_path}: UserWarning: (VIF/Tolerance) test only valid for n>=2 ... continuing anyway, n={len(results.columns)}"
    vif_tol_data = pd.DataFrame()
    vif_tol_data["Variable"] = results.columns
    vif_tol_data["VIF"] = [variance_inflation_factor(results.values, i) for i in range(results.shape[1])]
    vif_tol_data["Tolerance"] = 1 / vif_tol_data["VIF"]
    return vif_tol_data
