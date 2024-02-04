from IPython.display import display
from functools import wraps

import os
import pandas as pd
import numpy as np

import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor

from typing import Union
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
Results = Union[RegressionResultsWrapper, GLMResultsWrapper]

current_file_path = os.path.abspath(__file__)


def display_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print('\n=== %s ===' % func.__name__)
        data = func(*args, **kwargs)
        display(data)
        return data
    return wrapper


def law_func(results: Results, family='OLS') -> str:
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


def criteria_test(results: Results) -> str:
    # Вдруг понадобится - (AIC и BIC есть в summary)
    # output = (f'Criteria of informativeness:\n'
    #           f'AIC (Akaikes Information Criterion): {results.info_criteria("AIC")}\n'
    #           f'BIC (Bayesian Information Criterion): {results.info_criteria("BIC")}\n'
    #           f'HQIC (Hannan-Quinn Information Criterion): {results.info_criteria("HQIC")}\n')

    output = (f'Mean squared error the model:           {results.mse_model}\n'
              f'Mean squared error of the residuals:    {results.mse_resid}\n'
              f'Total mean squared error:               {results.mse_total}\n'
              f'Mean absolute error:                    {np.mean(np.abs(results.resid))}')
    return output


def breuschpagan_test(results: Results) -> str:
    """ Проводим тест Бройша-Пагана (Breusch-Pagan test) на гетероскедастичность. """
    het_test = sms.het_breuschpagan(results.resid, results.model.exog)
    output = (f'Breusch-Pagan test:\n'
              f'LM statistic:                   {het_test[0]:.3f}   LM-Test p-value:                 {het_test[1]:.3f}\n'
              f'F-statistic:                    {het_test[2]:.3f}   F-Test p-value:                  {het_test[3]:.3f}')
    return output


def white_test(results: Results) -> str:
    """ Проводим тест Уайта (White test) на гетероскедастичность. """
    het_test = sms.het_white(results.resid_deviance, results.model.exog)
    output = (f'White test:\n'
              f'LM statistic:                   {het_test[0]:.3f}   F-statistic:                    {het_test[2]:.3f}\n'
              f'LM-Test p-value:                {het_test[1]:.3f}   F-Test p-value:                 {het_test[3]:.3f}')
    return output


@display_decorator
def wald_test(results: Results, use_f: bool = True) -> pd.DataFrame:
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


@display_decorator
def vif_tol_test(results: Results, exclude_const: bool = False) -> pd.DataFrame:
    """ Проверяем модель на мультиколлинеарность данных. """
    results = pd.DataFrame(results.model.exog[:, exclude_const:], columns=results.model.exog_names[exclude_const:])
    if len(results.columns) < 2:
        raise f"{current_file_path}: UserWarning: (VIF/Tolerance) test only valid for n>=2 ... continuing anyway, n={len(results.columns)}"
    df = pd.DataFrame()
    df["Variable"] = results.columns
    df["VIF"] = [variance_inflation_factor(results.values, i) for i in range(results.shape[1])]
    df["Tolerance"] = 1 / df["VIF"]
    return df


@display_decorator
def summary_frame(results: Results) -> pd.DataFrame:
    """ Таблица влиятельности для каждого наблюдения. """
    influence = results.get_influence()
    df = influence.summary_frame()
    # influence_measures = df.sort_values("cooks_d", ascending=False)
    return df


@display_decorator
def outlier_test(results: Results, method: str = 'bonferroni') -> pd.DataFrame:
    """ Тест на наличие выбросов. """
    df = results.outlier_test(method=method)
    return df
