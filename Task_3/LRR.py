from IPython.display import display
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from scipy.stats import boxcox
from scipy.stats import zscore

import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import ProbPlot

sns.set()
pd.options.display.expand_frame_repr = False


# Подготовка данных
def norm_data(df, pass_columns=None):
    """ Нормализуем данные """
    if pass_columns is None:
        pass_columns = []
    for col in df.columns:
        label_encoder = LabelEncoder()
        scaler = MinMaxScaler()
        if df[col].dtype.kind in 'O':
            df[col] = label_encoder.fit_transform(df[col])
        elif df[col].dtype.kind in 'iufc':
            if (df[col].min() != 0 and df[col].max() != 1) and col not in pass_columns:
                df[col] = scaler.fit_transform(df[[col]])
            else:
                pass
    return df


# Линейная регрессия
class LinearRegressionResearch:
    def __init__(self, df, column, influence_measures_filename=None):
        self.filename = influence_measures_filename

        self.df = df
        self.column = column
        self.x = df.drop(columns=column)
        self.y = df[column]

        self.model = sm.OLS.from_formula(self.formula(), data=df)
        self.results = self.model.fit()
        self.influence = self.results.get_influence()

        self.residuals = self.results.resid
        
    def formula(self):
        x_columns = "+".join(self.x.columns)
        return f'{self.column} ~ {x_columns}'

    def info(self):
        """ Вызов основной информации для анализа """
        # Использование библиотеки statsmodels для получения summary
        print(self.results.summary(title=self.column))

        # Вывод уравнения(закона) регрессии
        intercept = self.results.params[0]
        coefficients = self.results.params[1:]
        output_str = f'Law:\n{self.column} = '
        for i, c in enumerate(self.results.params.index[1:]):
            output_str += f'({coefficients[i]}) * {c} + '
            if i % 2 != 0:
                output_str += '\n'
        output_str += f'({intercept})'
        print(output_str)

        # Проведём анализ дисперсии модели
        print('==============================================================================')
        anova_result = anova_lm(self.results)
        display(anova_result)

        # Получение мер влиятельности для каждого наблюдения
        print('==============================================================================')
        influence_measures = self.influence.summary_frame()
        if self.filename is not None:
            influence_measures.to_csv(f'{self.filename}.csv', index=False)
        display(influence_measures)

        # Проводим тест на наличие выбросов.
        outlier_test_results = self.results.outlier_test(method='bonferroni')
        display(outlier_test_results)

    def draw_plots(self):
        """ Рисуем графики необходимые для анализа """
        # Scatter plots
        scatter_plots = sns.pairplot(self.df, x_vars=self.df.columns, y_vars=self.df.columns, kind='reg')
        scatter_plots.fig.suptitle("Pair-plot with Regression Lines", y=1, fontsize=20)
        plt.show()

        # Residuals vs Fitted
        plt.scatter(self.results.predict(self.x), self.residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals vs Fitted')
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.show()

        # Residuals vs Fitted
        plt.scatter(self.results.predict(self.x), np.sqrt(zscore(self.residuals)))
        plt.title('Scale-Location')
        plt.xlabel('Fitted values')
        plt.ylabel('\u221AStandardized residuals')
        plt.show()

        # Normal Q-Q plot
        QQ = ProbPlot(zscore(self.residuals))
        QQ.qqplot(line='45', alpha=0.5, lw=1)
        plt.title('Normal Q-Q')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Standardized residuals')
        plt.show()

        # Residuals vs Leverage plot
        fig, ax = plt.subplots(figsize=(12, 8))
        sm.graphics.influence_plot(self.results, criterion="cooks", size=25, plot_alpha=0.5, ax=ax)
        plt.title('Residuals vs Leverage', fontsize=20)
        plt.show()

    def run_tests(self):
        """ Запуск дополнительных тестов """
        print('==============================================================================')

        # Проводим тест Бройша-Пагана (Breusch-Pagan test) на гетероскедастичность.
        het_test = sms.het_breuschpagan(self.residuals, self.model.exog)
        print(f'Breusch-Pagan test: \n'
              f'LM statistic: {het_test[0]}      LM-Test p-value: {het_test[1]:} \n'
              f'F-statistic: {het_test[2]}       F-Test p-value: {het_test[3]:}')

        print('==============================================================================')

        # Проверяем модель на мультиколлинеарность данных.
        vif_tol_data = pd.DataFrame()
        vif_tol_data["Variable"] = self.x.columns
        vif_tol_data["VIF"] = [variance_inflation_factor(self.x.values, i) for i in range(self.x.shape[1])]
        vif_tol_data["Tolerance"] = 1 / vif_tol_data["VIF"]
        display(vif_tol_data)

    def stepwise_selection(self, criteria: str = 'AIC'):
        """
            Улучшаем модель при помощи:
                ::AIC (Akaike Information Criterion)
                                или
                ::BIC (Bayesian Information Criterion)
        """

        output = (f'                                STEPS {criteria}                              \n'
                  f'==============================================================================\n')

        remaining_features = list(self.x.columns)

        best_model = self.results
        best_criterion = best_model.aic if criteria == 'AIC' else best_model.bic

        k = True
        drop_index = None
        while k:
            k = False
            output += (f'Selected Features: {remaining_features} \n'
                       f'{criteria}: {best_criterion} \n')

            for index in range(len(remaining_features)):
                features = remaining_features[:index] + remaining_features[(index + 1):]
                model = sm.OLS(self.y, sm.add_constant(self.x[features])).fit()
                criterion = model.aic if criteria == 'AIC' else model.bic

                if criterion < best_criterion:
                    k = True
                    best_criterion = criterion
                    best_model = model
                    drop_index = index

            if k:
                remaining_features.pop(drop_index)

        print(output)
        return best_model, remaining_features


