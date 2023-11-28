from IPython.display import display
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.evaluate import bias_variance_decomp
from scipy.stats import zscore

import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import ProbPlot

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

sns.set()
pd.options.display.expand_frame_repr = False


class RidgeRegressionResearch:
    def __init__(self, df, column, influence_measures_filename=None):
        self.filename = influence_measures_filename

        self.df = df
        self.column = column
        self.x = df.drop(columns=column)
        self.y = df[column]

        self.model = sm.OLS.from_formula(self.formula(), data=df)
        self.results = self.model.fit_regularized(alpha=1, L1_wt=1e-6, refit=True)
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
        coefficients = self.results.params[1:]
        coefficients_names = self.x.columns
        output_str = f'Law:\n{self.column} = '
        for i, c in enumerate(coefficients_names):
            output_str += f'({coefficients[i]}) * {c}'
            if i < len(coefficients_names) - 1:
                output_str += ' + '
            if i % 2 != 0:
                output_str += '\n'
        print(output_str)

        # Проведём анализ дисперсии модели
        print('==============================================================================')
        anova_result = anova_lm(self.results)
        display(anova_result)

        # # Получение мер влиятельности для каждого наблюдения
        # print('==============================================================================')
        # influence_measures = self.influence.summary_frame()
        # if self.filename is not None:
        #     influence_measures.to_csv(f'{self.filename}.csv', index=False)
        # display(influence_measures)

        # # Проводим тест на наличие выбросов.
        # outlier_test_results = self.results.outlier_test(method='bonferroni')
        # display(outlier_test_results)

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

        # # Residuals vs Leverage plot
        # fig, ax = plt.subplots(figsize=(12, 8))
        # sm.graphics.influence_plot(self.results, criterion="cooks", size=25, plot_alpha=0.5, ax=ax)
        # plt.title('Residuals vs Leverage', fontsize=20)
        # plt.show()

        # Оцените Ridge-модель с кросс-валидацией
        alphas = np.logspace(-6, 6, 13)
        ridge_model = RidgeCV(alphas=alphas, store_cv_values=True)
        ridge_model.fit(self.x, self.y)

        # print(ridge_model.coef_, self.results.params[1:])
        # mse_ridge = ((self.y - ridge_model.predict(self.x)) ** 2).mean()
        # mse_ols = ((self.y - self.results.predict(self.x)) ** 2).mean()
        # print(mse_ridge, mse_ols)

        # Cross-Validation plot
        plt.figure(figsize=(10, 6))
        cv_mean = np.mean(ridge_model.cv_values_, axis=0)
        cv_std = np.std(ridge_model.cv_values_, axis=0)

        best_alpha = ridge_model.alpha_
        min_mse = np.min(cv_mean)

        plt.semilogx(ridge_model.alphas, cv_mean, marker='o', zorder=1)
        plt.scatter(best_alpha, min_mse, color='red', marker='x', zorder=2)
        plt.text(best_alpha, min_mse, f'Best alpha: {best_alpha}\nMin MSE: {min_mse:.4f}', verticalalignment='top',
                 horizontalalignment='left', color='red')
        plt.fill_between(ridge_model.alphas, cv_mean - cv_std, cv_mean + cv_std, alpha=0.2)

        plt.xlabel('log(alpha)')
        plt.ylabel('Mean Squared Error')
        plt.title('Cross-Validation Plot for Ridge Regression')
        plt.legend(['CV Mean', 'CV Interval', 'Best Point'], loc='lower right', frameon=True)
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
                model = sm.OLS(self.y, sm.add_constant(self.x[features])).fit_regularized(alpha=1, L1_wt=1e-6,
                                                                                          refit=True)
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
