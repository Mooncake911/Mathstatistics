from IPython.display import display
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge, RidgeCV

import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import ProbPlot

from importlib import reload
import mathstats as mth
import mathstatsplots as mth_plot
reload(mth)
reload(mth_plot)

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
        sep_str = '=============================================================================='
        summary = self.results.summary(title=self.column)

        coefficients = self.results.params
        coefficients_names = ['Intercept'] + list(self.x.columns)
        output_str = f'Law:\n{self.column} = '
        for i, c in enumerate(coefficients_names):
            output_str += f'({coefficients[i]}) * {c}'
            if i < len(coefficients_names) - 1:
                output_str += ' + '
            if i % 2 != 0:
                output_str += '\n'

        law_str = output_str[:-1]
        het_str = mth.breuschpagan_test(self.residuals, self.model)  # тест на гетероскедастичность
        summary.add_extra_txt([law_str, sep_str, het_str])
        print(summary)

        print(sep_str)
        vif_tol_data = mth.vif_tol_test(self.x)  # тест на мультиколлинеарность
        display(vif_tol_data)

        print(sep_str)
        anova_data = anova_lm(self.results)  # анализ дисперсии модели
        display(anova_data)

    def draw_plots(self):
        """ Рисуем графики необходимые для анализа """
        # Scatter plots
        scatter_plots = sns.pairplot(self.df, x_vars=self.df.columns, y_vars=self.df.columns, kind='reg')
        scatter_plots.fig.suptitle("Pair-plot with Regression Lines", y=1, fontsize=20)
        plt.show()

        # Other plots
        mth_plot.plot_residuals(self.results.predict(self.x), self.residuals)
        mth_plot.qq_plot(self.residuals)

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
        plt.legend(['CV Mean', 'Best Point', 'CV Interval'], loc='lower right', frameon=True)
        plt.show()

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
