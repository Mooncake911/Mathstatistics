from IPython.display import display
import seaborn as sns
import pandas as pd

import statsmodels.api as sm
# from statsmodels.stats.anova import anova_lm
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations_with_replacement

from importlib import reload
import mathstatsplots as mth_plot
import mathstats as mth
reload(mth_plot)
reload(mth)

sns.set()
pd.options.display.expand_frame_repr = False


# Гребневая регрессия
class RidgeRegressionResearch:
    def __init__(self, x, y, degree=1, alpha=1):
        self.degree = degree
        poly = PolynomialFeatures(degree=self.degree)

        self.y = y

        self.column_names = ['const']
        for d in range(1, self.degree + 1):
            combinations = list(combinations_with_replacement(list(x.columns), d))
            self.column_names += ['&'.join(comb) for comb in combinations]

        self.x = pd.DataFrame(poly.fit_transform(x.values), columns=self.column_names)

        self.alpha = alpha
        self.model = sm.OLS(self.y, self.x)
        self.results = self.model.fit_regularized(alpha=self.alpha, L1_wt=0, refit=True)
        self.y_pred = self.results.predict(self.x)

        # self.influence = self.results.get_influence()
        self.residuals = self.results.model.endog - self.y_pred

    def info(self):
        sep_str = '=============================================================================='
        # summary = self.results.summary()
        # law_str = mth.law_func(self.results)    # формула
        # het_str = mth.breuschpagan_test(self.results)   # тест на гетероскедастичность
        # summary.add_extra_txt([law_str, sep_str, het_str])
        # print(summary)

        print(sep_str)
        vif_tol_data = mth.vif_tol_test(self.results)  # тест на мультиколлинеарность
        display(vif_tol_data)

        # print(sep_str)
        # wald_data = mth.wald_test(self.results)  # анализ дисперсии модели
        # display(wald_data)

        # print(sep_str)
        # anova_data = anova_lm(self.results)  # (не совсем) анализ дисперсии модели
        # display(anova_data)

        # print(sep_str)
        # influence_measures = self.influence.summary_frame()  # таблицы влиятельности для каждого наблюдения
        # if self.filename is not None:
        #     influence_measures.to_csv(f'{self.filename}.csv', index=False)
        # display(influence_measures)

        # print(sep_str)
        # outlier_test_results = self.results.outlier_test(method='bonferroni')  # тест на наличие выбросов.
        # display(outlier_test_results)

    def draw_plots(self):
        """ Рисуем графики необходимые для анализа """
        df = pd.concat([self.y, self.x.drop(columns='const')], axis=1)
        mth_plot.pair_scatter_plots(df=df, degree=self.degree, alpha=self.alpha)
        mth_plot.residuals_plot(self.y_pred, self.residuals)
        mth_plot.cross_validation_plot(self.x, self.y)
        mth_plot.qq_plot(self.residuals)

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
        best_criterion = 100

        k = True
        drop_index = None
        while k:
            k = False
            output += (f'Selected Features: {remaining_features[1:]}\n'
                       f'{criteria}: {best_criterion}\n')

            for index in range(1, len(remaining_features)):  # идём с 1 чтобы не удалить константу
                features = remaining_features[:index] + remaining_features[(index + 1):]
                model = sm.OLS(self.y, self.x[features]).fit_regularized(alpha=self.alpha, L1_wt=1e-6, refit=True)
                criterion = model.aic if criteria == 'AIC' else model.bic

                if criterion < best_criterion:
                    k = True
                    best_criterion = criterion
                    best_model = model
                    drop_index = index

            if k:
                remaining_features.pop(drop_index)

        print(output)
        return pd.concat([self.y, self.x[remaining_features[1:]]], axis=1)
