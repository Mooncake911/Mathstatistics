from IPython.display import display
import seaborn as sns
import pandas as pd

import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations_with_replacement

from importlib import reload
import mathstatsplots as mth_plot
import mathstats as mth
reload(mth_plot)
reload(mth)

sns.set()
pd.options.display.expand_frame_repr = False

logit = sm.genmod.families.links.Logit()
probit = sm.genmod.families.links.Probit()


# Биномиальная регрессия
class BinomialRegressionResearch:
    def __init__(self, x, y, family=logit, degree=1, threshold=0.5):
        poly = PolynomialFeatures(degree=degree)

        self.y = y

        self.column_names = ['const']
        for d in range(1, degree + 1):
            combinations = list(combinations_with_replacement(list(x.columns), d))
            self.column_names += ['&'.join(comb) for comb in combinations]

        self.x = pd.DataFrame(poly.fit_transform(x.values), columns=self.column_names)

        self.family = family
        self.model = sm.GLM(self.y, self.x, family=sm.families.Binomial(link=self.family))
        self.results = self.model.fit()
        self.y_prob = self.results.predict(self.x)
        self.y_pred = (self.y_prob > threshold).astype(int)

        self.influence = self.results.get_influence()
        self.residuals = self.results.resid_deviance

    def formula(self):
        x_columns = ''
        for c in self.x.columns:
            if isinstance(self.x[c].dtype, pd.api.types.CategoricalDtype):
                x_columns += f'C({c})+'
            else:
                x_columns += f'{c}+'
        return f'{self.column} ~ {x_columns[:-1]}'

    def info(self):
        sep_str = '=============================================================================='
        summary = self.results.summary(title=self.y.name)
        law_str = mth.law_func(self.results, family='GLM')  # формула
        het_str = mth.white_test(self.results)  # тест на гетероскедастичность
        summary.add_extra_txt([law_str, sep_str, het_str])
        print(summary)

        print(sep_str)
        vif_tol_data = mth.vif_tol_test(self.results)  # тест на мультиколлинеарность
        display(vif_tol_data)

        print(sep_str)
        wald_data = mth.wald_test(self.results)  # анализ дисперсии модели
        display(wald_data)

        print(sep_str)
        influence_measures = self.influence.summary_frame()  # таблица влиятельности для каждого наблюдения
        # influence_measures = influence_measures.sort_values("cooks_d", ascending=False)
        display(influence_measures)

    def draw_plots(self):
        """ Рисуем графики необходимые для анализа """
        mth_plot.confusion_matrix_plot(self.y, self.y_pred)
        mth_plot.residuals_plot(self.y_prob, self.residuals)
        mth_plot.influence_plot(self.influence)
        mth_plot.qq_plot(self.residuals)
        mth_plot.roc_plot(self.y, self.y_prob)

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
            output += (f'Selected Features: {remaining_features[1:]}\n'
                       f'{criteria}: {best_criterion}\n')

            for index in range(1, len(remaining_features)):  # идём с 1 чтобы не удалить константу
                features = remaining_features[:index] + remaining_features[(index + 1):]
                model = sm.GLM(self.y, self.x[features], family=sm.families.Binomial(link=self.family)).fit()
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
