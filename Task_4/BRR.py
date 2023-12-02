from IPython.display import display
import pandas as pd

import seaborn as sns

from importlib import reload
import mathstats as mth
import mathstatsplots as mth_plot
reload(mth)
reload(mth_plot)

import statsmodels.api as sm
logit = sm.genmod.families.links.Logit()
probit = sm.genmod.families.links.Probit()

sns.set()
pd.options.display.expand_frame_repr = False


# Биномиальная регрессия
class BinomialRegressionResearch:
    def __init__(self, df, column, family=logit, threshold=0.5, influence_measures_filename=None):
        self.family = family
        self.filename = influence_measures_filename

        self.column = column
        self.x = df.drop(columns=column)
        self.y = df[column]

        self.model = sm.GLM.from_formula(self.formula(), family=sm.families.Binomial(link=self.family), data=df)
        self.results = self.model.fit()

        self.influence = self.results.get_influence()
        self.residuals = self.results.resid_deviance
        self.exogenous = self.results.model.exog

        self.y_prob = self.results.predict(self.x)
        self.y_pred = (self.y_prob > threshold).astype(int)

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
        summary = self.results.summary(title=self.column)
        law_str = mth.law_func(self.column, self.results)  # формула
        het_str = mth.white_test(self.residuals, self.exogenous)  # тест на гетероскедастичность
        summary.add_extra_txt([law_str, sep_str, het_str])
        print(summary)

        print(sep_str)
        vif_tol_data = mth.vif_tol_test(self.x)  # тест на мультиколлинеарность
        display(vif_tol_data)

        print(sep_str)
        wald_data = mth.wald_test(self.results)  # анализ дисперсии модели
        display(wald_data)

        print(sep_str)
        influence_measures = self.influence.summary_frame()  # таблица влиятельности для каждого наблюдения
        # influence_measures = influence_measures.sort_values("cooks_d", ascending=False)
        if self.filename is not None:
            influence_measures.to_csv(f'{self.filename}.csv', index=False)
        display(influence_measures)

    def draw_plots(self):
        """ Рисуем графики необходимые для анализа """
        mth_plot.confusion_matrix_plot(self.y, self.y_pred)
        mth_plot.plot_residuals(self.y_prob, self.residuals)
        mth_plot.plot_influence(self.influence)
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
            output += (f'Selected Features: {remaining_features} \n'
                       f'{criteria}: {best_criterion} \n')

            for index in range(len(remaining_features)):
                features = remaining_features[:index] + remaining_features[(index + 1):]
                model = sm.GLM(self.y, sm.add_constant(self.x[features]),
                               family=sm.families.Binomial(link=self.family)).fit()
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
