from IPython.display import display
import seaborn as sns
import pandas as pd

import statsmodels.api as sm
# from statsmodels.stats.anova import anova_lm

from importlib import reload
import mathstatsplots as mth_plot
import mathstats as mth
reload(mth_plot)
reload(mth)

sns.set()
pd.options.display.expand_frame_repr = False


# Линейная регрессия
class LinearRegressionResearch:
    def __init__(self, df, column):
        self.df = df
        self.column = column
        self.x = df.drop(columns=column)
        self.y = df[column]

        self.model = sm.OLS.from_formula(self.formula(), data=df)
        self.results = self.model.fit()
        self.y_pred = self.results.predict(self.x)

        self.influence = self.results.get_influence()
        self.residuals = self.results.resid

    def formula(self):
        x_columns = "+".join(self.x.columns)
        return f'{self.column} ~ {x_columns}'

    def info(self):
        sep_str = '=============================================================================='
        summary = self.results.summary(title=self.column)
        law_str = mth.law_func(self.results)  # формула
        het_str = mth.breuschpagan_test(self.results)  # тест на гетероскедастичность
        summary.add_extra_txt([law_str, sep_str, het_str])
        print(summary)

        print(sep_str)
        vif_tol_data = mth.vif_tol_test(self.x)  # тест на мультиколлинеарность
        display(vif_tol_data)

        print(sep_str)
        wald_data = mth.wald_test(self.results)  # анализ дисперсии модели
        display(wald_data)

        # print(sep_str)
        # anova_data = anova_lm(self.results, test="F")  # (не совсем) анализ дисперсии модели
        # display(anova_data)

        print(sep_str)
        influence_measures = self.influence.summary_frame()  # таблицы влиятельности для каждого наблюдения
        display(influence_measures)

        print(sep_str)
        outlier_test_results = self.results.outlier_test(method='bonferroni')  # тест на наличие выбросов.
        display(outlier_test_results)

    def draw_plots(self):
        """ Рисуем графики необходимые для анализа """
        mth_plot.pair_scatter_plots(self.df)
        mth_plot.residuals_plot(self.y_pred, self.residuals)
        mth_plot.influence_plot(self.influence)
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
