from IPython.display import display

import statsmodels.api as sm

from importlib import reload
import regression as reg
import mathstatsplots as mth_plot
import mathstats as mth
reload(reg)
reload(mth_plot)
reload(mth)


# Линейная регрессия
class LinearRegressionResearch(reg.Model):
    def __init__(self, x, y, degree=1):
        super().__init__(model=sm.OLS, x=x, y=y, degree=degree)
        self.y_pred = self.results.predict(self.x)
        self.influence = self.results.get_influence()
        self.residuals = self.results.resid

    def info(self):
        """ Выводим необходимые тесты и статистику """
        sep_str = '=================================================================='
        summary = self.results.summary2(title=self.y.name, alpha=0.05, float_format='%.4f')
        summary.add_text(mth.law_func(self.results))  # формула
        summary.add_text(sep_str)
        summary.add_text(mth.breuschpagan_test(self.results))  # тест на гетероскедастичность
        summary.add_text(sep_str)
        summary.add_text(mth.criteria_test(self.results))
        print(summary)

        print(sep_str)
        vif_tol_data = mth.vif_tol_test(self.results)  # тест на мультиколлинеарность
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
        mth_plot.pair_scatter_plots(df=self.df, degree=self.degree)
        mth_plot.residuals_plot(self.y_pred, self.residuals)
        mth_plot.influence_plot(self.influence)
        mth_plot.qq_plot(self.residuals)
