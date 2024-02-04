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
        self.residuals = self.results.resid

    def info(self):
        """ Выводим необходимые тесты и статистики """
        summary = self.results.summary2(title=self.y.name, alpha=0.05, float_format='%.4f')
        summary.add_text(mth.law_func(self.results))
        summary.add_text(mth.breuschpagan_test(self.results))
        summary.add_text(mth.criteria_test(self.results))
        print(summary)

        mth.vif_tol_test(self.results)
        mth.wald_test(self.results)
        mth.summary_frame(self.results)
        mth.outlier_test(self.results)

    def draw_plots(self):
        """ Рисуем графики необходимые для анализа """
        mth_plot.pair_scatter_plots(df=self.df, degree=self.degree)
        mth_plot.residuals_plot(self.y_pred, self.residuals)
        mth_plot.influence_plot(self.results)
        mth_plot.qq_plot(self.residuals)
