import statsmodels.api as sm

from importlib import reload
import regression as reg
import mathstatsplots as mth_plot
import mathstats as mth

reload(reg)
reload(mth_plot)
reload(mth)

logit = sm.genmod.families.links.Logit()
probit = sm.genmod.families.links.Probit()


# Биномиальная регрессия
class BinomialRegressionResearch(reg.Model):
    def __init__(self, x, y, degree=1, family=logit, threshold=0.5):
        super().__init__(model=sm.GLM, x=x, y=y, degree=degree,
                         model_params={'family': sm.families.Binomial(link=family)})
        self.y_prob = self.results.predict(self.x)
        self.y_pred = (self.y_prob > threshold).astype(int)

        self.residuals = self.results.resid_deviance

    def info(self):
        """ Выводим необходимые тесты и статистику """
        sep_str = '=============================================================================='
        summary = self.results.summary(title=self.y.name)
        law_str = mth.law_func(self.results, family='GLM')
        het_str = mth.white_test(self.results)
        summary.add_extra_txt([het_str, sep_str, law_str])
        print(summary)

        mth.vif_tol_test(self.results)
        mth.wald_test(self.results)
        mth.summary_frame(self.results)

    def draw_plots(self):
        """ Рисуем графики необходимые для анализа """
        mth_plot.confusion_matrix_plot(self.y, self.y_pred)
        mth_plot.residuals_plot(self.y_prob, self.residuals)
        mth_plot.influence_plot(self.results)
        mth_plot.qq_plot(self.residuals)
        mth_plot.roc_plot(self.y, self.y_prob)
