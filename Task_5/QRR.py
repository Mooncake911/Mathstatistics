from IPython.display import display

import statsmodels.api as sm

from importlib import reload
import regression as reg
import mathstatsplots as mth_plot
import mathstats as mth
reload(reg)
reload(mth_plot)
reload(mth)


# Квантильная регрессия
class QuantRegressionResearch(reg.Model):
    def __init__(self, x, y, degree=1, q=0.5, q_step=0.1):
        super().__init__(model=sm.QuantReg, x=x, y=y, degree=degree,
                         fit_params={'q': q, 'method': 'powell'})
        self.q_step = q_step
        self.quantile = q
        self.y_pred = self.results.predict(self.x)
        # self.influence = self.results.get_influence()
        self.residuals = self.results.resid

    def info(self):
        """ Выводим необходимые тесты и статистику """
        sep_str = '==============================================================='
        summary = self.results.summary2(title=self.y.name, alpha=0.05, float_format='%.4f')
        law_str = mth.law_func(self.results)  # формула
        het_str = mth.breuschpagan_test(self.results)  # тест на гетероскедастичность
        summary.add_text(sep_str + '\n' + het_str + '\n' + sep_str + '\n' + law_str)
        print(summary)

        print(sep_str)
        vif_tol_data = mth.vif_tol_test(self.results)  # тест на мультиколлинеарность
        display(vif_tol_data)

        print(sep_str)
        wald_data = mth.wald_test(self.results)  # анализ дисперсии модели
        display(wald_data)

        # print(sep_str)
        # anova_data = anova_lm(self.results)  # анализ дисперсии модели
        # display(anova_data)

    def draw_plots(self):
        """ Рисуем графики необходимые для анализа """
        mth_plot.pair_scatter_plots(df=self.df, degree=self.degree, q=self.quantile)
        mth_plot.residuals_plot(self.y_pred, self.residuals)
        mth_plot.params_quantiles_plot(x=self.x, y=self.y, q_step=self.q_step, params_names=self.results.params.index)
        mth_plot.qq_plot(self.residuals)
