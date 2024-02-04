import statsmodels.api as sm

from importlib import reload
import regression as reg
import mathstatsplots as mth_plot
import mathstats as mth
reload(reg)
reload(mth_plot)
reload(mth)


# Гребневая регрессия
class RidgeRegressionResearch(reg.Model):
    def __init__(self, x, y, degree=1, alpha=1):
        super().__init__(model=sm.OLS, x=x, y=y, degree=degree)
        self.alpha = alpha
        self.results = sm.OLS(self.y, self.x).fit_regularized(alpha=self.alpha, L1_wt=0, refit=True)
        self.y_pred = self.results.predict(self.x)
        self.residuals = self.results.model.endog - self.y_pred

    def info(self):
        """ Выводим необходимые тесты и статистику """
        sep_str = '=============================================================================='
        # summary = self.results.summary()
        # law_str = mth.law_func(self.results)    # формула
        # het_str = mth.breuschpagan_test(self.results)   # тест на гетероскедастичность
        # summary.add_extra_txt([law_str, sep_str, het_str])
        # print(summary)

        mth.vif_tol_test(self.results)

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
        mth_plot.pair_scatter_plots(df=self.df, degree=self.degree, alpha=self.alpha)
        mth_plot.residuals_plot(self.y_pred, self.residuals)
        mth_plot.cross_validation_plot(self.x, self.y)
        mth_plot.qq_plot(self.residuals)
