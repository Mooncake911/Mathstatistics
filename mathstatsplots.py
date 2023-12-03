import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from scipy.stats import zscore
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot


def plot_residuals(y_pred, residuals):
    """ Просто график распределения остатков """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs Fitted
    axes[0].scatter(y_pred, residuals)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_title('Residuals vs Fitted')
    axes[0].set_xlabel('Fitted values')
    axes[0].set_ylabel('Residuals')

    # Scale-Location
    axes[1].scatter(y_pred, zscore(residuals))
    axes[1].set_title('Scale-Location')
    axes[1].set_xlabel('Fitted values')
    axes[1].set_ylabel('\u221AStandardized residuals')

    plt.tight_layout()
    plt.show()


def plot_influence(influence):
    """ График влияния в регрессии """
    fig, ax = plt.subplots(figsize=(12, 8))
    influence.plot_influence(criterion="cooks", size=25, plot_alpha=0.5, ax=ax)
    plt.title('Residuals vs Leverage', fontsize=20)
    plt.show()


def qq_plot(residuals):
    """ Normal Q-Q plot """
    QQ = ProbPlot(zscore(residuals))
    QQ.qqplot(line='45', alpha=0.5, lw=1)
    plt.title('Normal Q-Q')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Standardized residuals')
    plt.show()


def roc_plot(y, y_prob):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    """ ROC-кривая """
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    axes[0].fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.2, label='Random')
    axes[0].scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red',
                    label=f'Best Threshold: {optimal_threshold:.2f}')
    axes[0].text(0.5, 0.5, f'AUC = {roc_auc:0.2f}', fontsize=20,
                 horizontalalignment='center', verticalalignment='center')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curve')
    axes[0].legend(loc='lower right', frameon=True)

    """ Построение графика кривой точности-полноты """
    precision, recall, thresholds = precision_recall_curve(y, y_prob)
    average_precision = average_precision_score(y, y_prob)
    optimal_idx = np.argmax(2 * (precision * recall) / (precision + recall))
    optimal_threshold = thresholds[optimal_idx]
    axes[1].step(recall, precision, color='b', alpha=0.2, where='post', label='PR curve')
    axes[1].fill_between(recall, precision, step='post', alpha=0.2, color='b')
    axes[1].scatter(recall[optimal_idx], precision[optimal_idx], marker='o', color='red',
                label=f'Best Threshold: {optimal_threshold:.2f}')
    axes[1].text(0.5, 0.5, f'AP = {average_precision:0.2f}', fontsize=20,
                 horizontalalignment='center', verticalalignment='center')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Precision-Recall (PR) Curve')
    axes[1].legend(loc='lower right', frameon=True)

    plt.tight_layout()
    plt.show()


def confusion_matrix_plot(y, y_pred):
    """ Матрица Несоответствий """
    conf_matrix = confusion_matrix(y, y_pred)
    cnf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    true_class_names = ['True', 'False']
    predicted_class_names = ['Predicted True', 'Predicted False']
    df_cnf_matrix = pd.DataFrame(conf_matrix, index=true_class_names, columns=predicted_class_names)
    df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, index=true_class_names, columns=predicted_class_names)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Confusion Matrix')
    sns.heatmap(df_cnf_matrix, annot=True, ax=axes[0])
    axes[0].title.set_text('Perceptron: values')
    sns.heatmap(df_cnf_matrix_percent, annot=True, ax=axes[1])
    axes[1].title.set_text('Perceptron: %')


def cross_validation_plot(x, y):
    """ График кросс-валидации alpha"""
    alphas = np.logspace(-6, 6, 13)
    ridge_model = RidgeCV(alphas=alphas, store_cv_values=True)
    ridge_model.fit(x, y)

    # Раньше была проверка что statsmodels и sklearn строят одинаковую Ridge-регрессию
    # print(ridge_model.coef_, self.results.params[1:])
    # mse_ridge = ((self.y - ridge_model.predict(self.x)) ** 2).mean()
    # mse_ols = ((self.y - self.results.predict(self.x)) ** 2).mean()
    # print(mse_ridge, mse_ols)

    plt.figure(figsize=(12, 8))
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


def params_quantiles_plot(x, y, q_step, params_names):
    # Coefficients vs Quantiles plot
    quantiles = np.arange(q_step, 1, q_step)

    fig, axs = plt.subplots(len(params_names), 1, figsize=(12, 20))
    fig.suptitle('Coefficients vs Quantiles\n')

    params_groups = np.empty((len(quantiles), len(params_names)))
    for i, quantile in enumerate(quantiles):
        quant_reg = sm.QuantReg(y, sm.add_constant(x))
        quant_reg_results = quant_reg.fit(q=quantile)
        params_groups[i] = quant_reg_results.params

    ols_reg_results = sm.OLS(y, sm.add_constant(x)).fit()
    add_params = ols_reg_results.params
    add_conf_intervals = ols_reg_results.conf_int()

    for i, p in enumerate(params_names):
        axs[i].set_xlim(0, 1)
        coefficients = params_groups[:, i]
        axs[i].plot(quantiles, coefficients, marker='o')
        axs[i].axhline(add_params[i], color='r')
        axs[i].fill_between(x=axs[i].get_xlim(),
                            y1=add_conf_intervals[0][i],
                            y2=add_conf_intervals[1][i],
                            color='darkred', alpha=0.1, lw=2, linestyle='--', edgecolor='red')
        axs[i].set_title(p)

    plt.tight_layout()
    plt.show()
