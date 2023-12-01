import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from scipy.stats import zscore
from sklearn.metrics import confusion_matrix, roc_curve, auc

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


def roc_plot(fpr, tpr):
    """ ROC-кривая """
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right', frameon=True)
    plt.show()


def confusion_matrix_plot(y, y_pred):
    """ Матрица Несоответствий """
    conf_matrix = confusion_matrix(y, y_pred)
    cnf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    true_class_names = ['True', 'False']
    predicted_class_names = ['Predicted True', 'Predicted False']
    df_cnf_matrix = pd.DataFrame(conf_matrix, index=true_class_names, columns=predicted_class_names)
    df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, index=true_class_names, columns=predicted_class_names)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Confusion Matrix')
    sns.heatmap(df_cnf_matrix, annot=True, ax=axes[0])
    axes[0].title.set_text('Perceptron: values')
    sns.heatmap(df_cnf_matrix_percent, annot=True, ax=axes[1])
    axes[1].title.set_text('Perceptron: %')
