from IPython.display import display
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc

import statsmodels.api as sm
logit = sm.genmod.families.links.Logit()
probit = sm.genmod.families.links.Probit()
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set()
pd.options.display.expand_frame_repr = False


def norm_data(df, pass_columns=None):
    """ Нормализуем данные """
    if pass_columns is None:
        pass_columns = []
    for col in df.columns:
        label_encoder = LabelEncoder()
        scaler = MinMaxScaler()
        if df[col].dtype.kind in 'O':
            df[col] = label_encoder.fit_transform(df[col])
        elif df[col].dtype.kind in 'iufc':
            if (df[col].min() != 0 and df[col].max() != 1) and col not in pass_columns:
                df[col] = scaler.fit_transform(df[[col]])
            else:
                pass
    return df


# Биномиальная регрессия
class BinomialRegressionResearch:
    def __init__(self, df, column, family=None, threshold=None, influence_measures_filename=None):
        self.family = family
        self.filename = influence_measures_filename

        self.column = column
        self.x = df.drop(columns=column)
        self.y = df[column]

        self.model = sm.GLM.from_formula(self.formula(), family=sm.families.Binomial(link=self.family), data=df)
        self.results = self.model.fit()
        self.influence = self.results.get_influence()

        self.y_prob = self.results.predict(self.x)
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y, self.y_prob)
        self.roc_auc = auc(self.fpr, self.tpr)
        self.y_pred = self.optimal_y_pred() if threshold is None else (self.y_prob > threshold).astype(int)

    def formula(self):
        x_columns = ''
        for c in self.x.columns:
            if isinstance(self.x[c].dtype, pd.api.types.CategoricalDtype):
                x_columns += f'C({c})+'
            else:
                x_columns += f'{c}+'
        return f'{self.column} ~ {x_columns[:-1]}'

    def optimal_y_pred(self):
        # Находим оптимальное пороговое значение
        optimal_idx = np.argmax(self.tpr - self.fpr)
        optimal_threshold = self.thresholds[optimal_idx]
        return (self.y_prob > optimal_threshold).astype(int)

    def info(self):
        """ Вызов основной информации для анализа """
        # Использование библиотеки statsmodels для получения summary
        print(self.results.summary(title=self.column))

        # Вывод уравнения(закона) регрессии
        intercept = self.results.params[0]
        coefficients = self.results.params[1:]
        output_str = f'Law:\n{self.column} = '
        for i, c in enumerate(self.results.params.index[1:]):
            output_str += f'({coefficients[i]}) * {c} + '
            if i % 2 != 0:
                output_str += '\n'
        output_str += f'({intercept})'
        print(output_str)

        # Получение мер влиятельности для каждого наблюдения
        print('==============================================================================')
        influence_measures = self.influence.summary_frame()
        # influence_measures = influence_measures.sort_values("cooks_d", ascending=False)
        if self.filename is not None:
            influence_measures.to_csv(f'{self.filename}.csv', index=False)
        display(influence_measures)

    def draw_plots(self, ):
        """ Рисуем графики необходимые для анализа """
        # Residuals vs Leverage plot
        fig, ax = plt.subplots(figsize=(12, 8))
        fig = self.influence.plot_influence(criterion="cooks", size=25, plot_alpha=0.5, ax=ax)
        fig.tight_layout(pad=1.0)

        # Confusion Matrix plot
        conf_matrix = confusion_matrix(self.y, self.y_pred)
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

        # ROC plot
        plt.figure(figsize=(5, 5))
        plt.plot(self.fpr, self.tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {self.roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right', frameon=True)
        plt.show()

    def run_tests(self):
        """ Запуск дополнительных тестов """
        # Проверяем модель на мультиколлинеарность данных.
        print('==============================================================================')
        vif_tol_data = pd.DataFrame()
        vif_tol_data["Variable"] = self.x.columns
        vif_tol_data["VIF"] = [variance_inflation_factor(self.x.values, i) for i in range(self.x.shape[1])]
        vif_tol_data["Tolerance"] = 1 / vif_tol_data["VIF"]
        display(vif_tol_data)

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
                model = sm.GLM(self.y, sm.add_constant(self.x[features]), family=sm.families.Binomial(link=self.family)).fit()
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
