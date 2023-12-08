import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations_with_replacement


class Model:
    def __init__(self, model, x, y, degree, model_params=None, fit_params=None):
        self.model = model
        self.degree = degree
        self.fit_params = fit_params if fit_params is not None else {}
        self.model_params = model_params if model_params is not None else {}

        # Создаём полином степени degree и присваиваем имена для всех независимых переменных
        poly = PolynomialFeatures(degree=self.degree)
        self.column_names = ['const']
        for d in range(1, self.degree + 1):
            combinations = list(combinations_with_replacement(list(x.columns), d))
            self.column_names += ['&'.join(comb) for comb in combinations]
        self.y = y
        self.x = pd.DataFrame(poly.fit_transform(x.values), columns=self.column_names)

        self.df = pd.concat([self.y, self.x.drop(columns='const')], axis=1)

        self.results = self.model(self.y, self.x, **self.model_params).fit(**self.fit_params)

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
            output += (f'Selected Features: {remaining_features[1:]}\n'
                       f'{criteria}: {best_criterion}\n')

            for index in range(1, len(remaining_features)):  # идём с 1 чтобы не удалить константу
                features = remaining_features[:index] + remaining_features[(index + 1):]
                model = self.model(self.y, self.x[features], **self.model_params).fit(**self.fit_params)
                criterion = model.aic if criteria == 'AIC' else model.bic

                if criterion < best_criterion:
                    k = True
                    best_criterion = criterion
                    best_model = model
                    drop_index = index

            if k:
                remaining_features.pop(drop_index)

        print(output)
        return pd.concat([self.y, self.x[remaining_features[1:]]], axis=1)
