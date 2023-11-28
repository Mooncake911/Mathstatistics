from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


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
