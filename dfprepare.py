from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def norm_data(df, ignore_columns=()):
    """ Нормализуем данные """
    df = df.copy()

    label_encoder = LabelEncoder()
    scaler = MinMaxScaler()
    for col in df.columns:
        df_kind = df[col].dtype.kind
        if col in ignore_columns:
            pass
        elif df_kind in 'b':
            df[col] = df[col].replace({True: 1, False: 0})
        elif df_kind in 'O':
            df[col] = label_encoder.fit_transform(df[col])
        elif df_kind in 'iufc':
            df[col] = scaler.fit_transform(df[[col]])

    return df
