import pandas as pd
import numpy as np

def load_and_preprocess_data(path):
    # Load dataset
    df = pd.read_csv(path)

    # Replace zeros with NaN for specific columns
    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

    # Fill missing values with median
    for col in cols_with_zero:
        df[col].fillna(df[col].median(), inplace=True)

    # Features and target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    return X, y
