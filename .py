import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
import os

os.makedirs("models", exist_ok=True)


def train_and_save(csv_path, model_path, label_col):
    print(f"\nTraining model for: {csv_path}")

    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        print(f"❌ ERROR: Label column '{label_col}' not found!")
        print("Columns =", list(df.columns))
        return

    X = df.drop(label_col, axis=1)
    y = df[label_col]

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    print("Numeric:", list(numeric_features))
    print("Categorical:", list(categorical_features))

    # Preprocessing: Impute + Scale/Encode
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42))
        ]
    )

    pipeline.fit(X, y)

    joblib.dump(pipeline, model_path)
    print(f"✅ Saved: {model_path}")


# -------------------- TRAIN MODELS --------------------

train_and_save("diabetes.csv", "models/diabetes_pipeline.pkl", "Outcome")

train_and_save("heart.csv", "models/heart_pipeline.pkl", "target")

train_and_save("liver.csv", "models/liver_pipeline.pkl", "Dataset")

train_and_save("kidney.csv", "models/kidney_pipeline.pkl", "classification")

print("\n🎉 All models trained successfully (with NaN handling)!")
