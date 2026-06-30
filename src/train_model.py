from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


def train_model(preprocessor, X_train, X_test, y_train, y_test):

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    print("\nModel Performance")
    print("-----------------------")
    print("R² Score :", r2_score(y_test, predictions))
    print("Mean Absolute Error :", mean_absolute_error(y_test, predictions))

    joblib.dump(pipeline, "models/student_model.pkl")

    return pipeline