from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def create_preprocessor(data):

    X = data.drop("Exam_Score", axis=1)

    categorical_columns = X.select_dtypes(include=["object"]).columns

    numerical_columns = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            ),
            (
                "num",
                "passthrough",
                numerical_columns
            )
        ]
    )

    return preprocessor