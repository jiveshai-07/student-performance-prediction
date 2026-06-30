from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import create_preprocessor
from src.train_model import train_model

# Load dataset
data = load_data("data/StudentPerformanceFactors.csv")

# Separate Features and Target
X = data.drop("Exam_Score", axis=1)
y = data["Exam_Score"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create preprocessing pipeline
preprocessor = create_preprocessor(data)

# Train the model
model = train_model(
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
)

print("\nPipeline Model Trained Successfully!")