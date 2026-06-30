from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train_model import train_model

data = load_data("data/StudentPerformanceFactors.csv")

X_train, X_test, y_train, y_test = preprocess_data(data)

model = train_model(X_train, y_train)

print("Model trained successfully!")