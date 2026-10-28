import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model = joblib.load('iris_model.pkl')

test_df = pd.read_csv("Iris - Iris.csv")
test_df = test_df.drop("Id", axis=1)
X_test = test_df.drop("Species", axis=1)
y_test = test_df["Species"]
pred=model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(accuracy)

assert accuracy >=0.98
assert len(pred) == len(y_test)
print("All tests passed successfully!")