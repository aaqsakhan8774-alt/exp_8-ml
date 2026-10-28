import joblib 
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load('iris_model.pkl')

@app.route("/")
def home():
    return jsonify({"message": "App is running..."})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    s_l=data.get("sepal_length")
    s_w=data.get("sepal_width")
    p_l=data.get("petal_length")
    p_w=data.get("petal_width")
    pred=model.predict([[s_l, s_w, p_l, p_w]])
    return jsonify({"prediction": pred[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001) 