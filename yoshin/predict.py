from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# モデルの読み込み
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            features = [float(data[f"A{i}"]) for i in range(1, 16)]
            pred = model.predict_proba([features])[0][1]
            return jsonify({'score': pred})
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    elif request.method == 'GET':
        try:
            features = [float(request.args.get(f"A{i}")) for i in range(1, 16)]
            pred = model.predict_proba([features])[0][1]
            return f"score: {pred}"
        except Exception as e:
            return f"エラー: {str(e)}", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
