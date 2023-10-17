import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

#Load model from pickle
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    
    # Make predictions using your loaded model (rfc)
    predictions = model.predict(features)

    formatted_price = "Rs {:,.2f}/-".format(predictions[0])

    return render_template('index.html', prediction_text="The price of the car is {}".format(formatted_price))

if __name__ == '__main__':
    app.run(debug=True)
