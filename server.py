from flask import Flask
from flask import render_template
from flask import request
from reshape import mnist_preprocess
from flask import jsonify
import numpy as np
app = Flask(__name__)

def relu(x):
    return (x > 0) * x

def softmax(x):
  numerator = np.exp(x)
  denominator = np.sum(np.exp(x))
  y = numerator / denominator
  return y

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/mnist_playground/", methods=["GET", "POST"])
def mnist_playground():
    positions = request.json
    image28by28 = mnist_preprocess(input_data=positions)
    weights_1, weights_2 = load_model()
    output = predict(image28by28, weights_1, weights_2)
    np.set_printoptions(suppress=True, precision=2, floatmode='fixed')
    inner_list = output[0]
    prediction_dictionary = {}
    for i in range(0, 10):
        prediction_dictionary[i] = inner_list[i]
    print(prediction_dictionary)
    return jsonify(prediction_dictionary)

def load_model(): 
    file = np.load("mnistV1.npz")
    weights_1 = file["weights_1"]
    weights_2 = file["weights_2"]
    return[weights_1, weights_2]

def predict(image28by28, weights_1, weights_2):
    reshapedimage784 = image28by28.reshape(1, 784)
    layer_1 = reshapedimage784
    layer_2 = relu(np.dot(layer_1, weights_1)) 
    layer_3 = softmax(np.dot(layer_2, weights_2))
    return layer_3 * 100

app.run(debug=True)