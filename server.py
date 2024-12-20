from flask import Flask
from flask import render_template
from flask import request
from reshape import mnist_preprocess
app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/mnist_playground/", methods=["GET", "POST"])
def mnist_playground():
    positions = request.json
    mnist_preprocess(input_data=positions)
    print(positions)
    return "<h1>Hello</h1>"

app.run(debug=True)