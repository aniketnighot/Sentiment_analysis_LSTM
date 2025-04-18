# app.py
from flask import Flask, render_template, request
from utils import load_model_and_vocab, preprocess
import torch
import torch.nn.functional as F

app = Flask(__name__)
model, vocab = load_model_and_vocab()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        review = request.form["review"]
        input_tensor = preprocess(review, vocab)
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            prediction = "Positive" if prob > 0.5 else "Negative"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
