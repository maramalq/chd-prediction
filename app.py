import os
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pickle

# Create flask app
template_dir = os.path.abspath('./')
flask_app = Flask(__name__, template_folder=template_dir)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/results")
def Results():
    result = request.args['result']  # counterpart for url_for()
    result = session['result']  # counterpart for session
    return render_template("results.html", result=result)
    # return render_template("results.html", prediction_text)

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    print(prediction)
    if prediction == 0:
        result = "The patient does not have CHD"
    else:
        result = "The patient have CHD"
    session['result'] = result
    #return render_template("index.html", result=result)
    return redirect(url_for("Results", result = result))
    # return redirect("results.html", prediction_text = result)



if __name__ == "__main__":
    flask_app.secret_key = 'super secret key'
    flask_app.config['SESSION_TYPE'] = 'filesystem'
    flask_app.run()