import numpy as np
from flask import Flask, request, render_template, send_file
import pickle
import warnings
import pandas as pd
warnings.simplefilter("ignore", UserWarning)
import cat_dic as cat

# Create flask app
app = Flask(__name__)


# pediction function
def ValuePredictor(to_predict_list):
	to_predict = np.array(to_predict_list).reshape(1, 9)
	loaded_model = pickle.load(open("model.pkl", "rb"))
	result = loaded_model.predict(to_predict)
	return result[0]

@app.route("/")
def Home():
    print('Request for index page received')
    return render_template("upload.html")

@app.route("/ind")
def ind():
    print('Request for individual received')    
    return render_template("index.html", name = cat.name.items(), loc = cat.location.items(), cui = cat.cuisines.items(), type = cat.type.items(), city = cat.city.items() )

@app.route("/result", methods = ["POST"])
def result():
    print('Request for predict page received')
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
    return render_template("result.html", prediction_text = result)

@app.route("/predict_file", methods = ["POST"])
def predict_file():
    print('Request for Batch Prediction received')
    loaded_model = pickle.load(open("model.pkl", "rb"))
    df_in = pd.read_csv(request.files.get("file"))
    import batch as bt
    category_col = ['name','location','type','city','cuisines']
    for col in category_col:
        df_in[col] = df_in[col].map(bt.mapping_dict[col])
    result = loaded_model.predict(df_in)
    df_in['Rating'] = result.tolist()
    df_in.to_csv("Predicted.csv")
    return send_file('Predicted.csv', download_name='Predicted.csv')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)