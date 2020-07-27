import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open(r'C:\Users\Anushka Anil\Desktop\Diabetes\decision.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    features_value = [np.array(features)]
    features_name = ['preg', ' plas', 'pres', 'Skin ', 'test', 'mass', 'pedi', 'age']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)    
    if output == 1:
        res_val = "** Diabetes **"
    if output==0:
        res_val = "No Diabetes"
        

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
