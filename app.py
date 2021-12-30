import numpy as np
from flask import Flask,request,jsonify
import pickle

app = Flask(__name__)
# read our pickle file and label our logisticmodel as model
model = pickle.load(open('lightgbm_6features.pkl', 'rb'))

@app.route('/')
def home():
    return "connected to Heroku API"

@app.route('/predict',methods=['POST'])
def predict():
    acc_x = request.form.get('acc_x')
    acc_y = request.form.get('acc_y')
    acc_z = request.form.get('acc_z')
    gyr_x = request.form.get('gyr_x')
    gyr_y = request.form.get('gyr_x')
    gyr_z = request.form.get('gyr_x')
    input_query = np.array([[acc_x,acc_y,acc_z,
                             gyr_x,gyr_y,gyr_z]])
    result = model.predict(input_query)
    return jsonify({'placement':str(result)})

app.run()
