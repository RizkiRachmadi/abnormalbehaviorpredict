import numpy as np
from flask import Flask,request,jsonify
import pickle
import lightgbm

app = Flask(__name__)
model = pickle.load(open('models/lightgbm_6features.pkl', 'rb'))

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
    input_query = input_query.reshape(1,-1)
    result = model.predict(input_query)
    return (str(result))

if __name__ == '__main__':
    app.run()
