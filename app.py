#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = {'predicted': [prediction[0]]}

    return render_template('index.html', prediction_text='CO2 Emission of the vehicle is :{}'.format(output))

@app.route('/processjson', methods = ['POST'])
def processjson():
    data = request.get_json()

    # pull the inputs
    enginesize = data['enginesize']
    cylinders = data['cylinders']
    fuel = data['fuel']

    # get the prediction
    int_features = [enginesize, cylinders, fuel]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    
    return jsonify({'result' : 'Success!', 'enginesize': enginesize, 'cylinders': cylinders, 'fuel':fuel, 'prediction': int(prediction)})



if __name__ == "__main__":
    app.run(debug=True)
