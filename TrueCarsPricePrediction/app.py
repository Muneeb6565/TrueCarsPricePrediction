from flask import Flask, render_template,request
from Feature_Manipulator import Feature_Manipulator
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('PIPELINE.pkl', 'rb'))

@app.route('/', methods = ['GET','POST'])
def home():
    return render_template('prediction_page.html')

@app.route('/result', methods = ['POST'])
def data_incomming(mileage=None):

    dict = {
        'Price': int(request.form.get('price')),
        'Year': int(request.form.get('year')),
        'Mileage': int(request.form.get('mileage')),
        'City': request.form.get('city'),
        'Vin': request.form.get('vin'),
        'Make': request.form.get('make'),
        'Model': request.form.get('model'),
    }
    dict = pd.DataFrame.from_dict([dict])
    prediction = model.predict(dict)
    print(prediction)
    prediction = int(prediction[0])

    return render_template('result.html',label = prediction)

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=8080)


