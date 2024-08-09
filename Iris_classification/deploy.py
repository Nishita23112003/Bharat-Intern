from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib as joblib
import os

model = joblib.load('saved_model1.pkl')
scaler = joblib.load('scaler.save')

app = Flask(__name__)

IMG_FOLDER = os.path.join('static', 'IMG')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        sl = request.form['SepalLength']
        sw = request.form['SepalWidth']
        pl = request.form['PetalLength']
        pw = request.form['PetalWidth']
        data = np.array([[sl, sw, pl, pw]], dtype=float)
        x = scaler.transform(data)
        prediction = model.predict(x)
        
        class_names = ['setosa', 'versicolor', 'virginica']
        predicted_class = class_names[prediction[0]]
        
        image = predicted_class + '.png'
        image = os.path.join(app.config['UPLOAD_FOLDER'], image)
        
        return render_template('index.html', prediction=predicted_class, image=image)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

