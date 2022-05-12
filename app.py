# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
import os
import keras as k
from PIL import Image
import PIL
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODEL -----------------------------------------------
# Loading crop recommendation model
crop_recommendation_model_path = 'models/crop_pred.sav'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

crop_recommendation_data_path = 'Data/data.csv'
# lets read the dataset
data = pd.read_csv(crop_recommendation_data_path)

# Loading plant disease classification model
disease_classes = ['Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Soybean___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

MODEL_PATH = 'models/disease_selected_100.h5'

disease_model = None
if os.path.exists(MODEL_PATH):
    disease_model = k.models.load_model(MODEL_PATH)
    print("model found")
else:
    print("Model not found")

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Agro Culture - Home'
    return render_template('index.html', title=title)

@ app.route('/check')
def check():
    title = 'Agro Culture - Home'
    return render_template('dropdown.html', title=title)

# render crop recommendation form page
@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Agro Culture - Crop Recommendation'
    return render_template('crop.html', title=title)


# render crop recommendation form page
@ app.route('/crop_cluster')
def crop_cluster():
    title = 'Crop __________ Recommendation'
    return render_template('crop_cluster.html', title=title)

# render crop recommendation form page
@ app.route('/requirement')
def requirement():
    title = 'Crop Requirement Recommendation'
    return render_template('requirement.html', title=title)

# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page
@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Agro Culture - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        print("*"*30)
        print(N, P, K, rainfall,ph)

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]

        return render_template('crop-result.html', prediction=final_prediction, title=title)

# render crop cluster recommendation result page
@ app.route('/crop_cluster_result', methods=['POST'])
def crop_clustering():
    title = 'Agro Culture - Crop Clustering Recommendation'

    if request.method == 'POST':
        crop = request.form.get("crop")
        final_prediction = ""

        clusters = {
                    "c1":{'papaya', 'coffee', 'rice', 'coconut', 'jute', 'pigeonpeas'},
                    "c2": {'orange', 'mothbeans', 'kidneybeans', 'mango', 'blackgram', 'mungbean', 'chickpea', 'lentil', 'pomegranate'},
                    "c3":{'apple', 'grapes'},
                    "c4":{'muskmelon', 'banana', 'maize', 'cotton', 'watermelon'}
                  }
        for i in clusters:
            if crop in clusters[i]:
                final_prediction = ", ".join(clusters[i]-{crop})
                break

        return render_template('crop_cluster_result.html', prediction=final_prediction, crop=crop, title=title)

# render crop cluster recommendation result page
@ app.route('/crop_requirement_result', methods=['POST'])
def crop_requirement():
    title = 'Agro Culture - Crop Clustering Recommendation'

    if request.method == 'POST':
        crop = request.form.get("crop")
        final_prediction = "requirements _________"

        conditions = {'N': "Nitrogen", 'P': "Phosphorous", 'K': "Potassium", 'temperature': "Tempature", 'ph': "PH",
                      'humidity': "Relative Humidity", 'rainfall': "Rainfall"}
        x = data[data['label'] == crop]
        res = {}
        for key, value in conditions.items():
            res[key] = [round(x[key].min(), 2), round(x[key].mean(), 2), round(x[key].max(), 2)]

    return render_template('requirement_result.html', condition=res, crop=crop, data=conditions, title=title)


def predict_image(img, model=disease_model):
    img = Image.open(io.BytesIO(img))
    size = (64, 64)
    img = img.resize(size, PIL.Image.ANTIALIAS)
    test_image = np.expand_dims(img, axis=0)
    result = model.predict(test_image)
    return result


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Agro Culture - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)
            print(prediction)
            prediction = prediction.tolist()[0]
            prediction = disease_classes[prediction.index(max(prediction))]
            return render_template('disease-result.html', prediction=prediction, title=title)
        except Exception as e:
            print(e)
    return render_template('disease.html', title=title)



# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
