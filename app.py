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


# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)

@ app.route('/check')
def check():
    title = 'Harvestify - Home'
    return render_template('dropdown.html', title=title)

# render crop recommendation form page
@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
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
    title = 'Harvestify - Crop Recommendation'

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
    title = 'Harvestify - Crop Clustering Recommendation'

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
    title = 'Harvestify - Crop Clustering Recommendation'

    if request.method == 'POST':
        crop = request.form.get("crop")
        final_prediction = "requirements _________"

        conditions = {'N': "Nitrogen", 'P': "Phosphorous", 'K': "Potassium", 'temperature': "Tempature", 'ph': "PH",
                      'humidity': "Relative Humidity", 'rainfall': "Rainfall"}
        x = data[data['label'] == crop]
        res = {}
        for key, value in conditions.items():
            res[key] = [round(x[key].min(), 2), round(x[key].mean(), 2), round(x[key].max(), 2)]

    return render_template('requirement_result.html', condition=res, crop=crop, title=title)



# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
