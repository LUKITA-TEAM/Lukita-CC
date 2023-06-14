from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import io
from tensorflow.keras.utils import load_img, img_to_array
from google.cloud import storage
from PIL import Image

#Load Dataset
credentials_path = 'credentials.json'
client = storage.Client.from_service_account_json(credentials_path)

bucket_name = 'dataset-lukita'
bucket = client.get_bucket(bucket_name)

#Load Model ML
model = tf.keras.models.load_model('model.h5')

#OUTPUT PREDICT SIMILAR IMAGE, CLASS, DESCRIPTION
class_names = ['Abstract', 'Fauvism', 'Fiber Art', 'Japanese', 'Pop Art', 'Romanticism']   

image_dim = 224

def transform_image(pillow_image):
    data = img_to_array(pillow_image)
    data = np.expand_dims(data, axis = 0)
    data = tf.image.resize(data, [image_dim, image_dim])
    return data

def predict(x):
    predictions = model(x)
    predictions = tf.nn.softmax(predictions)
    pred0 = predictions[0]
    label0 = np.argmax(pred0)
    return label0

def process_file(file):
    image_bytes = file.read()
    pillow_img = Image.open(io.BytesIO(image_bytes))
    tensor = transform_image(pillow_img)
    prediction = predict(tensor)
    return prediction
    
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            #Prediction & Explanation
            prediction = process_file(file)
            result = class_names[prediction]
            if result == 'Abstract':
                explanation = "Abstract is art that does not attempt to represent an accurate depiction of a visual reality but instead use shapes, colours, forms and gestural marks to achieve its effect"
            elif result == 'Fauvism':
                explanation ="Fauvism is the name applied to the work produced by a group of artists (which included Henri Matisse and André Derain) from around 1905 to 1910, which is characterised by strong colours and fierce brushwork."
            elif result == 'Fiber':
                explanation = "Fiber is a style of fine art which uses textiles such as fabric, yarn, and natural and synthetic fibers. It focuses on the materials and on the manual labor involved as part of its significance."
            elif result == 'Japanese':
                explanation = "Japanese is art from Japan from the Muromachi period (1392-1573) all the way to the Shōwa period (1926-1989)"
            elif result == 'Pop Art':
                explanation = "Pop Art is popular (designed for a mass audience), Transient (short-term solution), Expendable (easily forgotten), Low cost, Mass produced, Young (aimed at youth), Witty, Sexy, Gimmicky, Glamorous, Big business"
            else:
                explanation = "Romanticism (also the Romantic era or the Romantic period) was an artistic, literary, musical and intellectual movement that originated in Europe toward the end of the 18th century and in most areas was at its peak in the approximate period from 1800 to 1850. Romanticism was characterized by its emphasis on emotion and individualism as well as glorification of all the past and nature, preferring the medieval rather than the classical."
            
            #Related Image
            blobs = bucket.list_blobs(prefix=result + '/')
            
            image_data = []
            # Iterate over the blobs
            for blob in blobs:
                # Download the image blob
                image_data.append(blob.public_url)

            data = {
                'Prediction': result, 
                'Explanation': explanation, 
                'Related Image': image_data
                }
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"

@app.route("/galeri", methods=["GET"])
def galeri():
    types = ['Abstract', 'Fauvism', 'Fiber Art', 'Japanese', 'Pop Art', 'Romanticism']
    links = []
    for type in types:
        blobs = bucket.list_blobs(prefix=type + '/')
        for blob in blobs:
            links.append({
                "type": type,
                "url": blob.public_url
            })
    return jsonify(links)

if  __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))