import tensorflow as tf
from tensorflow import keras
import pydicom as dcm
import numpy as np
import io

#web frameworks
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
import uvicorn
import aiohttp
import asyncio

import os
import sys
import base64 
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
keras.mixed_precision.set_global_policy('mixed_float16')
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = Starlette()
base_model_path = './base-cnn-model/'
saved_model_path = './rsna-rnn-model/'
rnn_model = keras.models.load_model(saved_model_path)
cnn_model = keras.models.load_model(base_model_path)
extractor = keras.models.Sequential(cnn_model.layers[:-1])

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

def get_center_and_width(dicom):
    return tuple([int(x[0]) if type(x) == dcm.multival.MultiValue else int(x) for x in [dicom.WindowCenter, dicom.WindowWidth]])

def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    if mi == ma:
        return img-mi
    return (img - mi) / (ma - mi)

def window_filter(img, center, width, slope, intercept):
    out = np.copy(img)
    out = out*slope + intercept
    lowest_visible = center - width//2
    highest_visible = center + width//2
    
    out[out < lowest_visible] = lowest_visible
    out[out > highest_visible] = highest_visible
    return normalize_minmax(out) * 255

def get_img_tensor(img_path):
    dicom = dcm.dcmread(img_path, force=True)
    
    img = dicom.pixel_array
    center, width = get_center_and_width(dicom)
    slope, intercept = dicom.RescaleSlope, dicom.RescaleIntercept
    brain = window_filter(img, 40, 80, slope, intercept)
    subdural = window_filter(img, 80, 200, slope, intercept)
    tissue = window_filter(img, 40, 380, slope, intercept)
    
    return np.stack([brain, subdural, tissue], axis=2).astype(np.float32) / 255.

    

@app.route("/upload", methods = ["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)

@app.route("/classify-url", methods = ["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)

def predict_image_from_bytes(bytes):
    #load byte data into a stream
    img_file = io.BytesIO(bytes)
    try:
        img_data = get_img_tensor(img_file)
        img_file = Image.fromarray((img_data*255).astype(np.uint8))
        img_file.save("img.png")
        img_uri = base64.b64encode(open("img.png", 'rb').read()).decode('utf-8')
        
        data = tf.expand_dims(tf.cast(img_data, tf.float32), axis=0)
        feat = extractor.predict(data)
        pred = rnn_model.predict(tf.expand_dims(feat, axis=0)).flatten()
        pred_array = np.array(pred >= 0.5, dtype=np.bool)
        
        
    except:
        try:
            img_file = Image.open(img_file)
            img_file.save("img.png")
            img_uri = base64.b64encode(open("img.png", 'rb').read()).decode('utf-8')

            data = tf.expand_dims(tf.convert_to_tensor(np.array(img_file), dtype=tf.float32) / 255., axis=0)
            feat = extractor.predict(data)
            pred = rnn_model.predict(tf.expand_dims(feat, axis=0)).flatten()
            pred_array = np.array(pred >= 0.5, dtype=np.bool)
        except:
            return HTMLResponse(
            """
            <html>
                <body>
                    <h3> Your file was the wrong type! Please use either DICOM (.dcm) or PNG (.png) files.</h3>
                </body>
            </html>
            """
            )

    
    return HTMLResponse(
        f"""
        <html>
            <body>
                <h2> Results </h2>
                <h3> Raw probabilities: {pred} </h3>
                <h3> Predictions: {pred_array} </h3>
                <br>
                <h3> <u>Presence of intracranial hemorrhaging:</u> </h3>
                <p>&nbsp;Any: {pred_array[0]}</p>
                <p>&nbsp;Epidural: {pred_array[1]}</p>
                <p>&nbsp;Intraparenchymal: {pred_array[2]}</p>
                <p>&nbsp;Intraventricular: {pred_array[3]}</p>
                <p>&nbsp;Subarachnoid: {pred_array[4]}</p>
                <p>&nbsp;Subdural: {pred_array[5]}</p>
                <br>
                <p> <u>Your image:</u> </p> 
                <figure class="figure">
                    <img src="data:image/png;base64, {img_uri}" class = "figure-img">
                </figure>
            </body>
        </html>
        """
        )
        
@app.route("/")
def form(request):
        return HTMLResponse(
            """
            <h1> RSNA Intracranial Hemorrhage Detection ML Service</h1>
            <h3> By Keenan Hom </h3>
            
            <form action="/upload" method = "post" enctype = "multipart/form-data">
                <u> Select picture to upload: </u> <br> <p>
                1. <input type="file" name="file"><br><p>
                2. <input type="submit" value="Predict">
            </form>
            <br>
            """)
        
@app.route("/form")
def redirect_to_homepage(request):
        return RedirectResponse("/")
        
if __name__ == "__main__":
    if "serve" in sys.argv:
        port = int(os.environ.get("PORT", 8008)) 
        uvicorn.run(app, host = "0.0.0.0", port = port)
