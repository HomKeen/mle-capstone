#web frameworks
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
import uvicorn
import aiohttp
import asyncio

from PIL import Image
from tensorflow import keras
import tensorflow as tf
import pydicom as dcm
import numpy as np
import io
import os
import sys
import base64 


#set up GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
keras.mixed_precision.set_global_policy('mixed_float16') #for extra speed, if applicable
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    #if a GPU is available, then allow full access to its memory
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = Starlette()
base_model_path = './base-cnn-model/'
saved_model_path = './rsna-rnn-model/'
rnn_model = keras.models.load_model(saved_model_path)
cnn_model = keras.models.load_model(base_model_path)
extractor = keras.models.Sequential(cnn_model.layers[:-1])


"""
The next 3 functions are for processing DICOM files. They combine to filter the raw pixel values into 3
windows, which are then assembled as a 3 channel image and saved as RGB.
"""
def get_center_and_width(dicom):
    return tuple([int(x[0]) if type(x) == dcm.multival.MultiValue else int(x) for x in [dicom.WindowCenter, dicom.WindowWidth]])

#normalize an image to be in the range [0,1]
def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    if mi == ma:
        return img-mi
    return (img - mi) / (ma - mi)

#apply a window filter to an image, based on some parameters contained in the DICOM file
def window_filter(img, center, width, slope, intercept):
    out = np.copy(img)
    out = out*slope + intercept
    lowest_visible = center - width//2
    highest_visible = center + width//2
    
    out[out < lowest_visible] = lowest_visible
    out[out > highest_visible] = highest_visible
    return normalize_minmax(out) * 255

#assembles the filtered image and returns it as a np.ndarray, given the path to a DICOM file
def get_img_tensor(img_path):
    dicom = dcm.dcmread(img_path, force=True)
    
    img = dicom.pixel_array
    center, width = get_center_and_width(dicom)
    slope, intercept = dicom.RescaleSlope, dicom.RescaleIntercept
    brain = window_filter(img, 40, 80, slope, intercept)
    subdural = window_filter(img, 80, 200, slope, intercept)
    tissue = window_filter(img, 40, 380, slope, intercept)
    
    return np.stack([brain, subdural, tissue], axis=2).astype(np.float32) / 255.


async def get_bytes(url):
    """
    Retrieves the raw bytes from an upload call
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()
    
@app.route("/upload", methods = ["POST"])
async def upload(request):
    """
    Retrieves data from a user upload and applies the model prediction to it.
    """
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)



def predict_image_from_bytes(bytes):
    """
    Given a stream of raw byte data representing a user's uploaded file, validate the file's type
    and then predict the presence of ICH.
    """
    #load byte data into a stream
    img_file = io.BytesIO(bytes)
    try:
        #try to first load as a DICOM file
        #save the DICOM file locally in order to display it on the results page
        img_data = get_img_tensor(img_file)
        img_file = Image.fromarray((img_data*255).astype(np.uint8))
        img_file.save("img.png")
        img_uri = base64.b64encode(open("img.png", 'rb').read()).decode('utf-8')
        
        #compute the prediction
        data = tf.expand_dims(tf.cast(img_data, tf.float32), axis=0)
        feat = extractor.predict(data)
        pred = rnn_model.predict(tf.expand_dims(feat, axis=0)).flatten()
        #get the binary predictions
        pred_array = np.array(pred >= 0.5, dtype=np.bool)
        
        
    except:
        try:
            #if not a DICOM file, try to laod as a PNG
            #save the PNG file locally in order to display it on the results page
            img_file = Image.open(img_file)
            img_file.save("img.png")
            img_uri = base64.b64encode(open("img.png", 'rb').read()).decode('utf-8')
            
            #compute the prediction
            data = tf.expand_dims(tf.convert_to_tensor(np.array(img_file), dtype=tf.float32) / 255., axis=0)
            feat = extractor.predict(data)
            pred = rnn_model.predict(tf.expand_dims(feat, axis=0)).flatten()
            #get the binary predictions
            pred_array = np.array(pred >= 0.5, dtype=np.bool)
        except:
            #if the PNG is invalid or the file is of some other type, return an error message.
            return HTMLResponse(
            """
            <html>
                <body>
                    <h3> Your file was the wrong type! Please use either DICOM (.dcm) or PNG (.png) files.</h3>
                </body>
            </html>
            """
            )

    #upon successful prediction, return some basic HTML containing the prediction results.
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
    """
    Basic HTML homepage for the UI
    """
    
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
    #run the app through container port 8008, which has been exposed by the Dockerfile.
    port = int(os.environ.get("PORT", 8008)) 
    uvicorn.run(app, host = "0.0.0.0", port = port)
