import pydicom as dcm
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

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

img_data = get_img_tensor('./ID_000280440.dcm')
print(img_data.shape)
img_file = Image.fromarray((img_data*255).astype(np.uint8))
img_file.save("img.png")
img_uri = base64.b64encode(open("img.png", 'rb').read()).decode('utf-8')
