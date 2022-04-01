# This script should be run immediately after unzipping the RSNA files
import numpy as np
import pydicom as dcm
import os
import pandas as pd
import sys
import tensorflow as tf
import PIL
import warnings
import csv

from tqdm import tqdm
from PIL import Image
from joblib import Parallel, delayed


img_dir_train = '/home/jupyter/rsna-intracranial-hemorrhage-detection/stage_2_train/'
png_dir_train = '/home/jupyter/rsna-intracranial-hemorrhage-detection/stage_2_train_imgs/'
img_dir_test = '/home/jupyter/rsna-intracranial-hemorrhage-detection/stage_2_test/'
png_dir_test = '/home/jupyter/rsna-intracranial-hemorrhage-detection/stage_2_test_imgs'
warnings.filterwarnings('ignore')

def is_invalid(img_path):
    try:
        dicom = dcm.dcmread(img_path, force=True)
        if dicom.pixel_array.shape != (512, 512):
            return True
        return False
    except:
        return True
    
def remove_if_invalid(img_path):
    if is_invalid(img_path):
        os.remove(img_path)
        return 1
    return 0


Parallel(n_jobs=-1)(delayed(remove_if_invalid)(img_dir_train+img_name) for img_name in tqdm(os.listdir(img_dir_train)))


"""
Part 2 of the script: extracts each DICOM file's pixel data, applies the three filters, and saves the result as an RGB PNG image
"""

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
    
    return np.stack([brain, subdural, tissue], axis=2).astype(np.int8)

def write_to_png(img_id, png_dir, img_dir):
    try:
        img_array = get_img_tensor(img_dir+img_id+'.dcm')
        if img_array.shape == (512,512,3):
            img = Image.fromarray(img_array, 'RGB')
            img.save(png_dir+img_id+'.png')
        
    except Exception as e:
        pass

def write_all_to_png(png_dir, img_dir):
    present = set(map(lambda x: x.split('.')[0], os.listdir(png_dir)))
    total = set(map(lambda x: x.split('.')[0], os.listdir(img_dir)))
    to_compute = list(total.difference(present))
    _ = Parallel(n_jobs=-1)(delayed(write_to_png)(img_name, png_dir, img_dir) for img_name in tqdm(to_compute))        

write_all_to_png(png_dir_train, img_dir_train)
write_all_to_png(png_dir_test, img_dir_test)


"""
Part 3 of the script: assemble all DICOM slices that belong the a CT scan, and write out their 
image IDs along with the patient ID to a CSV file. It will also save the (x,y,z) coordinates of each image 
and save this metadata into another csv file
"""


out_file_train_scans = '/home/jupyter/rsna-intracranial-hemorrhage-detection/train_ct_scans.csv'
out_file_test_scans = '/home/jupyter/rsna-intracranial-hemorrhage-detection/test_ct_scans.csv'
out_file_train_coords = '/home/jupyter/rsna-intracranial-hemorrhage-detection/train_ct_coords.csv'
out_file_test_coords = '/home/jupyter/rsna-intracranial-hemorrhage-detection/test_ct_coords.csv'

scans = {}
img_coords = []


def add_scans_and_process_coord(img_dir, img_name):
    d = dcm.dcmread(img_dir + img_name)
    img_id, patient_id = img_name.split('.')[0], d.PatientID
    coords = list(d.ImagePositionPatient)
    img_coords.append([img_id] + coords)
    
    del coords, d
    if patient_id in scans:
        scans[patient_id].append(img_id)
    else:
        scans[patient_id] = [img_id]
    del img_id, patient_id

def write_scans_and_coords(img_dir, scans_path, coords_path):
    _ = Parallel(n_jobs=-1, backend='threading', batch_size=5, require='sharedmem')(delayed(add_scans_and_process_coord)(img_dir, img_file_name) for img_file_name in tqdm(os.listdir(img_dir)))
    #write scans
    with open(scans_path, 'w') as output:
        writer = csv.writer(output)
        print(f'Writing scans to {scans_path}')
        for patient_id in tqdm(scans):
            writer.writerow([patient_id] + scans[patient_id])
    
    #write coords
    with open(coords_path, 'w') as output:
        writer = csv.writer(output)
        print(f'Writing coords to {coords_path}')
        for row in tqdm(img_coords):
            writer.writerow(row)


write_scans_and_coords(img_dir_train, out_file_train_scans, out_file_train_coords)
write_scans_and_coords(img_dir_test, out_file_test_scans, out_file_test_coords)
