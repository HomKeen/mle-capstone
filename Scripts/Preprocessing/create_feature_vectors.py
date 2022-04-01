"""
This module will create feature vectors and save each one in a NPY file, by extracting the feature vector after 
passing a DICOM image through the CNN feature exetractor, which has been pre-tuned.
"""


import os
import sys
import tensorflow as tf
import numpy as np

from PIL import Image
from tqdm import tqdm
from math import ceil
from joblib import Parallel, delayed
from tensorflow import keras


feature_dir = '/home/jupyter/extracted-features/' #directory to store all NPY files containing extracted features
train_img_dir = '/home/jupyter/rsna-intracranial-hemorrhage-detection/stage_2_train_imgs/' #directory containing all DICOM training images as PNGs
extractor_path = '/home/jupyter/base-cnn-model/checkpoint.ckpt/' #directory containing the model of the base CNN feature extractor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
keras.mixed_precision.set_global_policy('mixed_float16')
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

base_model = keras.models.load_model(extractor_path)
extractor = keras.models.Sequential(base_model.layers[:-1])


def get_img_tensor(img_path):
    return tf.convert_to_tensor(np.asarray(Image.open(img_path), dtype=np.float32) / 255.)


def precompute_features(img_dir):
    """
    Passes all the images through the extractor model first before training and save them in feature_dir as 
    feature vectors.
    """

    to_compute = set(x.split('.')[0] for x in os.listdir(train_img_dir))
    present = set(x.split('.')[0] for x in os.listdir(feature_dir))

    print(f'{len(to_compute.intersection(present))} feature vectors already present')
    to_compute = list(to_compute.difference(present))
    print(f'Computing {len(to_compute)} new feature vectors...')
    compute_batch_size = 200

    if len(to_compute) == 0:
        return

    for i in tqdm(range(ceil(len(to_compute)//compute_batch_size)+1)):
        batch_names = to_compute[i*compute_batch_size : (i+1)*compute_batch_size]
        batch = np.array(Parallel(n_jobs=-1, backend='threading')(delayed(get_img_tensor)(img_dir+img_id+'.png') for img_id in batch_names if img_id))
        try:
            batch = extractor.predict(batch)
        except:
            print("ERROR")
            print(batch.shape)
            
        for i,feat_vec in enumerate(batch):
            np.save(feature_dir+batch_names[i]+'.npy', feat_vec)

precompute_features(train_img_dir)
