"""
OVERVIEW:
This prototype is a fully functional custom class that trains the RNN classifier on slice windows surrounding a given CT image slice (i.e., the nearby slices in the CT scan). 
It uses extracted feature vectors from a previously tuned CNN (using ResNet weights) which predicted intracranial hemorrhaging (ICH) on only single slices, instead of slice windows. 
The idea is to use nearby slices to make a more informed decision on the presence of ICH, while also using extracted features to drastically reduce computational demand. 
The output is a prediction of ICH for all slices in the window, and the network updates its weights based on the loss of every prediction in the slice. 
We may also use this network to predict on entire CT scans at once, however this is computational infeasible during training but may be helpful during testing. 
Given a directory of preprocessed data and some CSV files for metadata and labels, the model can be constructed and trained in a few lines. 
It may also be instantiated by a previously trained instance of this model to continue training. 

SCALABILITY:
As the training dataset alone contains 750,000+ DICOM images, each containing a 512x512 pixel image, 
I had to design this model to be able to handle a very large amount of input from the ground up. 

Since each DICOM image has a varying range of pixel values, I normalized each into [0,255], computed their window images,
and saved it as a 512x512x3 RGB image in a PNG file, allowing for faster access than a DICOM file. I used ResNet-101 ImageNet weights from TensorFlow (to reduce training time) 
and tuned them to make a decent prediction on single DICOM images. This network is trained on a subset of the network until convergence, 
and it is not necessary to train it further since its only use beyond this point is a feature extractor. 
The RNN can be trained on a random subset of the data of desired size, or on the whole dataset, making it very scalable. 
The model will first compute all the feature vectors on the subset/set of data, store them in NPY files, and access them during training;
 this precomputation means that we can train much quicker than if we compute each feature vector as we go. Additionally, as the slice windows may overlap, 
 we want to avoid re-computing features for the same image. The model is also designed to be trained on an NVIDIA GPU.

All these implementation decisions contribute to a scalable model. I had to make a compromise for the sake of computational speed
 by using feature vectors from a tuned CNN, however when tuning this CNN it was clear that its predictions where only decent; 
 this may have an impact on the quality of the feature vectors. I also require lots of storage space in order to store the feature vectors, 
 which was another compromise for speed.

"""


import pydicom as dcm
import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
import math
import csv
import time

from math import ceil
from tqdm import tqdm
from PIL import Image
from tensorflow import keras
from tensorflow.keras.layers import *
from joblib import Parallel, delayed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
keras.mixed_precision.set_global_policy('mixed_float16')
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_img_dir = '~/rsna-intracranial-hemorrhage-detection/stage_2_train_imgs/'
train_label_path = '~/rsna-intracranial-hemorrhage-detection/train_labels.csv'
train_ct_path = '~/rsna-intracranial-hemorrhage-detection/train_ct_scans.csv'
train_coord_path = '~/rsna-intracranial-hemorrhage-detection/train_ct_coords.csv'
base_model_path = '~/base-cnn-model/checkpoint.ckpt'

saved_model_path = '~/Notebooks/Training/rnn-model-checkpoints/checkpoint.ckpt' 

test_img_dir = '~/rsna-intracranial-hemorrhage-detection/stage_2_test_imgs/'

n_epochs = 6



def get_img_tensor(img_path):
    return tf.convert_to_tensor(np.asarray(Image.open(img_path), dtype=np.float32) / 255.)


class RSNASequence(keras.utils.Sequence):
    """
    A keras Sequence which provides training data to the model
    """
    def __init__(self, labels, train_cutoff, batch_size, extractor, img_dir, n_slices,                 train_img_ct, train_img_ct_ind, feature_dir='~/extracted-features/', ):
        self.x = None
        self.y = labels #DataFrame of all labels for the whole training set
        self.train_cutoff = train_cutoff #number of images to predict for in one epoch
        self.batch_size = batch_size #number of image clusters to train on in one batch
        self.img_dir = img_dir #directory containing PNG images for training
        self.n_slices = n_slices #number of slices both above and below, to collect in a single training data point
        self.extractor = extractor #extractor model
        self.feature_dir = feature_dir #directory containing .npy files which are feature vectors of each image. The directory may be incomplete;
                                        #if so, this class will automatically generate feature vectors as needed
        self.train_img_ct = train_img_ct
        self.train_img_ct_ind = train_img_ct_ind
        
        self.on_epoch_end() #initialize a random subset of data
    
    def get_nearby_slices_names(self, img_id, n_slices):
        """
        :img_id (str): the ID of the slice to find nearby slices for
        :n_slices (int): number of slices, both above and below, to return.
        
        :return: list of 2*n_slices + 1 IDs of nearby images
        
        Gets a list of names of the nearby image slices, rather than their feature vectors or raw image values
        """
        ct_ind, ind = self.train_img_ct_ind[img_id]['ct_ind'], self.train_img_ct_ind[img_id]['ind']
        ct = self.train_img_ct[ct_ind]
        n = len(ct)
        low, high = ind-n_slices, ind+n_slices
        if low < 0:
            high += abs(low)
            low = 0
        elif high >= n:
            low -= high - (n-1)
            high = n-1
        return ct[low:high+1]
    
    def get_nearby_slices(self, img_id, n_slices):
        '''
        :img_id (str): the ID of the slice to find nearby slices for
        :n_slices (int): number of slices, both above and below, to return.
        
        :return: list of 2*n_slices + 1 images as TensorFlow tensor
        
        Will retrieve n_slices slices from BOTH above and below the given image. If there is not enough space, it will
        add more slices either below (if the image is near the top of the scan) or above (if the image is near the bottom of the scan).
        Exactly 2*n_slices + 1 images will be returned.
        '''
        ct_ind, ind = self.train_img_ct_ind[img_id]['ct_ind'], self.train_img_ct_ind[img_id]['ind']
        ct = self.train_img_ct[ct_ind]
        n = len(ct)
        low, high = ind-n_slices, ind+n_slices
        if low < 0:
            high += abs(low)
            low = 0
        elif high >= n:
            low -= high - (n-1)
            high = n-1
        return [get_img_tensor(self.train_img_dir+img_id+'.png') for img_id in ct[low:high+1]]

    def get_nearby_slices_features(self, img_id, n_slices):
        """
        :img_id (str): the ID of the slice to find nearby slices for
        :n_slices (int): number of slices, both above and below, to return.
        
        :return: list of 2*n_slices + 1 feature vectors arranged in a 2D Tensor
        
        Gets the feature vectors of nearby slices rather than their raw images. Results in better performance.
        """
        ct_ind, ind = self.train_img_ct_ind[img_id]['ct_ind'], self.train_img_ct_ind[img_id]['ind']
        ct = self.train_img_ct[ct_ind]
        n = len(ct)
        low, high = ind-n_slices, ind+n_slices
        if low < 0:
            high += abs(low)
            low = 0
        elif high >= n:
            low -= high - (n-1)
            high = n-1

        res = []
        for img_id in ct[low:high+1]:
            try:
                res.append(np.load(self.feature_dir+img_id+'.npy'))
            except:
                pass
        return tf.squeeze(res)
        
    def precompute_features(self):
        """
        Passes all the images through the extractor model first before training and save them in self.feature_dir as 
        feature vectors.
        """
        return
        to_compute = set()
        present = set(x.split('.')[0] for x in os.listdir(self.feature_dir))
        print('Collecting necessary slices...')
        for img_id in tqdm(self.x):
            ct = self.train_img_ct[self.train_img_ct_ind[img_id]['ct_ind']]
            to_compute.update(ct)
            
        print(f'{len(to_compute.intersection(present))} feature vectors already present')
        to_compute = list(to_compute.difference(present))
        print(f'Computing {len(to_compute)} new feature vectors...')
        compute_batch_size = 200
        
        if len(to_compute) == 0:
            return
        
        for i in tqdm(range(ceil(len(to_compute)//compute_batch_size)+1)):
            batch_names = to_compute[i*compute_batch_size : (i+1)*compute_batch_size]
            batch = np.array(Parallel(n_jobs=-1, backend='threading')(delayed(get_img_tensor)(self.img_dir+img_id+'.png') for img_id in batch_names))
            try:
                batch = self.extractor.predict(batch)
            except Exception as e:
                print(e)
                sys.exit(1)
            for i,feat_vec in enumerate(batch):
                np.save(self.feature_dir+batch_names[i]+'.npy', feat_vec)
            
        
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x_names = [self.get_nearby_slices_names(img_id, self.n_slices) for img_id in batch_x]
        batch_y = tf.convert_to_tensor([np.array([self.y[img_id] for img_id in slices]).flatten() for slices in batch_x_names])
        batch_x = tf.convert_to_tensor([[np.squeeze(np.load(self.feature_dir+img_id+'.npy')) for img_id in slices] for slices in batch_x_names])
        return batch_x, batch_y
    
    def on_epoch_end(self):
        ind = np.random.choice(list(range(len(os.listdir(self.img_dir)))), size=self.train_cutoff, replace=False)
        self.x = [img_name.split('.')[0] for img_name in np.array(os.listdir(self.img_dir))[ind]]



class RSNAModel:
    def __init__(self, train_img_dir, train_label_path, train_ct_path, train_coord_path, base_model_path, feature_dir):
        self.train_img_dir = train_img_dir #directory containing all 3-channel PNG images of scans
        self.train_label_path = train_label_path #path to CSV file containing binary encodings of the labels
        self.train_ct_path = train_ct_path #path to CSV where each line lists all image IDs in a certain CT scan
        self.train_coord_path = train_coord_path #path to CSV where each line contains an image ID and its (x,y,z) coordinates in its CT 
        self.base_model_path = base_model_path #path to the base CNN model which will be used to extract features 
        self.feature_dir = feature_dir #directory containing .npy files which are feature vectors of each image. The directory may be incomplete;
                                        #if so, this model will automatically generate feature vectors as needed
    
        self.train_img_ct = None
        self.train_img_ct_ind = None
        self.labels = None
        self.extractor = None
        self.model = None
        self.callbacks = None
    
        self.assemble_ct_scans()
        self.retrieve_labels()
        self.retrieve_feature_extractor()
        
    def assemble_ct_scans(self):
        """
        Collects image IDs from the same CT scan and stores them in a dictionary in sorted order (increasing z-value)
        """
        self.train_img_ct = {} # scan index : list of image IDs in the scan
        self.train_img_ct_ind = {} #image ID : {"ct_ind": index of the CT scan this image belongs to (key in train_img_ct), "ind": index in the list of slices}
        i = 0
        train_img_coords = pd.read_csv(self.train_coord_path, index_col=0, names=['x','y','z'])
        def populate_ct_info(row,i):
            #takes in a list of Image IDs of slices in a CT scan
            row = row[1:]
            row.sort(key=lambda x: train_img_coords.loc[x]['z'])
            for slice_ind, img_id in enumerate(row):
                self.train_img_ct_ind[img_id] = {'ct_ind': i, 'ind': slice_ind}
            self.train_img_ct[i] = row

        print('\nAssembling CT scans\' metadata...')
        with open(self.train_ct_path) as scans:
            reader = csv.reader(scans, delimiter=',')
            Parallel(n_jobs=-1, backend='threading', require='sharedmem', batch_size=75)(delayed(populate_ct_info)(row,i) for i, row in tqdm(list(enumerate(list(reader)))))
            
    def retrieve_labels(self):
        """
        Retrieves labels from the train_label_path and stores them in a DataFrame
        """
        labels = pd.read_csv(self.train_label_path)
        self.labels = {l[0]: l[1:].astype(np.int8) for l in labels.to_numpy()}
        
    def retrieve_feature_extractor(self):
        """
        Retrieves the model to be used for feature extraction
        """
        base_model = keras.models.load_model(self.base_model_path)
        self.extractor = keras.models.Sequential(base_model.layers[:-1])
    
    def initialize_model(self, saved_model_path=None):
        """
        :saved_model_path (str): The directory containing files for the saved model, including 'saved_model.pb'. If None,
        this method will create a new model.
        
        Initialize the RNN model for training, either by creating a new one or using a previously
        saved model path.
        """
        if not saved_model_path: 
            #create a new model if no checkpoint was provided
            self.model = keras.Sequential([Bidirectional(LSTM(512, return_sequences=True, name='lstm0')),
                              Bidirectional(LSTM(512, return_sequences=True, name='lstm1')),
                              Bidirectional(GRU(256, return_sequences=True, name='gru0')),
                              Conv1D(6, 1, padding='same', activation='sigmoid'),
                              Flatten()
                             ])
            print('Created new model')
        else:
            self.model = keras.models.load_model(saved_model_path)
            print(f'Found model at {saved_model_path}')
        
        self.model.build(input_shape=(None, 19,2048))
        self.model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False), 
              metrics=['binary_accuracy', 
                       keras.metrics.AUC(multi_label=True, num_labels=114, from_logits=False),
                       keras.metrics.Precision(), keras.metrics.Recall()],
              optimizer=keras.optimizers.Nadam(learning_rate=3e-5))
        
    
    def train(self, batch_size=32, n_batches=2000, n_slices=9, n_epochs=2, save_path=None):
        if not self.model:
            print("ERROR: please call self.initialize_model first before training")
            return
        
        print(f'Beginning training: {n_batches} batches of size {batch_size} for {n_epochs} epochs')
        print(f'Will save model to {save_path} every epoch' if save_path else 'WARNING: no model save path specified; model will not be saved!')
        
        train_cutoff = batch_size*n_batches
        train_sequence = RSNASequence(self.labels, train_cutoff, batch_size, self.extractor, self.train_img_dir, n_slices,                                      self.train_img_ct, self.train_img_ct_ind, self.feature_dir)
        if save_path:
            self.callbacks = [keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                 save_weights_only=False,
                                                 verbose=1)]
        
        self.model.fit(x=train_sequence, epochs=n_epochs, callbacks=self.callbacks)
        


model = RSNAModel(train_img_dir, train_label_path, train_ct_path, train_coord_path, base_model_path, feature_dir='~/extracted-features/')
#set to None in order to train a new model from scratch, or set to the saved_model_path to retrieve a previous model.
# model.initialize_model(None)
model.initialize_model(saved_model_path)
model.train(save_path=saved_model_path, n_epochs=n_epochs)

# After training, another python file would be used to load the trained model and handle API calls.
