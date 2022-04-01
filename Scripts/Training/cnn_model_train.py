"""
This module is for fine-tuning TensorFlow's ResNet101 model to predict the class of a single slice of a CT scan. 
Its results are not expected to be acceptable, but will serve as a feature extractor and input for another model.
"""

import tensorflow as tf
import pydicom as dcm
import pandas as pd
import numpy as np
import os
import math

from tensorflow import keras
from tensorflow.keras.layers import *
from PIL import Image


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
keras.mixed_precision.set_global_policy('mixed_float16')



png_path = '~/rsna-intracranial-hemorrhage-detection/stage_2_train_imgs/'
label_path = '~/rsna-intracranial-hemorrhage-detection/train_labels.csv'
model_save_path = '~/base-cnn-model/'

batch_size = 32 
#Training the whole dataset takes ~9 hours, so we cut it short. Set this to -1 to train on the whole dataset
#Before each epoch, we randomly select (without replacement) from the whole training dataset.
max_num_batches = 1200 
n_epochs = 3

labels = pd.read_csv(label_path)
labels = {l[0]: l[1:].astype(np.int8) for l in labels.to_numpy()}

#retrieve the pretrained model
inception = keras.applications.resnet.ResNet101(include_top=False, input_shape=(512,512,3), weights='imagenet', 
                                                       classes=6)

def get_img_tensor(img_path):
    return tf.convert_to_tensor(np.asarray(Image.open(img_path), dtype=np.float32) / 255.)

def get_center_and_width(dicom):
    return tuple([int(x[0]) if type(x) == dcm.multival.MultiValue else int(x) for x in [dicom.WindowCenter, dicom.WindowWidth]])
    
def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    return (img - mi) / (ma - mi)

def window_filter(img, center, width, slope, intercept):
    out = np.copy(img)
    out = out*slope + intercept
    lowest_visible = center - width//2
    highest_visible = center + width//2
    
    out[out < lowest_visible] = lowest_visible
    out[out > highest_visible] = highest_visible
    return normalize_minmax(out)
    
def standardize(img):
    m, std = img.mean(), img.std()
    return (img - m) / std


model = keras.models.Sequential()
model.add(inception)
model.add(GlobalAveragePooling2D())
model.add(Dense(6, activation='sigmoid'))

model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False), 
              metrics=['binary_accuracy', 
                       keras.metrics.AUC(multi_label=True, num_labels=6, from_logits=False),
                       keras.metrics.Precision(), keras.metrics.Recall()],
             optimizer=keras.optimizers.Adam(learning_rate=1e-4))

cp_callback = keras.callbacks.ModelCheckpoint(filepath='reproduce_training_2/checkpoint.ckpt',
                                                 save_weights_only=True,
                                                 verbose=1)


class RSNASequence(keras.utils.Sequence):
    def __init__(self, all_x, labels, img_dir, batch_size):
        """
        
        :all_x (list of str): list of image file names of each training vector (as a PNG) in the entire dataset
        :labels (dict): maps DICOM image IDs (str) to 1D np.array of bool of length 6, representing the label of each training image
        :img_dir (str): absolute directory containing the PNG images of each training image.
        :batch_size (int): number of training images to include in a single batch
        """
        self.x = all_x
        self.labels = labels
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.on_epoch_end() #compute the first set of training batches
        
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = [self.labels[img_id] for img_id in batch_x]
        
        return (tf.stack([get_img_tensor(self.img_dir+img_path+'.png') for img_path in batch_x], axis=0), 
               tf.convert_to_tensor(batch_y))
    
    def on_epoch_end(self):
        ind = np.random.choice(list(range(len(os.listdir(self.img_dir)))), size=train_cutoff, replace=False)
        self.x = [img_name.split('.')[0] for img_name in np.array(os.listdir(self.img_dir))[ind]]
    

total_train_vec = len(os.listdir(png_path))
train_cutoff = batch_size*max_num_batches if max_num_batches >= 0 else total_train_vec #number of data vectors to train on

ind = np.random.choice(list(range(total_train_vec)), size=train_cutoff, replace=False)
train_sequence = RSNASequence(np.array(os.listdir(png_path))[ind], labels, png_path, 32)


model.fit(x=train_sequence, epochs=n_epochs, callbacks=[cp_callback])
model.save(model_save_path)

"""
We can see that training on just a few thousand of the 700k+ DICOM training images yields a 96% accuracy and an AUC of 0.92 on the training dataset, using ResNet101 pretrained weights as a base and fine tuning using a custom softmax binary cross-entropy layer. These preliminary results are very promising, however it is unknown whether the model can generalize to the entire training set let alone the test set. Additionally, the recall of ~0.5 is rather low, reflecting the highly imbalanced nature of the dataset. However, I believe that I have succeeded in reproducing one part of the research that I have conducted- all solutions I found used a CNN with pretrained weights as the first stage for their model. While reproducing the entirety of their research would take up to a week of coding, debugging, and training (not to mention money); I believe that my results are sufficient for this capstone submission (which predicted a 5-20 hour timeframe). 

When testing full models, I plan to start by training at least 1 epoch on the full dataset with the CNN, then expirement with feeding the results and the extracted features (taken from the CNN's pennultimate layer) to feed into:
1. An ensemble method of boosting or bagging tree methods (as one of my sources did)
2. A RNN, perhaps a LSTM
3. Another CNN
4. An ensemble of RNNs and/or CNNs

Such experimentation will take lots of time and I worry that it will bring me above the $300 credit limit alloted to me for Paperspace.

"""
