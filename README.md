# RSNA Intracranial Hemorrhage (ICH) Prediction Model
Production repository for my UCSD Extension MLE course capstone project. Access the API here: 

<http://65.49.54.233>

For instructions on usage, see below.

## Overview

This repository hosts the scripts that I used to retrieve, clean, and process the data (courtesy of the [RSNA Intracranial Hemorrhage Detection Kaggle Dataset](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data), as well as train the model (in the "Scripts" folder). It also contains the Dockerfile and other necessary files for running my model with an API in a Docker container for getting predictions (in the "app" folder). Currently, my Docker container is hosted on a [Paperspace](https://www.paperspace.com/) CORE Virtual Machine, and is accessible to anyone on the internet. Additionally, you can use this repository to replicate my work with your own Docker container.

My RSNA model classifies into six classes, and a data matrix may be a member of multiple classes. There are five classes representing five subtypes of ICH: epidural, intraparenchymal, intraventricular, subarachnoid, and subdural; a sixth class called "Any" represents the presence of any type of ICH. The model returns a vector of length 6 with values between 0 and 1 representing probabilities; a probability value of >0.5 is interpreted as TRUE, i.e. the corresponding type of ICH is predicted to be present.

Note: The RSNA dataset contains around 750k training images and 122k testing, amounting to about 453GB of data. This is absolutely not going to fit on GitHub, so I provide instructions on how to download the data for yourself if desired.

## Recreating my project

My project used a GPU to train TensorFlow models, then saved these models in TensorFlow SavedModel format. Using Starlette (a lightweight ASGI framework, and alternative to Flask), a Python script runs inside the Docker container and provides the necessary API call actions.

### From my pretrained models

All of the files necessary for running my Docker container is in the "app" folder. The folder contains the Dockerfile, requirements.txt, the folders containing the necessary trained models in TensorFlow format, as well as some convenience shell scripts. Simply download the app folder and move it to wherever you want; I recommend a VM running on a cloud platform for easy access over the internet- but make sure that your VM allows HTTP and HTTPS traffic, otherwise this won't work. First, you will need the appropriate TensorFlow docker image. From the terminal, run:
`docker pull tensorflow/tensorflow:latest-gpu-jupyter`. Then, in the "app" folder call `./build_and_run.sh` from the terminal and the application will be spun up. If you want to stop the Docker container and remove its image, run `./remove_docker.sh` from the terminal.

### From scratch

If you wish to train from scratch, you can look in the "Scripts" folder. There are several requirements for training, as the dataset is vast and complex:
1. A CUDA-enabled GPU (the faster, the better)
2. TensorFlow 2 and the appropriate version of CUDA installed (depends on your GPU)
3. Python 3; latest versions of pip, numpy, pandas, Pydicom, pillow, tqdm, and joblib installed
4. At least 700GB of free disk space
5. At least 16GB of memory 

Your machine In a terminal on the machine you wish to train on, do the following tasks:
1. Run `kaggle competitions download -c rsna-intracranial-hemorrhage-detection` to download the zipped. IMPORTANT: if you don't yet have a Kaggle API token set up on your machine, refer to [the Kaggle docs](https://www.kaggle.com/docs/api) for more information.
2. Run `pip install unzip` and then `unzip rsna-intracranial-hemorrhage-detection.zip` to unzip the files into "rsna-intracranial-hemorrhage-detection"
3. Copy the "Scripts" folder onto your machine, in the same directory as "rsna-intracranial-hemorrhage-detection"
4. From the terminal, in the "Scripts/Preprocessing" folder, run `python extract_dicoms.py`, then `python create_labels.py`. This will extract the labels and necessary metadata, as well as process the data into PNG files. 
5. In the same directory, run `python cnn_model_train.py`. This will fine tune a ResNet-101 model and will serve as the feature extractor. Next, run `python create_feature_vectors.py` to use the tuned CNN model to save the extracted features of each data file into an NPY file. This will reduce the computational burden on the RNN model.
6. Finally, in the same directory, run `python train_model.py` to train the RNN model using the extracted features from the CNN model.
7. You will end up with the folders "base-cnn-model" and "rsna-rnn-model" which contain the CNN and RNN models, respectively, in TensorFlow SavedModel format. These folders are identical in function to the folders in the "app" folder with the same name in this repository.
8. Follow directions from the above section ("From my pretrained models") to deploy the model into a Docker container.

## Using the API

Using a browser, navigate to my VM's external IP:
<http://65.49.54.233>
Then, you should see the following simple GUI:

![alt text](https://github.com/HomKeen/mle-capstone/blob/main/gui-view.png)

The API accepts only 2 types of files, in specific format:
1. A DICOM file (.dcm) 
2. A PNG file (.png) in a 512x512 RGB image, where the 3 color channels represent the bone, subdural, and soft tissue windows assembled into one image (learn more about windowing in radiology [here](https://radiopaedia.org/articles/windowing-ct?lang=us)). 

If the user provides an invalid file type or a file with invalid contents, the API will return an error message prompting them to try again.

Simply click the "Choose File" button, upload the appropriate DICOM or PNG file from your local system, and then click the "Predict". The webpage should change and return your predictions, for example:

![alt text](https://github.com/HomKeen/mle-capstone/blob/main/prediction-results.png)

The API will return the raw probabilities returned by the model, the list of ICH predictions as booleans, the interpreted results, and the image you uploaded for reference. For example, the above CT slice was predicted to have a 0.7173 probability of having intraparenchymal ICH, interpreted as TRUE.
