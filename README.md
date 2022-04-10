# RSNA Intracranial Hemorrhage (ICH) Prediction Model
Production repository for my UCSD Extension MLE course capstone project.

## Overview

This repository hosts the scripts that I used to retrieve, clean, and process the data (courtesy of the [RSNA Intracranial Hemorrhage Detection Kaggle Dataset](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data), as well as train the model (in the "Scripts" folder). It also contains the Dockerfile and other necessary files for running my model with an API in a Docker container for getting predictions (in the "app" folder). Currently, my Docker container is hosted on a [Paperspace](https://www.paperspace.com/) CORE Virtual Machine, and is accessible to anyone on the internet. Additionally, you can use this repository to replicate my work with your own Docker container.

My RSNA model classifies into six classes, and a data matrix may be a member of multiple classes. There are five classes representing five subtypes of ICH: epidural, intraparenchymal, intraventricular, subarachnoid, and subdural; a sixth class called "Any" represents the presence of any type of ICH. The model returns a vector of length 6 with values between 0 and 1 representing probabilities; a probability value of >0.5 is interpreted as TRUE, i.e. the corresponding type of ICH is predicted to be present.

## Using the API

Using a browser, navigate to my VM's external IP:
<http://65.49.54.233>
Then, you should see the following simple GUI:

![alt text](https://github.com/HomKeen/mle-capstone/blob/main/gui-view.png)

The API accepts only 2 types of files, in specific format:
1. A DICOM file (.dcm) 
2. A PNG file (.png) in a 512x512 RGB image, where the 3 color channels represent the bone, subdural, and soft tissue windows assembled into one image (learn more about windowing in radiology [here](https://radiopaedia.org/articles/windowing-ct?lang=us)).

Simply click the "Choose File" button, upload the appropriate DICOM or PNG file from your local system, and then click the "Predict". The webpage should change and return your predictions, for example:

![alt text](https://github.com/HomKeen/mle-capstone/blob/main/prediction-results.png)

The API will return the raw probabilities returned by the model, the list of ICH predictions as booleans, the interpreted results, and the image you uploaded for reference. For example, the above CT slice was predicted to have a 0.7173 probability of having intraparenchymal ICH, interpreted as TRUE.

## 
