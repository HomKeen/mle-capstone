# RSNA Intracranial Hemorrhage (ICH) Prediction Model
Production repository for my UCSD Extension MLE course capstone project.

## Overview

This repository hosts the scripts that I used to retrieve, clean, and process the data (courtesy of the [RSNA Intracranial Hemorrhage Detection Kaggle Dataset](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data), as well as train the model (in the "Scripts" folder). It also contains the Dockerfile and other necessary files for running my model with an API in a Docker container for getting predictions (in the "app" folder).

Currently, my Docker container is hosted on a [Paperspace](https://www.paperspace.com/) CORE Virtual Machine, and is accessible to anyone on the internet. Additionally, you can use this repository to replicate my work with your own Docker container.

## Using the API

Using a browser, navigate to my VM's external IP:
<http://65.49.54.233>
Then, you should see the following simple GUI:

