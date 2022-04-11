FROM tensorflow/tensorflow:latest-gpu-jupyter

ADD requirements.txt requirements.txt
ADD app.py app.py
ADD ./rsna-rnn-model/ rsna-rnn-model/
ADD ./base-cnn-model base-cnn-model/

# Install required libraries
RUN pip install -r requirements.txt

EXPOSE 8008

# Start the server
CMD ["python", "app.py"]