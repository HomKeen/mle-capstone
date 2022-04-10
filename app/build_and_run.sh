sudo docker image build -t app:latest .
sudo docker run -d -p 80:8008 --name rsna --gpus all app:latest
