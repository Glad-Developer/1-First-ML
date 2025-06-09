FROM python:3.8-slim-buster
WORKDIR /app

COPY requirements.txt /app/ 

RUN apt-get update && apt-get install unzip libsm6 libxext6 ffmpeg awscli && pip install -r requirements.txt

COPY . /app

CMD [ "python3","app.py" ]