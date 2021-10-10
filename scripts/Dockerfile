FROM python:3.7

COPY . /app
WORKDIR /app

RUN apt-get update && \
    apt-get -y install ffmpeg && \
    pip install --quiet --user --no-warn-script-location -r requirements.txt

CMD python dualmerger.py