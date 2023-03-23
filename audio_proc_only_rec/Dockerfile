ARG BASE_CONTAINER=ubuntu
ARG UBUNTU_VERSION=20.04

FROM $BASE_CONTAINER:$UBUNTU_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 -y

RUN apt-get install -y python3 python3-pip

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY __init__.py /opt/__init__.py

COPY recorder.py /opt/recorder.py

COPY commands.py /opt/commands.py

ENTRYPOINT ["python3", "opt/commands.py"]
