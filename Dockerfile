FROM tensorflow/tensorflow:latest-gpu

ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN set -x && \
    apt update && \
    apt install -y --no-install-recommends \
        jupyter \
        git\
        wget\
        build-essential \
        apt-utils \
        ca-certificates \
        curl \
        software-properties-common \
        libopencv-dev \ 
        cmake \
        swig \
        wget \
        unzip \
        tmux

RUN pip3 install pip --upgrade

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /home/app/

RUN pip install pipenv


COPY requirements.txt ./tmp/requirements.txt
WORKDIR /tmp
RUN pip install -r requirements.txt

WORKDIR /home/app

