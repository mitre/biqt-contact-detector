FROM tensorflow/tensorflow:2.3.0-gpu as biqt_contactlens_trainer 

# Install python3.7
RUN apt-get update \
	&& apt-get install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget autoconf libbz2-dev liblzma-dev -y 

RUN wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
RUN tar xzf Python-3.7.9.tgz
RUN cd Python-3.7.9 \
    && ./configure --enable-shared --enable-optimizations \
    && make -j$(nproc) altinstall \
    && cd .. \
    && rm -rf Python-3.7.9.tgz Python-3.7.9 \
	&& ldconfig

COPY requirements.txt requirements.txt
RUN pip3.7 install -r requirements.txt


