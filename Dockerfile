FROM ubuntu:18.04

# Install dependencies.
RUN apt-get -y update
RUN apt-get -y install gcc
RUN apt-get -y install git
RUN apt-get -y install wget
RUN apt-get -y install libosmesa6-dev
RUN apt-get -y install python3.6 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 --version
RUN which python3

# Install sumo.
RUN apt-get -y install cmake swig libgtest-dev
RUN apt-get -y install autoconf libtool pkg-config libgdal-dev libxerces-c-dev
RUN apt-get -y install libproj-dev libfox-1.6-dev libxml2-dev libxslt1-dev
RUN apt-get -y install build-essential curl unzip flex bison
RUN pip install cmake cython

RUN mkdir -p $HOME/sumo_binaries/bin
WORKDIR $HOME/sumo_binaries/bin
RUN wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.4.0/binaries-ubuntu1804.tar.xz
RUN tar -xf binaries-ubuntu1804.tar.xz
RUN rm binaries-ubuntu1804.tar.xz
RUN chmod +x *
ENV PATH "$HOME/sumo_binaries/bin:$PATH"
ENV SUMO_HOME "$HOME/sumo_binaries/bin"
RUN echo $SUMO_HOME
RUN echo $PATH
RUN sumo

WORKDIR /

# Install Flow.
RUN git clone https://username:password@github.com/CIRCLES-consortium/flow
WORKDIR $HOME/flow
RUN git checkout controller_env
RUN pip3 install --use-deprecated=legacy-resolver -e .

WORKDIR /

# Install il-traffic.
RUN git clone https://github.com/AboudyKreidieh/il-traffic
RUN mv il-traffic/* .
RUN pip3 install --use-deprecated=legacy-resolver -e .
