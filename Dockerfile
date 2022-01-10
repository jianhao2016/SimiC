FROM satijalab/seurat:3.2.3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils \
    pkg-config build-essential python3 python3-pip python3-setuptools \
    python3-dev wget software-properties-common libssl-dev curl \
    libcurl4-openssl-dev liblapack-dev libopenblas-base libopenblas-dev \
    gfortran libpng-dev libcairo2-dev libxml2-dev libfontconfig1-dev \
    cmake libmagick++-dev libpq-dev libhdf5-dev git libgsl-dev locate

# installing python libraries
RUN pip3 install pandas==0.25.3
RUN pip3 install numpy==1.19.5
RUN pip3 install freetype-py==2.2.0
RUN pip3 install cvxpy==1.0.10
RUN pip3 install scikit-learn==0.22.2.post1
RUN pip3 install scipy==1.1.0
RUN pip3 install matplotlib==3.0.2
RUN pip3 install requests==2.21.0
RUN pip3 install jupyter==1.0.0
RUN pip3 install ipdb==0.11

WORKDIR /root
RUN git clone https://github.com/jianhao2016/SimiC.git
WORKDIR /root/SimiC

RUN python3 setup.py install

WORKDIR /root/SimiC/Tutorial/Data
RUN wget https://databank.illinois.edu/datafiles/78ic1/download
RUN rm clonalKinetics_Example_data_description.txt ClonalKinetics_filtered.DF_data_description.txt
RUN unzip download

RUN R -e "install.packages('hues')"

WORKDIR /root/SimiC

