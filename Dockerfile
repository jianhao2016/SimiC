FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils pkg-config build-essential python3.6 python3-pip python3-setuptools python3-dev wget software-properties-common libssl-dev curl libcurl4-openssl-dev liblapack-dev libopenblas-base libopenblas-dev gfortran libpng-dev libcairo2-dev libxml2-dev libfontconfig1-dev libmagick++-dev libpq-dev libhdf5-dev

# Install R4
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"

RUN apt-get install -y --no-install-recommends r-base r-base-core r-recommended r-base-dev


ADD requirements.txt .
ADD requirements.r .

# installing python libraries
#RUN pip3 install -r requirements.txt
# installing r libraries
#RUN Rscript requirements.r


#COPY /home/tereshkova/data/gserranos/GraCa/Data/Nets/GoldStandard/NR/bind_orfhsa_drug_nr.txt /Data/NR/bind_orfhsa_drug_nr.txt
#RUN head /Data/bind_orfhsa_drug_nr.txt
# add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/


#  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9



# ** testing if installed package keeps a record of temporary installation path
# * DONE (Seurat)

# The downloaded source packages are in
# 	'/tmp/Rtmpp8BmwV/downloaded_packages'
# Warning messages:
# 1: In install.packages("Seurat", dependencies = TRUE) :
#   installation of package 'RcppGSL' had non-zero exit status
# 2: In install.packages("Seurat", dependencies = TRUE) :
#   installation of package 'hdf5r' had non-zero exit status
# 3: In install.packages("Seurat", dependencies = TRUE) :
#   installation of package 'RcppZiggurat' had non-zero exit status
# 4: In install.packages("Seurat", dependencies = TRUE) :
#   installation of package 'mutoss' had non-zero exit status
# 5: In install.packages("Seurat", dependencies = TRUE) :
#   installation of package 'Rfast' had non-zero exit status
# 6: In install.packages("Seurat", dependencies = TRUE) :
#   installation of package 'Rfast2' had non-zero exit status
# 7: In install.packages("Seurat", dependencies = TRUE) :
#   installation of package 'metap' had non-zero exit status