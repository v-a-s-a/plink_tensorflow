Bootstrap: docker
From: tensorflow/tensorflow:latest

%setup
%test
%environment
%runscript
%post
    # prep
    apt-get -y update 
    apt-get install -y --force-yes make 
    apt-get install -y vim wget python3 python3-pip git 
    apt-get install -y --force-yes python-dev python-numpy python-matplotlib python-h5py 
    apt-get install -y --force-yes python-setuptools 
    pip install --upgrade pip 
    pip install --upgrade pandas 
    pip install --upgrade scikit-learn
    pip install --upgrade numpy

    # install python packages
    pip install --upgrade pandas-plink
    
