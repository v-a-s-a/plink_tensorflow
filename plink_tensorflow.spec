Bootstrap: docker
From: tensorflow/tensorflow:latest-py3

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
    apt-get install -y --force-yes python3-tk
    #pip3 install --upgrade pip setuptools
    pip3 install --upgrade pandas 
    pip3 install --upgrade scikit-learn
    pip3 install --upgrade numpy

    # install python packages
    pip3 install --upgrade dask[dataframe]
    pip3 install --upgrade pandas-plink
    pip3 install --upgrade ipdb
    pip3 install --user --upgrade tfp-nightly 
    
