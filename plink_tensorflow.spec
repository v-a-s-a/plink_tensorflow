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
    apt-get install -y --force-yes python3-dev 

    # install python packages
    pip3 install --upgrade 
    pip3 install --upgrade scikit-learn dask[dataframe] ipdb
    pip3 install --no-cache-dir --upgrade pandas-plink
    pip3 install --upgrade tfp-nightly 
