Bootstrap: docker
From: index.docker.io/tensorflow/tensorflow:1.7.0-py3

%setup
%test
%environment
%runscript
%post
    # prep
    apt-get -y update 
    apt-get install -y vim wget git
    apt-get install -y python3-dev python3 python3-pip python-setuptools 

    # install python packages
    pip3 install --upgrade 
    pip3 install --upgrade scikit-learn dask[dataframe] ipdb
    pip3 install --no-cache-dir --upgrade pandas-plink
    pip3 install --upgrade tfp-nightly 
