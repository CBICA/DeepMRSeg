FROM tensorflow/tensorflow:2.3.3-gpu

LABEL IMAGE="deepmrseg"
LABEL VERSION="1.0.0.Alpha2"
LABEL CI_IGNORE="True"

RUN apt-key del 3bf863cc
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y \
    git \
    htop \
    zip \
    unzip \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install scipy==1.5.4 matplotlib==3.3.3 dill==0.3.4 h5py==2.10.0 hyperopt==0.2.5 keras==2.6.0 numpy==1.18.5 protobuf==3.17.3 pymongo==3.12.0 scikit-learn==0.24.2 nibabel==3.2.1 resource==0.2.1 networkx==2.5.1 pandas

RUN mkdir /DeepMRSeg-package

RUN git clone  https://github.com/CBICA/DeepMRSeg.git /DeepMRSeg-package

RUN cd /DeepMRSeg-package && python setup.py install

CMD ["/bin/bash"]
