# DeepMRSeg

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/abb2c20d73ed464180494bf2fed3f0eb)](https://app.codacy.com/gh/CBICA/DeepMRSeg?utm_source=github.com&utm_medium=referral&utm_content=CBICA/DeepMRSeg&utm_campaign=Badge_Grade_Settings)

DeepMRSeg is a Python-based package for MRI image segmentation. The package is developed and maintained by the [Center for Biomedical Image Computing and Analytics (CBICA)](https://www.cbica.upenn.edu/) at the University of Pennsylvania. As the name implies, main modules of DeepMRSeg are built upon Deep Learning models that perform a set of image segmentation steps on MRI scans.

DeepMRSeg aims to provide users a ***robust***, ***accurate*** and ***user-friendly*** toolset for performing common segmentation tasks in neuroimaging. In order to meet these challenges, the development of DeepMRSeg was guided by following set of fundamental principles:

* ***Efficient network architecture:*** DeepMRSeg uses a modified UNet architecture that combines an ensemble of learners for a robust segmentation  _[1]_.
* ***Model repository with extensively trained models:*** We provide a set of pre-trained models for various segmentation tasks. We applied model training using ***_very large and diverse MRI datasets_*** with carefully curated and verified ground-truth labels.
* ***Easy installation and application:*** Using a few simple commands, users can easily install DeepMRSeg on different platforms, download pre-trained models, and apply these models on their images.
* ***Extensibility:*** DeepMRSeg is built using a generic network architecture and a software package that allows extending it with minimal efforts. The model repository will grow in the future with regular addition of new models and tasks.

## Supported Platforms
We have tested DeepMRSeg on the following platforms: 
-   Windows 10 Enterprise x64
-   Ubuntu 18.04.3 , 20.04.2

## Prerequisities
-   [Python 3](https://www.python.org/downloads/)
-   If you prefer conda, you may install it from [here](https://www.anaconda.com/products/individual)

## Installation Instructions

### 1) Direct installation at default location 
```
git clone  https://github.com/CBICA/DeepMRSeg.git
cd DeepMRSeg
python setup.py install
```

### 2) Installation in conda environment
```
conda create --name DeepMRSeg python=3.7.9
conda activate DeepMRSeg
```
Then follow steps from [direct installation](#direct-installation-at-default-location)

## Usage

DeepMRSeg commands are called on the command prompt or on Anaconda prompt. After installation of the package, users can apply a segmentation task using a pre-trained model (testing), or train their own model using a custom training dataset (training). 

For training:

```
deepmrseg_train
```

For testing:

```
deepmrseg_test 
```


Pre-trained models are hosted in [DeepMRSeg-Models repository](https://github.com/CBICA/DeepMRSeg-Models). Users can download a model from GitHub, save it in a local folder, and indicate the model path when calling _deepmrseg_test_

, or use the command provided for downloading a model for a specific segmentation task:

```
deepmrseg_loadmodel --task [taskname]
```




## License

## How to cite DeepMRSeg

## Publications

[1] Doshi, Jimit, et al. _DeepMRSeg: A convolutional deep neural network for anatomy and abnormality segmentation on MR images._ arXiv preprint arXiv:1907.02110 (2019).

## Authors and Contributors

The DeepMRSeg package is currently developed and maintained by:

Others who contributed to the project are:

## Grant support

Development of the DeepMRSeg package is supported by the following grants:

## Disclaimer
-   The software has been designed for research purposes only and has neither been reviewed nor approved for clinical use by the Food and Drug Administration (FDA) or by any other federal/state agency.
-   This code (excluding dependent libraries) is governed by the license provided in https://www.med.upenn.edu/cbica/software-agreement.html unless otherwise specified.
-   By using DeepMRSeg, you agree to the license as stated in the [license file](https://github.com/CBICA/DeepMRSeg/blob/main/LICENSE).

## Contact
For more information, please contact <a href="mailto:software@cbica.upenn.edu">CBICA Software</a>.
