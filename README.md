# DeepMRSeg

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/abb2c20d73ed464180494bf2fed3f0eb)](https://app.codacy.com/gh/CBICA/DeepMRSeg?utm_source=github.com&utm_medium=referral&utm_content=CBICA/DeepMRSeg&utm_campaign=Badge_Grade_Settings)

DeepMRSeg is a Python-based package for MRI image segmentation. The package is developed and maintained by the [Center for Biomedical Image Computing and Analytics (CBICA)](https://www.cbica.upenn.edu/) at the University of Pennsylvania. As the name implies, main modules of DeepMRSeg are built upon Deep Learning models that perform a set of image segmentation steps on MRI scans.

DeepMRSeg aims to provide users a ***robust***, ***accurate*** and ***user-friendly*** toolset for performing common segmentation tasks in neuroimaging. In order to meet these challenges, the development of DeepMRSeg was guided by the following set of fundamental principles:

* ***Efficient network architecture:*** DeepMRSeg uses a modified UNet architecture that combines an ensemble of learners for a robust segmentation  _[1]_.
* ***Model repository with extensively trained models:*** We provide a set of pre-trained models for various segmentation tasks. We applied model training using ***_very large and diverse MRI datasets_*** with carefully curated and verified ground-truth labels.
* ***Easy installation and application:*** Using a few simple commands, users can easily install DeepMRSeg on different platforms, download pre-trained models, and apply these models on their images. Most applications require no or minimal pre-processing; thus, users can directly apply them on raw scans.
* ***Extensibility:*** DeepMRSeg is built using a generic network architecture and a software package that allows extending it with minimal effort. The model repository will grow in the future with regular addition of new models and tasks.

## Supported Platforms
We have tested DeepMRSeg on the following platforms: 
-   Windows 10 Enterprise x64
-   Ubuntu 18.04.3 , 20.04.2
It may also work on other platforms.

## Prerequisities
-   [Python 3](https://www.python.org/downloads/)
-   If you prefer conda, you may install it from [here](https://www.anaconda.com/products/individual)

## Installation Instructions

#### 1) Direct installation at default location 
```
git clone  https://github.com/CBICA/DeepMRSeg.git
cd DeepMRSeg
python setup.py install #install DeepMRSeg and its dependencies
```

#### 2) Installation in conda environment
```
conda create --name DeepMRSeg python=3.7.9
conda activate DeepMRSeg
```
Then follow steps from [direct installation](#direct-installation-at-default-location)

## Usage

After installation of the package, users can call DeepMRSeg commands on the command prompt (or on Anaconda prompt).

#### Pre-trained models:

Pre-trained models for testing are hosted in [DeepMRSeg-Models repository](https://github.com/CBICA/DeepMRSeg-Models). Users can manually download a model from the model repository into a local folder.

Alternatively, the model can be downloaded to a pre-defined local folder (_~/.deepmrseg/trained_models_) automatically using the command:

```
deepmrseg_downloadmodel
```

#### Training and testing:

Users can train their own model using a custom training dataset (training):

```
deepmrseg_train
```

or apply a pre-trained model on their image(s) (testing):

```
deepmrseg_test 
```

Note that _deepmrseg_train_ and _deepmrseg_test_ are generic commands that allow users to run training and testing in an exhaustive way by supplying a set of user arguments.

#### Applying a task:

Alternatively, we provide a simplified interface for the application of a specific segmentation task on user data:

```
deepmrseg_apply
```

Note that _deepmrseg_apply_ is a wrapper to _deepmrseg_test_, which calls it with a pre-defined model automatically downloaded using _deepmrseg_downloadmodel_.

#### Examples:

We provide here few examples using minimal argument sets as a quick reference. These examples also show 3 possible I/O options provided for different use cases (single subject, batch processing using an image list and batch processing of images in a folder).

Please see the user manual (or call the command with the _-h_ option) for details of the complete command line arguments for _deepmrseg_train_ and _deepmrseg_test_.

```
# Download pre-trained models
deepmrseg_downloadmodel --model dlicv 		## Tissue segmentation model
deepmrseg_downloadmodel --model wmlesion  	## ROI segmentation model

# Segment single subject (single-modality task)
deepmrseg_apply --model dlicv --inImg subj1_T1.nii.gz --outImg subj1_T1_DLICV.nii.gz     

# Segment single subject (multi-modality task)
#   Img names for different modalities are entered as repeated args in the same order
#   used in model training
deepmrseg_apply --model wmlesion --inImg subj1_FL.nii.gz --inImg subj1_T1.nii.gz --outImg subj1_T1_WMLES.nii.gz     

# Batch processing of multiple subjects using a subject list
#   User provides a csv file with columns: ID,InputMod1,InputMod2,...,OutputImage
deepmrseg_apply --model dlicv --sList subjectList.csv

# Batch processing of multiple subjects in input folder 
#   Testing is applied individually to all images with the given suffix in the input folder
deepmrseg_apply --model dlicv --inDir myindir --outDir myoutdir --inSuff _T1.nii.gz --outSuff _DLICV.nii.gz

```

## License

## How to cite DeepMRSeg

## Publications

_[1] Doshi, Jimit, et al. DeepMRSeg: A convolutional deep neural network for anatomy and abnormality segmentation on MR images. arXiv preprint arXiv:1907.02110 (2019)_.

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
