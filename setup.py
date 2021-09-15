#!/usr/bin/env python3
"""Setup tool for DeepMRSeg."""
__author__ 	= 'Ashish Singh'

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DeepMRSeg",
    version="0.1.0",
    author="Jimit Doshi",
    author_email="software@cbica.upenn.edu",
    description="Deep Learning based MR image Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CBICA/DeepMRSeg",
    packages=setuptools.find_packages(),
	install_requires=[
	'scipy==1.5.4',
	'matplotlib==3.3.3',
	'tensorflow==2.3.3',
	'tensorflow-addons==0.13.0',
	'dill==0.3.4',
	'h5py==2.10.0',
	'hyperopt==0.2.5',
	'keras==2.6.0',
	'numpy==1.18.5',
	'protobuf==3.17.3',
	'pymongo==3.12.0',
	'scikit-learn==0.24.2',
	'nibabel==3.2.1',
	'resource==0.2.1',
	'networkx==2.5.1'
	],
	entry_points = {
    'console_scripts': ['deepmrseg_train=DeepMRSeg.deepmrseg_train:_main', 'deepmrseg_test=DeepMRSeg.deepmrseg_test:_main', 'deepmrseg_downloadmodel=DeepMRSeg.deepmrseg_downloadmodel:_main', 'deepmrseg_apply=DeepMRSeg.deepmrseg_apply:_main'],
    },
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)

