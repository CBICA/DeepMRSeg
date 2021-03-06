#!/usr/bin/env python3
"""
@author: Ashish Singh
"""
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
	'tensorflow~=2.2.0',
	'tensorflow-addons==0.11.2',
	'dill', 
	'h5py==2.10.0', 
	'hyperopt', 
	'keras',
#	'pandas', 
	'numpy~=1.18.4',
	'protobuf',
	'pymongo',
	'scikit-learn',
#	'seaborn',
	'scipy==1.4.1',
	'nibabel',
	'resource'
	],
	entry_points = {
    'console_scripts': ['deepmrseg_train=DeepMRSeg.deepmrseg_train:_main', 
			'deepmrseg_test=DeepMRSeg.deepmrseg_test:_main'],
    },
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)

