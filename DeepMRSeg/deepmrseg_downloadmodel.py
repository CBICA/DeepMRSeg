#!/usr/bin/python3

# deepmrseg

import argparse as _argparse
import os as _os
import urllib.request
from urllib.parse import urlparse
import zipfile
import sys as _sys

##############################################################
## This is a dictionary that keeps the saved models for now
mdlurl = 'https://github.com/CBICA/DeepMRSeg-Models/raw/main/models'

modelDict = {}
modelDict['dlicv'] = mdlurl + '/DLICV/DeepMRSeg_DLICV_v1.0.zip'
modelDict['muse'] = mdlurl + '/MUSE/DeepMRSeg_MUSE_v1.0.zip'
modelDict['tissueseg'] = mdlurl + '/TissueSeg/DeepMRSeg_TissueSeg_v1.0.zip'

##############################################################

## Path to saved models
DEEPMRSEG = _os.path.expanduser(_os.path.join('~', '.deepmrseg'))
MDL_DIR = _os.path.join(DEEPMRSEG, 'trained_models')

def _main():
	"""Main program for the script to download pre-trained models."""
	
	argv = _sys.argv

	exeName = _os.path.basename(argv[0])

	descTxt = '{prog} downloads pre-trained models for DeepMRSeg'.format(prog=exeName)

	epilogTxt = '''Example:
  ## Download tissue segmentation model
  {prog} --model tissueseg
  '''.format(prog=exeName)

	parser = _argparse.ArgumentParser( formatter_class=_argparse.RawDescriptionHelpFormatter, \
		description=descTxt, epilog=epilogTxt )

	parser.add_argument("--model", default=None, type=str, \
		help=	'Name of the model. Options are: ' \
			+ '[' + ', '.join(modelDict.keys()) + ']. (REQUIRED)')

	inargs = parser.parse_args()

	## Check args
	if inargs.model is None:
		parser.print_help( _sys.stderr )
		print('ERROR: Missing required arg: model')
		_sys.exit(1)
	
	if inargs.model not in modelDict.keys():
		parser.print_help( _sys.stderr )
		print('\nERROR: Model not found: ' + inargs.model) 
		_sys.exit(1)

	## Download model
	mdlurl = modelDict[inargs.model]
	mdlfname = _os.path.basename(urlparse(mdlurl).path)
	outFile = _os.path.join(MDL_DIR , inargs.model, mdlfname)

	if _os.path.isdir(outFile.replace('.zip', '')):
		print("Model already downloaded: " + outFile.replace('.zip', ''))

	else:
		print("Loading model: " + inargs.model)

		outPath = _os.path.join(MDL_DIR , inargs.model)
		if not _os.path.exists(outPath):
			_os.makedirs(outPath)
			print('Created dir : ' + outPath)

		urllib.request.urlretrieve(mdlurl, outFile)
		print('Downloaded model : ' + outFile)

		with zipfile.ZipFile(outFile, 'r') as fzip:
			fzip.extractall(outPath)
			print('Unzipped model : ' + outFile.replace('.zip', ''))

		_os.remove(outFile)

if __name__ == "__main__":
	main()
